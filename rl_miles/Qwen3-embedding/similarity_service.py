#!/usr/bin/env python3
"""High-throughput similarity API backed by independent vLLM replicas.

External callers still send one HTTP request per pair:

    POST /similarity {"text1": "...", "text2": "..."}

The service uses a short collection window to batch nearby pairs internally.
Each internal batch is sent to one TP=1 replica, selected by the smallest
queue plus active-batch load. This keeps the public API independent-request
friendly while reducing HTTP and scheduler overhead for short similarity
queries.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

for key in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
):
    os.environ.pop(key, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"


def parse_endpoints() -> list[str]:
    raw = os.environ.get("EMBEDDING_API_BASES", "")
    if not raw:
        raw = os.environ.get(
            "EMBEDDING_API_BASE",
            "http://127.0.0.1:8200,http://127.0.0.1:8201,"
            "http://127.0.0.1:8202,http://127.0.0.1:8203",
        )
    endpoints = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    if not endpoints:
        raise RuntimeError("EMBEDDING_API_BASES does not contain an endpoint")
    return endpoints


MODEL = os.environ.get("EMBEDDING_MODEL_NAME", "Qwen3-8B-Embedding")
MAX_BATCH = max(1, int(os.environ.get("SIMILARITY_MAX_BATCH", "128")))
BATCH_TIMEOUT = max(0.0, float(os.environ.get("SIMILARITY_BATCH_TIMEOUT", "0.005")))
QUEUE_LIMIT = max(1, int(os.environ.get("SIMILARITY_QUEUE_LIMIT", "5000")))
REQUEST_TIMEOUT = max(10.0, float(os.environ.get("SIMILARITY_REQUEST_TIMEOUT", "1200")))
TRUNCATE_PROMPT_TOKENS = int(os.environ.get("TRUNCATE_PROMPT_TOKENS", "32768"))
TRUNCATION_SIDE = os.environ.get("TRUNCATION_SIDE", "right")


class SimilarityRequest(BaseModel):
    text1: str = Field(min_length=1)
    text2: str = Field(min_length=1)


class SimilarityResponse(BaseModel):
    similarity: float
    dimension: int


class _Item:
    __slots__ = ("text1", "text2", "future")

    def __init__(self, text1: str, text2: str, future: asyncio.Future[Any]):
        self.text1 = text1
        self.text2 = text2
        self.future = future


@dataclass
class _Worker:
    endpoint: str
    queue: asyncio.Queue[_Item]
    client: httpx.AsyncClient
    active_batches: int = 0
    completed_batches: int = 0
    completed_items: int = 0

    @property
    def load(self) -> int:
        return self.queue.qsize() + self.active_batches * MAX_BATCH


workers: list[_Worker] = []
worker_tasks: list[asyncio.Task[None]] = []
worker_lock = asyncio.Lock()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding dimensions do not match")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return min(1.0, max(0.0, dot / (left_norm * right_norm)))


def dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


async def choose_worker() -> _Worker:
    async with worker_lock:
        if not workers:
            raise HTTPException(status_code=503, detail="embedding workers are unavailable")
        return min(workers, key=lambda worker: worker.load)


async def worker_loop(worker: _Worker) -> None:
    loop = asyncio.get_running_loop()
    while True:
        first = await worker.queue.get()
        items = [first]
        deadline = loop.time() + BATCH_TIMEOUT
        while len(items) < MAX_BATCH:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                items.append(await asyncio.wait_for(worker.queue.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break

        worker.active_batches += 1
        try:
            texts: list[str] = []
            for item in items:
                texts.extend((item.text1, item.text2))
            payload = {"model": MODEL, "input": texts, "encoding_format": "float"}
            if TRUNCATE_PROMPT_TOKENS:
                payload["truncate_prompt_tokens"] = TRUNCATE_PROMPT_TOKENS
                if TRUNCATION_SIDE:
                    payload["truncation_side"] = TRUNCATION_SIDE
            response = await worker.client.post(
                "/v1/embeddings",
                json=payload,
            )
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda item: item["index"])
            if len(data) != len(texts):
                raise RuntimeError(
                    f"embedding response length {len(data)} != request length {len(texts)}"
                )
            for index, item in enumerate(items):
                left = data[index * 2]["embedding"]
                right = data[index * 2 + 1]["embedding"]
                result = SimilarityResponse(
                    similarity=cosine_similarity(left, right),
                    dimension=len(left),
                )
                if not item.future.done():
                    item.future.set_result(dump_model(result))
            worker.completed_batches += 1
            worker.completed_items += len(items)
        except Exception as exc:  # noqa: BLE001 - propagate to every waiter
            for item in items:
                if not item.future.done():
                    item.future.set_exception(exc)
        finally:
            worker.active_batches -= 1
            for _ in items:
                worker.queue.task_done()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global workers, worker_tasks
    limits = httpx.Limits(
        max_connections=MAX_BATCH + 8,
        max_keepalive_connections=MAX_BATCH + 8,
        keepalive_expiry=60.0,
    )
    endpoints = parse_endpoints()
    workers = [
        _Worker(
            endpoint=endpoint,
            queue=asyncio.Queue(maxsize=QUEUE_LIMIT),
            client=httpx.AsyncClient(
                base_url=endpoint,
                timeout=httpx.Timeout(
                    REQUEST_TIMEOUT,
                    connect=10.0,
                    read=REQUEST_TIMEOUT,
                    write=REQUEST_TIMEOUT,
                    pool=REQUEST_TIMEOUT,
                ),
                limits=limits,
                trust_env=False,
            ),
        )
        for endpoint in endpoints
    ]
    worker_tasks = [
        asyncio.create_task(worker_loop(worker), name=f"similarity-worker-{index}")
        for index, worker in enumerate(workers)
    ]
    yield
    for task in worker_tasks:
        task.cancel()
    await asyncio.gather(*worker_tasks, return_exceptions=True)
    for worker in workers:
        await worker.client.aclose()
    workers = []
    worker_tasks = []


app = FastAPI(title="Qwen3-8B-Embedding similarity service", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok" if workers else "starting",
        "model": MODEL,
        "workers": [
            {
                "endpoint": worker.endpoint,
                "queue": worker.queue.qsize(),
                "active_batches": worker.active_batches,
                "completed_items": worker.completed_items,
            }
            for worker in workers
        ],
    }


@app.post("/similarity", response_model=SimilarityResponse)
async def similarity(request: SimilarityRequest) -> dict[str, Any]:
    worker = await choose_worker()
    if worker.queue.full():
        raise HTTPException(status_code=429, detail="similarity queue is full")
    future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    item = _Item(request.text1, request.text2, future)
    await worker.queue.put(item)
    try:
        return await future
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - convert backend failures to HTTP
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("SIMILARITY_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("SIMILARITY_PORT", "8100"))
    )
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
