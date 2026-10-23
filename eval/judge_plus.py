#!/usr/bin/env python3
"""Judge existing predictions with an API judge and write judge-plus.jsonl.

No rollout is performed. Each run_dir/predictions.jsonl is consumed and a
sibling judge-plus.jsonl is produced.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from eval_config import DEFAULT_EVAL_DATA, DEFAULT_JUDGE_MODEL, PROJECT_ROOT
from run_eval import (
    CORE_DIMENSIONS,
    DEFAULT_WEIGHTS,
    _disable_proxy_env,
    _judge_one,
    weighted_score,
)

EVAL_ROOT = Path(__file__).resolve().parent
DEFAULT_TOKENIZER = PROJECT_ROOT / "models" / "Qwen3.5-4B"
DEFAULT_MAX_JUDGE_TOKENS = 16_384
DEFAULT_CONCURRENCY = 64
DEFAULT_JUDGE_MAX_TOKENS = 4096


@dataclass
class JudgeSample:
    index: int
    domain: str
    response: str
    ground_truth: str
    judge_scores: dict[str, float] = field(default_factory=dict)
    judge_weighted_score: float = 0.0
    judge_reward: float = 0.0
    judge_raw_output: str = ""
    judge_error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    parser.add_argument("--eval-data", type=Path, default=DEFAULT_EVAL_DATA)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--api-base-url", default=os.environ.get("JUDGE_PLUS_API_BASE_URL", ""))
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_PLUS_MODEL", DEFAULT_JUDGE_MODEL))
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--judge-max-tokens", type=int, default=DEFAULT_JUDGE_MAX_TOKENS)
    parser.add_argument("--max-judge-tokens", type=int, default=DEFAULT_MAX_JUDGE_TOKENS)
    parser.add_argument("--dirs", nargs="*", default=None, help="optional run_dir names to process instead of auto-discovering")
    parser.add_argument("--limit", type=int, default=None, help="smoke-test: judge only first N predictions per directory")
    parser.add_argument("--output-name", default="judge-plus.jsonl")
    parser.add_argument("--shard-dir-name", default="judge_plus_shards")
    args = parser.parse_args()
    if not args.api_base_url.strip():
        parser.error("Provide --api-base-url or JUDGE_PLUS_API_BASE_URL for your judge service")
    return args


def load_tokenizer(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"tokenizer not found: {path}")
    try:
        return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)


def truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    if not text or max_tokens <= 0:
        return text
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=False)


def clean_thinking_text(text: str) -> str:
    # Preserve the actual observation block when present; otherwise remove
    # paired thinking blocks and stray think tags without discarding any
    # non-thinking content.
    marker = "###[Observation]"
    if marker in text:
        return text[text.find(marker) :].strip()
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def load_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def load_shards(shard_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    all_shards: dict[int, dict[str, Any]] = {}
    successful: dict[int, dict[str, Any]] = {}
    if not shard_dir.exists():
        return successful, all_shards
    for path in sorted(shard_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            index = int(record.get("index", -1))
        except (TypeError, ValueError):
            continue
        all_shards[index] = record
        if record.get("judge_scores"):
            successful[index] = record
    return successful, all_shards


def make_judge_record(prediction: dict[str, Any], sample: JudgeSample) -> dict[str, Any]:
    record = dict(prediction)
    record["response"] = sample.response
    record["judge_scores"] = sample.judge_scores
    record["judge_weighted_score"] = sample.judge_weighted_score
    record["judge_reward"] = sample.judge_reward
    record["judge_raw_output"] = sample.judge_raw_output
    record["judge_error"] = sample.judge_error
    record["judge_fallback"] = bool(record.get("judge_fallback", False))
    return record


def write_shard(shard_dir: Path, record: dict[str, Any]) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    index = int(record.get("index", 0))
    path = shard_dir / f"{index:08d}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_final(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: int(item.get("index", 0))):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


async def judge_directory(
    args: argparse.Namespace,
    tokenizer,
    meta_by_index: dict[int, dict[str, Any]],
    client: AsyncOpenAI,
    run_dir: Path,
) -> None:
    prediction_path = run_dir / "predictions.jsonl"
    output_path = run_dir / args.output_name
    shard_dir = run_dir / args.shard_dir_name
    if not prediction_path.exists():
        print(f"[judge-plus] skip missing predictions {run_dir}", flush=True)
        return

    predictions = load_records(prediction_path, args.limit)
    successful, all_existing = load_shards(shard_dir)
    pending_predictions = [
        item for item in predictions if int(item.get("index", -1)) not in successful
    ]
    print(
        f"[judge-plus] start dir={run_dir} samples={len(predictions)} "
        f"existing={len(successful)} pending={len(pending_predictions)}",
        flush=True,
    )

    if pending_predictions:
        samples: list[JudgeSample] = []
        predictions_by_index: dict[int, dict[str, Any]] = {}
        for prediction in pending_predictions:
            index = int(prediction.get("index", -1))
            predictions_by_index[index] = prediction
            meta = meta_by_index.get(index) or {}
            response = clean_thinking_text(str(prediction.get("response") or ""))
            response = truncate_to_tokens(tokenizer, response, args.max_judge_tokens)
            ground_truth = truncate_to_tokens(
                tokenizer,
                str(meta.get("ground_truth") or ""),
                args.max_judge_tokens,
            )
            samples.append(
                JudgeSample(
                    index=index,
                    domain=str(meta.get("domain") or "web"),
                    response=response,
                    ground_truth=ground_truth,
                )
            )

        judge_args = SimpleNamespace(
            judge_model=args.judge_model,
            judge_max_tokens=args.judge_max_tokens,
            judge_concurrency=args.concurrency,
        )
        semaphore = asyncio.Semaphore(args.concurrency)
        base_urls = [args.api_base_url.rstrip("/")]
        clients = [client]

        done_count = 0
        total_pending = len(pending_predictions)

        async def guarded(sample: JudgeSample, counter: int) -> tuple[JudgeSample, dict[str, Any]]:
            result = await _judge_one(clients, base_urls, judge_args, semaphore, sample, counter)
            record = make_judge_record(predictions_by_index[sample.index], result)
            write_shard(shard_dir, record)
            return result, record

        tasks = [
            asyncio.create_task(guarded(sample, counter))
            for counter, sample in enumerate(samples)
        ]
        for task in asyncio.as_completed(tasks):
            await task
            done_count += 1
            if done_count % 25 == 0 or done_count == total_pending:
                print(
                    f"[judge-plus] dir={run_dir.name} judged {done_count}/{total_pending}",
                    flush=True,
                )

    _, all_existing = load_shards(shard_dir)
    records_by_index: dict[int, dict[str, Any]] = {}
    for item in predictions:
        records_by_index[int(item.get("index", -1))] = dict(item)
    for index, record in all_existing.items():
        records_by_index[index] = record
    final_records = [records_by_index[index] for index in sorted(records_by_index)]
    write_final(output_path, final_records)
    print(
        f"[judge-plus] complete dir={run_dir} output={output_path} rows={len(final_records)}",
        flush=True,
    )


async def async_main(args: argparse.Namespace) -> int:
    tokenizer = load_tokenizer(args.tokenizer)
    meta_records = load_records(args.eval_data)
    meta_by_index = {
        int(record.get("index", index)): record
        for index, record in enumerate(meta_records)
    }
    print(
        f"[judge-plus] eval_data={args.eval_data} meta_rows={len(meta_records)} "
        f"tokenizer={args.tokenizer}",
        flush=True,
    )

    timeout = httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=60.0)
    http_client = httpx.AsyncClient(timeout=timeout, trust_env=False)
    client = AsyncOpenAI(
        base_url=args.api_base_url.rstrip("/"),
        api_key=os.environ.get("JUDGE_PLUS_API_KEY", "EMPTY"),
        max_retries=0,
        http_client=http_client,
    )

    if args.dirs:
        run_dirs = [Path(item) for item in args.dirs]
    else:
        run_dirs = sorted(
            path.parent
            for path in args.eval_root.rglob("predictions.jsonl")
            if "4b-lite" not in path.parts
        )

    print(
        f"[judge-plus] judge_model={args.judge_model} api_base_url={args.api_base_url} "
        f"concurrency={args.concurrency} dirs={len(run_dirs)}",
        flush=True,
    )
    for idx, run_dir in enumerate(run_dirs, 1):
        try:
            await judge_directory(args, tokenizer, meta_by_index, client, run_dir)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[judge-plus] error dir={run_dir} {type(exc).__name__}: {exc}",
                flush=True,
            )
        print(f"[judge-plus] progress dirs {idx}/{len(run_dirs)}", flush=True)

    await client.close()
    await http_client.aclose()
    print("[judge-plus] all_done", flush=True)
    return 0


def main() -> int:
    _disable_proxy_env()
    args = parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
