#!/usr/bin/env python3
"""Formal vLLM world-model evaluation runner.

Pipeline:
  1. Build system+prompt+``###[Observation]`` with the model chat template.
  2. Truncate the user turn from the left when the rendered prompt exceeds the
     configured input token budget.
  3. Serve the selected checkpoint with vLLM on the configured local GPUs and
     call the OpenAI-compatible ``/v1/completions`` endpoint. Generation stops
     on ``<|im_end|>`` (token id 248046) and the configured output limit.
  4. Judge every prediction against its ground-truth observation with the
     remote OpenAI-compatible LLM judge service.

This module is self-contained: it vendors judge prompts under ``prompts/`` and
does not import the training codebase at runtime.
"""

from __future__ import annotations

import argparse
from thinking import parse_think_mode
import asyncio
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import httpx
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from eval_config import (
    CORE_DIMENSIONS,
    DEFAULT_EVAL_DATA,
    DEFAULT_JUDGE_API_KEY,
    DEFAULT_JUDGE_BASE_URL,
    DEFAULT_JUDGE_CONCURRENCY,
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_TIMEOUT,
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_MAX_RESPONSE_TOKENS,
    DEFAULT_PROMPT_SAFETY_MARGIN,
    DEFAULT_REQUEST_CONCURRENCY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_VLLM_BIN,
    DEFAULT_WEIGHTS,
    MODELS,
    OBS_MARKER,
    OUTPUT_ROOT,
    PROMPT_ROOT,
    ModelSpec,
)


@dataclass
class EvalSample:
    index: int
    system: str
    prompt_text: str
    ground_truth: str
    rendered_prompt: str
    source: Any = None
    domain: str = ""
    dataset: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    prompt_truncated: bool = False
    response: str = ""
    cot: str = ""
    response_length: int = 0
    status: str = "pending"
    finish_reason: str = ""
    error: str = ""
    judge_scores: dict[str, float] = field(default_factory=dict)
    judge_weighted_score: float = 0.0
    judge_reward: float = 0.0
    judge_raw_output: str = ""
    judge_error: str = ""

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.pop("system", None)
        record.pop("prompt_text", None)
        record.pop("ground_truth", None)
        record.pop("source", None)
        record.pop("domain", None)
        record.pop("dataset", None)
        record.pop("messages", None)
        record.pop("prompt_truncated", None)
        record.pop("rendered_prompt", None)
        return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="Qwen3.5-4B")
    parser.add_argument("--checkpoint", type=Path, default=None, help="override model checkpoint/HF directory")
    parser.add_argument("--template-model", type=Path, default=None, help="override tokenizer/template model directory")
    parser.add_argument("--eval-data", type=Path, default=DEFAULT_EVAL_DATA)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--base-port", type=int, default=38000)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    parser.add_argument("--max-response-tokens", type=int, default=DEFAULT_MAX_RESPONSE_TOKENS)
    parser.add_argument("--prompt-safety-margin", type=int, default=DEFAULT_PROMPT_SAFETY_MARGIN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--dtype", default=None, help="optional vLLM dtype override, e.g. auto, half, bfloat16")
    parser.add_argument("--kv-cache-dtype", default=None, help="optional vLLM KV cache dtype override, e.g. fp8")
    parser.add_argument("--quantization", default=None, help="optional vLLM quantization override, e.g. fp8")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--request-concurrency", type=int, default=DEFAULT_REQUEST_CONCURRENCY)
    parser.add_argument("--server-timeout", type=int, default=3600)
    parser.add_argument("--rollout-base-url", default=os.environ.get("ROLLOUT_BASE_URL", ""), help="OpenAI-compatible rollout server base URL; if set, skip local server launch")
    parser.add_argument("--rollout-model-name", default=os.environ.get("ROLLOUT_MODEL_NAME", ""), help="model name sent to the rollout server")
    parser.add_argument("--judge-base-url", default=os.environ.get("JUDGE_BASE_URL", DEFAULT_JUDGE_BASE_URL))
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    parser.add_argument("--judge-api-key", default=os.environ.get("JUDGE_API_KEY", DEFAULT_JUDGE_API_KEY))
    parser.add_argument("--judge-max-tokens", type=int, default=int(os.environ.get("JUDGE_MAX_TOKENS", DEFAULT_JUDGE_MAX_TOKENS)))
    parser.add_argument("--judge-concurrency", type=int, default=int(os.environ.get("JUDGE_CONCURRENCY", DEFAULT_JUDGE_CONCURRENCY)))
    parser.add_argument("--vllm-bin", default=os.environ.get("VLLM_BIN", str(DEFAULT_VLLM_BIN)))
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="keep existing per-sample shards and skip completed indices")
    parser.add_argument("--run-name", default=None, help="override the output subdirectory name")
    parser.add_argument("--think-mode", type=parse_think_mode, choices=("auto", "on", "off"), default=os.environ.get("THINK_MODE", "auto"), help="CoT generation: on enables thinking, off disables it, auto uses model defaults; independent of judge thinking")
    parser.set_defaults(inline_judge=True)
    parser.add_argument("--inline-judge", dest="inline_judge", action="store_true", help="judge each sample immediately after generation")
    parser.add_argument("--no-inline-judge", dest="inline_judge", action="store_false", help="disable per-sample inline judging")
    return parser.parse_args()


def _disable_proxy_env() -> None:
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def _find_free_port(start: int) -> int:
    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"no free port found starting at {start}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_no, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[eval] skipping invalid JSON at {path}:{line_no}: {exc}", file=sys.stderr)
    return rows


def ensure_observation_marker(prompt: str) -> str:
    if not prompt:
        return OBS_MARKER + "\n"
    if prompt.rstrip().endswith(OBS_MARKER):
        return prompt
    suffix = "\n" if not prompt.endswith("\n") else ""
    return f"{prompt}{suffix}{OBS_MARKER}\n"


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _render_completion_prompt(
    tokenizer,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
    append_assistant_prompt: bool = True,
) -> str:
    rendered = str(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    )
    if append_assistant_prompt and not add_generation_prompt and not rendered.endswith("<|im_start|>assistant\n"):
        rendered += "<|im_start|>assistant\n"
    return rendered


def _truncate_user_message(
    tokenizer,
    system: str,
    user: str,
    max_tokens: int,
    add_generation_prompt: bool = False,
    append_assistant_prompt: bool = True,
) -> tuple[list[dict[str, str]], bool]:
    user_text = ensure_observation_marker(user)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]
    rendered = _render_completion_prompt(tokenizer, messages, add_generation_prompt, append_assistant_prompt)
    if _token_count(tokenizer, rendered) <= max_tokens:
        return messages, False

    marker = OBS_MARKER + "\n"
    if user_text.rstrip().endswith(OBS_MARKER):
        marker_start = user_text.rfind(OBS_MARKER)
        body = user_text[:marker_start]
        suffix = user_text[marker_start:]
        if not suffix.endswith("\n"):
            suffix += "\n"
    else:
        body = user_text
        suffix = ""

    system_ids = tokenizer.encode(system, add_special_tokens=False)
    body_ids = tokenizer.encode(body, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix or marker, add_special_tokens=False)
    overhead = _token_count(tokenizer, rendered) - len(system_ids) - len(body_ids) - len(suffix_ids)
    max_body_tokens = max(1, max_tokens - len(system_ids) - len(suffix_ids) - overhead)

    kept = body_ids[-max_body_tokens:]
    kept_text = tokenizer.decode(kept, skip_special_tokens=False)
    for _ in range(8):
        candidate_user = kept_text + (suffix or marker)
        candidate_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": candidate_user},
        ]
        candidate_rendered = _render_completion_prompt(
            tokenizer,
            candidate_messages,
            add_generation_prompt,
            append_assistant_prompt,
        )
        if _token_count(tokenizer, candidate_rendered) <= max_tokens or len(kept) <= 1:
            return candidate_messages, True
        excess = _token_count(tokenizer, candidate_rendered) - max_tokens
        drop = min(len(kept) - 1, excess + max(1, len(kept) // 50))
        kept = kept[drop:]
        kept_text = tokenizer.decode(kept, skip_special_tokens=False)

    candidate_user = kept_text + (suffix or marker)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": candidate_user},
    ], True


def build_samples(
    records: list[dict[str, Any]],
    tokenizer,
    max_prompt_tokens: int,
    add_generation_prompt: bool = False,
    append_assistant_prompt: bool = True,
) -> list[EvalSample]:
    samples: list[EvalSample] = []
    for index, record in enumerate(records):
        system = str(record.get("system", "") or "")
        prompt = str(record.get("prompt", "") or "")
        ground_truth = str(record.get("ground_truth", "") or "")
        messages, truncated = _truncate_user_message(
            tokenizer,
            system,
            prompt,
            max_prompt_tokens,
            add_generation_prompt,
            append_assistant_prompt,
        )
        rendered_prompt = _render_completion_prompt(
            tokenizer,
            messages,
            add_generation_prompt,
            append_assistant_prompt,
        )
        samples.append(
            EvalSample(
                index=index,
                system=system,
                prompt_text=prompt,
                ground_truth=ground_truth,
                rendered_prompt=rendered_prompt,
                source=record.get("source"),
                domain=str(record.get("domain", "") or ""),
                dataset=str(record.get("dataset", "") or ""),
                messages=messages,
                prompt_truncated=truncated,
            )
        )
    return samples


def _server_command(args: argparse.Namespace, spec: ModelSpec, port: int) -> list[str]:
    vllm_bin = args.vllm_bin
    if not vllm_bin:
        vllm_bin = shutil.which("vllm") or "vllm"
    engine = spec.engine
    command = [
        str(vllm_bin),
        "serve",
        str(spec.checkpoint),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        spec.name,
        "--tensor-parallel-size",
        str(args.num_gpus or engine.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len or engine.max_model_len),
        "--trust-remote-code",
        "--generation-config",
        "vllm",
    ]
    command.extend(engine.extra_server_args)
    if args.dtype:
        command.extend(["--dtype", args.dtype])
    if args.kv_cache_dtype:
        command.extend(["--kv-cache-dtype", args.kv_cache_dtype])
    if args.quantization:
        command.extend(["--quantization", args.quantization])
    return command


def _is_agentworld_model(spec: ModelSpec) -> bool:
    lowered = spec.name.lower()
    return "agentworld" in lowered or "agent-world" in lowered


def _uses_generation_prompt(spec: ModelSpec) -> bool:
    lowered = spec.name.lower()
    return _is_agentworld_model(spec) or lowered.startswith(("deepseek-v4-flash", "glm-"))


def _append_assistant_prompt(spec: ModelSpec) -> bool:
    return not spec.name.lower().startswith("glm-")


def _uses_qwen_chat_completion(spec: ModelSpec) -> bool:
    return _is_agentworld_model(spec) or spec.name.startswith(("Qwen3.5", "Qwen3.8", "DeepSeek-V4-Flash-0731"))


def _generation_thinking_enabled(args: argparse.Namespace, spec: ModelSpec) -> bool:
    if args.think_mode == "on":
        return True
    if args.think_mode == "off":
        return False
    return _is_agentworld_model(spec) or ("122b" in spec.name.lower() and "a10b" in spec.name.lower())


def _generation_extra_body(args: argparse.Namespace, spec: ModelSpec) -> dict[str, Any]:
    extra_body: dict[str, Any] = {}
    if args.top_k > 0 and not spec.name.startswith("DeepSeek-V4-Flash-0731"):
        extra_body["top_k"] = args.top_k
    if getattr(args, "repetition_penalty", 1.0) not in (None, 1.0):
        extra_body["repetition_penalty"] = args.repetition_penalty
    if spec.stop_token_ids:
        extra_body["stop_token_ids"] = list(spec.stop_token_ids)
    if _uses_qwen_chat_completion(spec):
        extra_body["chat_template_kwargs"] = {
            "enable_thinking": _generation_thinking_enabled(args, spec)
        }
    else:
        extra_body.update(
            {
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                "add_special_tokens": False,
            }
        )
    return extra_body
def _launch_server(args: argparse.Namespace, spec: ModelSpec, port: int, log_dir: Path) -> subprocess.Popen:
    log_path = log_dir / f"vllm_{spec.name}.log"
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("NO_PROXY", "localhost,127.0.0.1,0.0.0.0")
    env.setdefault("no_proxy", env["NO_PROXY"])
    env.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    num_gpus = args.num_gpus or spec.engine.tensor_parallel_size
    env.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(num_gpus)))
    env.setdefault("VLLM_USE_V1", "1")
    env.setdefault("VLLM_DISABLE_LOG_STATS", "0")
    command = _server_command(args, spec, port)
    print(f"[eval] launching vLLM: {' '.join(command)}", flush=True)
    log_file = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=env)


async def _wait_server(base_url: str, process: subprocess.Popen, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
        last_error = ""
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"vLLM server exited early: {base_url}, returncode={process.returncode}")
            try:
                response = await client.get(f"{base_url}/health")
                if response.status_code == 200:
                    return
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(2.0)
    raise RuntimeError(f"Timed out waiting for vLLM server {base_url}; last_error={last_error}")


def _terminate_server(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
    deadline = time.time() + 30
    while process.poll() is None and time.time() < deadline:
        time.sleep(0.5)
    if process.poll() is None:
        process.kill()


def _shard_path(shard_dir: Path, index: int) -> Path:
    return shard_dir / f"{index:08d}.json"


def _write_sample_shard(sample: EvalSample, shard_dir: Path) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = _shard_path(shard_dir, sample.index)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(sample.to_record(), ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _append_sample_jsonl(sample: EvalSample, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(sample.to_record(), ensure_ascii=False) + "\n")


def _load_sharded_samples(shard_dir: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if not shard_dir.exists():
        return samples
    for path in sorted(shard_dir.glob("*.json")):
        try:
            samples.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] skipping unreadable shard {path}: {exc}", file=sys.stderr, flush=True)
    samples.sort(key=lambda item: int(item.get("index", 0)))
    return samples


def _load_completed_judge_samples(shard_dir: Path) -> list[dict[str, Any]]:
    samples = _load_sharded_samples(shard_dir)
    completed = []
    skipped = 0
    for sample in samples:
        if sample.get("judge_fallback") is True or not sample.get("judge_scores"):
            skipped += 1
            continue
        completed.append(sample)
    if skipped:
        print(f"[eval] ignoring non-LLM judge shards={skipped} dir={shard_dir}", flush=True)
    return completed


def _apply_shard(base_sample: EvalSample, shard_record: dict[str, Any]) -> EvalSample:
    for field_name in (
        "response",
        "cot",
        "response_length",
        "status",
        "finish_reason",
        "error",
        "judge_scores",
        "judge_weighted_score",
        "judge_reward",
        "judge_raw_output",
        "judge_error",
    ):
        if field_name in shard_record:
            setattr(base_sample, field_name, shard_record[field_name])
    return base_sample


def _merge_samples(original_samples: list[EvalSample], sharded_samples: list[dict[str, Any] | EvalSample]) -> list[EvalSample]:
    by_index = {sample.index: sample for sample in original_samples}
    for item in sharded_samples:
        if isinstance(item, EvalSample):
            by_index[item.index] = item
            continue
        index = int(item.get("index", 0))
        if index in by_index:
            _apply_shard(by_index[index], item)
    return [by_index[index] for index in sorted(by_index)]


async def _generate_one(
    client: AsyncOpenAI,
    args: argparse.Namespace,
    spec: ModelSpec,
    sample: EvalSample,
    rollout_model_name: str,
) -> EvalSample:
    last_error = ""
    for attempt in range(1, 4):
        try:
            if _uses_qwen_chat_completion(spec):
                response = await client.chat.completions.create(
                    model=rollout_model_name,
                    messages=sample.messages,
                    max_tokens=args.max_response_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop=list(spec.stop_strings) or None,
                    extra_body=_generation_extra_body(args, spec),
                )
                choices = response.choices or []
                if not choices:
                    raise RuntimeError("Empty vLLM chat completion response")
                choice = choices[0]
                message = choice.message
                raw_response = getattr(message, "content", None) or ""
                raw_cot = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None) or ""
                cot, observation = split_thinking_and_response(raw_response)
                sample.cot = raw_cot or cot
                sample.response = observation
            else:
                response = await client.completions.create(
                    model=rollout_model_name,
                    prompt=sample.rendered_prompt,
                    max_tokens=args.max_response_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop=list(spec.stop_strings) or None,
                    extra_body=_generation_extra_body(args, spec),
                )
                choices = response.choices or []
                if not choices:
                    raise RuntimeError("Empty vLLM completion response")
                choice = choices[0]
                sample.response = choice.text or ""
                sample.cot = ""
            usage = getattr(response, "usage", None) or {}
            completion_tokens = getattr(usage, "completion_tokens", None) if not isinstance(usage, dict) else usage.get("completion_tokens")
            sample.response_length = completion_tokens if isinstance(completion_tokens, int) else len(sample.response)
            finish_reason = getattr(choice, "finish_reason", None)
            sample.finish_reason = str(finish_reason or "")
            if finish_reason == "length":
                sample.status = "truncated"
            elif finish_reason == "abort":
                sample.status = "aborted"
            else:
                sample.status = "completed"
            return sample
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 3:
                await asyncio.sleep(1.5 * attempt)
            continue

    sample.response = ""
    sample.cot = ""
    sample.response_length = 0
    sample.status = "failed"
    sample.error = last_error
    return sample


async def run_generation(
    args: argparse.Namespace,
    spec: ModelSpec,
    samples: list[EvalSample],
    run_root: Path,
    rollout_base_url: str = "",
    rollout_model_name: str = "",
) -> tuple[list[EvalSample], float, str]:
    log_dir = run_root / "server_logs"
    prediction_path = run_root / "predictions.jsonl"
    prediction_shard_dir = run_root / "prediction_shards"
    log_dir.mkdir(parents=True, exist_ok=True)
    existing = {int(sample.get("index", -1)): sample for sample in _load_sharded_samples(prediction_shard_dir)} if args.resume else {}
    pending_samples = [sample for sample in samples if sample.index not in existing]
    if existing:
        print(f"[eval] resume generation shards={len(existing)} pending={len(pending_samples)}", flush=True)
    if not pending_samples:
        return _merge_samples(samples, list(existing.values())), 0.0, rollout_base_url or ""

    process: subprocess.Popen | None = None
    if rollout_base_url:
        base_urls = _parse_rollout_urls(rollout_base_url)
        print(f"[eval] using external rollout servers at {', '.join(base_urls)}", flush=True)
    else:
        port = _find_free_port(args.base_port)
        base_urls = [f"http://127.0.0.1:{port}"]
        process = _launch_server(args, spec, port, log_dir)
        await _wait_server(base_urls[0], process, args.server_timeout)
        print(f"[eval] server ready at {base_urls[0]}", flush=True)

    http_clients: list[httpx.AsyncClient] = []
    clients: list[AsyncOpenAI] = []
    for base_url in base_urls:
        http_client = httpx.AsyncClient(timeout=3600.0, trust_env=False)
        http_clients.append(http_client)
        clients.append(AsyncOpenAI(base_url=f"{base_url.rstrip('/')}/v1", api_key="EMPTY", max_retries=0, http_client=http_client))
    semaphore = asyncio.Semaphore(args.request_concurrency)
    rollout_start = time.perf_counter()
    rollout_model_name = rollout_model_name or spec.name

    async def guarded(sample: EvalSample, ordinal: int) -> EvalSample:
        client = clients[ordinal % len(clients)]
        async with semaphore:
            return await _generate_one(client, args, spec, sample, rollout_model_name)

    results: list[EvalSample] = []
    try:
        tasks = [asyncio.create_task(guarded(sample, index)) for index, sample in enumerate(pending_samples)]
        done_count = 0
        for task in asyncio.as_completed(tasks):
            sample = await task
            results.append(sample)
            _write_sample_shard(sample, prediction_shard_dir)
            _append_sample_jsonl(sample, prediction_path)
            done_count += 1
            if done_count % 25 == 0 or done_count == len(pending_samples):
                print(f"[eval] generated {done_count}/{len(pending_samples)}", flush=True)
    finally:
        for client in clients:
            await client.close()
        for http_client in http_clients:
            await http_client.aclose()
        if process is not None:
            _terminate_server(process)
    rollout_seconds = time.perf_counter() - rollout_start
    merged = _merge_samples(samples, [*existing.values(), *results])
    _write_jsonl(merged, prediction_path)
    return merged, rollout_seconds, ",".join(base_urls)


def _parse_judge_urls(raw_value: str) -> list[str]:
    return [item.strip() for item in (raw_value or "").split(",") if item.strip()]


def _parse_rollout_urls(raw_value: str) -> list[str]:
    return _parse_judge_urls(raw_value)


def _create_judge_clients(args: argparse.Namespace) -> tuple[list[str], list[AsyncOpenAI]]:
    base_urls = _parse_judge_urls(args.judge_base_url) or _parse_judge_urls(DEFAULT_JUDGE_BASE_URL)
    timeout = httpx.Timeout(connect=30.0, read=DEFAULT_JUDGE_TIMEOUT, write=60.0, pool=60.0)
    clients: list[AsyncOpenAI] = []
    for base_url in base_urls:
        http_client = httpx.AsyncClient(timeout=timeout, trust_env=False)
        clients.append(
            AsyncOpenAI(base_url=base_url, api_key=args.judge_api_key, max_retries=0, http_client=http_client)
        )
    return base_urls, clients


async def _close_judge_clients(clients: list[AsyncOpenAI]) -> None:
    await asyncio.gather(*(client.close() for client in clients), return_exceptions=True)
    for client in clients:
        http_client = getattr(client, "_client", None)
        if hasattr(http_client, "aclose"):
            await http_client.aclose()


def _extract_json_object(text: str) -> str | None:
    code_blocks = list(re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE))
    for match in reversed(code_blocks):
        candidate = match.group(1).strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate
    end = text.rfind("}")
    if end == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(end, -1, -1):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "}":
            depth += 1
        elif char == "{":
            depth -= 1
            if depth == 0:
                return text[index : end + 1]
    return None


def _coerce_score(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = int(value)
    elif isinstance(value, str):
        match = re.search(r"[1-5]", value)
        if not match:
            return None
        score = int(match.group(0))
    else:
        return None
    return score if 1 <= score <= 5 else None


def parse_judge_output(raw_output: str) -> dict[str, Any]:
    json_text = _extract_json_object(raw_output)
    if not json_text:
        return {"success": False, "parse_error": "no_json_object"}
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return {"success": False, "parse_error": f"json_decode_error: {exc}"}
    scores = parsed.get("scores") if isinstance(parsed, dict) else None
    if not isinstance(scores, dict):
        return {"success": False, "parse_error": "missing_or_invalid_scores"}
    coerced = {
        dimension: score
        for dimension, score in ((key, _coerce_score(value)) for key, value in scores.items())
        if score is not None
    }
    if not coerced:
        return {"success": False, "parse_error": "missing_or_invalid_scores"}
    return {"success": True, "scores": coerced}


def weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    total = 0.0
    total_weight = 0.0
    for dimension, weight in weights.items():
        value = scores.get(dimension)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        total += value * weight
        total_weight += weight
    return total / total_weight if total_weight else 0.0


def normalize_judge_score(raw_score: float) -> float:
    return max(0.0, min(1.0, (raw_score - 1.0) / 4.0))


def _remove_thinking_tags(text: str) -> str:
    if not text:
        return text

    tags = []
    for match in re.finditer(r"<think>", text, re.IGNORECASE):
        tags.append((match.start(), "open"))
    for match in re.finditer(r"</think>", text, re.IGNORECASE):
        tags.append((match.start(), "close"))
    if not tags:
        return text

    tags.sort(key=lambda item: item[0])
    ranges_to_remove = []
    used_indices = set()

    for index, (position, tag_type) in enumerate(tags):
        if index in used_indices or tag_type != "open":
            continue
        for candidate_index in range(index + 1, len(tags)):
            if candidate_index in used_indices:
                continue
            candidate_position, candidate_type = tags[candidate_index]
            if candidate_type == "close":
                ranges_to_remove.append((position, candidate_position + len("</think>")))
                used_indices.update((index, candidate_index))
                break
        else:
            ranges_to_remove.append((position, len(text)))
            used_indices.add(index)

    for index, (position, tag_type) in enumerate(tags):
        if index in used_indices or tag_type != "close":
            continue
        ranges_to_remove.append((0, position + len("</think>")))
        used_indices.add(index)

    ranges_to_remove.sort()
    merged_ranges = []
    for start, end in ranges_to_remove:
        if merged_ranges and start <= merged_ranges[-1][1]:
            merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
        else:
            merged_ranges.append((start, end))

    result_parts = []
    previous_end = 0
    for start, end in merged_ranges:
        if start > previous_end:
            result_parts.append(text[previous_end:start])
        previous_end = end
    if previous_end < len(text):
        result_parts.append(text[previous_end:])
    return "".join(result_parts).strip()


def split_thinking_and_response(text: str) -> tuple[str, str]:
    """Split Qwen3.5 think-on content into (cot, observation).

    The SGLang server returns the thinking block inline before
    ``###[Observation]`` rather than in ``reasoning_content``.  Keep the
    observation in ``response`` and store the pre-observation text in ``cot``.
    """
    if not text:
        return "", ""
    marker_positions = [match.start() for match in re.finditer(re.escape(OBS_MARKER), text)]
    if not marker_positions:
        return "", text.strip()

    # The real observation follows the final ``</think>``.  The model often
    # mentions ``###[Observation]`` inside its thinking block, so use the
    # first marker after that close tag when it is present.
    think_close = text.lower().rfind("</think>")
    marker_pos = None
    if think_close >= 0:
        for position in reversed(marker_positions):
            if position > think_close:
                marker_pos = position
                break
        if marker_pos is None:
            cot = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
            return cot, ""
    else:
        marker_pos = marker_positions[-1]

    cot = text[:marker_pos]
    cot = re.sub(r"</?think>", "", cot, flags=re.IGNORECASE).strip()
    response = text[marker_pos:].strip()
    return cot, response


def strip_observation_marker(text: str) -> str:
    if not text:
        return ""
    cleaned = _remove_thinking_tags(text)
    cleaned = re.sub(r"<\|im_(?:start|end)\|>", "", cleaned)
    cleaned = cleaned.strip()
    observation_newline_marker = f"{OBS_MARKER}\n"
    if cleaned.startswith(observation_newline_marker):
        cleaned = cleaned[len(observation_newline_marker):]
    elif cleaned.startswith(OBS_MARKER):
        cleaned = cleaned[len(OBS_MARKER):]
    return cleaned.lstrip(" \t\r\n")


def _judge_content(text: str) -> str:
    cleaned = strip_observation_marker(text)
    cleaned = cleaned.strip()
    if not cleaned:
        return ""
    return f"**World Model Output (Simulated):**\n```\n{cleaned}\n```"


def _ground_truth_content(text: str) -> str:
    cleaned = strip_observation_marker(text)
    cleaned = cleaned.strip()
    if not cleaned:
        return ""
    return f"**Ground Truth (Real Output):**\n```\n{cleaned}\n```"


def _truncate_before_second_observation(text: str) -> str:
    if not text:
        return text
    first = text.find(OBS_MARKER)
    if first < 0:
        return text
    second = text.find(OBS_MARKER, first + len(OBS_MARKER))
    if second < 0:
        return text
    return text[:second].rstrip()


def _load_judge_system_prompt(domain: str) -> str:
    if domain not in {"android", "mcp", "os", "search", "swe", "terminal", "web"}:
        domain = "web"
    path = PROMPT_ROOT / domain / "judge_system_prompt.txt"
    return path.read_text(encoding="utf-8").strip()


def _load_judge_user_template() -> str:
    return (PROMPT_ROOT / "judge_user_prompt.txt").read_text(encoding="utf-8").strip()


def _build_judge_messages(domain: str, predicted: str, ground_truth: str) -> list[dict[str, str]]:
    system = _load_judge_system_prompt(domain)
    user = _load_judge_user_template().format(
        predicted_observation=_judge_content(predicted),
        ground_truth=_ground_truth_content(ground_truth),
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


async def _judge_one(
    clients: list[AsyncOpenAI],
    base_urls: list[str],
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    sample: EvalSample,
    counter: int,
) -> EvalSample:
    if not sample.response or not sample.ground_truth:
        sample.judge_reward = 0.0
        sample.judge_weighted_score = 0.0
        sample.judge_scores = {}
        sample.judge_error = "missing predicted response or ground-truth label"
        return sample

    sample.response = _truncate_before_second_observation(sample.response)
    sample.ground_truth = _truncate_before_second_observation(sample.ground_truth)
    if not sample.response or not sample.ground_truth:
        sample.judge_reward = 0.0
        sample.judge_weighted_score = 0.0
        sample.judge_scores = {}
        sample.judge_error = "missing predicted response or ground-truth label after observation truncation"
        return sample

    messages = _build_judge_messages(sample.domain, sample.response, sample.ground_truth)
    for attempt in range(1, 4):
        position = (counter + attempt) % len(clients)
        client = clients[position]
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=args.judge_model,
                    messages=messages,
                    max_tokens=args.judge_max_tokens,
                    temperature=0.0,
                    extra_body={"enable_thinking": False},
                )
                raw_output = response.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                sample.judge_error = f"{type(exc).__name__}: {exc}"
                if attempt < 3:
                    await asyncio.sleep(1.0 * attempt)
                continue
        if not raw_output:
            sample.judge_raw_output = ""
            sample.judge_scores = {}
            sample.judge_weighted_score = 0.0
            sample.judge_reward = 0.0
            sample.judge_error = "empty judge output"
            return sample

        parsed = parse_judge_output(raw_output)
        if not parsed.get("success") or not parsed.get("scores"):
            sample.judge_raw_output = raw_output
            sample.judge_scores = {}
            sample.judge_weighted_score = 0.0
            sample.judge_reward = 0.0
            sample.judge_error = parsed.get("parse_error", "judge parsing failed")
            return sample

        sample.judge_raw_output = raw_output
        sample.judge_scores = {dimension: float(parsed["scores"].get(dimension, 0.0)) for dimension in CORE_DIMENSIONS}
        sample.judge_weighted_score = weighted_score(sample.judge_scores, DEFAULT_WEIGHTS)
        sample.judge_reward = normalize_judge_score(sample.judge_weighted_score)
        sample.judge_error = ""
        return sample

    sample.judge_raw_output = ""
    sample.judge_scores = {}
    sample.judge_weighted_score = 0.0
    sample.judge_reward = 0.0
    sample.judge_error = sample.judge_error or "remote judge unavailable after retries"
    return sample


async def run_judge(
    args: argparse.Namespace,
    samples: list[EvalSample],
    run_root: Path,
) -> list[EvalSample]:
    judged_path = run_root / "judged.jsonl"
    judged_shard_dir = run_root / "judge_shards"
    existing = {int(sample.get("index", -1)): sample for sample in _load_completed_judge_samples(judged_shard_dir)} if args.resume else {}
    pending_samples = [sample for sample in samples if sample.index not in existing]
    if existing:
        print(f"[eval] resume judge shards={len(existing)} pending={len(pending_samples)}", flush=True)
    if not pending_samples:
        merged = _merge_samples(samples, list(existing.values()))
        _write_jsonl(merged, judged_path)
        return merged
    base_urls = _parse_judge_urls(args.judge_base_url) or _parse_judge_urls(DEFAULT_JUDGE_BASE_URL)
    timeout = httpx.Timeout(connect=30.0, read=DEFAULT_JUDGE_TIMEOUT, write=60.0, pool=60.0)
    clients: list[AsyncOpenAI] = []
    for base_url in base_urls:
        http_client = httpx.AsyncClient(timeout=timeout, trust_env=False)
        clients.append(
            AsyncOpenAI(base_url=base_url, api_key=args.judge_api_key, max_retries=0, http_client=http_client)
        )
    semaphore = asyncio.Semaphore(args.judge_concurrency)
    judged: list[EvalSample] = []
    try:
        tasks = [
            asyncio.create_task(_judge_one(clients, base_urls, args, semaphore, sample, index))
            for index, sample in enumerate(pending_samples)
        ]
        done_count = 0
        for task in asyncio.as_completed(tasks):
            sample = await task
            judged.append(sample)
            _write_sample_shard(sample, judged_shard_dir)
            _append_sample_jsonl(sample, judged_path)
            done_count += 1
            if done_count % 25 == 0 or done_count == len(pending_samples):
                print(f"[eval] judged {done_count}/{len(pending_samples)}", flush=True)
    finally:
        await asyncio.gather(*(client.close() for client in clients), return_exceptions=True)
        for client in clients:
            http_client = getattr(client, "_client", None)
            if hasattr(http_client, "aclose"):
                await http_client.aclose()
    merged = _merge_samples(samples, [*existing.values(), *judged])
    _write_jsonl(merged, judged_path)
    return merged


async def run_generation_with_inline_judge(
    args: argparse.Namespace,
    spec: ModelSpec,
    samples: list[EvalSample],
    run_root: Path,
    rollout_base_url: str = "",
    rollout_model_name: str = "",
) -> tuple[list[EvalSample], float, str]:
    log_dir = run_root / "server_logs"
    prediction_path = run_root / "predictions.jsonl"
    judged_path = run_root / "judged.jsonl"
    prediction_shard_dir = run_root / "prediction_shards"
    judged_shard_dir = run_root / "judge_shards"
    log_dir.mkdir(parents=True, exist_ok=True)

    existing_judged = {int(sample.get("index", -1)): sample for sample in _load_completed_judge_samples(judged_shard_dir)} if args.resume else {}
    existing_predictions = {int(sample.get("index", -1)): sample for sample in _load_sharded_samples(prediction_shard_dir)} if args.resume else {}
    pending_samples = [sample for sample in samples if sample.index not in existing_judged]
    if existing_judged:
        print(f"[eval] resume inline judged shards={len(existing_judged)} pending={len(pending_samples)}", flush=True)
    if not pending_samples:
        merged = _merge_samples(samples, list(existing_judged.values()))
        _write_jsonl(merged, prediction_path)
        _write_jsonl(merged, judged_path)
        return merged, 0.0, rollout_base_url or ""

    process: subprocess.Popen | None = None
    if rollout_base_url:
        base_urls = _parse_rollout_urls(rollout_base_url)
        print(f"[eval] using external rollout servers at {', '.join(base_urls)}", flush=True)
    else:
        port = _find_free_port(args.base_port)
        base_urls = [f"http://127.0.0.1:{port}"]
        process = _launch_server(args, spec, port, log_dir)
        await _wait_server(base_urls[0], process, args.server_timeout)
        print(f"[eval] server ready at {base_urls[0]}", flush=True)

    generation_http_clients: list[httpx.AsyncClient] = []
    generation_clients: list[AsyncOpenAI] = []
    for base_url in base_urls:
        generation_http_client = httpx.AsyncClient(timeout=3600.0, trust_env=False)
        generation_http_clients.append(generation_http_client)
        generation_clients.append(AsyncOpenAI(base_url=f"{base_url.rstrip('/')}/v1", api_key="EMPTY", max_retries=0, http_client=generation_http_client))
    judge_base_urls, judge_clients = _create_judge_clients(args)
    generation_semaphore = asyncio.Semaphore(args.request_concurrency)
    judge_semaphore = asyncio.Semaphore(args.judge_concurrency)
    rollout_start = time.perf_counter()
    rollout_model_name = rollout_model_name or spec.name

    async def guarded(sample: EvalSample, ordinal: int) -> EvalSample:
        if sample.index in existing_predictions:
            _apply_shard(sample, existing_predictions[sample.index])
        else:
            async with generation_semaphore:
                generation_client = generation_clients[ordinal % len(generation_clients)]
                sample = await _generate_one(generation_client, args, spec, sample, rollout_model_name)
            _write_sample_shard(sample, prediction_shard_dir)
            _append_sample_jsonl(sample, prediction_path)
        sample = await _judge_one(judge_clients, judge_base_urls, args, judge_semaphore, sample, ordinal)
        _write_sample_shard(sample, judged_shard_dir)
        _append_sample_jsonl(sample, judged_path)
        return sample

    judged: list[EvalSample] = []
    try:
        done_count = 0
        if args.request_concurrency == 1 and args.judge_concurrency == 1:
            for index, pending_sample in enumerate(pending_samples):
                sample = await guarded(pending_sample, index)
                judged.append(sample)
                done_count += 1
                if done_count % 25 == 0 or done_count == len(pending_samples):
                    print(f"[eval] generated+judged {done_count}/{len(pending_samples)}", flush=True)
        else:
            tasks = [asyncio.create_task(guarded(sample, index)) for index, sample in enumerate(pending_samples)]
            for task in asyncio.as_completed(tasks):
                sample = await task
                judged.append(sample)
                done_count += 1
                if done_count % 25 == 0 or done_count == len(pending_samples):
                    print(f"[eval] generated+judged {done_count}/{len(pending_samples)}", flush=True)
    finally:
        for generation_client in generation_clients:
            await generation_client.close()
        for generation_http_client in generation_http_clients:
            await generation_http_client.aclose()
        await _close_judge_clients(judge_clients)
        if process is not None:
            _terminate_server(process)
    rollout_seconds = time.perf_counter() - rollout_start
    merged = _merge_samples(samples, [*existing_judged.values(), *judged])
    _write_jsonl(merged, prediction_path)
    _write_jsonl(merged, judged_path)
    return merged, rollout_seconds, ",".join(base_urls)


def _write_jsonl(samples: Iterable[EvalSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as sink:
        for sample in samples:
            sink.write(json.dumps(sample.to_record(), ensure_ascii=False) + "\n")


def _subset_name(sample: EvalSample) -> str:
    source = str(sample.source or "")
    if source.startswith("OOD"):
        return "OOD"
    if source.startswith("ID"):
        return "ID"
    return "UNKNOWN"


def _compute_metrics(samples: list[EvalSample], model_name: str) -> dict[str, Any]:
    total = len(samples)
    completed = sum(1 for item in samples if item.status == "completed")
    truncated = sum(1 for item in samples if item.status == "truncated")
    failed = sum(1 for item in samples if item.status in {"failed", "aborted"})
    prompt_truncated = sum(1 for item in samples if item.prompt_truncated)
    judged_samples = [item for item in samples if item.judge_scores]
    judged_count = len(judged_samples)
    judge_fallback_count = sum(1 for item in samples if getattr(item, "judge_fallback", False))
    mean_reward = sum(item.judge_reward for item in judged_samples) / max(judged_count, 1)
    mean_weighted = sum(item.judge_weighted_score for item in judged_samples) / max(judged_count, 1)
    total_response_tokens = sum(int(item.response_length or 0) for item in samples)
    domain_metrics: dict[str, dict[str, Any]] = {}
    for item in samples:
        domain = item.domain or "unknown"
        bucket = domain_metrics.setdefault(
            domain,
            {
                "count": 0,
                "judged_count": 0,
                "reward_sum": 0.0,
                "weighted_sum": 0.0,
                "dimension_sums": {dimension: 0.0 for dimension in CORE_DIMENSIONS},
            },
        )
        bucket["count"] += 1
        if item.judge_scores:
            bucket["judged_count"] += 1
            bucket["reward_sum"] += item.judge_reward
            bucket["weighted_sum"] += item.judge_weighted_score
            for dimension in CORE_DIMENSIONS:
                if dimension in item.judge_scores:
                    bucket["dimension_sums"][dimension] += item.judge_scores[dimension]
    dimensions: dict[str, float] = {}
    for dimension in CORE_DIMENSIONS:
        values = [item.judge_scores[dimension] for item in samples if dimension in item.judge_scores]
        dimensions[dimension] = sum(values) / len(values) if values else 0.0
    domain_weighted_scores = [
        values["weighted_sum"] / max(values["judged_count"], 1)
        for values in domain_metrics.values()
        if values["judged_count"] > 0
    ]
    domain_reward_scores = [
        values["reward_sum"] / max(values["judged_count"], 1)
        for values in domain_metrics.values()
        if values["judged_count"] > 0
    ]
    domain_mean_weighted = sum(domain_weighted_scores) / len(domain_weighted_scores) if domain_weighted_scores else 0.0
    domain_mean_reward = sum(domain_reward_scores) / len(domain_reward_scores) if domain_reward_scores else 0.0
    judged_domains = [values for values in domain_metrics.values() if values["judged_count"] > 0]
    domain_dimension_means: dict[str, float] = {}
    for dimension in CORE_DIMENSIONS:
        domain_means = [
            values["dimension_sums"][dimension] / values["judged_count"]
            for values in judged_domains
        ]
        domain_dimension_means[dimension] = sum(domain_means) / len(domain_means) if domain_means else 0.0
    return {
        "model": model_name,
        "num_samples": total,
        "completed": completed,
        "truncated": truncated,
        "failed": failed,
        "prompt_truncated": prompt_truncated,
        "judged_count": judged_count,
        "judge_fallback_count": judge_fallback_count,
        "mean_judge_reward": mean_reward,
        "mean_weighted_score": mean_weighted,
        "mean_judge_reward_domain_avg": domain_mean_reward,
        "mean_weighted_score_domain_avg": domain_mean_weighted,
        "mean_weighted_score_domain_avg_100": domain_mean_weighted * 20.0,
        "mean_scores": dimensions,
        "mean_scores_domain_avg": domain_dimension_means,
        "total_response_tokens": total_response_tokens,
        "domain_summary": {
            domain: {
                "count": values["count"],
                "judged_count": values["judged_count"],
                "mean_reward": values["reward_sum"] / max(values["judged_count"], 1),
                "mean_weighted_score": values["weighted_sum"] / max(values["judged_count"], 1),
            }
            for domain, values in sorted(domain_metrics.items())
        },
    }


def summarize(samples: list[EvalSample], run_root: Path) -> dict[str, Any]:
    metrics = _compute_metrics(samples, run_root.name)
    split_samples: dict[str, list[EvalSample]] = {}
    for sample in samples:
        split_samples.setdefault(_subset_name(sample), []).append(sample)
    split_summary = {
        name: _compute_metrics(items, run_root.name)
        for name, items in sorted(split_samples.items())
    }
    metrics["split_summary"] = split_summary

    (run_root / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    for split_name, split_metric in split_summary.items():
        (run_root / f"metrics_{split_name}.json").write_text(
            json.dumps(split_metric, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return metrics


def main() -> int:
    args = parse_args()
    _disable_proxy_env()
    if not args.eval_data.exists():
        raise FileNotFoundError(args.eval_data)

    spec = MODELS[args.model]
    if args.checkpoint is not None or args.template_model is not None:
        spec = replace(
            spec,
            checkpoint=args.checkpoint or spec.checkpoint,
            template_model=args.template_model or args.checkpoint or spec.template_model,
        )
    if not spec.checkpoint.exists():
        raise FileNotFoundError(f"model checkpoint not found: {spec.checkpoint}")
    if not spec.template_model.exists():
        raise FileNotFoundError(f"template model not found: {spec.template_model}")

    if args.num_gpus is None:
        args.num_gpus = spec.engine.tensor_parallel_size
    if args.max_model_len is None:
        args.max_model_len = spec.engine.max_model_len
    if _is_agentworld_model(spec):
        if args.temperature == DEFAULT_TEMPERATURE:
            args.temperature = 0.6
        if args.top_p == DEFAULT_TOP_P:
            args.top_p = 0.95
        if args.top_k == DEFAULT_TOP_K:
            args.top_k = 20

    output_root = Path(args.output_root)
    if args.run_name is not None:
        run_root = output_root / args.run_name
    elif output_root.name == spec.name:
        run_root = output_root
    else:
        run_root = output_root / spec.name
    if not args.dry_run and not args.resume and run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "server_logs").mkdir(parents=True, exist_ok=True)

    print(f"[eval] model={spec.name}", flush=True)
    print(f"[eval] checkpoint={spec.checkpoint}", flush=True)
    print(f"[eval] eval_data={args.eval_data}", flush=True)
    print(f"[eval] output_root={run_root}", flush=True)
    print(f"[eval] max_model_len={args.max_model_len} max_prompt_tokens={args.max_prompt_tokens} max_response_tokens={args.max_response_tokens}", flush=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.template_model),
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(str(spec.template_model), trust_remote_code=True)
    if spec.chat_template is not None and spec.chat_template.exists():
        tokenizer.chat_template = spec.chat_template.read_text(encoding="utf-8")
    records = _load_jsonl(args.eval_data)
    if args.eval_limit is not None:
        records = records[: args.eval_limit]
    prompt_budget = min(
        args.max_prompt_tokens,
        max(1, args.max_model_len - args.max_response_tokens - args.prompt_safety_margin),
    )
    samples = build_samples(
        records,
        tokenizer,
        prompt_budget,
        add_generation_prompt=_uses_generation_prompt(spec),
        append_assistant_prompt=_append_assistant_prompt(spec),
    )
    print(f"[eval] samples={len(samples)} prompt_budget={prompt_budget}", flush=True)

    if args.dry_run:
        port = _find_free_port(args.base_port)
        print(f"[eval] dry-run server command: {' '.join(_server_command(args, spec, port))}", flush=True)
        return 0

    rollout_base_url = args.rollout_base_url.strip()
    rollout_model_name = args.rollout_model_name.strip() or spec.name

    if args.inline_judge and not args.skip_judge:
        samples, rollout_seconds, rollout_ref = asyncio.run(
            run_generation_with_inline_judge(
                args,
                spec,
                samples,
                run_root,
                rollout_base_url=rollout_base_url,
                rollout_model_name=rollout_model_name,
            )
        )
    else:
        samples, rollout_seconds, rollout_ref = asyncio.run(
            run_generation(
                args,
                spec,
                samples,
                run_root,
                rollout_base_url=rollout_base_url,
                rollout_model_name=rollout_model_name,
            )
        )
    print(f"[eval] generation complete seconds={rollout_seconds:.1f} server={rollout_ref}", flush=True)

    if args.inline_judge and not args.skip_judge:
        pass
    elif not args.skip_judge:
        samples = asyncio.run(run_judge(args, samples, run_root))
    else:
        _write_jsonl(samples, run_root / "predictions.jsonl")

    metrics = summarize(samples, run_root)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"[eval] complete model={spec.name} mean_reward={metrics['mean_judge_reward']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
