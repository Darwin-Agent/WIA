#!/usr/bin/env python3
"""Streaming evaluation worker.

For every sample:
  1. generate one rollout response against an already-running SGLang server;
  2. immediately judge the generated response with two user-configured
     OpenAI-compatible judge services in parallel;
  3. persist prediction/judged/judge-plus shards so the run is resumable.

All three operations share concurrency=64 by default.
"""

from __future__ import annotations

import argparse
from thinking import parse_think_mode
import asyncio
import copy
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from eval_config import DEFAULT_EVAL_DATA
from run_eval import (
    CORE_DIMENSIONS,
    DEFAULT_WEIGHTS,
    EvalSample,
    MODELS,
    _append_sample_jsonl,
    _append_assistant_prompt,
    _apply_shard,
    _close_judge_clients,
    _compute_metrics,
    _create_judge_clients,
    _generate_one,
    _judge_one,
    _load_jsonl,
    _load_sharded_samples,
    _merge_samples,
    _parse_rollout_urls,
    _subset_name,
    _uses_generation_prompt,
    _write_jsonl,
    _write_sample_shard,
    replace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="Qwen3.5-4B")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--template-model", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, default=DEFAULT_EVAL_DATA)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--rollout-base-url", required=True)
    parser.add_argument("--rollout-model-name", default=None)
    parser.add_argument("--max-model-len", type=int, default=81920)
    parser.add_argument("--max-prompt-tokens", type=int, default=65536)
    parser.add_argument("--max-response-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--request-concurrency", type=int, default=64)
    parser.add_argument("--think-mode", type=parse_think_mode, choices=("auto", "on", "off"), default=os.environ.get("THINK_MODE", "off"), help="CoT generation: on enables thinking, off disables it, auto uses model defaults; independent of judge thinking")
    parser.add_argument("--judge-base-url", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-api-key", default=os.environ.get("JUDGE_API_KEY", "EMPTY"))
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--judge-concurrency", type=int, default=64)
    parser.add_argument("--judge-plus-base-url", required=True)
    parser.add_argument("--judge-plus-model", required=True)
    parser.add_argument("--judge-plus-max-tokens", type=int, default=4096)
    parser.add_argument("--judge-plus-concurrency", type=int, default=64)
    parser.add_argument("--max-judge-input-tokens", type=int, default=16384)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def disable_proxy_env() -> None:
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


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


def write_metrics(samples: list[EvalSample], run_root: Path, prefix: str = "") -> None:
    full = _compute_metrics(samples, run_root.name)
    full_name = "metrics.json" if not prefix else f"metrics_{prefix}.json"
    (run_root / full_name).write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")

    splits: dict[str, list[EvalSample]] = {}
    for sample in samples:
        splits.setdefault(_subset_name(sample), []).append(sample)
    for split_name, split_samples in sorted(splits.items()):
        split_metric = _compute_metrics(split_samples, run_root.name)
        split_name_path = f"metrics_{split_name}.json" if not prefix else f"metrics_{split_name}_{prefix}.json"
        (run_root / split_name_path).write_text(json.dumps(split_metric, ensure_ascii=False, indent=2), encoding="utf-8")


async def async_main(args: argparse.Namespace) -> int:
    disable_proxy_env()
    tokenizer = load_tokenizer(args.template_model)

    spec = MODELS[args.model]
    spec = replace(
        spec,
        checkpoint=args.checkpoint,
        template_model=args.template_model,
    )
    if not spec.checkpoint.exists():
        raise FileNotFoundError(f"model checkpoint not found: {spec.checkpoint}")
    if not spec.template_model.exists():
        raise FileNotFoundError(f"template model not found: {spec.template_model}")

    run_root = args.output_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "server_logs").mkdir(parents=True, exist_ok=True)

    records = _load_jsonl(args.eval_data)
    if args.limit is not None:
        records = records[: args.limit]
    prompt_budget = min(
        args.max_prompt_tokens,
        max(1, args.max_model_len - args.max_response_tokens),
    )
    samples = build_stream_samples(
        records,
        tokenizer,
        prompt_budget,
        spec,
    )

    prediction_path = run_root / "predictions.jsonl"
    judged_path = run_root / "judged.jsonl"
    plus_path = run_root / "judge-plus.jsonl"
    prediction_shard_dir = run_root / "prediction_shards"
    judged_shard_dir = run_root / "judge_shards"
    plus_shard_dir = run_root / "judge_plus_shards"

    existing_predictions: dict[int, dict] = {}
    existing_flash: dict[int, dict] = {}
    existing_plus: dict[int, dict] = {}
    if args.resume:
        existing_predictions = {int(item.get("index", -1)): item for item in _load_sharded_samples(prediction_shard_dir)}
        existing_flash = {int(item.get("index", -1)): item for item in _load_sharded_samples(judged_shard_dir)}
        existing_plus = {int(item.get("index", -1)): item for item in _load_sharded_samples(plus_shard_dir)}

    pending = []
    for sample in samples:
        need_generate = sample.index not in existing_predictions
        need_flash = sample.index not in existing_flash or not existing_flash[sample.index].get("judge_scores")
        need_plus = sample.index not in existing_plus or not existing_plus[sample.index].get("judge_scores")
        if need_generate or need_flash or need_plus:
            pending.append((sample, need_generate, need_flash, need_plus))

    print(
        f"[streaming] run={args.run_name} samples={len(samples)} pending={len(pending)} "
        f"request_concurrency={args.request_concurrency} judge_concurrency={args.judge_concurrency} "
        f"judge_plus_concurrency={args.judge_plus_concurrency}",
        flush=True,
    )

    rollout_urls = _parse_rollout_urls(args.rollout_base_url)
    if not rollout_urls:
        raise ValueError(f"invalid --rollout-base-url: {args.rollout_base_url}")
    generation_clients: list[AsyncOpenAI] = []
    generation_http_clients: list[httpx.AsyncClient] = []
    for base_url in rollout_urls:
        http_client = httpx.AsyncClient(timeout=3600.0, trust_env=False)
        generation_http_clients.append(http_client)
        generation_clients.append(
            AsyncOpenAI(base_url=f"{base_url.rstrip('/')}/v1", api_key="EMPTY", max_retries=0, http_client=http_client)
        )

    flash_judge_ns = SimpleNamespace(
        judge_base_url=args.judge_base_url,
        judge_model=args.judge_model,
        judge_api_key=args.judge_api_key,
        judge_max_tokens=args.judge_max_tokens,
        judge_concurrency=args.judge_concurrency,
    )
    flash_urls, flash_clients = _create_judge_clients(flash_judge_ns)

    plus_timeout = httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=60.0)
    plus_http_client = httpx.AsyncClient(timeout=plus_timeout, trust_env=False)
    plus_clients = [
        AsyncOpenAI(
            base_url=args.judge_plus_base_url.rstrip("/"),
            api_key=os.environ.get("JUDGE_PLUS_API_KEY", "EMPTY"),
            max_retries=0,
            http_client=plus_http_client,
        )
    ]
    plus_urls = [args.judge_plus_base_url.rstrip("/")]
    plus_judge_ns = SimpleNamespace(
        judge_model=args.judge_plus_model,
        judge_max_tokens=args.judge_plus_max_tokens,
    )

    generation_semaphore = asyncio.Semaphore(args.request_concurrency)
    flash_semaphore = asyncio.Semaphore(args.judge_concurrency)
    plus_semaphore = asyncio.Semaphore(args.judge_plus_concurrency)

    generation_args = SimpleNamespace(
        max_response_tokens=args.max_response_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        think_mode=args.think_mode,
    )
    rollout_model_name = args.rollout_model_name or spec.name

    def failed_judge_sample(template: EvalSample, exc: BaseException) -> EvalSample:
        failed = copy.deepcopy(template)
        failed.judge_raw_output = ""
        failed.judge_scores = {}
        failed.judge_weighted_score = 0.0
        failed.judge_reward = 0.0
        failed.judge_error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        return failed

    async def safe_judge_one(
        kind: str,
        clients: list[AsyncOpenAI],
        urls: list[str],
        judge_ns: SimpleNamespace,
        semaphore: asyncio.Semaphore,
        judge_sample: EvalSample,
        index: int,
    ) -> EvalSample:
        try:
            return await _judge_one(clients, urls, judge_ns, semaphore, judge_sample, index)
        except BaseException as exc:  # noqa: BLE001
            print(
                f"[streaming] isolated {kind} judge failure index={index}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return failed_judge_sample(judge_sample, exc)

    async def process_one(sample: EvalSample, need_generate: bool, need_flash: bool, need_plus: bool) -> None:
        try:
            base_sample = copy.deepcopy(sample)
            if need_generate:
                try:
                    async with generation_semaphore:
                        base_sample = await _generate_one(
                            generation_clients[0],
                            generation_args,
                            spec,
                            base_sample,
                            rollout_model_name,
                        )
                except BaseException as exc:  # noqa: BLE001
                    base_sample.status = "failed"
                    base_sample.error = f"{type(exc).__name__}: {exc}"
                try:
                    _write_sample_shard(base_sample, prediction_shard_dir)
                    _append_sample_jsonl(base_sample, prediction_path)
                except BaseException as exc:  # noqa: BLE001
                    print(
                        f"[streaming] prediction write failure index={sample.index}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
            else:
                existing = existing_predictions[sample.index]
                _apply_shard(base_sample, existing)

            flash_sample = copy.deepcopy(base_sample) if need_flash else None
            plus_sample = copy.deepcopy(base_sample) if need_plus else None
            if flash_sample is not None:
                flash_sample.response = truncate_to_tokens(tokenizer, flash_sample.response, args.max_judge_input_tokens)
                flash_sample.ground_truth = truncate_to_tokens(tokenizer, flash_sample.ground_truth, args.max_judge_input_tokens)
            if plus_sample is not None:
                plus_sample.response = truncate_to_tokens(tokenizer, plus_sample.response, args.max_judge_input_tokens)
                plus_sample.ground_truth = truncate_to_tokens(tokenizer, plus_sample.ground_truth, args.max_judge_input_tokens)

            async def run_flash() -> EvalSample | None:
                if flash_sample is None:
                    return None
                return await safe_judge_one(
                    "judged", flash_clients, flash_urls, flash_judge_ns, flash_semaphore,
                    flash_sample, sample.index,
                )

            async def run_plus() -> EvalSample | None:
                if plus_sample is None:
                    return None
                return await safe_judge_one(
                    "judge-plus", plus_clients, plus_urls, plus_judge_ns, plus_semaphore,
                    plus_sample, sample.index,
                )

            flash_result, plus_result = await asyncio.gather(
                run_flash(), run_plus(), return_exceptions=True
            )

            if isinstance(flash_result, BaseException) or flash_result is None:
                if flash_sample is not None:
                    flash_result = failed_judge_sample(flash_sample, flash_result or RuntimeError("judged task cancelled"))
            if isinstance(plus_result, BaseException) or plus_result is None:
                if plus_sample is not None:
                    plus_result = failed_judge_sample(plus_sample, plus_result or RuntimeError("judge-plus task cancelled"))

            if flash_result is not None:
                try:
                    _write_sample_shard(flash_result, judged_shard_dir)
                    _append_sample_jsonl(flash_result, judged_path)
                except BaseException as exc:  # noqa: BLE001
                    print(
                        f"[streaming] judged write failure index={sample.index}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
            if plus_result is not None:
                try:
                    _write_sample_shard(plus_result, plus_shard_dir)
                    _append_sample_jsonl(plus_result, plus_path)
                except BaseException as exc:  # noqa: BLE001
                    print(
                        f"[streaming] judge-plus write failure index={sample.index}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
        except BaseException as exc:  # noqa: BLE001
            print(
                f"[streaming] isolated sample failure index={sample.index}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    done_count = 0
    total_pending = len(pending)
    tasks = [
        asyncio.create_task(process_one(sample, need_generate, need_flash, need_plus))
        for sample, need_generate, need_flash, need_plus in pending
    ]
    try:
        for task in asyncio.as_completed(tasks):
            await task
            done_count += 1
            if done_count % 25 == 0 or done_count == total_pending:
                print(f"[streaming] {args.run_name} processed {done_count}/{total_pending}", flush=True)
    finally:
        for client in generation_clients:
            await client.close()
        for http_client in generation_http_clients:
            await http_client.aclose()
        await _close_judge_clients(flash_clients)
        await plus_clients[0].close()
        await plus_http_client.aclose()

    prediction_records = _load_sharded_samples(prediction_shard_dir)
    flash_records = _load_sharded_samples(judged_shard_dir)
    plus_records = _load_sharded_samples(plus_shard_dir)

    prediction_samples = _merge_samples(copy.deepcopy(samples), prediction_records)
    flash_samples = _merge_samples(copy.deepcopy(samples), flash_records)
    plus_samples = _merge_samples(copy.deepcopy(samples), plus_records)

    _write_jsonl(prediction_samples, prediction_path)
    _write_jsonl(flash_samples, judged_path)
    _write_jsonl(plus_samples, plus_path)

    write_metrics(flash_samples, run_root, prefix="")
    write_metrics(plus_samples, run_root, prefix="plus")

    print(
        f"[streaming] complete run={args.run_name} predictions={len(prediction_records)} "
        f"flash_judged={len(flash_records)} plus_judged={len(plus_records)}",
        flush=True,
    )
    return 0


def build_stream_samples(records, tokenizer, prompt_budget, spec):
    from run_eval import build_samples
    return build_samples(
        records,
        tokenizer,
        prompt_budget,
        add_generation_prompt=_uses_generation_prompt(spec),
        append_assistant_prompt=_append_assistant_prompt(spec),
    )


def main() -> int:
    args = parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
