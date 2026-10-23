from __future__ import annotations

import asyncio
import copy
import os
from pathlib import Path
from typing import Any

from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
)
from miles.rollout.fully_async_rollout import FullyAsyncRolloutFn
from miles.rollout.inference_rollout.inference_rollout_common import generate_and_rm_group
from miles.utils.types import Sample

from world_model_rl.dataloader import WorldModelRLDataLoader, think_mode_enabled
from world_model_rl.jsonl_writer import write_eval_samples, write_training_samples


class JsonlFullyAsyncRolloutFn:
    def __init__(self, input: RolloutFnConstructorInput):
        self.args = input.args
        self._inner = FullyAsyncRolloutFn(input)
        self._eval_prompt_cache: dict[Any, list[dict[str, Any]]] = {}

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            if input.generate_state is not None:
                results = await _run_world_model_eval(input.generate_state, self._eval_prompt_cache)
            else:
                self._inner._producer_resumed.clear()
                try:
                    results = await _run_world_model_eval(self._inner.state, self._eval_prompt_cache)
                finally:
                    self._inner._producer_resumed.set()
            self._write_eval_output(results, input)
            return RolloutFnEvalOutput(data=results)

        output = await self._inner(input)
        write_training_samples(output.samples, self.args, input.rollout_id)
        return output

    def dispose(self) -> None:
        dispose = getattr(self._inner, "dispose", None)
        if dispose is not None:
            dispose()

    def _write_eval_output(self, data: dict[str, dict[str, Any]], input: RolloutFnEvalInput) -> None:
        for dataset_name, dataset_output in data.items():
            samples = dataset_output.get("samples") if isinstance(dataset_output, dict) else None
            if samples is None:
                continue
            write_eval_samples(samples, self.args, input.rollout_id, dataset_name)


def _limit_from_env(name: str) -> int | None:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return None
    return int(raw_value)


async def _run_world_model_eval(state, prompt_cache: dict[Any, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    args = state.args
    eval_data_path = os.environ.get("WM_EVAL_DATA_PATH", "").strip()
    if not eval_data_path:
        raise ValueError("WM_EVAL_DATA_PATH is not set")

    eval_limit = _limit_from_env("WM_EVAL_DATA_LIMIT")
    template_model = getattr(args, "hf_checkpoint", None)
    max_prompt_tokens = getattr(args, "eval_max_prompt_len", None) or getattr(args, "rollout_max_prompt_len", 32768)
    max_response_tokens = getattr(args, "eval_max_response_len", None) or getattr(args, "rollout_max_response_len", 8192)
    n_samples = getattr(args, "n_samples_per_eval_prompt", 1)

    cache_key = (Path(eval_data_path).resolve(), template_model, max_prompt_tokens, eval_limit)
    if cache_key not in prompt_cache:
        loader = WorldModelRLDataLoader(
            input_path=Path(eval_data_path),
            template_model=template_model,
            max_prompt_tokens=max_prompt_tokens,
            limit=eval_limit,
            include_think=think_mode_enabled(getattr(args, "think_mode", None)),
        )
        prompt_cache[cache_key] = list(loader)

    base_sampling_params = dict(
        temperature=getattr(args, "eval_temperature", None) or getattr(args, "rollout_temperature", 1.0),
        top_p=getattr(args, "eval_top_p", None) or getattr(args, "rollout_top_p", 1.0),
        top_k=getattr(args, "eval_top_k", None) or getattr(args, "rollout_top_k", 0),
        max_new_tokens=max_response_tokens,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    tasks = []
    sample_index = 0
    for item in prompt_cache[cache_key]:
        prompt_sample = Sample(
            prompt=item["prompt"],
            label=item["label"],
            metadata=item.get("metadata", {}),
            index=sample_index,
        )
        sample_index += 1
        group = []
        for _ in range(n_samples):
            clone = copy.deepcopy(prompt_sample)
            clone.generate_function_path = getattr(args, "custom_generate_function_path", None)
            group.append(clone)
        tasks.append(asyncio.create_task(generate_and_rm_group(state, group, base_sampling_params, evaluation=True)))

    data = []
    for task in asyncio.as_completed(tasks):
        group = await task
        data.extend(group)

    data.sort(key=lambda sample: sample.index)
    dataset_name = Path(eval_data_path).stem
    return {
        dataset_name: {
            "rewards": [sample.reward for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }
