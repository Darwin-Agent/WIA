from __future__ import annotations

import asyncio
import copy
import os
from pathlib import Path

from miles.rollout.base_types import RolloutFnEvalOutput
from miles.rollout.sglang_rollout import generate_and_rm_group
from miles.utils.types import Sample

from world_model_rl.dataloader import WorldModelRLDataLoader, think_mode_enabled
from world_model_rl.jsonl_writer import write_eval_samples


def _limit_from_env(name: str) -> int | None:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return None
    return int(raw_value)


def raw_eval_rollout(args, rollout_id, data_source, evaluation: bool = True):
    return asyncio.run(_raw_eval_rollout_async(args, rollout_id, data_source, evaluation=evaluation))


async def _raw_eval_rollout_async(args, rollout_id, data_source, evaluation: bool = True):
    del data_source
    del evaluation

    eval_data_path = os.environ.get("WM_EVAL_DATA_PATH", "").strip()
    if not eval_data_path:
        raise ValueError("WM_EVAL_DATA_PATH is not set")

    eval_limit = _limit_from_env("WM_EVAL_DATA_LIMIT")
    template_model = getattr(args, "hf_checkpoint", None)
    max_prompt_tokens = getattr(args, "eval_max_prompt_len", None) or getattr(args, "rollout_max_prompt_len", 32768)
    max_response_tokens = getattr(args, "eval_max_response_len", None) or getattr(args, "rollout_max_response_len", 8192)
    n_samples = getattr(args, "n_samples_per_eval_prompt", 1)

    loader = WorldModelRLDataLoader(
        input_path=Path(eval_data_path),
        template_model=template_model,
        max_prompt_tokens=max_prompt_tokens,
        limit=eval_limit,
        include_think=think_mode_enabled(getattr(args, "think_mode", None)),
    )

    dataset_name = Path(eval_data_path).stem
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
    for item in loader:
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
        tasks.append(asyncio.create_task(generate_and_rm_group(args, group, base_sampling_params, evaluation=True)))

    data = []
    for coro in asyncio.as_completed(tasks):
        group = await coro
        data.extend(group)

    data.sort(key=lambda sample: sample.index)
    write_eval_samples(data, args, rollout_id, dataset_name)
    result = {
        dataset_name: {
            "rewards": [sample.reward for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }
    return RolloutFnEvalOutput(data=result, metrics={"num_samples": len(data), "dataset": dataset_name})
