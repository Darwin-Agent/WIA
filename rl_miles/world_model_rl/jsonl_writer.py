from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from miles.utils.types import Sample

CORE_JUDGE_DIMENSIONS = [
    "format",
    "factuality",
    "consistency",
    "realism",
    "quality",
    "efficiency",
]


def _output_dir(args: Any) -> Path:
    override = os.environ.get("WM_ROLLOUT_OUTPUT_DIR", "").strip()
    if override:
        return Path(override)
    save_dir = getattr(args, "save", None)
    if save_dir:
        return Path(save_dir) / "rollout_jsonl"
    return Path.cwd() / "rollout_jsonl"


def _iter_samples(samples: Any):
    if samples is None:
        return
    for item in samples:
        if isinstance(item, (list, tuple)):
            yield from _iter_samples(item)
        else:
            yield item


def _status_value(sample: Sample):
    status = getattr(sample, "status", None)
    return getattr(status, "value", status)


def _json_default(value: Any):
    status = getattr(value, "value", None)
    if status is not None:
        return status
    return str(value)


def sample_to_record(sample: Sample, rollout_id: Any, evaluation: bool) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    judge_scores = metadata.get("judge_scores") if isinstance(metadata.get("judge_scores"), dict) else {}
    similarity_score = metadata.get("similarity_score")

    record = {
        "rollout_id": rollout_id,
        "evaluation": evaluation,
        "index": sample.index,
        "group_index": sample.group_index,
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
        "reward": sample.reward,
        "status": _status_value(sample),
        "metadata": metadata,
        "judge_enabled": metadata.get("judge_enabled"),
        "judge_skipped": metadata.get("judge_skipped"),
        "judge_success": metadata.get("judge_success"),
        "judge_error": metadata.get("judge_error"),
        "judge_scores": judge_scores,
        "judge_reward": metadata.get("judge_reward"),
        "judge_rule_based_reward": metadata.get("judge_rule_based_reward"),
        "judge_weighted_score": metadata.get("judge_weighted_score"),
        "judge_weight": metadata.get("judge_weight"),
        "judge_raw_output": metadata.get("judge_raw_output"),
        "judge_reason": metadata.get("judge_reason"),
        "fused_reward": metadata.get("fused_reward"),
        "fused_reward_formula": metadata.get("fused_reward_formula"),
        "think_mode": metadata.get("think_mode"),
        "similarity_enabled": metadata.get("similarity_enabled"),
        "similarity_success": metadata.get("similarity_success"),
        "similarity_weight": metadata.get("similarity_weight"),
        "similarity_score": similarity_score,
        "similarity_error": metadata.get("similarity_error"),
    }

    for dimension in CORE_JUDGE_DIMENSIONS:
        record[f"judge_{dimension}"] = judge_scores.get(dimension)

    return record


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def append_jsonl_record(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        sink.flush()


def write_training_samples(samples: Any, args: Any, rollout_id: Any) -> Path:
    records = [sample_to_record(sample, rollout_id, evaluation=False) for sample in _iter_samples(samples)]
    path = _output_dir(args) / f"rollout_{int(rollout_id):09d}.jsonl"
    _write_jsonl(records, path)
    return path


def write_eval_samples(samples: Any, args: Any, rollout_id: Any, dataset_name: str) -> Path:
    records = [sample_to_record(sample, rollout_id, evaluation=True) for sample in _iter_samples(samples)]
    safe_dataset = dataset_name.replace("/", "_")
    path = _output_dir(args) / f"eval_{safe_dataset}_{int(rollout_id):09d}.jsonl"
    _write_jsonl(records, path)
    return path
