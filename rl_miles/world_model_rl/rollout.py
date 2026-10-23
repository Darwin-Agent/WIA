from __future__ import annotations

from typing import Any

from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.sglang_rollout import generate_rollout as miles_generate_rollout
from world_model_rl.jsonl_writer import write_eval_samples, write_training_samples


def raw_train_rollout(args, rollout_id, data_source, evaluation=False):
    """Wrap miles' default training rollout and dump the generated samples to JSONL."""
    if evaluation:
        return miles_generate_rollout(args, rollout_id, data_source, evaluation=True)

    output = miles_generate_rollout(args, rollout_id, data_source, evaluation=False)
    samples = output.samples if isinstance(output, RolloutFnTrainOutput) else output
    write_training_samples(samples, args, rollout_id)
    return output
