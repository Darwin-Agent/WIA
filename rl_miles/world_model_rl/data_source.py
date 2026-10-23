from __future__ import annotations

import copy
import random
import os
from pathlib import Path

from miles.rollout.data_source import DataSource
from miles.utils.types import Sample

from world_model_rl.dataloader import WorldModelRecord, _fit_prompt, parse_record, read_jsonl, think_mode_enabled, validate_prepared_think_mode


class WorldModelRLJsonlDataSource(DataSource):
    """Read raw RL JSONL and render prompts on the fly."""

    def __init__(self, args):
        self.args = args
        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        self.metadata = {}
        self.sample_buffer = []

        input_path = Path(args.prompt_data)
        template_model = getattr(args, "template_model", None) or getattr(args, "hf_checkpoint", None)
        max_prompt_tokens = getattr(args, "rollout_max_prompt_len", 32768)
        include_think = think_mode_enabled(getattr(args, "think_mode", None))
        raw_limit = os.environ.get("MILES_PROMPT_DATA_LIMIT", "").strip()
        limit = int(raw_limit) if raw_limit else None

        self.input_path = input_path
        self.template_model = template_model
        self.max_prompt_tokens = max_prompt_tokens
        self.include_think = include_think
        self._tokenizer = None

        self.dataset = []
        for record in read_jsonl(input_path, limit):
            parsed = parse_record(record)
            if parsed is None:
                continue
            is_pre_rendered = (
                "prompt" in record
                and "label" in record
                and "ground_truth" not in record
                and "messages" not in record
            )
            self.dataset.append((parsed, is_pre_rendered))

        if self.args.rollout_shuffle:
            self._shuffle_dataset()

        if not self.dataset:
            raise ValueError(
                f"No usable samples loaded from {input_path}; expected ground_truth/prompt-label/messages JSONL records."
            )

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.template_model, trust_remote_code=True)
        return self._tokenizer

    def _shuffle_dataset(self):
        seed = int(getattr(self.args, "rollout_seed", 0) or 0) + self.epoch_id
        random.Random(seed).shuffle(self.dataset)

    def _make_sample(self, item: tuple[WorldModelRecord, bool]) -> Sample:
        parsed, is_pre_rendered = item
        if is_pre_rendered:
            validate_prepared_think_mode(parsed.metadata, self.include_think)
            prompt = parsed.user
            metadata = dict(parsed.metadata)
        else:
            metadata = dict(parsed.metadata)
            rendered, truncated = _fit_prompt(
                self.tokenizer,
                parsed.system,
                parsed.user,
                self.max_prompt_tokens,
                self.include_think,
            )
            if truncated:
                metadata["prompt_truncated"] = True
            prompt = rendered
        return Sample(prompt=prompt, label=parsed.label, metadata=metadata)

    def _make_samples(self, items: list[tuple[WorldModelRecord, bool]]) -> list[Sample]:
        return [self._make_sample(item) for item in items]

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        if not self.dataset:
            return []
        samples = self.sample_buffer[:num_samples]
        del self.sample_buffer[: len(samples)]
        num_samples -= len(samples)
        if num_samples == 0:
            return samples

        if self.sample_offset + num_samples <= len(self.dataset):
            prompt_samples = self._make_samples(self.dataset[self.sample_offset : self.sample_offset + num_samples])
            self.sample_offset += num_samples
        else:
            prompt_sample_items = self.dataset[self.sample_offset :]
            num_samples -= len(prompt_sample_items)
            self.epoch_id += 1
            if self.args.rollout_shuffle:
                self._shuffle_dataset()
            prompt_sample_items += self.dataset[:num_samples]
            prompt_samples = self._make_samples(prompt_sample_items)
            self.sample_offset = num_samples

        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        self.sample_buffer.extend(samples)

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return
        path = Path(self.args.save) / f"rollout/global_dataset_state_dict_{rollout_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch_state = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
        }
        import torch

        torch.save(torch_state, path)

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset or self.args.load is None:
            return
        path = Path(self.args.load) / f"rollout/global_dataset_state_dict_{rollout_id}.pt"
        if not path.exists():
            return
        import torch

        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})
        if self.args.rollout_shuffle:
            self._shuffle_dataset()

    def __len__(self) -> int:
        return len(self.dataset)
