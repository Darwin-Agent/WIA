"""Data loading utilities for the world-model RL corpus.

The native training/test files are jsonl records of the form::

    {
        "system": "<world-model system prompt>",
        "prompt": "<current turn / action prompt>",
        "ground_truth": "<target observation>",
        "source": "...",
        "domain": "...",
        "dataset": "..."
    }

Legacy SFT-style ``messages`` records are also accepted, where the last
assistant message is treated as the ground-truth observation.

miles consumes an already-rendered prompt string. For native and legacy
message records, this loader appends the ``###[Observation]`` marker to the
user content and renders with ``template_model`` (default: the configured
HF checkpoint). It attempts to fit the token budget by
left-truncating user tokens; system/template overhead is not truncated.
Pre-rendered ``prompt``/``label`` records are passed through without template
rendering, marker insertion, or token-budget enforcement by this loader.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

OBS_MARKER = "###[Observation]"

DEFAULT_TEMPLATE_MODEL = os.environ.get("WM_TEMPLATE_MODEL", "")


def think_mode_enabled(override: Optional[bool | str] = None) -> bool:
    """Resolve whether rollout prompts should include the Qwen thinking prefix.

    Explicit call-site overrides take precedence, then ``WM_THINK_MODE`` (set
    by the launch scripts from ``THINK_MODE``), then the raw ``THINK_MODE``
    variable for local/debug use.
    """
    if override is not None:
        if isinstance(override, str):
            normalized = override.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return bool(override)

    for env_name in ("WM_THINK_MODE", "THINK_MODE"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        normalized = str(raw_value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False

    return False


@dataclass
class WorldModelRecord:
    """Parsed and tokenizer-independent world-model record."""

    system: str
    user: str
    label: str
    metadata: Dict[str, Any]


def read_jsonl(path: str | Path, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Yield parsed JSON records from a jsonl file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8") as source:
        for line_index, line in enumerate(source):
            if limit is not None and line_index >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON at %s:%d: %s", path, line_index, exc)


def ensure_observation_marker(prompt: str) -> str:
    """Ensure a trailing observation marker; leave an existing suffix unchanged."""
    if not prompt:
        return OBS_MARKER + "\n"
    if prompt.rstrip().endswith(OBS_MARKER):
        return prompt
    suffix = "\n" if not prompt.endswith("\n") else ""
    return f"{prompt}{suffix}{OBS_MARKER}\n"


def validate_prepared_think_mode(metadata: dict, include_think: bool) -> None:
    mode = metadata.get("think_mode")
    if mode is not None and (not isinstance(mode, bool) or mode != include_think):
        raise ValueError("Prepared prompt CoT mode does not match THINK_MODE; regenerate with prepare_rl_data.py --think-mode on/off or use raw JSONL")


def _metadata_for(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return metadata, preferring an explicit metadata dict over top-level tags."""
    explicit = record.get("metadata")
    if isinstance(explicit, dict):
        return explicit
    return _metadata_from(record)


def parse_record(record: Dict[str, Any]) -> Optional[WorldModelRecord]:
    """Parse native ground-truth, pre-rendered prompt/label, or legacy messages."""
    if "ground_truth" in record and "prompt" in record:
        return _parse_native_record(record)
    if "messages" in record:
        return _parse_messages_record(record)
    if "prompt" in record and "label" in record:
        prompt = str(record.get("prompt", "") or "")
        label = str(record.get("label", "") or "")
        if not prompt or not label:
            return None
        return WorldModelRecord(
            system="",
            user=prompt,
            label=label,
            metadata=_metadata_for(record),
        )
    return None


def _metadata_from(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "domain": str(record.get("domain", "") or ""),
        "dataset": str(record.get("dataset", "") or ""),
        "source": str(record.get("source", "") or ""),
    }


def _parse_native_record(record: Dict[str, Any]) -> Optional[WorldModelRecord]:
    system = str(record.get("system", "") or "")
    prompt = str(record.get("prompt", "") or "")
    ground_truth = str(record.get("ground_truth", "") or "")
    if not prompt or not ground_truth:
        return None
    return WorldModelRecord(
        system=system,
        user=ensure_observation_marker(prompt),
        label=ground_truth,
        metadata=_metadata_from(record),
    )


def _find_last_assistant(messages: List[Dict[str, Any]]) -> Optional[int]:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return index
    return None


def _parse_messages_record(record: Dict[str, Any]) -> Optional[WorldModelRecord]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    assistant_index = _find_last_assistant(messages)
    if assistant_index is None or assistant_index <= 0:
        return None

    system_parts: List[str] = []
    user_parts: List[str] = []
    for message in messages[:assistant_index]:
        content = str(message.get("content", "") or "")
        if message.get("role") == "system":
            system_parts.append(content)
        else:
            user_parts.append(content)

    ground_truth = str(messages[assistant_index].get("content", "") or "")
    if not user_parts or not ground_truth:
        return None

    return WorldModelRecord(
        system="\n\n".join(system_parts),
        user=ensure_observation_marker("\n\n".join(user_parts)),
        label=ground_truth,
        metadata=_metadata_from(record),
    )


class WorldModelRLDataLoader:
    """Load raw world-model records and emit miles-ready samples.

    Each emitted sample is a dict with ``prompt`` (rendered string),
    ``label`` and ``metadata``. Native/legacy records use ``template_model``
    or DEFAULT_TEMPLATE_MODEL; pre-rendered prompt/label records bypass
    rendering and truncation.
    """

    def __init__(
        self,
        input_path: str | Path,
        template_model: Optional[str] = None,
        max_prompt_tokens: int = 32768,
        limit: Optional[int] = None,
        include_think: bool = True,
    ):
        self.input_path = Path(input_path)
        self.template_model = template_model or DEFAULT_TEMPLATE_MODEL
        self.max_prompt_tokens = max_prompt_tokens
        self.limit = limit
        self.include_think = include_think
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            if not self.template_model:
                raise ValueError("Set template_model/WM_TEMPLATE_MODEL to a local HF tokenizer directory")
            from miles.utils.processing_utils import load_tokenizer

            self._tokenizer = load_tokenizer(
                self.template_model,
                trust_remote_code=True,
                fix_mistral_regex=True,
            )
        return self._tokenizer

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return self.iter_samples()

    def iter_samples(self) -> Iterator[Dict[str, Any]]:
        for record in read_jsonl(self.input_path, self.limit):
            parsed = parse_record(record)
            if parsed is None:
                continue

            # Prompt/label records from the RL corpora are already fully rendered
            # (for example, ChatML strings). Do not re-apply the chat template.
            if (
                "prompt" in record
                and "label" in record
                and "ground_truth" not in record
                and "messages" not in record
            ):
                validate_prepared_think_mode(parsed.metadata, self.include_think)
                yield {
                    "prompt": parsed.user,
                    "label": parsed.label,
                    "metadata": parsed.metadata,
                }
                continue

            rendered, truncated = _fit_prompt(
                self.tokenizer,
                parsed.system,
                parsed.user,
                self.max_prompt_tokens,
                include_think=self.include_think,
            )
            if truncated:
                parsed.metadata["prompt_truncated"] = True
            yield {
                "prompt": rendered,
                "label": parsed.label,
                "metadata": parsed.metadata,
            }


def _render_messages(tokenizer, system: str, user: str, include_think: bool) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if include_think:
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        )
    return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)) + "<|im_start|>assistant\n"


def _num_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _fit_prompt(tokenizer, system: str, user: str, max_tokens: int, include_think: bool) -> Tuple[str, bool]:
    """Render system+user and try to fit by left-truncating only user tokens.

    This is best-effort: system/template overhead is preserved, at least one
    user token is kept, and the correction loop is capped at eight iterations.
    """
    rendered = _render_messages(tokenizer, system, user, include_think)
    if _num_tokens(tokenizer, rendered) <= max_tokens:
        return rendered, False

    system_ids = tokenizer.encode(system, add_special_tokens=False)
    user_ids = tokenizer.encode(user, add_special_tokens=False)
    overhead = _num_tokens(tokenizer, rendered) - len(system_ids) - len(user_ids)

    max_user_tokens = max(1, max_tokens - len(system_ids) - overhead)
    kept = user_ids[-max_user_tokens:]
    kept_text = tokenizer.decode(kept, skip_special_tokens=False)

    for _ in range(8):
        rendered = _render_messages(tokenizer, system, kept_text, include_think)
        n_tokens = _num_tokens(tokenizer, rendered)
        if n_tokens <= max_tokens or len(kept) <= 1:
            break
        excess = n_tokens - max_tokens
        drop = min(len(kept) - 1, excess + max(1, len(kept) // 50))
        kept = kept[drop:]
        kept_text = tokenizer.decode(kept, skip_special_tokens=False)

    return rendered, True
