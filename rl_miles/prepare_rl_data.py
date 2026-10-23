#!/usr/bin/env python3
"""Prepare world-model RL prompt data for miles-megatron.

Input (current RL corpus):

    {"system": "...", "prompt": "...", "ground_truth": "...",
     "source": "...", "domain": "...", "dataset": "..."}

Input (legacy SFT ``messages`` format) is also accepted.

The prompt is rendered once with the selected HF chat template and explicit
CoT setting using the same renderer as the Miles data loader. The generated string is written
directly to ``prompt`` and miles is launched *without* ``--apply-chat-template``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

OBS_MARKER = "###[Observation]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="RL/SFT jsonl input file.")
    parser.add_argument("--output", required=True, help="Output RL prompt jsonl.")
    parser.add_argument("--template-model", required=True, help="Local HF tokenizer/chat-template directory")
    parser.add_argument("--think-mode", choices=("on", "off"), default="off", help="Bake the selected CoT mode into prepared prompts; regenerate to change it")
    parser.add_argument("--max-prompt-tokens", type=int, default=32768)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N records.")
    return parser.parse_args()


def _load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _num_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def ensure_observation_marker(prompt: str) -> str:
    if not prompt:
        return OBS_MARKER + "\n"
    if prompt.rstrip().endswith(OBS_MARKER):
        return prompt
    suffix = "\n" if not prompt.endswith("\n") else ""
    return f"{prompt}{suffix}{OBS_MARKER}\n"


# Keep preparation and on-the-fly Miles data loading on the same rendering path.
from world_model_rl.dataloader import _render_messages, _fit_prompt


def _build_from_new_format(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    system = str(record.get("system", "") or "")
    prompt = str(record.get("prompt", "") or "")
    ground_truth = str(record.get("ground_truth", "") or "")
    if not prompt or not ground_truth:
        return None
    metadata = {
        "domain": str(record.get("domain", "") or ""),
        "dataset": str(record.get("dataset", "") or ""),
        "source": str(record.get("source", "") or ""),
    }
    return {
        "system": system,
        "user": ensure_observation_marker(prompt),
        "label": ground_truth,
        "metadata": metadata,
    }


def _find_last_assistant(messages: List[Dict[str, Any]]) -> Optional[int]:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return index
    return None


def _build_from_messages_format(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    assistant_index = _find_last_assistant(messages)
    if assistant_index is None or assistant_index <= 0:
        return None
    prompt_messages = [dict(message) for message in messages[:assistant_index]]
    system_parts: List[str] = []
    user_parts: List[str] = []
    for message in prompt_messages:
        content = str(message.get("content", "") or "")
        if message.get("role") == "system":
            system_parts.append(content)
        else:
            user_parts.append(content)
    ground_truth = str(messages[assistant_index].get("content", "") or "")
    if not user_parts or not ground_truth:
        return None
    user = ensure_observation_marker("\n\n".join(user_parts))
    metadata = {
        "domain": str(record.get("domain", "") or ""),
        "dataset": str(record.get("dataset", "") or ""),
        "source": str(record.get("source", "") or ""),
    }
    return {
        "system": "\n\n".join(system_parts),
        "user": user,
        "label": ground_truth,
        "metadata": metadata,
    }


def build_sample(tokenizer, record: Dict[str, Any], max_prompt_tokens: int, include_think: bool = False) -> Optional[Dict[str, Any]]:
    if "ground_truth" in record and "prompt" in record:
        parsed = _build_from_new_format(record)
    else:
        parsed = _build_from_messages_format(record)
    if parsed is None:
        return None

    rendered, truncated = _fit_prompt(
        tokenizer,
        parsed["system"],
        parsed["user"],
        max_prompt_tokens,
        include_think=include_think,
    )
    parsed["metadata"]["think_mode"] = include_think
    out = {
        "prompt": rendered,
        "label": parsed["label"],
        "metadata": parsed["metadata"],
    }
    if truncated:
        out["metadata"]["prompt_truncated"] = True
    return out


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(args.template_model)
    accepted = 0
    skipped = 0
    truncated = 0

    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as sink:
        for line_index, line in enumerate(source):
            if args.limit is not None and line_index >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON at line %d: %s", line_index, exc)
                skipped += 1
                continue

            out = build_sample(tokenizer, record, args.max_prompt_tokens, args.think_mode == "on")
            if out is None:
                skipped += 1
                continue
            if out["metadata"].get("prompt_truncated"):
                truncated += 1

            sink.write(json.dumps(out, ensure_ascii=False) + "\n")
            accepted += 1
            if accepted % 1000 == 0:
                logger.info(
                    "processed=%d accepted=%d skipped=%d truncated=%d",
                    accepted + skipped,
                    accepted,
                    skipped,
                    truncated,
                )

    logger.info(
        "done accepted=%d skipped=%d truncated=%d output=%s",
        accepted,
        skipped,
        truncated,
        output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
