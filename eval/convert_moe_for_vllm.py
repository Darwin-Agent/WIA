#!/usr/bin/env python3
"""Prepare a Qwen3.5-35B-A3B checkpoint mirror for vLLM serving.

The source checkpoint is the official post-trained multimodal model and uses
``Qwen3_5MoeForConditionalGeneration`` / ``qwen3_5_moe``. Current vLLM supports
that architecture directly, so this script creates a clean companion directory
whose tensors/tokenizer/config files are symlinks or copies from the source while
skipping download sidecars such as ``*.aria2``. The serving command should use
``--language-model-only`` for text-only evaluation.

Example:
    python convert_moe_for_vllm.py \
        --source models/Qwen3.5-35B-A3B \
        --dest   models/Qwen3.5-35B-A3B-vllm
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from eval_config import PROJECT_ROOT

EXPECTED_ARCHITECTURES = ["Qwen3_5MoeForConditionalGeneration"]
EXPECTED_MODEL_TYPE = "qwen3_5_moe"


def validate_source_config(source_config: dict) -> None:
    architectures = source_config.get("architectures")
    model_type = source_config.get("model_type")
    if architectures != EXPECTED_ARCHITECTURES or model_type != EXPECTED_MODEL_TYPE:
        raise ValueError(
            "source config does not look like the post-trained Qwen3.5 MoE checkpoint: "
            f"architectures={architectures!r} model_type={model_type!r}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "models" / "Qwen3.5-35B-A3B",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=PROJECT_ROOT / "models" / "Qwen3.5-35B-A3B-vllm",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    dest = args.dest.resolve()
    if not (source / "config.json").exists():
        raise FileNotFoundError(f"missing source config: {source / 'config.json'}")
    if dest.exists() and not args.force:
        print(f"[convert] dest already exists: {dest} (use --force to overwrite)")
        return 0

    if dest.exists() and args.force:
        import shutil

        shutil.rmtree(dest)

    dest.mkdir(parents=True, exist_ok=True)

    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    validate_source_config(source_config)
    (dest / "config.json").write_text(
        json.dumps(source_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for path in source.iterdir():
        if path.name == "config.json" or path.name.endswith(".aria2"):
            continue
        target = dest / path.name
        if target.exists() or target.is_symlink():
            continue
        os.symlink(path, target)

    print(f"[convert] wrote {dest}")
    print(f"[convert] architectures={source_config['architectures']}")
    print(f"[convert] model_type={source_config['model_type']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
