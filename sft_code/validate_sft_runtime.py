"""Check the Qwen3.5 model and minimum GatedDeltaNet CP capabilities before torchrun.

This is a local, single-process preflight, not a GPU/memory/convergence test.
--model-only checks config.json without importing the training frameworks.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


# (layers, hidden size, full-attention heads, linear key heads, linear value heads)
MODEL_SHAPES = {
    "4b": (32, 2560, 16, 16, 32),
    "9b": (32, 4096, 16, 16, 32),
    "27b": (64, 5120, 24, 16, 48),
    "35b_a3b": (40, 2048, 16, 16, 32),
}


def validate_model(model: Path, size: str, tp: int, pp: int, cp: int, ep: int) -> None:
    if min(tp, pp, cp, ep) < 1:
        raise ValueError("TP, PP, CP and EP must be positive integers")
    config = json.loads((model / "config.json").read_text())
    text = config.get("text_config", config)
    keys = ("num_hidden_layers", "hidden_size", "num_attention_heads", "linear_num_key_heads", "linear_num_value_heads")
    expected = dict(zip(keys, MODEL_SHAPES[size]))
    mismatches = {key: (text.get(key), value) for key, value in expected.items() if text.get(key) != value}
    if mismatches:
        raise ValueError(f"The checkpoint does not match Qwen3.5-{size}: {mismatches}")
    moe = size == "35b_a3b"
    architecture = "Qwen3_5MoeForConditionalGeneration" if moe else "Qwen3_5ForConditionalGeneration"
    if architecture not in config.get("architectures", []):
        raise ValueError(f"The default loader requires the full HF architecture {architecture}, not a text-only export")
    if moe:
        if text.get("num_experts") != 256 or 256 % ep:
            raise ValueError("35B-A3B requires 256 experts divisible by EP")
    elif ep != 1 or text.get("num_experts", 0):
        raise ValueError("A Dense checkpoint requires EP=1 and no MoE experts")
    if text["num_hidden_layers"] % pp or text["hidden_size"] % tp or text["num_attention_heads"] % tp:
        raise ValueError("Layer/PP or hidden-size/attention-head/TP partition is invalid")
    if text["linear_num_key_heads"] % (tp * cp) or text["linear_num_value_heads"] % (tp * cp):
        raise ValueError("Headwise GatedDeltaNet CP requires linear key/value heads divisible by TP*CP")


def validate_runtime(cp: int) -> None:
    # Imports inspect the selected installation; no distributed process group is started.
    for name in ("swift.cli._megatron.sft", "transformer_engine.pytorch", "fla", "causal_conv1d"):
        try:
            importlib.import_module(name)
        except Exception as exc:
            raise RuntimeError(f"Unable to import {name}: {type(exc).__name__}: {exc}") from exc
    try:
        importlib.import_module("mcore_bridge.model.mm_gpts.qwen3_5_gdn")
        bridge = importlib.import_module("mcore_bridge.model.modules.gated_delta_net")
        core = importlib.import_module("megatron.core.ssm.gated_delta_net")
    except Exception as exc:
        raise RuntimeError(f"A compatible Qwen3.5 MCore GDN bridge is required: {exc}") from exc
    if not hasattr(bridge, "GatedDeltaNet"):
        raise RuntimeError("mcore-bridge does not expose its GatedDeltaNet implementation")
    if cp > 1:
        missing = [name for name in ("tensor_a2a_cp2hp", "tensor_a2a_hp2cp", "get_parameter_local_cp")
                   if not callable(getattr(core, name, None))]
        if missing:
            raise RuntimeError(
                "Selected Megatron lacks GatedDeltaNet CP helpers: " + ", ".join(missing)
                + ". Select a compatible Megatron/mcore-bridge installation with SWIFT_VENV or MEGATRON_ROOT; "
                  "a CP1-only CPT environment is insufficient."
            )
    print(f"Validated minimum GDN CP capabilities; selected MCore module: {core.__file__}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-size", choices=MODEL_SHAPES, required=True)
    for name in ("tp", "pp", "cp", "ep"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--model-only", action="store_true")
    args = parser.parse_args()
    try:
        validate_model(args.model, args.model_size, args.tp, args.pp, args.cp, args.ep)
        if not args.model_only:
            validate_runtime(args.cp)
    except Exception as exc:
        parser.exit(1, f"SFT preflight failed: {exc}\n")
    print("SFT model/topology preflight passed (no GPU training performed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
