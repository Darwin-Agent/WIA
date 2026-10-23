"""Configuration and model registry for the world-model Eval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os

EVAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(EVAL_ROOT.parent)))
OUTPUT_ROOT = EVAL_ROOT / "outputs"
PROMPT_ROOT = Path(os.environ.get("JUDGE_PROMPT_DIR", str(EVAL_ROOT.parent / "prompts")))

DEFAULT_EVAL_DATA = Path(
    f"{PROJECT_ROOT}/data/id_test_data.jsonl"
)
ALT_EVAL_DATA = Path(f"{PROJECT_ROOT}/data/ood_test_data.jsonl")

DEFAULT_SOURCE_CODE_ROOT = Path(f"{PROJECT_ROOT}/rl_miles")
DEFAULT_RL_PYTHON = EVAL_ROOT / ".venv" / "bin" / "python"
DEFAULT_VLLM_BIN = Path(
    str(EVAL_ROOT / ".venv" / "bin" / "vllm")
)

OBS_MARKER = "###[Observation]"
IM_END = "<|im_end|>"
IM_START = "<|im_start|>"
END_OF_TEXT = "<|endoftext|>"
IM_END_TOKEN_ID = 248046
IM_START_TOKEN_ID = 248045
END_OF_TEXT_TOKEN_ID = 248044
DEFAULT_MAX_MODEL_LEN = 81920
DEFAULT_MAX_PROMPT_TOKENS = 65536
DEFAULT_MAX_RESPONSE_TOKENS = 16384
DEFAULT_PROMPT_SAFETY_MARGIN = 0
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = -1
DEFAULT_REQUEST_CONCURRENCY = 128

DEFAULT_JUDGE_BASE_URL = "http://<JUDGE_HOST>:30000/v1"
DEFAULT_JUDGE_MODEL = "DeepSeek-V4-Flash-0731"
DEFAULT_JUDGE_API_KEY = "EMPTY"
DEFAULT_JUDGE_MAX_TOKENS = 4096
DEFAULT_JUDGE_TIMEOUT = 1800.0
DEFAULT_JUDGE_CONCURRENCY = 64

CORE_DIMENSIONS = [
    "format",
    "factuality",
    "consistency",
    "realism",
    "quality",
    "efficiency",
]
DEFAULT_WEIGHTS = {
    "format": 1,
    "factuality": 2,
    "consistency": 2,
    "realism": 2,
    "quality": 2,
    "efficiency": 1,
}


@dataclass(frozen=True)
class EngineSpec:
    tensor_parallel_size: int = 8
    gpu_memory_utilization: float = 0.80
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    extra_server_args: tuple[str, ...] = (
        "--language-model-only",
        "--ignore-patterns",
        "re:visual.*",
    )


SAFE_QWEN35_SERVER_ARGS = (
    "--language-model-only",
    "--ignore-patterns",
    "re:visual.*",
    "--no-enable-prefix-caching",
    "--gdn-prefill-backend",
    "triton",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: Path
    template_model: Path
    engine: EngineSpec = field(default_factory=EngineSpec)
    stop_strings: tuple[str, ...] = (IM_END, IM_START, END_OF_TEXT)
    stop_token_ids: tuple[int, ...] = (IM_END_TOKEN_ID, IM_START_TOKEN_ID, END_OF_TEXT_TOKEN_ID)
    chat_template: Path | None = None


def _p(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


MODELS: dict[str, ModelSpec] = {
    "Qwen-AgentWorld": ModelSpec(
        name="Qwen-AgentWorld",
        checkpoint=_p("SRL", "Qwen-AgentWorld"),
        template_model=_p("SRL", "Qwen-AgentWorld"),
        engine=EngineSpec(
            tensor_parallel_size=4,
            max_model_len=262144,
            extra_server_args=(
                "--language-model-only",
                "--reasoning-parser",
                "qwen3",
            ),
        ),
    ),
    "qwen-agent-world-35b": ModelSpec(
        name="qwen-agent-world-35b",
        checkpoint=_p("models", "qwen-agent-world-35b"),
        template_model=_p("models", "qwen-agent-world-35b"),
        engine=EngineSpec(
            tensor_parallel_size=8,
            max_model_len=262144,
            extra_server_args=(
                "--language-model-only",
                "--reasoning-parser",
                "qwen3",
            ),
        ),
    ),
    "Qwen3.5-4B": ModelSpec(
        name="Qwen3.5-4B",
        checkpoint=_p("models", "Qwen3.5-4B"),
        template_model=_p("models", "Qwen3.5-4B"),
        engine=EngineSpec(extra_server_args=SAFE_QWEN35_SERVER_ARGS),
    ),
    "DeepSeek-V4-Flash-0731": ModelSpec(
        name="DeepSeek-V4-Flash-0731",
        checkpoint=_p("models", "DeepSeek-V4-Flash-0731"),
        template_model=_p("models", "DeepSeek-V4-Flash-0731"),
        engine=EngineSpec(
            tensor_parallel_size=8,
            max_model_len=81920,
        ),
        chat_template=_p("models", "DeepSeek-V4-Flash-0731", "chat_templates", "model.jinja"),
    ),
    "Qwen3.8-27B": ModelSpec(
        name="Qwen3.8-27B",
        checkpoint=_p("models", "Qwen3.8-27B"),
        template_model=_p("models", "Qwen3.8-27B"),
    ),
    "Qwen3.8-Flash-Next": ModelSpec(
        name="Qwen3.8-Flash-Next",
        checkpoint=_p("models", "Qwen3.8-Flash-Next"),
        template_model=_p("models", "Qwen3.8-Flash-Next"),
        engine=EngineSpec(
            tensor_parallel_size=8,
            max_model_len=81920,
            extra_server_args=(
                "--language-model-only",
                "--reasoning-parser",
                "qwen3",
            ),
        ),
    ),
    "GLM-5.3-Flash": ModelSpec(
        name="GLM-5.3-Flash",
        checkpoint=_p("models", "GLM-5.3-Flash"),
        template_model=_p("models", "GLM-5.3-Flash"),
        engine=EngineSpec(
            tensor_parallel_size=8,
            max_model_len=81920,
            extra_server_args=("--language-model-only",),
        ),
        stop_strings=(),
        stop_token_ids=(),
    ),
    "Qwen3.5-4B-Base": ModelSpec(
        name="Qwen3.5-4B-Base",
        checkpoint=_p("models", "Qwen3.5-4B-Base"),
        template_model=_p("models", "Qwen3.5-4B-Base"),
        engine=EngineSpec(extra_server_args=SAFE_QWEN35_SERVER_ARGS),
    ),
    "Qwen3.5-27B": ModelSpec(
        name="Qwen3.5-27B",
        checkpoint=_p("models", "Qwen3.5-27B"),
        template_model=_p("models", "Qwen3.5-27B"),
    ),
    "Qwen3.5-35B-A3B-vllm": ModelSpec(
        name="Qwen3.5-35B-A3B-vllm",
        checkpoint=_p("models", "Qwen3.5-35B-A3B-vllm"),
        template_model=_p("models", "Qwen3.5-35B-A3B-vllm"),
    ),
    "Qwen3.5-35B-A3B": ModelSpec(
        name="Qwen3.5-35B-A3B",
        checkpoint=_p("models", "Qwen3.5-35B-A3B"),
        template_model=_p("models", "Qwen3.5-35B-A3B"),
    ),
    "Qwen3.5-122B-A10B": ModelSpec(
        name="Qwen3.5-122B-A10B",
        checkpoint=_p("models", "Qwen3.5-122B-A10B"),
        template_model=_p("models", "Qwen3.5-122B-A10B"),
    ),
    "Qwen3.5-9B": ModelSpec(
        name="Qwen3.5-9B",
        checkpoint=_p("models", "Qwen3.5-9B"),
        template_model=_p("models", "Qwen3.5-9B"),
    ),
    "Qwen3.5-9B-Base": ModelSpec(
        name="Qwen3.5-9B-Base",
        checkpoint=_p("models", "Qwen3.5-9B-Base"),
        template_model=_p("models", "Qwen3.5-9B-Base"),
    ),
    "Qwen3.5-35B-A3B-Base": ModelSpec(
        name="Qwen3.5-35B-A3B-Base",
        checkpoint=_p("models", "Qwen3.5-35B-A3B-Base"),
        template_model=_p("models", "Qwen3.5-35B-A3B-Base"),
    ),
}
