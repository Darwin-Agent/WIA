import os
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _extend_if_set(args: list[str], option: str, value: str) -> None:
    if value:
        args.extend([option, value])


def _enabled(name: str, default: str = "0") -> bool:
    return _env(name, default) in {"1", "true", "TRUE", "yes", "YES"}


def main() -> None:
    from argparse import Namespace

    from sglang.cli.serve import serve

    script_dir = Path(__file__).resolve().parent
    chat_template = _env(
        "CHAT_TEMPLATE",
        str(script_dir / "chat_templates" / "deepseek_v4_flash.jinja"),
    )
    reasoning_parser = _env("REASONING_PARSER", "deepseek-v4")
    tool_call_parser = _env("TOOL_CALL_PARSER", "deepseekv4")
    dist_init_addr = _env("DIST_INIT_ADDR")
    if not dist_init_addr:
        master_addr = _env("MASTER_ADDR")
        master_port = _env("MASTER_PORT")
        if master_addr and master_port:
            dist_init_addr = f"{master_addr}:{master_port}"

    args = Namespace()
    extra = [
        "--model-path", _env("MODEL_PATH", "/preset-models"),
        "--served-model-name", _env("SERVED_MODEL_NAME", "DeepSeek-V4-Flash-0731"),
        "--host", _env("HOST", "0.0.0.0"),
        "--port", _env("PORT", "30000"),
        "--tensor-parallel-size", _env("TP_SIZE", _env("RESOURCE_GPU", "8")),
        "--context-length", _env("CONTEXT_LENGTH", "81920"),
        "--attention-backend", _env("ATTENTION_BACKEND", "dsv4"),
        "--moe-runner-backend", _env("MOE_RUNNER_BACKEND", "marlin"),
        "--chat-template", chat_template,
        "--reasoning-parser", reasoning_parser,
        "--tool-call-parser", tool_call_parser,
        "--default-chat-template-kwargs", '{"thinking": false}',
        "--mem-fraction-static", _env("MEM_FRACTION_STATIC", "0.80"),
        "--max-running-requests", _env("MAX_RUNNING_REQUESTS", "128"),
        "--enforce-disable-flashinfer-allreduce-fusion",
        "--trust-remote-code",
    ]
    if _enabled("ENABLE_DISTRIBUTED"):
        _extend_if_set(extra, "--dist-init-addr", dist_init_addr)
        _extend_if_set(extra, "--nnodes", _env("NNODES", _env("WORLD_SIZE")))
        _extend_if_set(extra, "--node-rank", _env("NODE_RANK", _env("RANK")))
    _extend_if_set(extra, "--mm-feature-transport", _env("MM_FEATURE_TRANSPORT"))
    _extend_if_set(extra, "--cpu-offload-gb", _env("CPU_OFFLOAD_GB"))
    if _env("ENABLE_MEMORY_SAVER") in {"1", "true", "TRUE", "yes", "YES"}:
        extra.append("--enable-memory-saver")
    if _enabled("ENABLE_TWO_BATCH_OVERLAP"):
        extra.append("--enable-two-batch-overlap")
    if _enabled("ENABLE_MIXED_CHUNK"):
        extra.append("--enable-mixed-chunk")
    if _enabled("ENABLE_DYNAMIC_CHUNKING"):
        extra.append("--enable-dynamic-chunking")
    _extend_if_set(
        extra,
        "--chunked-prefill-size",
        _env("CHUNKED_PREFILL_SIZE", "4096"),
    )
    _extend_if_set(
        extra,
        "--max-prefill-tokens",
        _env("MAX_PREFILL_TOKENS", "8192"),
    )
    _extend_if_set(
        extra,
        "--cuda-graph-backend-decode",
        _env("CUDA_GRAPH_BACKEND_DECODE", "full"),
    )
    _extend_if_set(
        extra,
        "--cuda-graph-backend-prefill",
        _env("CUDA_GRAPH_BACKEND_PREFILL", "disabled"),
    )
    serve(args, extra)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
