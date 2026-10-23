# DeepSeek-V4-Flash judge service

**English** | [简体中文](README_zh.md) · [Home](../../README.md) · [RL](../README.md) · [Configuration](../../docs/CONFIGURATION.md)

Provide OpenAI-compatible chat completions for RL and independent evaluation. This is a scoring service,
separate from policy rollout SGLang: use a dedicated environment and GPUs.

## Installation

Provide complete DeepSeek-V4-Flash weights, Python 3.12 and a compatible GPU/CUDA/PyTorch base stack.
From the repository root:

```bash
bash rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh install
```

The default `.venv` in this directory inherits system packages. Install in a dedicated container.
The recipe uses SGLang `a5c9636` plus [three patches](patches/README.md), the kernel package
`sglang-kernel==0.4.6.post1`, FlashInfer 0.6.13, Transformers 5.12.1 and xgrammar 0.2.1.
Do not mix these with the RL SGLang-Miles/FlashInfer stack.

## Start and verify

```bash
export MODEL_PATH=/path/to/DeepSeek-V4-Flash
export RESOURCE_GPU=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HOST=0.0.0.0 PORT=30000
export SERVED_MODEL_NAME=DeepSeek-V4-Flash-0731
KEEP_ENTRYPOINT_ALIVE=0 bash rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh serve
curl --noproxy '*' http://127.0.0.1:30000/health
curl --noproxy '*' http://127.0.0.1:30000/v1/models
```

`0.0.0.0` is a bind address, not a client destination. On training nodes configure a reachable host:

```bash
export JUDGE_BASE_URL=http://judge.example.invalid:30000/v1
export JUDGE_MODEL=DeepSeek-V4-Flash-0731
```

Replace the hostname, verify connectivity from training nodes, and match the model name returned by `/v1/models`.
Configure authentication at your service/gateway as needed; inject `JUDGE_API_KEY` at runtime.
Never expose unauthenticated scoring ports directly to the public internet.

## Parameters

| Variable | Default / purpose |
| --- | --- |
| `MODEL_PATH` | Set the actual weights path; `/preset-models` is only a container mount convention |
| `VENV_DIR` / `PYTHON_BIN` | Dedicated venv/interpreter; `USE_VENV=1` |
| `SGLANG_SRC` / `SGLANG_REPO` / `SGLANG_REF` | Source location, public origin and pinned revision; use a dedicated checkout |
| `RESOURCE_GPU` | Main shell entry derives TP from this, default 8; setting `TP_SIZE` alone does not override that entry |
| `CUDA_VISIBLE_DEVICES` | Must match the allocated GPU count |
| `HOST` / `PORT` / `SERVED_MODEL_NAME` | `0.0.0.0` / 30000 / `DeepSeek-V4-Flash-0731` |
| `CONTEXT_LENGTH` / `MEM_FRACTION_STATIC` | 81920 / 0.80 |
| `MAX_RUNNING_REQUESTS` | 128 |
| `CHUNKED_PREFILL_SIZE` / `MAX_PREFILL_TOKENS` | 4096 / 8192 |
| `ATTENTION_BACKEND` / `MOE_RUNNER_BACKEND` | `dsv4` / `marlin` |
| `CUDA_GRAPH_BACKEND_DECODE/PREFILL` | `full` / `disabled` |
| `ENABLE_TWO_BATCH_OVERLAP` / `ENABLE_MIXED_CHUNK` / `ENABLE_DYNAMIC_CHUNKING` | 0 / 0 / 1 in the main shell entry |
| `CHAT_TEMPLATE` / `REASONING_PARSER` / `TOOL_CALL_PARSER` | Bundled template / `deepseek-v4` / `deepseekv4` |
| `SKIP_RUNTIME_CHECK` | 0; do not bypass missing patches or incompatible imports with 1 |
| `STARTUP_TIMEOUT` / `HEALTH_INTERVAL` | 1800 / 10 seconds |
| `WARMUP_AFTER_READY` / `WARMUP_TIMEOUT` | 1 / 600 seconds |
| `KEEP_ENTRYPOINT_ALIVE` | 1; 0 returns after readiness while the service remains PID-managed |
| `LOG_FILE` / `PID_FILE` / `STOP_TIMEOUT` | Runtime paths / stop timeout 30 seconds |

See the [full index](../../docs/CONFIGURATION.md). Shell and Python defaults differ:
the direct Python runner also supports `ENABLE_DISTRIBUTED`, `DIST_INIT_ADDR`, `NNODES/NODE_RANK`
for actual multi-node model partitioning. This advanced path requires separate resource/network validation.

## Replicas and operations

`deploy_sglang_deepseek_v4_flash_4node.sh` / `start_sglang_deepseek_v4_flash_4node.sh`
run **independent model replicas** on each node, not one cross-node TP model.
Prepare environment, weights and rank per node. The node0..node3 scripts are compatibility aliases;
they do not fill ranks or addresses. RL can consume comma-separated judge URLs.

```bash
bash rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh status
bash rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh stop
```

Preserve any custom `PID_FILE/LOG_FILE/PORT` when managing the service.
Single-GPU/CPU-offload entries are experimental alternatives, not equivalent replacements for the default 8-GPU profile.
Do not publish runtime logs or deployment inventories.

## Subdirectories

- [chat_templates](chat_templates/README.md): served conversation template.
- [encoding](encoding/README.md): encoding/parsing reference.
- [patches](patches/README.md): pinned deployment patches.
