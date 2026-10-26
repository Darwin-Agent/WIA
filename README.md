# World Model

**English** | [简体中文](README_zh.md)

**Learn environment feedback from trajectories: CPT → SFT → asynchronous RL → evaluation.**

Train Qwen3.5 language world models to predict the next environment observation from a history of interactions.
This repository provides continual pretraining, supervised fine-tuning, Miles-based async GSPO,
reward services, and domain-aware evaluation in one source release.

[Quick start](#quick-start) · [Installation](ENVIRONMENT.md) · [Model configurations](#models-and-resources) ·
[Configuration reference](docs/CONFIGURATION.md) · [Services](#required-services) · [Documentation](docs/README.md)

## Required installation before training

**This is a source release, not a ready-to-run environment. CPT/SFT require ms-swift; RL requires Miles and its companion frameworks. Training cannot run without these dependencies.**
Choose the stage you need; you do not need every training framework just to run one stage.
After preparing the compatible GPU base described in [Installation](ENVIRONMENT.md), run from the repository root:

| Stage | Required installation command | Default framework source locations |
| --- | --- | --- |
| CPT | `bash cpt/setup_runtime.sh cpt` | `cpt/ms-swift`, `cpt/mcore-bridge` |
| SFT | `bash sft_code/setup_runtime.sh` | Shared Swift/bridge sources above; `rl_miles/Megatron-LM` for CP support |
| RL | `bash rl_miles/install_official_miles_env.sh` | `rl_miles/miles`, `rl_miles/Megatron-LM`, `rl_miles/Megatron-Bridge`, `rl_miles/sglang-miles` |

These installers fetch pinned sources, apply the included framework patches and install dependencies.
You do not need to manually clone arbitrary upstream versions. The RL installer calls `bootstrap_miles.sh` itself;
bootstrap alone prepares sources, **not** the Python/GPU environment. `--help` on these installers is read-only.
A matching existing environment can be reused after version/patch checks. Custom source/environment paths must be
consistent between installation and launch; see [installation and reuse](ENVIRONMENT.md#installation-and-reuse).
Weights, datasets, cluster configuration and required judge/similarity services must still be prepared separately.

## Features

- **CPT:** assistant-only or explicitly masked later-user + assistant training.
- **SFT:** CP4 Dense/MoE launchers with CPT checkpoint handoff guidance.
- **RL:** separate actor/rollout GPU pools, async generation, weight synchronization and resumable training.
- **Reward services:** DeepSeek-V4-Flash judge and optional Qwen3 Embedding similarity.
- **Evaluation:** local vLLM or user-configured compatible inference, CoT controls, domain metrics and resume.

The release includes source, launchers, prompts and framework patches. **Weights, datasets, virtual environments,
running services and credentials are not included.** Configure them before launching a job.

## Workflow

```text
Trajectories / messages → CPT → HF or compatible Megatron checkpoint
                                      ↓
                                  SFT (CP4)
                                      ↓ HF export
                              Miles async GSPO
                              ├─ Megatron actor
                              ├─ SGLang rollout
                              └─ judge + optional similarity
                                      ↓
                              Evaluation / reports
```

## Models and resources

Defaults assume 8 GPUs per node. **Dedicated judge/embedding GPUs are additional.** TP, PP and CP mean tensor,
pipeline and context parallelism; EP/ETP mean expert/expert-tensor parallelism. Dense DP does not divide by EP again.

### CPT

| Model | GPUs / nodes | Backend and parallelism | Dense DP | Global batch |
| --- | --- | --- | ---: | ---: |
| 4B | 64 / 8 | HF Swift; SP4, ZeRO-1 | 16 | 1024 |
| 9B | 64 / 8 | Megatron; TP4 / PP1 / CP1 | 16 | 1024 |
| 27B | 128 / 16 | Megatron; TP8 / PP1 / CP1 | 16 | 1024 |
| 35B-A3B | 128 / 16 | Megatron; TP2 / PP8 / CP1 / EP2 / ETP1 / SP | 8 | 1024 |

HF SP4 and Megatron's boolean sequence parallelism are different mechanisms. The 35B row uses
`cpt/train_qwen35_35b_a3b_cpt_128gpu.sh` with GBS 1024; EP is an expert-routing topology and is not
an additional factor in Dense DP. See [CPT](cpt/README.md) for the topology and accumulation settings.

### SFT

| Model | GPUs / nodes | TP / PP / CP | EP / ETP | Dense DP | Global batch |
| --- | --- | --- | --- | ---: | ---: |
| 4B | 8 / 1 | 2 / 1 / 4 | 1 / 1 | 1 | 128 |
| 9B | 8 / 1 | 2 / 1 / 4 | 1 / 1 | 1 | 128 |
| 27B | 64 / 8 | 4 / 4 / 4 | 1 / 1 | 1 | 128 |
| 35B-A3B | 32 / 4 | 2 / 4 / 4 | 8 / 1 | 1 | 128 |

Micro batch is 1 and accumulation is 128. A GatedDeltaNet CP-capable runtime is required. See [SFT](sft_code/README.md).

### RL

| Model | Total GPUs | Actor / rollout GPUs | Actor TP / PP / CP | EP / ETP | Actor Dense DP | GPUs/rollout engine × engines | Global batch |
| --- | ---: | --- | --- | --- | ---: | --- | ---: |
| 4B | 128 | 64 / 64 | 2 / 1 / 4 | 1 / 1 | 8 | 1 × 64 | 1024 |
| 9B | 128 | 64 / 64 | 2 / 2 / 4 | 1 / 1 | 4 | 2 × 32 | 2048 |
| 27B | 128 | 64 / 64 | 4 / 4 / 4 | 1 / 1 | 1 | 2 × 32 | 2048 |
| 35B-A3B | 128 | 64 / 64 | 2 / 4 / 4 | 8 / 1 | 2 | 2 × 32 | 2048 |
| 122B-A10B | 160 | 96 / 64 | 2 / 12 / 4 | 8 / 1 | 1 | 8 × 8 | 2048 |

122B-A10B has only a 160-GPU RL launcher; CPT/SFT launchers are not provided.
All RL profiles default to `MAX_WEIGHT_STALENESS=1` and `ASYNC_DATA_BUFFER_CAPACITY_FACTOR=1.0`.
Dynamic token batching determines actual microbatch partitions; static accumulation arithmetic is insufficient.
See [RL](rl_miles/README.md).

## Required services

| Stage | Training/inference runtime | Separately deployed services |
| --- | --- | --- |
| CPT / SFT | Swift, Megatron; DeepSpeed for 4B/35B CPT | None |
| RL: 4B / 9B / 27B / 35B-A3B | Ray + Megatron actor + SGLang rollout | LLM judge; similarity disabled by default |
| RL: 122B-A10B | Ray + Megatron actor + SGLang rollout | LLM judge + similarity API + embedding backend |
| Evaluation | Local vLLM or user-configured compatible server | LLM judge; `--skip-judge` generates without scoring |

Miles manages policy rollout; this is **not** the separately deployed DeepSeek judge.
Deployment: [DeepSeek judge](rl_miles/DS-V4-Flash/README.md) · [Embedding/similarity](rl_miles/Qwen3-embedding/README.md).

| Model | CoT | Judge / similarity weights | Similarity service |
| --- | --- | --- | --- |
| 4B / 9B / 27B / 35B-A3B | Off | 1.0 / 0.0 | Not required by default |
| 122B-A10B | On | 0.9 / 0.1 | Required by default |

RL multiplies the final reward by 0.9 when thinking exceeds **10,000 characters**. Independent evaluation does not.
See [reward configuration](rl_miles/README.md#reward-configuration) for failure fallbacks and weight overrides.

## Ablation scope

The published ablation tables provide complete CPT-to-SFT-to-RL stage comparisons for 4B, 9B, and 35B-A3B. For 27B, the release reports only the starting Qwen3.5-27B checkpoint and the final WIA-WM-27B checkpoint; intermediate stage rows are not included. Therefore, 27B results should be interpreted as a starting-checkpoint-to-final-model comparison, not as separate CPT, SFT, and RL contribution estimates.

## Quick start

### 1. Prepare an environment

Run commands from the repository root:

```bash
export PROJECT_ROOT="$PWD"
```

Follow [Installation](ENVIRONMENT.md). CPT/SFT use a CUDA12 stack; RL uses a separate CUDA13 stack.
Keep complete HF config, tokenizer and weight files together.

### 2. Fill in your resources

| Resource | CPT / SFT | RL | Evaluation |
| --- | --- | --- | --- |
| Model | `MODEL_PATH`; optional SFT `MCORE_MODEL` | `MILES_HF_CHECKPOINT`, `MILES_LOAD` | `--checkpoint`, optional `--template-model` |
| Data | `DATASET_PATH`; optional CPT `CACHED_DATASET` | `MILES_PROMPT_DATA`, `EVAL_DATA` | `--eval-data` |
| Outputs | `OUTPUT_DIR` | `MILES_SAVE` | `--output-root`, `--run-name` |
| Nodes | `NNODES`, `NODE_RANK`, `MASTER_ADDR` | `MASTER_ADDR`; per-node `WORLD_SIZE/RANK` or `HOSTFILE` | `--num-gpus` or `--rollout-base-url` |
| Judge | Not required | `JUDGE_BASE_URL`, `JUDGE_MODEL` | Corresponding environment variables or CLI |
| Similarity | Not required | `SIMILARITY_URL`, `SIMILARITY_ENABLED` | Not used by the main evaluator |

Data contracts: [CPT](cpt/README.md#data-and-loss-masks), [SFT](sft_code/README.md#data-and-checkpoints),
[RL](rl_miles/README.md#data-and-checkpoints). Replace `/path/to/...` and `*.example.invalid` examples.
Inject real credentials through the environment, never through committed source or configuration.

### 3. Launch a stage

- [CPT: preprocessing, caching and multi-node launch](cpt/README.md#quick-start)
- [SFT: load CPT weights and fine-tune with CP4](sft_code/README.md#quick-start)
- [RL: deploy scorers and launch the Ray cluster](rl_miles/README.md#quick-start)
- [Evaluation: generate, score, resume and report](eval/README.md#quick-start)

Inspect `--help` / `DRY_RUN=1` first, then validate model loading, communication and service requests on a small run.
RL dry-run still requires readable HF `config.json`. Dry-run does not establish GPU readiness.

## Configuration overrides

Environment defaults, model-wrapper overrides and supported CLI flags differ by entry point.
Use documented environment variables rather than assuming arbitrary flags are forwarded.
The [configuration reference](docs/CONFIGURATION.md) includes default expressions, entry-point scope and source locations.

## Repository map

| Directory | Purpose |
| --- | --- |
| [cpt](cpt/README.md) | Masks, cache export and four CPT launchers |
| [sft_code](sft_code/README.md) | Four SFT launchers and CP/topology validation |
| [rl_miles](rl_miles/README.md) | Five async GSPO profiles, data loader and rewards |
| [eval](eval/README.md) | Evaluation, dual judges, rejudging and reports |
| [prompts](prompts/README.md) | Judge templates for seven domains |
| [runtime](runtime/README.md) | Pinned sources, patches and runtime checks |
| [docs](docs/README.md) | Documentation navigation and parameter index |

## Operational boundaries

- GPU counts are allocations, not guarantees for every GPU architecture or memory size. Validate driver, CUDA, ABI and communication.
- Use consistent source, environment, model and data paths across nodes; do not transplant a venv across incompatible base images.
- Restrict Ray dashboards and scorer endpoints to trusted networks.
- Never commit datasets, weights, logs, responses, host inventories or secrets; ignore rules do not replace a privacy review.
- Third-party frameworks, models and datasets have their own terms. Check their permissions before use.
