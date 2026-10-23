# SFT: supervised fine-tuning

**English** | [简体中文](README_zh.md) · [Home](../README.md) · [Installation](../ENVIRONMENT.md) · [Configuration](../docs/CONFIGURATION.md)

Continue from CPT weights using Megatron-SWIFT and CP4 for long contexts. No external reward service is required.
SFT datasets normally enable assistant-only content loss.

## Required framework installation

**SFT requires ms-swift, mcore-bridge and a CP-capable Megatron runtime.** It cannot train from this ZIP alone.
After preparing the base environment in [Installation](../ENVIRONMENT.md), run from the repository root:

```bash
bash sft_code/setup_runtime.sh
```

This delegates to `cpt/setup_runtime.sh sft`, fetches/checks pinned Swift and bridge sources under `cpt/`, applies the
shared Swift patches and prepares CP-capable Megatron under `rl_miles/Megatron-LM`. It installs a separate `sft_code/.venv`;
the Megatron source location does **not** mean SFT requires the Miles RL environment or reward services.
You do not need to run the CPT installer first. Do not replace this environment with CPT's CP1-only setup.
For a custom environment, set `SFT_VENV` during installation and `SWIFT_VENV` to the same directory during training;
keep `SWIFT_ROOT` and `MEGATRON_ROOT` consistent. Reuse only after version/patch/import checks.
`bash sft_code/setup_runtime.sh --help` is read-only. See [installation and reuse](../ENVIRONMENT.md#installation-and-reuse).

## Models and defaults

| Model | Launcher | Nodes × GPUs | TP / PP / CP | EP / ETP | Dense DP | Global batch |
| --- | --- | --- | --- | --- | ---: | ---: |
| 4B | `train_qwen35_4b_sft_8gpu.sh` | 1 × 8 | 2 / 1 / 4 | 1 / 1 | 1 | 128 |
| 9B | `train_qwen35_9b_sft_8gpu.sh` | 1 × 8 | 2 / 1 / 4 | 1 / 1 | 1 | 128 |
| 27B | `train_qwen35_27b_sft_64gpu.sh` | 8 × 8 | 4 / 4 / 4 | 1 / 1 | 1 | 128 |
| 35B-A3B | `train_qwen35_35b_a3b_sft_32gpu.sh` | 4 × 8 | 2 / 4 / 4 | 8 / 1 | 1 | 128 |

Defaults: 2 epochs, micro batch 1, accumulation 128, length 65536, learning rate `1e-5`, cosine decay and CPU
optimizer offload. 35B expert DP = 32/(ETP1×EP8×PP4) = 1; CP is not multiplied into that expert grid.
Boolean sequence parallelism operates inside TP groups, not as another GPU dimension.

## Data and checkpoints

Use `messages` JSONL with the [CPT mask contract](../cpt/README.md#data-and-loss-masks). Assistant-only example:

```json
{"messages":[{"role":"user","content":"Predict the next observation.","loss_mask":false},{"role":"assistant","content":"The action succeeds.","loss_mask":true}]}
```

`TRAIN_USER_LOSS=true` only permits explicitly enabled later-user content. With false user masks, training remains
assistant-only. No separate SFT-only loss patch is used.

Two checkpoint paths are supported:

1. **Complete HF CPT export:** set `MODEL_PATH` to config, tokenizer and weights.
2. **Native Megatron checkpoint:** also set `MCORE_MODEL`; retain corresponding HF metadata/tokenizer in `MODEL_PATH`.

TP/PP/CP change between stages. Native resharding depends on the checkpoint format and runtime; export to HF first
if incompatible. Renaming or symlinking an arbitrary checkpoint does not convert it.
The launcher sets finetune, `no_load_optim=true`, `no_load_rng=true`: this initializes a stage, not full optimizer-state resume.

## Quick start

Install the [CP-capable SFT environment](../ENVIRONMENT.md). From the repository root, for 27B:

```bash
export MODEL_PATH=/path/to/cpt-27b-hf
export DATASET_PATH=/path/to/sft.jsonl
export OUTPUT_DIR=/path/to/checkpoints/sft-27b
export NNODES=8 NODE_RANK=0 RESOURCE_GPU=8 NPROC_PER_NODE=8
export MASTER_ADDR=rank0.example.invalid MASTER_PORT=29503
DRY_RUN=1 bash sft_code/train_qwen35_27b_sft_64gpu.sh
```

Replace the hostname, run on every node and vary `NODE_RANK`; remove dry-run after checking.
4B/9B use one node and may omit the master address (loopback default). 35B uses four nodes and its own launcher.
Do not add an outer torchrun. `WORLD_SIZE/RANK` are node-count/node-rank aliases.

## Overrides

Only `--help` is accepted as a CLI flag; configure training through environment variables.
Model wrappers fix the scale. Direct `train_sft.sh` invocation accepts `SFT_MODEL_SIZE=4b|9b|27b|35b_a3b`.
Exact defaults and additional parameters are in the [complete index](../docs/CONFIGURATION.md).

| Category | Variables and defaults |
| --- | --- |
| Paths | `MODEL_PATH`, optional `MCORE_MODEL`, `DATASET_PATH`, `OUTPUT_DIR`, `CACHE_ROOT` |
| Runtime | `SWIFT_ROOT`, `SWIFT_VENV` (default `sft_code/.venv`), optional `MEGATRON_ROOT`; installer venv override is `SFT_VENV` |
| Template | `MODEL_TYPE=qwen3_5` or `qwen3_5_moe`; `TEMPLATE=qwen3_5`, matching the checkpoint |
| Nodes | `NNODES/NODE_RANK`, `RESOURCE_GPU/NPROC_PER_NODE`, `CUDA_VISIBLE_DEVICES`, `MASTER_ADDR/MASTER_PORT` |
| Parallelism | `TENSOR_MODEL_PARALLEL_SIZE`, `PIPELINE_MODEL_PARALLEL_SIZE`, `CONTEXT_PARALLEL_SIZE=4`, `EXPERT_MODEL_PARALLEL_SIZE`, `EXPERT_TENSOR_PARALLEL_SIZE=1` |
| CP / SP | `CP_COMM_TYPE=p2p`, `USE_MCORE_GDN=1`; `SEQUENCE_PARALLEL=true` when TP>1 |
| Batch | `MICRO_BATCH_SIZE=1` (fallback `PER_DEVICE_TRAIN_BATCH_SIZE`), `GLOBAL_BATCH_SIZE=128` |
| Accumulation | Derived from global batch/(DP×micro batch); conflicting `GRADIENT_ACCUMULATION_STEPS` is rejected |
| Duration | `NUM_TRAIN_EPOCHS=2`; nonempty `TRAIN_ITERS` takes precedence |
| Length | `MAX_LENGTH=65536`, `TRUNCATION_STRATEGY=right` |
| Optimizer | `LEARNING_RATE=1e-5`, `MIN_LR=0`, `WARMUP_RATIO=0.03`, `WEIGHT_DECAY=0.1`, `MAX_GRAD_NORM=1.0` |
| Objective | `TRAIN_USER_LOSS=true`, `LOSS_SCALE=default`, `PACKING=false`, `PADDING_FREE=false` |
| Data loading | `DATASET_NUM_PROC=8`, `DATALOADER_NUM_WORKERS=4`, `LOAD_FROM_CACHE_FILE=false`, `DATASET_SHUFFLE=true` |
| Saving | `SAVE_STEPS=10`, `SAVE_TOTAL_LIMIT=20`, `LOGGING_STEPS=1`, `SAVE_SAFETENSORS=true` |
| Freezing | `FREEZE_VIT=true`, `FREEZE_ALIGNER=true`; full LLM fine-tuning |
| Offload | `OPTIMIZER_CPU_OFFLOAD=true`, `OPTIMIZER_OFFLOAD_FRACTION=1`, `USE_PRECISION_AWARE_OPTIMIZER=true` |

## Topology constraints

- GPU caps: 64 for 27B, 32 for the other sizes; these are not environment-overridable.
- GPUs must divide by TP×PP×CP; global batch must divide by Dense DP×micro batch.
- Layers divide across PP, hidden size and attention heads across TP.
- GDN key/value heads divide across TP×CP: do not reuse 27B CPT TP8 unchanged with CP4.
- `MAX_LENGTH` divides by 2×CP; CP>1 requires MCore GDN; only `CP_COMM_TYPE=p2p` is accepted.
- MoE additionally checks expert count, ETP and its expert rank grid. Arithmetic validity does not guarantee memory capacity.

## Outputs and next steps

Inspect checkpoints for complete config/tokenizer/weights before [RL](../rl_miles/README.md).
`SAVE_SAFETENSORS=true` alone is not a guarantee that every output is a portable full HF export.
`register_qwen3_5_text.py` is an optional HF helper, not loaded by the default Megatron entry.
