# CPT: continual pretraining

**English** | [简体中文](README_zh.md) · [Home](../README.md) · [Installation](../ENVIRONMENT.md) · [Configuration](../docs/CONFIGURATION.md)

Learn actions and environment feedback from trajectories. Message masks select assistant-only training or joint
later-user + assistant content training. No judge or similarity service is required.

## Required framework installation

**CPT cannot train without ms-swift and the matching training dependencies.** This source release does not bundle them.
After preparing the GPU base in [Installation](../ENVIRONMENT.md), run from the repository root:

```bash
bash cpt/setup_runtime.sh cpt
```

The installer fetches pinned Swift and mcore-bridge into `cpt/ms-swift` and `cpt/mcore-bridge`, applies the shipped
Swift patches and installs the CPT dependencies in `cpt/ms-swift-venv`. No manual clone of the latest Swift is needed.
The patches are required for the project's loss-mask behavior; an arbitrary upstream installation is not equivalent.
`bash cpt/setup_runtime.sh --help` explains paths without installing. Reuse a matching environment via `SWIFT_ROOT`
and `SWIFT_VENV`; use the same paths for installation and launch. Validate versions, patches and imports before reuse.
Models, datasets and multi-node resources are separate prerequisites; see [installation and reuse](../ENVIRONMENT.md#installation-and-reuse).

## Models and defaults

| Model | Launcher | Nodes × GPUs | Parallelism | Dense DP | Micro batch | Global batch | Accumulation |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 4B | `train_qwen35_4b_cpt_64gpu.sh` | 8 × 8 | HF SP4, ZeRO-1 | 16 | 1 | 1024 | 64 |
| 9B | `train_qwen35_9b_cpt_64gpu.sh` | 8 × 8 | TP4 / PP1 / CP1 | 16 | 1 | 1024 | 64 |
| 27B | `train_qwen35_27b_cpt_128gpu.sh` | 16 × 8 | TP8 / PP1 / CP1 | 16 | 1 | 1024 | 64 |
| 35B-A3B | `train_qwen35_35b_a3b_cpt_128gpu.sh` | 16 × 8 | TP2 / PP8 / CP1 / EP2 / ETP1 / SP | 8 | 1 | 1024 | 128 |

All profiles default to `MAX_LENGTH=32768`, `LEARNING_RATE=2e-5` and left truncation.
For Megatron, dense/attention DP = GPUs/(TP×PP×CP); accumulation = global batch/(dense DP×micro batch).
Do not divide Dense DP by EP again. HF SP for 4B is a separate sequence-sharding mechanism;
Megatron's boolean sequence parallelism operates inside TP groups.
All recommended CPT profiles use GBS 1024. The 4B profile halves the original multinode allocation from 128 to 64 GPUs
and doubles accumulation from 32 to 64, preserving GBS. The 35B Megatron profile uses dense DP8,
expert DP8 and accumulation128.
`train_qwen35_35b_a3b_cpt_128gpu.sh` uses TP2/PP8/CP1/EP2/ETP1, SP=true, dense/attention DP8,
expert DP8, micro batch1, GBS1024 and accumulation128. EP is not an additional divisor of dense DP;
it creates a separate expert-parallel group. Do not mix topology settings from different backends.

## Data and loss masks

Use JSONL with one `messages` record per line. Synthetic example:

```json
{"messages":[{"role":"system","content":"Predict environment feedback.","loss_mask":false},{"role":"user","content":"Initial state.","loss_mask":false},{"role":"assistant","content":"Perform an action.","loss_mask":true},{"role":"user","content":"The state changes.","loss_mask":true},{"role":"assistant","content":"Next action.","loss_mask":true}]}
```

- **Joint training:** mask system/first user; later content participates wherever its mask is true.
- **Assistant-only:** mask every user/system and enable desired assistant messages.
- Message-level `loss` or `loss_mask` accepts booleans or 0/1; a top-level mask list must have one item per message.
- Missing flags default to assistant-only. Conflicting fields, enabled system/first-user masks and token-array masks are rejected.
- `TRAIN_USER_LOSS=true` permits explicit later-user flags, not all user text. Use lowercase `true/false`.
  False restores assistant-only behavior. Masked messages remain input context.
- User role/template delimiters stay masked; native assistant template/EOS behavior is retained.

Validate before preprocessing to avoid non-strict framework processing filtering invalid rows:

```bash
python3 -S cpt/prepare_loss_masks.py --input /path/to/raw.jsonl --output /path/to/canonical.jsonl
```

## Quick start

Prepare the [CPT environment](../ENVIRONMENT.md). From the repository root, for 9B:

```bash
export MODEL_PATH=/path/to/qwen35-9b-hf
export DATASET_PATH='/path/to/cpt/*.jsonl'
export OUTPUT_DIR=/path/to/checkpoints/cpt-9b
export NNODES=8 NODE_RANK=0 RESOURCE_GPU=8 NPROC_PER_NODE=8
export MASTER_ADDR=rank0.example.invalid MASTER_PORT=29500
export USE_CACHED_DATASET=false
DRY_RUN=1 bash cpt/train_qwen35_9b_cpt_64gpu.sh
```

Replace the example hostname. Run once on every node with a unique `NODE_RANK` in `0..NNODES-1`.
Remove `DRY_RUN=1` to train. Do not wrap the launcher in another torchrun.
`WORLD_SIZE/RANK` are optional **node-count/node-rank** aliases, not worker counts;
they must agree with `NNODES/NODE_RANK` when both are supplied.

### Precompute caches

All recommended profiles read cached data by default. Set `USE_CACHED_DATASET=false` for direct JSONL loading, or export a cache first:

```bash
MODEL_PATH=/path/to/qwen35-9b-hf DATASET_PATH='/path/to/cpt/*.jsonl' \
OUTPUT_DIR=/path/to/cache/cpt-9b TRAIN_USER_LOSS=true \
bash cpt/preprocess_cpt_cache.sh
```

Here `OUTPUT_DIR` means the exported cache, not checkpoints. Then train with
`USE_CACHED_DATASET=true CACHED_DATASET=/path/to/cache/cpt-9b/train`.
Export is single-process, requires complete local model/tokenizer files and uses offline loading.
The 35B Megatron profile reads `DATASET_PATH` directly.
**Regenerate caches after changing the template, length, masks, patches or TRAIN_USER_LOSS.** Old labels are not rewritten.

## Overrides

Use `export VAR=value` or `VAR=value bash ...`. See the [complete index](../docs/CONFIGURATION.md)
for exact default expressions and scope. Do not assume arbitrary CLI flags are forwarded.

| Category | Variables and defaults |
| --- | --- |
| Model/data | `MODEL_PATH`: complete HF model; `DATASET_PATH`: JSONL/glob, quote globs to avoid premature expansion |
| Outputs/cache | `OUTPUT_DIR`, `CACHE_ROOT`; all recommended profiles: `USE_CACHED_DATASET`, `CACHED_DATASET`, `CACHED_DATASET_ROOT` |
| Environment | `PROJECT_ROOT`, `CODE_ROOT`, `SWIFT_ROOT`, `SWIFT_VENV`; repository-relative defaults |
| Nodes | `NNODES`, `NODE_RANK`, `RESOURCE_GPU`, `NPROC_PER_NODE`, `CUDA_VISIBLE_DEVICES`, `MASTER_ADDR/MASTER_PORT` |
| Objective | `TRAIN_USER_LOSS=true`; HF 4B `LOSS_SCALE=default`; understand interactions before changing loss scaling |
| Length | `MAX_LENGTH=32768`, `TRUNCATION_STRATEGY=left` |
| Duration | Configured through scripts/environment; Megatron supports `TRAIN_ITERS` |
| HF batch | 4B: micro batch1, accumulation64, SP4; conflicting accumulation is rejected |
| Megatron batch | `MICRO_BATCH_SIZE=1`, `GLOBAL_BATCH_SIZE` from table; require divisibility by DP×micro batch |
| Megatron topology | `TENSOR_MODEL_PARALLEL_SIZE`, `PIPELINE_MODEL_PARALLEL_SIZE`, `CONTEXT_PARALLEL_SIZE`; the 35B Megatron entry adds `EXPERT_MODEL_PARALLEL_SIZE`, `EXPERT_TENSOR_PARALLEL_SIZE` |
| Dense topology guard | 9B/27B require `ALLOW_TOPOLOGY_OVERRIDE=true` for nondefault grids; other divisibility constraints still apply |
| Learning rate | `LEARNING_RATE=2e-5`; Megatron `MIN_LR=2e-6`, `LR_WARMUP_FRACTION=0.03`; HF 4B `WARMUP_RATIO=0.03` |
| Saving | `SAVE_STEPS`: HF 4B 100, 9B/27B 500, 35B Megatron 200; HF also `SAVE_TOTAL_LIMIT=100` |
| Data workers | `DATASET_NUM_PROC`, `DATALOADER_NUM_WORKERS`; export defaults to 16 data processes, training to 8 |
| Processing cache | Training `LOAD_FROM_CACHE_FILE=false`; not synonymous with `USE_CACHED_DATASET` |
| HF performance | `DEEPSPEED=zero1`, `ATTN_IMPL=flash_attn`, `PACKING=false`, `GROUP_BY_LENGTH=false`; `PADDING_FREE=false` for 4B |
| Megatron offload | `OPTIMIZER_CPU_OFFLOAD`: false for 9B/27B, true for 35B Megatron; `OPTIMIZER_OFFLOAD_FRACTION=1`, `USE_PRECISION_AWARE_OPTIMIZER=true` |
| Dense preflight | `NCCL_PREFLIGHT_ONLY=true` launches GPU/communication workers unless also in dry-run mode |

Example, after configuring paths/nodes:

```bash
GLOBAL_BATCH_SIZE=1024 TRAIN_ITERS= \
bash cpt/train_qwen35_9b_cpt_64gpu.sh
```

## Outputs and next steps

Megatron CPT sets `save_safetensors=false`; do not assume its checkpoint directory is a full HF export.
Use a compatible native checkpoint or export HF config/tokenizer/weights before [SFT](../sft_code/README.md).
Compatibility aliases `train_cpt_swift.sh`, `train_cpt_megatron_dense.sh`, `train_cpt_megatron_moe.sh`
select 4B HF, 27B Megatron and 35B Megatron, not single-node smoke profiles.
The recommended 35B entry is `train_qwen35_35b_a3b_cpt_128gpu.sh`.
