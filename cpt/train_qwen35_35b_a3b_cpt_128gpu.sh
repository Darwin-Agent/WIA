#!/usr/bin/env bash
set -euo pipefail
# Adapted from Code/CPT/train_qwen35_35b_a3b_world_model_sft_megatron_16node.sh; private paths removed.
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Qwen3.5-35B-A3B CPT: 16 nodes x 8 GPUs = 128 GPUs; NUM_TRAIN_EPOCHS=1.
Megatron MoE: TP=2, PP=8, CP=1, EP=2, ETP=1, SP=true, dense/attention DP=8, expert DP=8, GBS=1024, accumulation=128.
This is the Megatron-SWIFT MoE profile for 35B-A3B CPT.
Run once per node, with NNODES=16, NODE_RANK=<rank> and MASTER_ADDR=<rank-0-host>.
WORLD_SIZE/RANK are supported node-count/node-rank aliases, not worker counts.
Set MODEL_PATH, DATASET_PATH, SWIFT_ROOT and SWIFT_VENV.
DRY_RUN=1 prints the resolved command without loading dependencies or launching workers.
See cpt/README.md for data, cache, topology and environment requirements.
USAGE
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CODE_ROOT="${CODE_ROOT:-${SCRIPT_DIR}}"
CPT_ROOT="${CPT_ROOT:-${PROJECT_ROOT}/models}"
SWIFT_ROOT="${SWIFT_ROOT:-${CODE_ROOT}/ms-swift}"
SWIFT_VENV="${SWIFT_VENV:-${CODE_ROOT}/ms-swift-venv}"

cd "${CODE_ROOT}"

MODEL_PATH="${MODEL_PATH:-${CPT_ROOT}/Qwen3.5-35B-A3B-Base}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/data/cpt_data*.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/cpt/qwen35_35b_a3b}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/cache/cpt}"


# Run once per node. WORLD_SIZE/RANK, if supplied, denote node count/rank.
# Do not invoke this entry from inside an outer torchrun.
if [[ -n "${LOCAL_RANK:-}" ]]; then
  echo "Run this entry once per node, not inside an outer torchrun" >&2
  exit 1
fi
RESOURCE_GPU="${RESOURCE_GPU:-${NPROC_PER_NODE:-8}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${RESOURCE_GPU}}"
if [[ -n "${WORLD_SIZE:-}" && -n "${NNODES:-}" && "${WORLD_SIZE}" != "${NNODES}" ]]; then
  echo "WORLD_SIZE (node count) and NNODES disagree" >&2; exit 1
fi
if [[ -n "${RANK:-}" && -n "${NODE_RANK:-}" && "${RANK}" != "${NODE_RANK}" ]]; then
  echo "RANK (node rank) and NODE_RANK disagree" >&2; exit 1
fi
NNODES="${NNODES:-${WORLD_SIZE:-16}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29500}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# torchrun will provide the actual global worker WORLD_SIZE/RANK.
unset WORLD_SIZE RANK
for name in RESOURCE_GPU NPROC_PER_NODE NNODES MASTER_PORT; do
  if [[ ! "${!name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer" >&2; exit 1
  fi
done
if [[ ! "${NODE_RANK}" =~ ^(0|[1-9][0-9]*)$ ]] || (( NODE_RANK >= NNODES )); then
  echo "NODE_RANK must be in [0, NNODES)" >&2; exit 1
fi
if (( RESOURCE_GPU != NPROC_PER_NODE || MASTER_PORT > 65535 )); then
  echo "RESOURCE_GPU must match NPROC_PER_NODE; MASTER_PORT must be <= 65535" >&2; exit 1
fi
if [[ "${NNODES}" == "1" ]]; then
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
elif [[ -z "${MASTER_ADDR}" || "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" || "${MASTER_ADDR}" == "::1" ]]; then
  echo "MASTER_ADDR must be a rank-0 address reachable from every node" >&2; exit 1
fi
IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#GPU_LIST[@]} != NPROC_PER_NODE )); then
  echo "NPROC_PER_NODE does not match CUDA_VISIBLE_DEVICES" >&2; exit 1
fi

MAX_LENGTH="${MAX_LENGTH:-32768}"
TRUNCATION_STRATEGY="${TRUNCATION_STRATEGY:-left}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
TRAIN_ITERS="${TRAIN_ITERS:-}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
# Qwen3.5-35B-A3B has 40 layers: eight stages give five layers per stage.
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-8}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-2}"
EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
# Leave empty for even pipeline partitioning; explicit uneven layouts are checked below.
DECODER_LAST_PIPELINE_NUM_LAYERS="${DECODER_LAST_PIPELINE_NUM_LAYERS:-}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MIN_LR="${MIN_LR:-2e-6}"
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.03}"
SAVE_STEPS="${SAVE_STEPS:-200}"
TRAIN_USER_LOSS="${TRAIN_USER_LOSS:-true}"
OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-true}"
OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1}"
USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-true}"
LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE:-false}"
DRY_RUN="${DRY_RUN:-false}"

for name in TENSOR_MODEL_PARALLEL_SIZE PIPELINE_MODEL_PARALLEL_SIZE CONTEXT_PARALLEL_SIZE EXPERT_MODEL_PARALLEL_SIZE EXPERT_TENSOR_PARALLEL_SIZE MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE; do
  if [[ ! "${!name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer" >&2; exit 1
  fi
done
# This model-specific launcher targets the 40-layer, 256-expert architecture.
if (( 16 % TENSOR_MODEL_PARALLEL_SIZE != 0 || 256 % EXPERT_MODEL_PARALLEL_SIZE != 0 )); then
  echo "TP must divide 16 attention heads and EP must divide 256 experts" >&2; exit 1
fi
if [[ -n "${DECODER_LAST_PIPELINE_NUM_LAYERS}" ]]; then
  if [[ ! "${DECODER_LAST_PIPELINE_NUM_LAYERS}" =~ ^[1-9][0-9]*$ ]] ||
     (( PIPELINE_MODEL_PARALLEL_SIZE <= 1 || DECODER_LAST_PIPELINE_NUM_LAYERS >= 40 )); then
    echo "Uneven PP requires PP > 1 and a last-stage layer count in [1, 39]" >&2; exit 1
  fi
  if (( (40 - DECODER_LAST_PIPELINE_NUM_LAYERS) % (PIPELINE_MODEL_PARALLEL_SIZE - 1) != 0 )); then
    echo "Remaining layers must divide evenly across the other pipeline stages" >&2; exit 1
  fi
elif (( 40 % PIPELINE_MODEL_PARALLEL_SIZE != 0 )); then
  echo "PP must divide the model's 40 layers unless a valid last-stage split is supplied" >&2; exit 1
fi
DATASET_FILES=( "${DATASET_PATH}" )

if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" ]]; then
if [[ ! -d "${SWIFT_ROOT}" ]]; then
  echo "ms-swift repo not found: ${SWIFT_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${SWIFT_VENV}/bin/activate" ]]; then
  echo "ms-swift venv not found: ${SWIFT_VENV}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "local model not found: ${MODEL_PATH}" >&2
  exit 1
fi
shopt -s nullglob
DATASET_FILES=( ${DATASET_PATH} )
shopt -u nullglob
if (( ${#DATASET_FILES[@]} == 0 )); then
  echo "dataset not found: ${DATASET_PATH}" >&2
  exit 1
fi

fi

WORLD_SIZE_TOTAL=$((NNODES * NPROC_PER_NODE))
MODEL_PARALLEL_SIZE=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))
EXPERT_PARALLEL_PRODUCT=$((EXPERT_TENSOR_PARALLEL_SIZE * EXPERT_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))
if (( WORLD_SIZE_TOTAL % MODEL_PARALLEL_SIZE != 0 )); then
  echo "WORLD_SIZE=${WORLD_SIZE_TOTAL} must be divisible by TP*PP*CP=${MODEL_PARALLEL_SIZE}" >&2
  exit 1
fi
if (( WORLD_SIZE_TOTAL % EXPERT_PARALLEL_PRODUCT != 0 )); then
  echo "WORLD_SIZE=${WORLD_SIZE_TOTAL} must be divisible by ETP*EP*PP*CP=${EXPERT_PARALLEL_PRODUCT}" >&2
  exit 1
fi
DATA_PARALLEL_SIZE=$((WORLD_SIZE_TOTAL / MODEL_PARALLEL_SIZE))
EXPERT_DATA_PARALLEL_SIZE=$((WORLD_SIZE_TOTAL / EXPERT_PARALLEL_PRODUCT))
STEP_BATCH_SIZE=$((MICRO_BATCH_SIZE * DATA_PARALLEL_SIZE))
if (( GLOBAL_BATCH_SIZE % STEP_BATCH_SIZE != 0 )); then
  echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by MICRO_BATCH_SIZE*DP=${STEP_BATCH_SIZE}" >&2
  exit 1
fi
if (( NODE_RANK < 0 || NODE_RANK >= NNODES )); then
  echo "NODE_RANK=${NODE_RANK} must be in [0, $((NNODES - 1))]" >&2
  exit 1
fi

if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" ]]; then
mkdir -p "${OUTPUT_DIR}" "${CACHE_ROOT}"/{huggingface/datasets,huggingface/transformers,modelscope,xdg,tmp}

source "${SWIFT_VENV}/bin/activate"
cd "${SWIFT_ROOT}"

python - <<'PY'
import importlib.util
import sys
missing = [name for name in ['mcore_bridge', 'megatron.core', 'transformer_engine'] if importlib.util.find_spec(name) is None]
if missing:
    print('Missing required packages: ' + ', '.join(missing), file=sys.stderr)
    sys.exit(1)
PY

fi

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy
unset USE_HF HF_ENDPOINT
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MODELSCOPE_OFFLINE=1
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_DATASETS_CACHE="${CACHE_ROOT}/huggingface/datasets"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface/transformers"
export MODELSCOPE_CACHE="${CACHE_ROOT}/modelscope"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TMPDIR="${CACHE_ROOT}/tmp"

export CUDA_VISIBLE_DEVICES NPROC_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
unset NCCL_ASYNC_ERROR_HANDLING NCCL_BLOCKING_WAIT
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-12}"
export USE_MCORE_GDN="${USE_MCORE_GDN:-0}"
if [[ "${TRAIN_USER_LOSS}" == "true" ]]; then
  export SWIFT_TRAIN_USER_LOSS=1
else
  unset SWIFT_TRAIN_USER_LOSS
fi
if [[ "${DRY_RUN:-0}" != "1" && "${DRY_RUN:-0}" != "true" ]]; then
  LOSS_CHECK_ARGS=()
  [[ "${SWIFT_TRAIN_USER_LOSS:-0}" == "1" ]] && LOSS_CHECK_ARGS+=(--train-user-loss)
  PYTHONPATH="${SWIFT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${SWIFT_VENV}/bin/python" \
    "${PROJECT_ROOT}/runtime/validate_swift.py" "${LOSS_CHECK_ARGS[@]}"
fi

ARGS=(
  sft
  --model "${MODEL_PATH}"
  --save_safetensors false
  --dataset "${DATASET_PATH}"
  --load_from_cache_file "${LOAD_FROM_CACHE_FILE}"
  --tuner_type full
  --freeze_llm false
  --freeze_vit true
  --freeze_aligner true
  --tensor_model_parallel_size "${TENSOR_MODEL_PARALLEL_SIZE}"
  --pipeline_model_parallel_size "${PIPELINE_MODEL_PARALLEL_SIZE}"
  --expert_model_parallel_size "${EXPERT_MODEL_PARALLEL_SIZE}"
  --expert_tensor_parallel_size "${EXPERT_TENSOR_PARALLEL_SIZE}"
  --moe_permute_fusion true
  --moe_grouped_gemm true
  --moe_shared_expert_overlap true
  --moe_aux_loss_coeff 1e-6
  --sequence_parallel true
  --context_parallel_size "${CONTEXT_PARALLEL_SIZE}"
  --recompute_granularity full
  --recompute_method uniform
  --recompute_num_layers 1
  --micro_batch_size "${MICRO_BATCH_SIZE}"
  --global_batch_size "${GLOBAL_BATCH_SIZE}"
  --finetune true
  --cross_entropy_loss_fusion true
  --gradient_accumulation_fusion false
  --masked_softmax_fusion false
  --lr "${LEARNING_RATE}"
  --lr_warmup_fraction "${LR_WARMUP_FRACTION}"
  --min_lr "${MIN_LR}"
  --output_dir "${OUTPUT_DIR}"
  --save_steps "${SAVE_STEPS}"
  --max_length "${MAX_LENGTH}"
  --truncation_strategy "${TRUNCATION_STRATEGY}"
  --packing false
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --dataset_num_proc "${DATASET_NUM_PROC}"
  --no_save_optim true
  --no_save_rng true
  --use_precision_aware_optimizer "${USE_PRECISION_AWARE_OPTIMIZER}"
  --optimizer_cpu_offload "${OPTIMIZER_CPU_OFFLOAD}"
  --optimizer_offload_fraction "${OPTIMIZER_OFFLOAD_FRACTION}"
  --attention_backend flash
)

if [[ -n "${DECODER_LAST_PIPELINE_NUM_LAYERS}" ]]; then
  ARGS+=(--decoder_last_pipeline_num_layers "${DECODER_LAST_PIPELINE_NUM_LAYERS}")
fi

if [[ -n "${TRAIN_ITERS}" ]]; then
  ARGS+=(--train_iters "${TRAIN_ITERS}")
else
  ARGS+=(--num_train_epochs "${NUM_TRAIN_EPOCHS}")
fi

echo "Running Megatron-SWIFT Qwen3.5-35B-A3B CPT"
echo "  model: ${MODEL_PATH}"
echo "  dataset: ${DATASET_PATH}"
echo "  dataset_files: ${#DATASET_FILES[@]}"
echo "  output: ${OUTPUT_DIR}"
echo "  nodes/rank: ${NNODES}/${NODE_RANK}"
echo "  nproc_per_node: ${NPROC_PER_NODE}"
echo "  master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  world_size: ${WORLD_SIZE_TOTAL}, dense/attention dp: ${DATA_PARALLEL_SIZE}, expert dp: ${EXPERT_DATA_PARALLEL_SIZE}"
echo "  max_length: ${MAX_LENGTH}"
echo "  parallel: TP=${TENSOR_MODEL_PARALLEL_SIZE} PP=${PIPELINE_MODEL_PARALLEL_SIZE} EP=${EXPERT_MODEL_PARALLEL_SIZE} ETP=${EXPERT_TENSOR_PARALLEL_SIZE} SP=true CP=${CONTEXT_PARALLEL_SIZE}"
echo "  decoder_last_pipeline_num_layers: ${DECODER_LAST_PIPELINE_NUM_LAYERS:-even partition}"
echo "  micro/global batch: ${MICRO_BATCH_SIZE}/${GLOBAL_BATCH_SIZE}"
echo "  gradient accumulation microbatches: $((GLOBAL_BATCH_SIZE / STEP_BATCH_SIZE))"
echo "  num_train_epochs: $([[ -n "${TRAIN_ITERS}" ]] && echo auto-disabled || echo "${NUM_TRAIN_EPOCHS}")"
echo "  train_iters: ${TRAIN_ITERS:-auto}"
echo "  optimizer_cpu_offload: ${OPTIMIZER_CPU_OFFLOAD}"

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
  printf 'Resolved command:'
  printf ' %q' megatron "${ARGS[@]}"
  printf '\n'
  exit 0
fi

megatron "${ARGS[@]}"
