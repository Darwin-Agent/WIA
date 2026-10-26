#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Qwen3.5-4B CPT: 8 nodes x 8 GPUs = 64 GPUs; NUM_TRAIN_EPOCHS=1.
HF Swift + ZeRO-1, SP=4, DP=16, accumulation=64, effective batch=1024.
Run once per node, with NNODES=8, NODE_RANK=<rank> and MASTER_ADDR=<rank-0-host>.
WORLD_SIZE/RANK are supported node-count/node-rank aliases, not worker counts.
Set MODEL_PATH, DATASET_PATH (or CACHED_DATASET), SWIFT_ROOT and SWIFT_VENV.
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

MODEL_PATH="${MODEL_PATH:-${CPT_ROOT}/Qwen3.5-4B-Base}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/data/cpt_data*.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/cpt/qwen35_4b}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/cache/cpt}"
CACHED_DATASET_ROOT="${CACHED_DATASET_ROOT:-${CACHE_ROOT}/qwen35_4b/cpt_data_cache}"
CACHED_DATASET="${CACHED_DATASET:-${CACHED_DATASET_ROOT}/train}"
USE_CACHED_DATASET="${USE_CACHED_DATASET:-true}"

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
NNODES="${NNODES:-${WORLD_SIZE:-8}}"
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
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-64}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
OPTIM="${OPTIM:-adamw_torch_fused}"
LOSS_SCALE="${LOSS_SCALE:-default}"
TRAIN_USER_LOSS="${TRAIN_USER_LOSS:-true}"
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
PACKING="${PACKING:-false}"
PADDING_FREE="${PADDING_FREE:-false}"
SEQUENCE_PARALLEL_SIZE="${SEQUENCE_PARALLEL_SIZE:-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
DRY_RUN="${DRY_RUN:-false}"
SAVE_STEPS="${SAVE_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-100}"
SPLIT_DATASET_RATIO="${SPLIT_DATASET_RATIO:-0}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
# Set GROUP_BY_LENGTH=true to restore the previous length-bucketed batching behavior.
GROUP_BY_LENGTH="${GROUP_BY_LENGTH:-false}"
DDP_TIMEOUT="${DDP_TIMEOUT:-18000000}"
DEEPSPEED="${DEEPSPEED:-zero1}"
LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE:-false}"
DATASET_SHUFFLE="${DATASET_SHUFFLE:-true}"
for name in SEQUENCE_PARALLEL_SIZE PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS; do
  if [[ ! "${!name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer" >&2; exit 1
  fi
done
if (( NPROC_PER_NODE % SEQUENCE_PARALLEL_SIZE != 0 )); then
  echo "NPROC_PER_NODE must be divisible by SEQUENCE_PARALLEL_SIZE" >&2; exit 1
fi
WORLD_SIZE_TOTAL=$((NNODES * NPROC_PER_NODE))
DATA_PARALLEL_SIZE=$((WORLD_SIZE_TOTAL / SEQUENCE_PARALLEL_SIZE))
EFFECTIVE_BATCH_SIZE=$((DATA_PARALLEL_SIZE * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
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
DATASET_FILES=()
if [[ "${USE_CACHED_DATASET}" == "true" ]]; then
  if [[ ! -d "${CACHED_DATASET}" ]]; then
    echo "cached dataset not found: ${CACHED_DATASET}" >&2
    echo "run first: bash ${CODE_ROOT}/preprocess_cpt_cache.sh" >&2
    exit 1
  fi
else
  shopt -s nullglob
  DATASET_FILES=( ${DATASET_PATH} )
  shopt -u nullglob
  if (( ${#DATASET_FILES[@]} == 0 )); then
    echo "dataset not found: ${DATASET_PATH}" >&2
    exit 1
  fi
fi
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CACHE_ROOT}"/{huggingface/datasets,huggingface/transformers,modelscope,xdg,tmp}

source "${SWIFT_VENV}/bin/activate"
cd "${SWIFT_ROOT}"

# The shell environment may contain stale proxy placeholders. Training uses local files only.
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

if [[ -n "${DEEPSPEED}" ]]; then
python - <<'PY'
import importlib.util
import sys
missing = []
for name in ['deepspeed']:
    if importlib.util.find_spec(name) is None:
        missing.append(name)
if missing:
    print('Missing required packages: ' + ', '.join(missing), file=sys.stderr)
    sys.exit(1)
PY
fi

fi

export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE
export NNODES
export NODE_RANK
export MASTER_ADDR
export MASTER_PORT
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
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
export SWIFT_SKIP_ZERO3_PARAM_INFO="${SWIFT_SKIP_ZERO3_PARAM_INFO:-1}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-12}"

ARGS=(
  sft
  --model "${MODEL_PATH}"
  --tuner_type full
  --torch_dtype bfloat16
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --optim "${OPTIM}"
  --warmup_ratio "${WARMUP_RATIO}"
  --weight_decay "${WEIGHT_DECAY}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --lr_scheduler_type cosine
  --check_model false
  --loss_scale "${LOSS_SCALE}"
  --attn_impl "${ATTN_IMPL}"
  --max_length "${MAX_LENGTH}"
  --truncation_strategy "${TRUNCATION_STRATEGY}"
  --packing "${PACKING}"
  --padding_free "${PADDING_FREE}"
  --gradient_checkpointing true
  --load_from_cache_file "${LOAD_FROM_CACHE_FILE}"
  --dataset_shuffle "${DATASET_SHUFFLE}"
  --split_dataset_ratio "${SPLIT_DATASET_RATIO}"
  --dataset_num_proc "${DATASET_NUM_PROC}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --group_by_length "${GROUP_BY_LENGTH}"
  --ddp_timeout "${DDP_TIMEOUT}"
  --save_steps "${SAVE_STEPS}"
  --logging_steps "${LOGGING_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --save_only_model true
  --freeze_llm false
  --freeze_vit true
  --freeze_aligner true
  --output_dir "${OUTPUT_DIR}"
)

if [[ "${USE_CACHED_DATASET}" == "true" ]]; then
  ARGS+=(--cached_dataset "${CACHED_DATASET}")
else
  ARGS+=(--dataset "${DATASET_FILES[@]}")
fi

if [[ -n "${DEEPSPEED}" ]]; then
  ARGS+=(--deepspeed "${DEEPSPEED}")
fi

if [[ "${SEQUENCE_PARALLEL_SIZE}" != "1" ]]; then
  ARGS+=(--sequence_parallel_size "${SEQUENCE_PARALLEL_SIZE}")
fi

echo "Running HF Swift Qwen3.5-4B CPT"
echo "  total GPUs / DP / effective batch: ${WORLD_SIZE_TOTAL}/${DATA_PARALLEL_SIZE}/${EFFECTIVE_BATCH_SIZE}"
echo "  epochs: ${NUM_TRAIN_EPOCHS}"
echo "  model: ${MODEL_PATH}"
echo "  dataset pattern: ${DATASET_PATH}"
echo "  dataset files: ${#DATASET_FILES[@]}"
echo "  use_cached_dataset: ${USE_CACHED_DATASET}"
echo "  cached_dataset: ${CACHED_DATASET}"
echo "  output: ${OUTPUT_DIR}"
echo "  cache_root: ${CACHE_ROOT}"
echo "  cuda: ${CUDA_VISIBLE_DEVICES}"
echo "  resource_gpu: ${RESOURCE_GPU}"
echo "  nnodes: ${NNODES}"
echo "  node_rank: ${NODE_RANK}"
echo "  master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  max_length: ${MAX_LENGTH}"
echo "  truncation_strategy: ${TRUNCATION_STRATEGY}"
echo "  lr: ${LEARNING_RATE}"
echo "  optim: ${OPTIM}"
echo "  loss_scale: ${LOSS_SCALE}"
echo "  train_user_loss: ${TRAIN_USER_LOSS}"
echo "  attn_impl: ${ATTN_IMPL}"
echo "  packing: ${PACKING}"
echo "  padding_free: ${PADDING_FREE}"
echo "  sequence_parallel_size: ${SEQUENCE_PARALLEL_SIZE}"
echo "  grad_accum: ${GRADIENT_ACCUMULATION_STEPS}"
echo "  skip_zero3_param_info: ${SWIFT_SKIP_ZERO3_PARAM_INFO}"
echo "  deepspeed: ${DEEPSPEED}"
echo "  load_from_cache_file: ${LOAD_FROM_CACHE_FILE}"
echo "  dataset_shuffle: ${DATASET_SHUFFLE}"
echo "  group_by_length: ${GROUP_BY_LENGTH}"
echo "  ddp_timeout: ${DDP_TIMEOUT}"
echo "  split_dataset_ratio: ${SPLIT_DATASET_RATIO}"

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
  printf 'Resolved command:'
  printf ' %q' swift "${ARGS[@]}"
  printf '\n'
  exit 0
fi
exec swift "${ARGS[@]}"
