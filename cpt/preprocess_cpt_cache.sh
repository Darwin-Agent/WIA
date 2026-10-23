#!/usr/bin/env bash
set -euo pipefail
trap 'echo "preprocess script failed at line ${LINENO} with exit code $?" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CODE_ROOT="${CODE_ROOT:-${SCRIPT_DIR}}"
CPT_ROOT="${CPT_ROOT:-${PROJECT_ROOT}/models}"
SWIFT_ROOT="${SWIFT_ROOT:-${CODE_ROOT}/ms-swift}"
SWIFT_VENV="${SWIFT_VENV:-${CODE_ROOT}/ms-swift-venv}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/cache/cpt}"
MODEL_PATH="${MODEL_PATH:-${CPT_ROOT}/Qwen3.5-4B-Base}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/data/cpt_data*.jsonl}"
# Set OUTPUT_DIR for each model; training reads <OUTPUT_DIR>/train.
OUTPUT_DIR="${OUTPUT_DIR:-${CACHE_ROOT}/qwen35_4b/cpt_data_cache}"

MAX_LENGTH="${MAX_LENGTH:-32768}"
TRUNCATION_STRATEGY="${TRUNCATION_STRATEGY:-left}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-16}"
LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE:-true}"
DATASET_SHUFFLE="${DATASET_SHUFFLE:-true}"
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
TRAIN_USER_LOSS="${TRAIN_USER_LOSS:-true}"
if [[ "${TRAIN_USER_LOSS}" == "true" ]]; then
  export SWIFT_TRAIN_USER_LOSS=1
else
  unset SWIFT_TRAIN_USER_LOSS
fi
export PYTHONPATH="${SWIFT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"


shopt -s nullglob
DATASET_FILES=( ${DATASET_PATH} )
shopt -u nullglob
if (( ${#DATASET_FILES[@]} == 0 )); then
  echo "dataset not found: ${DATASET_PATH}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${CACHE_ROOT}"/{huggingface/datasets,huggingface/transformers,modelscope,xdg,tmp}
source "${SWIFT_VENV}/bin/activate"
cd "${SWIFT_ROOT}"

# Keep preprocessing strictly single-process/non-DDP: no torchrun, no NCCL collectives.
unset NPROC_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT RANK LOCAL_RANK WORLD_SIZE
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
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-12}"

printf 'Export ms-swift cached dataset\n'
printf '  model: %s\n' "${MODEL_PATH}"
printf '  dataset files: %s\n' "${#DATASET_FILES[@]}"
printf '  cache_root: %s\n' "${CACHE_ROOT}"
printf '  output_dir: %s\n' "${OUTPUT_DIR}"
printf '  max_length: %s\n' "${MAX_LENGTH}"
printf '  truncation_strategy: %s\n' "${TRUNCATION_STRATEGY}"
printf '  dataset_num_proc: %s\n' "${DATASET_NUM_PROC}"
printf '  dataset_shuffle: %s\n' "${DATASET_SHUFFLE}"
printf '  attn_impl: %s\n' "${ATTN_IMPL}"
printf '  group_by_length: %s\n' 'not used during export; keep it in training script'

LOSS_CHECK_ARGS=()
[[ "${SWIFT_TRAIN_USER_LOSS:-0}" == "1" ]] && LOSS_CHECK_ARGS+=(--train-user-loss)
"${SWIFT_VENV}/bin/python" "${PROJECT_ROOT}/runtime/validate_swift.py" "${LOSS_CHECK_ARGS[@]}"
set -x
"${SWIFT_VENV}/bin/swift" export \
  --to_cached_dataset true \
  --model "${MODEL_PATH}" \
  --dataset "${DATASET_FILES[@]}" \
  --torch_dtype bfloat16 \
  --loss_scale default \
  --attn_impl "${ATTN_IMPL}" \
  --max_length "${MAX_LENGTH}" \
  --truncation_strategy "${TRUNCATION_STRATEGY}" \
  --packing false \
  --padding_free false \
  --load_from_cache_file "${LOAD_FROM_CACHE_FILE}" \
  --dataset_shuffle "${DATASET_SHUFFLE}" \
  --split_dataset_ratio 0 \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --output_dir "${OUTPUT_DIR}" \
  --exist_ok true
