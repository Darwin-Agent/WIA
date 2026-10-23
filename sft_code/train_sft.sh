#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Qwen3.5 Megatron SFT failed at line ${LINENO} with exit code $?" >&2' ERR

# Same Megatron-SWIFT backend as the Dense/MoE CPT launchers, with genuine CP.
# HF sequence_parallel_size is not context_parallel_size and is not used here.
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Qwen3.5 Megatron SFT: SFT_MODEL_SIZE=4b|9b|27b|35b_a3b (default: 9b).
4B/9B:    1 x 8 GPUs, TP2/PP1/CP4, DP1.
27B:      8 x 8 GPUs, TP4/PP4/CP4, DP1.
35B-A3B:  4 x 8 GPUs, TP2/PP4/CP4/EP8/ETP1, Dense DP1, expert DP1.
Defaults: global batch=128, micro batch=1, accumulation=128, epochs=2, length=65536.
Megatron sequence_parallel=true within TP; it does not add another GPU dimension.
GPU caps: 64 for 27B; 32 for the other sizes. Run this script once per node.
Set MODEL_PATH, DATASET_PATH, SWIFT_ROOT and SWIFT_VENV; optional MEGATRON_ROOT
selects a compatible Megatron source checkout. GatedDeltaNet CP support is required.
For native CPT weights, set MCORE_MODEL and keep MODEL_PATH as HF metadata/tokenizer.
DRY_RUN=1 checks topology and prints the command; it does not validate runtime/memory.
See sft_code/README.md for configuration, dependencies and checkpoint formats.
USAGE
  exit 0
fi
if (( $# != 0 )); then
  echo "Configure this launcher through environment variables; only --help is accepted" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/cpt}"
SWIFT_ROOT="${SWIFT_ROOT:-${CODE_ROOT}/ms-swift}"
SWIFT_VENV="${SWIFT_VENV:-${SCRIPT_DIR}/.venv}"
MEGATRON_ROOT="${MEGATRON_ROOT:-}"
SFT_MODEL_SIZE="${SFT_MODEL_SIZE:-9b}"
# Model-specific safety limits, not environment-overridable.
MAX_GPUS=32
case "${SFT_MODEL_SIZE}" in
  4b)
    DEFAULT_NNODES=1; DEFAULT_TP=2; DEFAULT_PP=1; DEFAULT_EP=1
    MODEL_LAYERS=32; MODEL_HIDDEN=2560; MODEL_HEADS=16; LINEAR_VALUE_HEADS=32
    DEFAULT_MODEL_TYPE=qwen3_5
    ;;
  9b)
    DEFAULT_NNODES=1; DEFAULT_TP=2; DEFAULT_PP=1; DEFAULT_EP=1
    MODEL_LAYERS=32; MODEL_HIDDEN=4096; MODEL_HEADS=16; LINEAR_VALUE_HEADS=32
    DEFAULT_MODEL_TYPE=qwen3_5
    ;;
  27b)
    # TP8 from CPT cannot be combined with headwise CP4: 16 GDN key heads / 32 is invalid.
    DEFAULT_NNODES=8; DEFAULT_TP=4; DEFAULT_PP=4; DEFAULT_EP=1
    MAX_GPUS=64
    MODEL_LAYERS=64; MODEL_HIDDEN=5120; MODEL_HEADS=24; LINEAR_VALUE_HEADS=48
    DEFAULT_MODEL_TYPE=qwen3_5
    ;;
  35b_a3b)
    DEFAULT_NNODES=4; DEFAULT_TP=2; DEFAULT_PP=4; DEFAULT_EP=8
    MODEL_LAYERS=40; MODEL_HIDDEN=2048; MODEL_HEADS=16; LINEAR_VALUE_HEADS=32
    DEFAULT_MODEL_TYPE=qwen3_5_moe
    ;;
  *) echo "Unsupported SFT_MODEL_SIZE: ${SFT_MODEL_SIZE}" >&2; exit 1 ;;
esac
LINEAR_KEY_HEADS=16
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/cpt-hf/qwen35_${SFT_MODEL_SIZE}}"
MCORE_MODEL="${MCORE_MODEL:-}"
MODEL_TYPE="${MODEL_TYPE:-${DEFAULT_MODEL_TYPE}}"
TEMPLATE="${TEMPLATE:-qwen3_5}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/data/sft_data.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/sft/qwen35_${SFT_MODEL_SIZE}}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/cache/sft/qwen35_${SFT_MODEL_SIZE}}"

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
NNODES="${NNODES:-${WORLD_SIZE:-${DEFAULT_NNODES}}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${SINGLE_NODE_MASTER_ADDR:-}}"
MASTER_PORT="${MASTER_PORT:-${SINGLE_NODE_MASTER_PORT:-29503}}"
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

# Context parallelism is explicit and distinct from Megatron's boolean SP flag.
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-${DEFAULT_TP}}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-${DEFAULT_PP}}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-4}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-${DEFAULT_EP}}"
EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
CP_COMM_TYPE="${CP_COMM_TYPE:-p2p}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-${PER_DEVICE_TRAIN_BATCH_SIZE:-1}}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
TRAIN_ITERS="${TRAIN_ITERS:-}"
MAX_LENGTH="${MAX_LENGTH:-65536}"
TRUNCATION_STRATEGY="${TRUNCATION_STRATEGY:-right}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MIN_LR="${MIN_LR:-0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LOSS_SCALE="${LOSS_SCALE:-default}"
PACKING="${PACKING:-false}"
PADDING_FREE="${PADDING_FREE:-false}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE:-false}"
DATASET_SHUFFLE="${DATASET_SHUFFLE:-true}"
SAVE_STEPS="${SAVE_STEPS:-10}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"
SAVE_SAFETENSORS="${SAVE_SAFETENSORS:-true}"
TRAIN_USER_LOSS="${TRAIN_USER_LOSS:-true}"
FREEZE_VIT="${FREEZE_VIT:-true}"
FREEZE_ALIGNER="${FREEZE_ALIGNER:-true}"
OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-true}"
OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1}"
USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-true}"
DRY_RUN="${DRY_RUN:-false}"
export USE_MCORE_GDN="${USE_MCORE_GDN:-1}"

# Reject obsolete HF controls rather than silently interpreting SP as CP.
for name in SEQUENCE_PARALLEL_SIZE DEEPSPEED; do
  if [[ -n "${!name:-}" ]]; then
    echo "${name} is an HF-only setting; this entry uses Megatron TP/PP/CP/EP" >&2; exit 1
  fi
done
for name in TENSOR_MODEL_PARALLEL_SIZE PIPELINE_MODEL_PARALLEL_SIZE CONTEXT_PARALLEL_SIZE EXPERT_MODEL_PARALLEL_SIZE EXPERT_TENSOR_PARALLEL_SIZE MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE MAX_LENGTH; do
  if [[ ! "${!name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer" >&2; exit 1
  fi
done
if [[ -n "${PER_DEVICE_TRAIN_BATCH_SIZE:-}" && "${PER_DEVICE_TRAIN_BATCH_SIZE}" != "${MICRO_BATCH_SIZE}" ]]; then
  echo "PER_DEVICE_TRAIN_BATCH_SIZE and MICRO_BATCH_SIZE disagree" >&2; exit 1
fi
WORLD_SIZE_TOTAL=$((NNODES * NPROC_PER_NODE))
if (( WORLD_SIZE_TOTAL > MAX_GPUS )); then
  echo "${SFT_MODEL_SIZE} SFT supports at most ${MAX_GPUS} GPUs; requested ${WORLD_SIZE_TOTAL}" >&2; exit 1
fi
MODEL_PARALLEL_SIZE=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))
if (( WORLD_SIZE_TOTAL % MODEL_PARALLEL_SIZE != 0 )); then
  echo "Total GPUs must be divisible by TP*PP*CP=${MODEL_PARALLEL_SIZE}" >&2; exit 1
fi
DP_SIZE=$((WORLD_SIZE_TOTAL / MODEL_PARALLEL_SIZE))
if (( MODEL_LAYERS % PIPELINE_MODEL_PARALLEL_SIZE != 0 || MODEL_HIDDEN % TENSOR_MODEL_PARALLEL_SIZE != 0 || MODEL_HEADS % TENSOR_MODEL_PARALLEL_SIZE != 0 )); then
  echo "Model layers must divide PP; hidden size and attention heads must divide TP" >&2; exit 1
fi
# The selected Qwen3.5 bridge uses headwise GDN CP, which shards heads over TP*CP.
GDN_HEAD_PARTITIONS=$((TENSOR_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))
if (( LINEAR_KEY_HEADS % GDN_HEAD_PARTITIONS != 0 || LINEAR_VALUE_HEADS % GDN_HEAD_PARTITIONS != 0 )); then
  echo "GatedDeltaNet key/value heads must be divisible by TP*CP=${GDN_HEAD_PARTITIONS}" >&2; exit 1
fi
if (( MAX_LENGTH % (2 * CONTEXT_PARALLEL_SIZE) != 0 )); then
  echo "MAX_LENGTH must be divisible by 2*CP for context-parallel sequence splitting" >&2; exit 1
fi
if (( CONTEXT_PARALLEL_SIZE > 1 )) && [[ "${USE_MCORE_GDN}" != "1" && "${USE_MCORE_GDN}" != "true" ]]; then
  echo "CP > 1 requires USE_MCORE_GDN=1; the legacy HF GDN bridge supports CP1 only" >&2; exit 1
fi
if [[ "${CP_COMM_TYPE}" != "p2p" ]]; then
  echo "This launcher validates CP_COMM_TYPE=p2p for full attention; other CP transports need separate validation" >&2; exit 1
fi
if (( TENSOR_MODEL_PARALLEL_SIZE > 1 )); then DEFAULT_SEQUENCE_PARALLEL=true; else DEFAULT_SEQUENCE_PARALLEL=false; fi
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-${DEFAULT_SEQUENCE_PARALLEL}}"
case "${SEQUENCE_PARALLEL}" in
  1|true) SEQUENCE_PARALLEL=true ;;
  0|false) SEQUENCE_PARALLEL=false ;;
  *) echo "SEQUENCE_PARALLEL must be true/false or 1/0" >&2; exit 1 ;;
esac
if (( TENSOR_MODEL_PARALLEL_SIZE == 1 )) && [[ "${SEQUENCE_PARALLEL}" == "true" ]]; then
  echo "Megatron sequence parallelism requires TP > 1" >&2; exit 1
fi
if [[ "${SFT_MODEL_SIZE}" == "35b_a3b" ]]; then
  if (( 256 % EXPERT_MODEL_PARALLEL_SIZE != 0 || 512 % EXPERT_TENSOR_PARALLEL_SIZE != 0 )); then
    echo "EP must divide 256 experts and ETP must divide the 512-wide expert FFN" >&2; exit 1
  fi
  # MCore's expert rank generator uses cp=1: do NOT multiply CP into this product.
  EXPERT_PARALLEL_SIZE=$((EXPERT_TENSOR_PARALLEL_SIZE * EXPERT_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE))
  if (( WORLD_SIZE_TOTAL % EXPERT_PARALLEL_SIZE != 0 )); then
    echo "Total GPUs must be divisible by ETP*EP*PP=${EXPERT_PARALLEL_SIZE}" >&2; exit 1
  fi
  EXPERT_DP_SIZE=$((WORLD_SIZE_TOTAL / EXPERT_PARALLEL_SIZE))
elif (( EXPERT_MODEL_PARALLEL_SIZE != 1 || EXPERT_TENSOR_PARALLEL_SIZE != 1 )); then
  echo "Dense models require EP=1 and ETP=1" >&2; exit 1
fi
STEP_BATCH_SIZE=$((DP_SIZE * MICRO_BATCH_SIZE))
if (( GLOBAL_BATCH_SIZE % STEP_BATCH_SIZE != 0 )); then
  echo "GLOBAL_BATCH_SIZE must be divisible by DP*MICRO_BATCH_SIZE=${STEP_BATCH_SIZE}" >&2; exit 1
fi
GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / STEP_BATCH_SIZE))
if [[ -n "${GRADIENT_ACCUMULATION_STEPS:-}" && "${GRADIENT_ACCUMULATION_STEPS}" != "${GRAD_ACCUM}" ]]; then
  echo "Megatron derives accumulation=${GRAD_ACCUM}; set GLOBAL_BATCH_SIZE instead of a conflicting GRADIENT_ACCUMULATION_STEPS" >&2; exit 1
fi

# A selected checkout takes precedence over a potentially CP-incompatible installed MCore.
export PYTHONPATH="${MEGATRON_ROOT:+${MEGATRON_ROOT}:}${SWIFT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES NPROC_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export FLA_TILELANG="${FLA_TILELANG:-1}"

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
export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
unset NCCL_ASYNC_ERROR_HANDLING NCCL_BLOCKING_WAIT
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-12}"
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
  --model_type "${MODEL_TYPE}"
  --template "${TEMPLATE}"
  --load_args false
  --tuner_type full
  --torch_dtype bfloat16
  --finetune true
  --no_load_optim true
  --no_load_rng true
  --freeze_llm false
  --freeze_vit "${FREEZE_VIT}"
  --freeze_aligner "${FREEZE_ALIGNER}"
  --tensor_model_parallel_size "${TENSOR_MODEL_PARALLEL_SIZE}"
  --pipeline_model_parallel_size "${PIPELINE_MODEL_PARALLEL_SIZE}"
  --context_parallel_size "${CONTEXT_PARALLEL_SIZE}"
  --cp_comm_type "${CP_COMM_TYPE}"
  --sequence_parallel "${SEQUENCE_PARALLEL}"
  --micro_batch_size "${MICRO_BATCH_SIZE}"
  --global_batch_size "${GLOBAL_BATCH_SIZE}"
  --recompute_granularity full
  --recompute_method uniform
  --recompute_num_layers 1
  --use_cpu_initialization true
  --cross_entropy_loss_fusion true
  --gradient_accumulation_fusion false
  --masked_softmax_fusion false
  --optimizer adam
  --lr "${LEARNING_RATE}"
  --min_lr "${MIN_LR}"
  --lr_decay_style cosine
  --lr_warmup_fraction "${WARMUP_RATIO}"
  --weight_decay "${WEIGHT_DECAY}"
  --clip_grad "${MAX_GRAD_NORM}"
  --use_distributed_optimizer true
  --use_precision_aware_optimizer "${USE_PRECISION_AWARE_OPTIMIZER}"
  --optimizer_cpu_offload "${OPTIMIZER_CPU_OFFLOAD}"
  --optimizer_offload_fraction "${OPTIMIZER_OFFLOAD_FRACTION}"
  --attention_backend flash
  --max_length "${MAX_LENGTH}"
  --truncation_strategy "${TRUNCATION_STRATEGY}"
  --loss_scale "${LOSS_SCALE}"
  --packing "${PACKING}"
  --padding_free "${PADDING_FREE}"
  --dataset "${DATASET_PATH}"
  --load_from_cache_file "${LOAD_FROM_CACHE_FILE}"
  --dataset_shuffle "${DATASET_SHUFFLE}"
  --split_dataset_ratio 0
  --dataset_num_proc "${DATASET_NUM_PROC}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --save_steps "${SAVE_STEPS}"
  --logging_steps "${LOGGING_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --save_safetensors "${SAVE_SAFETENSORS}"
  --no_save_optim true
  --no_save_rng true
  --output_dir "${OUTPUT_DIR}"
)
if [[ "${SFT_MODEL_SIZE}" == "35b_a3b" ]]; then
  ARGS+=(
    --expert_model_parallel_size "${EXPERT_MODEL_PARALLEL_SIZE}"
    --expert_tensor_parallel_size "${EXPERT_TENSOR_PARALLEL_SIZE}"
    --moe_token_dispatcher_type alltoall
    --moe_permute_fusion true
    --moe_grouped_gemm true
    --moe_shared_expert_overlap true
    --moe_aux_loss_coeff 1e-6
  )
fi
if [[ -n "${MCORE_MODEL}" ]]; then ARGS+=(--mcore_model "${MCORE_MODEL}"); fi
if [[ -n "${TRAIN_ITERS}" ]]; then
  ARGS+=(--train_iters "${TRAIN_ITERS}")
else
  ARGS+=(--num_train_epochs "${NUM_TRAIN_EPOCHS}")
fi
LAUNCH=( "${SWIFT_VENV}/bin/python" -m swift.cli._megatron.main )

echo "Running Megatron-SWIFT Qwen3.5-${SFT_MODEL_SIZE} SFT"
echo "  nodes/rank: ${NNODES}/${NODE_RANK}; GPUs/node: ${NPROC_PER_NODE}; total GPUs: ${WORLD_SIZE_TOTAL}"
echo "  parallel: TP=${TENSOR_MODEL_PARALLEL_SIZE} PP=${PIPELINE_MODEL_PARALLEL_SIZE} CP=${CONTEXT_PARALLEL_SIZE} SP=${SEQUENCE_PARALLEL}"
echo "  dense DP: ${DP_SIZE}; layers per pipeline stage: $((MODEL_LAYERS / PIPELINE_MODEL_PARALLEL_SIZE))"
if [[ "${SFT_MODEL_SIZE}" == "35b_a3b" ]]; then
  echo "  MoE: EP=${EXPERT_MODEL_PARALLEL_SIZE} ETP=${EXPERT_TENSOR_PARALLEL_SIZE} expert DP=${EXPERT_DP_SIZE}"
fi
echo "  micro/global batch: ${MICRO_BATCH_SIZE}/${GLOBAL_BATCH_SIZE}; accumulation: ${GRAD_ACCUM}"
echo "  epochs: $([[ -n "${TRAIN_ITERS}" ]] && echo auto-disabled || echo "${NUM_TRAIN_EPOCHS}"); train_iters: ${TRAIN_ITERS:-auto}"
echo "  max_length: ${MAX_LENGTH}; lr: ${LEARNING_RATE}"
echo "  model: ${MODEL_PATH}; mcore_model: ${MCORE_MODEL:-none}"
echo "  dataset: ${DATASET_PATH}; output: ${OUTPUT_DIR}"
echo "  master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  USE_MCORE_GDN: ${USE_MCORE_GDN}; optimizer_cpu_offload: ${OPTIMIZER_CPU_OFFLOAD}"
if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
  printf 'Resolved command:'
  printf ' %q' "${LAUNCH[@]}" "${ARGS[@]}"
  printf '\n'
  exit 0
fi

for directory in "${SWIFT_ROOT}" "${MODEL_PATH}"; do
  if [[ ! -d "${directory}" ]]; then echo "Required directory not found: ${directory}" >&2; exit 1; fi
done
if [[ ! -x "${SWIFT_VENV}/bin/python" || ! -f "${SWIFT_VENV}/bin/activate" ]]; then
  echo "A compatible Swift/Megatron venv is required at SWIFT_VENV=${SWIFT_VENV}" >&2; exit 1
fi
if [[ ! -f "${DATASET_PATH}" ]]; then echo "SFT JSONL not found: ${DATASET_PATH}" >&2; exit 1; fi
if [[ -n "${MEGATRON_ROOT}" && ! -d "${MEGATRON_ROOT}/megatron/core" ]]; then
  echo "MEGATRON_ROOT must contain megatron/core" >&2; exit 1
fi
if [[ -n "${MCORE_MODEL}" && ! -d "${MCORE_MODEL}" ]]; then
  echo "MCORE_MODEL checkpoint directory not found: ${MCORE_MODEL}" >&2; exit 1
fi
source "${SWIFT_VENV}/bin/activate"
# Before creating outputs or spawning distributed workers, reject incompatible model/runtime stacks.
"${SWIFT_VENV}/bin/python" "${SCRIPT_DIR}/validate_sft_runtime.py"   --model "${MODEL_PATH}" --model-size "${SFT_MODEL_SIZE}"   --tp "${TENSOR_MODEL_PARALLEL_SIZE}" --pp "${PIPELINE_MODEL_PARALLEL_SIZE}"   --cp "${CONTEXT_PARALLEL_SIZE}" --ep "${EXPERT_MODEL_PARALLEL_SIZE}"
mkdir -p "${OUTPUT_DIR}" "${CACHE_ROOT}"/{huggingface/datasets,huggingface/transformers,modelscope,xdg,tmp}
cd "${SWIFT_ROOT}"
# Swift's Megatron CLI launches torchrun using the exported node variables.
exec "${LAUNCH[@]}" "${ARGS[@]}"
