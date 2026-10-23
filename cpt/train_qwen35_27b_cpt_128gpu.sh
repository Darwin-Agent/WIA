#!/usr/bin/env bash
set -euo pipefail
# Adapted from Code/CPT/train_qwen35_27b_dense_world_model_sft_megatron_16node_gbs1024.sh; private paths removed.
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Qwen3.5-27B CPT: 16 nodes x 8 GPUs = 128 GPUs; NUM_TRAIN_EPOCHS=1.
Megatron: TP=8, PP=1, CP=1, SP=true, DP=16, global batch=1024.
Run once per node, with NNODES=16, NODE_RANK=<rank> and MASTER_ADDR=<rank-0-host>.
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

MODEL_PATH="${MODEL_PATH:-${CPT_ROOT}/Qwen3.5-27B}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/data/cpt_data*.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/cpt/qwen35_27b}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/cache/cpt}"
CACHED_DATASET_ROOT="${CACHED_DATASET_ROOT:-${CACHE_ROOT}/qwen35_27b/cpt_data_cache}"
CACHED_DATASET="${CACHED_DATASET:-${CACHED_DATASET_ROOT}/train}"
USE_CACHED_DATASET="${USE_CACHED_DATASET:-true}"

# CloudML's WORLD_SIZE/RANK variables denote node count/rank in the existing
# launch contract. Keep them as shell aliases and do not export them; torchrun
# supplies the true 128-process WORLD_SIZE to its children.
RESOURCE_GPU="${RESOURCE_GPU:-${NPROC_PER_NODE:-8}}"
if [[ -n "${WORLD_SIZE:-}" ]]; then
  if [[ -n "${NNODES:-}" && "${NNODES}" != "${WORLD_SIZE}" ]]; then
    echo "WORLD_SIZE=${WORLD_SIZE} and NNODES=${NNODES} disagree; WORLD_SIZE is the CloudML node count here" >&2
    echo "Run this entry once per node, not from inside an outer torchrun" >&2
    exit 1
  fi
  NNODES="${WORLD_SIZE}"
elif [[ -n "${NNODES:-}" ]]; then
  NNODES="${NNODES}"
else
  NNODES="16"
fi
if [[ -n "${RANK:-}" ]]; then
  if [[ -n "${NODE_RANK:-}" && "${NODE_RANK}" != "${RANK}" ]]; then
    echo "RANK=${RANK} and NODE_RANK=${NODE_RANK} disagree; RANK is the CloudML node rank here" >&2
    echo "Run this entry once per node, not from inside an outer torchrun" >&2
    exit 1
  fi
  NODE_RANK="${RANK}"
elif [[ -n "${NODE_RANK:-}" ]]; then
  NODE_RANK="${NODE_RANK}"
else
  NODE_RANK="0"
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-${RESOURCE_GPU}}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29500}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# Prevent CloudML's node-count aliases from leaking into torchrun children as
# their global WORLD_SIZE/RANK. torchrun will set those from NNODES/NODE_RANK.
if [[ -v WORLD_SIZE ]]; then
  export -n WORLD_SIZE
fi
if [[ -v RANK ]]; then
  export -n RANK
fi

MAX_LENGTH="${MAX_LENGTH:-32768}"
TRUNCATION_STRATEGY="${TRUNCATION_STRATEGY:-left}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
# Empty by default means train by NUM_TRAIN_EPOCHS. Set TRAIN_ITERS=<N> only
# for an intentional fixed-step smoke/debug run.
TRAIN_ITERS="${TRAIN_ITERS-}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-${NPROC_PER_NODE}}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MIN_LR="${MIN_LR:-2e-6}"
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.03}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
TRAIN_USER_LOSS="${TRAIN_USER_LOSS:-true}"
OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-false}"
OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1}"
USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-true}"
MAIN_GRADS_DTYPE="${MAIN_GRADS_DTYPE:-bf16}"
MAIN_PARAMS_DTYPE="${MAIN_PARAMS_DTYPE:-fp32}"
EXP_AVG_DTYPE="${EXP_AVG_DTYPE:-fp32}"
EXP_AVG_SQ_DTYPE="${EXP_AVG_SQ_DTYPE:-fp32}"
USE_CPU_INITIALIZATION="${USE_CPU_INITIALIZATION:-true}"
LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE:-false}"
NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-false}"
NO_SAVE_RNG="${NO_SAVE_RNG:-false}"
DRY_RUN="${DRY_RUN:-false}"
NCCL_PREFLIGHT_ONLY="${NCCL_PREFLIGHT_ONLY:-false}"
NCCL_PREFLIGHT_TENSOR_MIB="${NCCL_PREFLIGHT_TENSOR_MIB:-16}"
NCCL_PREFLIGHT_TIMEOUT_SEC="${NCCL_PREFLIGHT_TIMEOUT_SEC:-180}"
ALLOW_TOPOLOGY_OVERRIDE="${ALLOW_TOPOLOGY_OVERRIDE:-false}"
EXPECTED_NNODES="${EXPECTED_NNODES:-16}"
EXPECTED_GPUS_PER_NODE="${EXPECTED_GPUS_PER_NODE:-8}"
LOCAL_CACHE_BASE="${LOCAL_CACHE_BASE:-/tmp/world_model_cpt}"
LOCAL_TMP_ROOT="${LOCAL_TMP_ROOT:-${LOCAL_CACHE_BASE}/qwen35_27b_rank${NODE_RANK}}"
SHORT_TMPDIR="${SHORT_TMPDIR:-/tmp/q27b_r${NODE_RANK}}"

is_positive_integer() {
  [[ "${1}" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

for pair in \
  "RESOURCE_GPU:${RESOURCE_GPU}" \
  "NPROC_PER_NODE:${NPROC_PER_NODE}" \
  "NNODES:${NNODES}" \
  "EXPECTED_NNODES:${EXPECTED_NNODES}" \
  "EXPECTED_GPUS_PER_NODE:${EXPECTED_GPUS_PER_NODE}" \
  "TENSOR_MODEL_PARALLEL_SIZE:${TENSOR_MODEL_PARALLEL_SIZE}" \
  "PIPELINE_MODEL_PARALLEL_SIZE:${PIPELINE_MODEL_PARALLEL_SIZE}" \
  "CONTEXT_PARALLEL_SIZE:${CONTEXT_PARALLEL_SIZE}"; do
  key="${pair%%:*}"
  value="${pair#*:}"
  if ! is_positive_integer "${value}"; then
    echo "${key} must be a positive integer, got: ${value}" >&2
    exit 1
  fi
done
if ! is_nonnegative_integer "${NODE_RANK}"; then
  echo "NODE_RANK must be a non-negative integer, got: ${NODE_RANK}" >&2
  exit 1
fi
if ! is_positive_integer "${MASTER_PORT}" || (( MASTER_PORT > 65535 )); then
  echo "MASTER_PORT must be in [1, 65535], got: ${MASTER_PORT}" >&2
  exit 1
fi

if [[ "${NPROC_PER_NODE}" != "${RESOURCE_GPU}" ]]; then
  echo "RESOURCE_GPU=${RESOURCE_GPU} must match NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
  exit 1
fi
if (( NODE_RANK >= NNODES )); then
  echo "NODE_RANK=${NODE_RANK} must be in [0, $((NNODES - 1))]" >&2
  exit 1
fi
if [[ "${NNODES}" != "1" && -z "${MASTER_ADDR}" ]]; then
  echo "MASTER_ADDR is required for multi-node training (rank-0 host/IP)" >&2
  exit 1
fi
if [[ -z "${MASTER_ADDR}" ]]; then
  MASTER_ADDR="127.0.0.1"
fi
if [[ "${NNODES}" != "1" &&
      ( "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" || "${MASTER_ADDR}" == "::1" ) ]]; then
  echo "MASTER_ADDR=${MASTER_ADDR} is a loopback address; use a rank-0 address reachable from all nodes" >&2
  exit 1
fi

if [[ "${ALLOW_TOPOLOGY_OVERRIDE}" != "1" && "${ALLOW_TOPOLOGY_OVERRIDE}" != "true" ]]; then
  if [[ "${NNODES}" != "${EXPECTED_NNODES}" || "${NPROC_PER_NODE}" != "${EXPECTED_GPUS_PER_NODE}" ]]; then
    echo "This entry defaults to ${EXPECTED_NNODES} nodes x ${EXPECTED_GPUS_PER_NODE} GPUs" >&2
    echo "Set ALLOW_TOPOLOGY_OVERRIDE=true only for a smaller script-path smoke" >&2
    exit 1
  fi
  if [[ "${TENSOR_MODEL_PARALLEL_SIZE}" != "${NPROC_PER_NODE}" ||
        "${PIPELINE_MODEL_PARALLEL_SIZE}" != "1" ||
        "${CONTEXT_PARALLEL_SIZE}" != "1" ]]; then
    echo "Dense 96-GPU topology requires TP=${NPROC_PER_NODE}, PP=1, CP=1" >&2
    echo "Set ALLOW_TOPOLOGY_OVERRIDE=true only for an intentional topology experiment" >&2
    exit 1
  fi
fi

if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
  if (( ${#GPU_LIST[@]} != NPROC_PER_NODE )); then
    echo "NPROC_PER_NODE=${NPROC_PER_NODE} does not match CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
  fi
fi

WORLD_SIZE_TOTAL=$((NNODES * NPROC_PER_NODE))
MODEL_PARALLEL_SIZE=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))
if (( WORLD_SIZE_TOTAL % MODEL_PARALLEL_SIZE != 0 )); then
  echo "total processes ${WORLD_SIZE_TOTAL} must be divisible by TP*PP*CP=${MODEL_PARALLEL_SIZE}" >&2
  exit 1
fi
DATA_PARALLEL_SIZE=$((WORLD_SIZE_TOTAL / MODEL_PARALLEL_SIZE))
if ! is_positive_integer "${MICRO_BATCH_SIZE}" || ! is_positive_integer "${GLOBAL_BATCH_SIZE}"; then
  echo "MICRO_BATCH_SIZE and GLOBAL_BATCH_SIZE must be positive integers" >&2
  exit 1
fi
STEP_BATCH_SIZE=$((MICRO_BATCH_SIZE * DATA_PARALLEL_SIZE))
if (( GLOBAL_BATCH_SIZE % STEP_BATCH_SIZE != 0 )); then
  echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by MICRO_BATCH_SIZE*DP=${STEP_BATCH_SIZE}" >&2
  exit 1
fi
GRAD_ACCUM_MICROBATCHES=$((GLOBAL_BATCH_SIZE / STEP_BATCH_SIZE))

DATASET_FILES=( "${DATASET_PATH}" )
if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" ]]; then
if [[ ! -d "${CODE_ROOT}" ]]; then
  echo "code root not found: ${CODE_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${SWIFT_ROOT}" ]]; then
  echo "ms-swift repo not found: ${SWIFT_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${SWIFT_VENV}/bin/activate" ]]; then
  echo "Qwen3.5 venv not found: ${SWIFT_VENV}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Dense model config not found: ${MODEL_PATH}/config.json" >&2
  exit 1
fi
shopt -s nullglob
DATASET_FILES=( ${DATASET_PATH} )
shopt -u nullglob
if [[ "${USE_CACHED_DATASET}" == "true" ]]; then
  if [[ ! -d "${CACHED_DATASET}" ]]; then
    echo "cached dataset not found: ${CACHED_DATASET}" >&2
    exit 1
  fi
elif (( ${#DATASET_FILES[@]} == 0 )); then
  echo "dataset not found: ${DATASET_PATH}" >&2
  exit 1
fi

# Validate the exact Dense model and every safetensors shard before launching
# 128 workers. A partial shared mount should fail locally instead of hanging
# the whole job during model initialization.
MODEL_CONFIG_PATH="${MODEL_PATH}/config.json" \
TENSOR_MODEL_PARALLEL_SIZE_CHECK="${TENSOR_MODEL_PARALLEL_SIZE}" \
PIPELINE_MODEL_PARALLEL_SIZE_CHECK="${PIPELINE_MODEL_PARALLEL_SIZE}" \
python - <<'PY'
import json
import os

path = os.environ["MODEL_CONFIG_PATH"]
with open(path, "r", encoding="utf-8") as handle:
    config = json.load(handle)
text_config = config.get("text_config", config)
model_type = text_config.get("model_type", config.get("model_type"))
architectures = config.get("architectures", [])
if model_type not in {"qwen3_5", "qwen3_5_text"}:
    raise SystemExit(f"Qwen3.5 Dense model expected, got model_type={model_type!r}")
if not any("Qwen3_5ForConditionalGeneration" in arch for arch in architectures):
    raise SystemExit(f"Dense architecture expected, got architectures={architectures!r}")
moe_keys = {
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
}
moe_fields = sorted(key for key in moe_keys if key in text_config)
if moe_fields:
    raise SystemExit(f"MoE fields found in Dense config: {moe_fields}")
expected = {
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_attention_heads": 24,
    "num_hidden_layers": 64,
    "num_key_value_heads": 4,
    "vocab_size": 248320,
}
mismatches = {
    key: (text_config.get(key), wanted)
    for key, wanted in expected.items()
    if text_config.get(key) != wanted
}
if mismatches:
    details = ", ".join(
        f"{key}={actual!r} (expected {wanted!r})"
        for key, (actual, wanted) in mismatches.items()
    )
    raise SystemExit("Qwen3.5-27B Dense config mismatch: " + details)
tp = int(os.environ["TENSOR_MODEL_PARALLEL_SIZE_CHECK"])
pp = int(os.environ["PIPELINE_MODEL_PARALLEL_SIZE_CHECK"])
if text_config["hidden_size"] % tp:
    raise SystemExit(f"hidden_size is not divisible by TP={tp}")
if text_config["num_attention_heads"] % tp:
    raise SystemExit(f"num_attention_heads is not divisible by TP={tp}")
if text_config["num_hidden_layers"] % pp:
    raise SystemExit(f"num_hidden_layers is not divisible by PP={pp}")
index_path = os.path.join(os.path.dirname(path), "model.safetensors.index.json")
if os.path.isfile(index_path):
    with open(index_path, "r", encoding="utf-8") as index_handle:
        index = json.load(index_handle)
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    missing = [
        shard for shard in shard_names
        if not os.path.isfile(os.path.join(os.path.dirname(path), shard))
        or os.path.getsize(os.path.join(os.path.dirname(path), shard)) == 0
    ]
    if missing:
        raise SystemExit("Missing/empty safetensors shard(s): " + ", ".join(missing))
    print(f"Validated Dense config and safetensors index: {len(shard_names)} shards")
else:
    print("Validated Dense config; no safetensors index was present")
PY

mkdir -p "${OUTPUT_DIR}" "${CACHE_ROOT}"/{huggingface/datasets,huggingface/transformers,modelscope,xdg} \
  "${LOCAL_TMP_ROOT}"/{huggingface/datasets,huggingface/transformers,modelscope,xdg,torchinductor,triton,torch_extensions} \
  "${SHORT_TMPDIR}"

source "${SWIFT_VENV}/bin/activate"
export PATH="${SWIFT_VENV}/bin:${PATH}"
cd "${SWIFT_ROOT}"
export PYTHONPATH="${SWIFT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
hash -r

echo "Using SWIFT_VENV=${SWIFT_VENV}"
echo "Using python=${SWIFT_VENV}/bin/python"
"${SWIFT_VENV}/bin/python" - <<'PY'
import importlib.metadata as im
import sys
print(f"Resolved python executable: {sys.executable}")
for name in ("torch", "transformers", "megatron-core", "mcore-bridge", "causal-conv1d"):
    try:
        print(f"Resolved {name}={im.version(name)}")
    except im.PackageNotFoundError:
        print(f"Resolved {name}=not-installed")
PY

# Run this on every node before either the NCCL preflight or model launch.
"${SWIFT_VENV}/bin/python" - <<'PY'
import importlib
import sys
from importlib.metadata import version

from packaging.version import Version


required = [
    "swift",
    "transformers.models.qwen3_5",
    "mcore_bridge",
    "megatron.core",
    "transformer_engine.pytorch",
    "fla",
    "tilelang",
    "causal_conv1d",
]
failed = {}
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        failed[name] = f"{type(exc).__name__}: {exc}"
if failed:
    for name, error in failed.items():
        print(f"Unable to import {name}: {error}", file=sys.stderr)
    sys.exit(1)
for distribution, minimum in (
    ("transformers", "5.9.0"),
    ("flash-linear-attention", "0.4.2"),
    ("tilelang", "0.1.9"),
    ("causal-conv1d", "1.4.0"),
):
    installed = Version(version(distribution))
    if installed < Version(minimum):
        print(f"{distribution}>={minimum} required; found {installed}", file=sys.stderr)
        sys.exit(1)
print(
    "Validated training environment: "
    + " ".join(
        f"{distribution}={version(distribution)}"
        for distribution in (
            "torch",
            "transformers",
            "megatron-core",
            "mcore-bridge",
            "transformer-engine",
            "flash-linear-attention",
            "tilelang",
            "causal-conv1d",
        )
    )
)
PY

fi

if [[ "${NCCL_PREFLIGHT_ONLY}" == "1" || "${NCCL_PREFLIGHT_ONLY}" == "true" ]]; then
  # Do not wrap the training entry in an outer torchrun. This is the one
  # explicit preflight launch; the normal Megatron entry launches its own
  # torchrun from NPROC_PER_NODE/NNODES/NODE_RANK.
  export CUDA_VISIBLE_DEVICES NPROC_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT
  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    printf "Resolved NCCL preflight command:"
    printf " %q" "${SWIFT_VENV}/bin/python" -m torch.distributed.run \
      --nproc_per_node "${NPROC_PER_NODE}" --nnodes "${NNODES}" \
      --node_rank "${NODE_RANK}" --master_addr "${MASTER_ADDR}" \
      --master_port "${MASTER_PORT}" "${SCRIPT_DIR}/dense_multinode_preflight.py"
    printf "\n"
    exit 0
  fi
  exec "${SWIFT_VENV}/bin/python" -m torch.distributed.run \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${SCRIPT_DIR}/dense_multinode_preflight.py" \
    --expected-world-size "${WORLD_SIZE_TOTAL}" \
    --expected-local-world-size "${NPROC_PER_NODE}" \
    --tensor-mib "${NCCL_PREFLIGHT_TENSOR_MIB}" \
    --timeout-sec "${NCCL_PREFLIGHT_TIMEOUT_SEC}"
fi

# Keep compiled kernels rank/node-specific. NCCL transport settings supplied
# by the platform (for example bond0 and IB HCA lists) are intentionally
# inherited instead of hard-coding an interface here.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy
unset USE_HF HF_ENDPOINT
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MODELSCOPE_OFFLINE=1
export HF_HOME="${LOCAL_TMP_ROOT}/huggingface"
export HF_DATASETS_CACHE="${LOCAL_TMP_ROOT}/huggingface/datasets"
export TRANSFORMERS_CACHE="${LOCAL_TMP_ROOT}/huggingface/transformers"
export MODELSCOPE_CACHE="${LOCAL_TMP_ROOT}/modelscope"
export XDG_CACHE_HOME="${LOCAL_TMP_ROOT}/xdg"
export TMPDIR="${SHORT_TMPDIR}"
export TORCHINDUCTOR_CACHE_DIR="${LOCAL_TMP_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${LOCAL_TMP_ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${LOCAL_TMP_ROOT}/torch_extensions"

export CUDA_VISIBLE_DEVICES NPROC_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
unset NCCL_BLOCKING_WAIT NCCL_ASYNC_ERROR_HANDLING
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-12}"
export USE_MCORE_GDN="${USE_MCORE_GDN:-1}"
export FLA_TILELANG="${FLA_TILELANG:-1}"
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
  --load_from_cache_file "${LOAD_FROM_CACHE_FILE}"
  --tuner_type full
  --freeze_llm false
  --freeze_vit true
  --freeze_aligner true
  --tensor_model_parallel_size "${TENSOR_MODEL_PARALLEL_SIZE}"
  --pipeline_model_parallel_size "${PIPELINE_MODEL_PARALLEL_SIZE}"
  --context_parallel_size "${CONTEXT_PARALLEL_SIZE}"
  --sequence_parallel true
  --recompute_granularity full
  --recompute_method uniform
  --recompute_num_layers 1
  --micro_batch_size "${MICRO_BATCH_SIZE}"
  --global_batch_size "${GLOBAL_BATCH_SIZE}"
  --finetune true
  --use_cpu_initialization "${USE_CPU_INITIALIZATION}"
  --cross_entropy_loss_fusion true
  --gradient_accumulation_fusion false
  --masked_softmax_fusion false
  --lr "${LEARNING_RATE}"
  --lr_warmup_fraction "${LR_WARMUP_FRACTION}"
  --min_lr "${MIN_LR}"
  --output_dir "${OUTPUT_DIR}"
  --save_steps "${SAVE_STEPS}"
  --logging_steps "${LOGGING_STEPS}"
  --max_length "${MAX_LENGTH}"
  --truncation_strategy "${TRUNCATION_STRATEGY}"
  --packing false
  --padding_free true
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --dataset_num_proc "${DATASET_NUM_PROC}"
  --no_save_optim "${NO_SAVE_OPTIM}"
  --no_save_rng "${NO_SAVE_RNG}"
  --use_precision_aware_optimizer "${USE_PRECISION_AWARE_OPTIMIZER}"
  --main_grads_dtype "${MAIN_GRADS_DTYPE}"
  --main_params_dtype "${MAIN_PARAMS_DTYPE}"
  --exp_avg_dtype "${EXP_AVG_DTYPE}"
  --exp_avg_sq_dtype "${EXP_AVG_SQ_DTYPE}"
  --optimizer_cpu_offload "${OPTIMIZER_CPU_OFFLOAD}"
  --optimizer_offload_fraction "${OPTIMIZER_OFFLOAD_FRACTION}"
  --attention_backend flash
)

if [[ "${USE_CACHED_DATASET}" == "true" ]]; then
  ARGS+=(--cached_dataset "${CACHED_DATASET}")
else
  ARGS+=(--dataset "${DATASET_FILES[@]}")
fi

if [[ -n "${TRAIN_ITERS}" ]]; then
  ARGS+=(--train_iters "${TRAIN_ITERS}")
else
  ARGS+=(--num_train_epochs "${NUM_TRAIN_EPOCHS}")
fi

echo "Running Megatron-SWIFT Qwen3.5-27B Dense 16-node entry"
echo "  model: ${MODEL_PATH}"
echo "  dataset pattern: ${DATASET_PATH}"
echo "  dataset files: ${#DATASET_FILES[@]}"
echo "  use_cached_dataset: ${USE_CACHED_DATASET}"
echo "  cached_dataset: ${CACHED_DATASET}"
echo "  output: ${OUTPUT_DIR}"
echo "  cache_root: ${CACHE_ROOT}"
echo "  cuda: ${CUDA_VISIBLE_DEVICES}"
echo "  nodes/rank: ${NNODES}/${NODE_RANK}"
echo "  nproc_per_node: ${NPROC_PER_NODE}"
echo "  master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  total processes: ${WORLD_SIZE_TOTAL}, dp: ${DATA_PARALLEL_SIZE}"
echo "  max_length: ${MAX_LENGTH}"
echo "  parallel: TP=${TENSOR_MODEL_PARALLEL_SIZE} PP=${PIPELINE_MODEL_PARALLEL_SIZE} CP=${CONTEXT_PARALLEL_SIZE} SP=true"
echo "  expert parallelism: disabled (Dense model, EP=1 ETP=1)"
echo "  micro/global batch: ${MICRO_BATCH_SIZE}/${GLOBAL_BATCH_SIZE}"
echo "  gradient accumulation microbatches: ${GRAD_ACCUM_MICROBATCHES}"
echo "  train_iters: ${TRAIN_ITERS:-auto}"
echo "  num_train_epochs: $([[ -n "${TRAIN_ITERS}" ]] && echo auto-disabled || echo "${NUM_TRAIN_EPOCHS}")"
echo "  NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME:-platform-default}"
echo "  NCCL_IB_HCA: ${NCCL_IB_HCA:-platform-default}"
echo "  FLA TileLang backend: ${FLA_TILELANG}"
echo "  local_tmp_root: ${LOCAL_TMP_ROOT}"
echo "  tmpdir: ${TMPDIR}"
echo "  hf_home: ${HF_HOME}"
echo "  modelscope_cache: ${MODELSCOPE_CACHE}"

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
  printf "Resolved command:"
  printf " %q" "${SWIFT_VENV}/bin/python" -m swift.cli._megatron.main "${ARGS[@]}"
  printf "\n"
  exit 0
fi

"${SWIFT_VENV}/bin/python" -m swift.cli._megatron.main "${ARGS[@]}"
