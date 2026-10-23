#!/usr/bin/env bash
set -euo pipefail

# Shared miles-megatron GSPO launcher for the Qwen3.5 single-turn world model.
#
# Model-specific wrappers set MODEL_* variables and then call this file.
#
# Important environment overrides:
#   MILES_LOAD             initial checkpoint (HF dir for the first run, or a
#                          Megatron save dir to resume)
#   MILES_HF_CHECKPOINT    HF checkpoint used for tokenizer/architecture/SGLang
#   MILES_SAVE             Megatron save directory
#   MILES_PROMPT_DATA      raw RL jsonl; prompts are rendered on the fly
#   MASTER_ADDR/HOSTFILE   multi-node Ray cluster controls
#   MASTER_PORT            Ray head port for the cluster join address
#   WORLD_SIZE/RANK/RESOURCE_GPU
#                          CPT-style node env contract; when present without
#                          HOSTFILE, each node starts its own Ray worker and
#                          rank-0 submits the job.
#   DRY_RUN=1              print the resolved command and exit

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Usage:
  bash run_qwen35_4b_gspo_128gpu_async.sh
  bash run_qwen35_9b_gspo_128gpu_async.sh
  bash run_qwen35_27b_gspo_128gpu_async.sh
  bash run_qwen35_35b_a3b_gspo_128gpu_async.sh
  bash run_qwen35_122b_a10b_gspo_160gpu_async.sh

THINK_MODE defaults to 0 for 4B/9B/27B/35B-A3B and 1 for 122B-A10B.
Set THINK_MODE=0/1 to override. See README.md for reward and service settings.

For a local argument smoke test (no Ray/GPU launch):
  DRY_RUN=1 bash run_qwen35_4b_gspo_128gpu_async.sh
USAGE
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RL_CODE_ROOT="${MILES_CODE_ROOT:-${SCRIPT_DIR}}"
MILES_REPO="${MILES_REPO:-${RL_CODE_ROOT}/miles}"
MEGATRON_REPO="${MILES_MEGATRON_ROOT:-${MEGATRON_REPO:-${RL_CODE_ROOT}/Megatron-LM}}"
SGLANG_REPO="${MILES_SGLANG_ROOT:-${SGLANG_REPO:-${RL_CODE_ROOT}/sglang-miles}}"
MILES_VENV="${MILES_VENV:-${RL_CODE_ROOT}/.venv}"
if [[ -x "${MILES_VENV}/bin/python" ]]; then
  PYTHON_BIN="${MILES_VENV}/bin/python"
  RAY_BIN="${MILES_VENV}/bin/ray"
else
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
  RAY_BIN="${RAY_BIN:-$(command -v ray || true)}"
  RAY_BIN="${RAY_BIN:-ray}"
fi
MILES_EXTRA_PYTHONPATH="${MILES_EXTRA_PYTHONPATH:-}"
RUNTIME_PYTHONPATH="${MEGATRON_REPO}:${SGLANG_REPO}/python:${SGLANG_REPO}/sgl-kernel/python:${MILES_REPO}:${SCRIPT_DIR}:${RL_CODE_ROOT}${MILES_EXTRA_PYTHONPATH:+:${MILES_EXTRA_PYTHONPATH}}"
DRY_RUN="${DRY_RUN:-0}"
RAY_STOP_BEFORE_START="${RAY_STOP_BEFORE_START:-0}"
DEBUG_ROLLOUT_ONLY="${DEBUG_ROLLOUT_ONLY:-0}"
DEBUG_TRAIN_ONLY="${DEBUG_TRAIN_ONLY:-0}"
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-1}"
COLOCATE="${COLOCATE:-1}"
if [[ "${DEBUG_ROLLOUT_ONLY}" == "1" || "${DEBUG_ROLLOUT_ONLY}" == "true" ]] && \
   [[ "${DEBUG_TRAIN_ONLY}" == "1" || "${DEBUG_TRAIN_ONLY}" == "true" ]]; then
  echo "DEBUG_ROLLOUT_ONLY and DEBUG_TRAIN_ONLY cannot both be enabled." >&2
  exit 1
fi

# User-specified hyper-parameters. The defaults implement the requested
# 128-GPU setting; they can be overridden for single-machine smoke tests.
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"
MIN_LR="${MIN_LR:-0.0}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.00}"
USE_KL_LOSS="${USE_KL_LOSS:-0}"
REF_LOAD="${REF_LOAD:-}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
NUM_ROLLOUT="${NUM_ROLLOUT:-}"
NUM_EPOCH="${NUM_EPOCH:-}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-32768}"
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-8192}"
SEQ_LEN="${SEQ_LEN:-40960}"
EVAL_MAX_PROMPT_TOKENS="${EVAL_MAX_PROMPT_TOKENS:-65536}"
EVAL_MAX_CONTEXT_LEN="${EVAL_MAX_CONTEXT_LEN:-81920}"
QKV_FORMAT="${QKV_FORMAT:-thd}"

if [[ -z "${ROLLOUT_STOP_TOKEN_IDS:-}" ]]; then
  ROLLOUT_STOP_TOKEN_IDS="248046"
fi

if [[ "${DEBUG_TRAIN_ONLY}" == "1" || "${DEBUG_TRAIN_ONLY}" == "true" ]]; then
  NUM_ROLLOUT="${DEBUG_TRAIN_NUM_ROLLOUT:-0}"
fi

COLOCATE_ARGS=()
if [[ "${COLOCATE}" == "1" || "${COLOCATE}" == "true" ]]; then
  COLOCATE_ARGS+=(--colocate)
fi

# Keep train prompt+response within the training context.
TRAIN_RESPONSE_BUDGET=$((SEQ_LEN - MAX_PROMPT_TOKENS))
if (( TRAIN_RESPONSE_BUDGET <= 0 )); then
  echo "MAX_PROMPT_TOKENS must be smaller than SEQ_LEN" >&2
  exit 1
fi
if (( MAX_RESPONSE_TOKENS > TRAIN_RESPONSE_BUDGET )); then
  echo "Adjusting MAX_RESPONSE_TOKENS from ${MAX_RESPONSE_TOKENS} to ${TRAIN_RESPONSE_BUDGET} to reserve 1 token" >&2
  MAX_RESPONSE_TOKENS="${TRAIN_RESPONSE_BUDGET}"
fi

# Shared defaults.
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.75}"
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-64}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-64}"
if [[ ! -v EVAL_DATA ]]; then
  EVAL_DATA="${PROJECT_ROOT}/data/rl/datasets/world_model_rl/test.jsonl"
fi
EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
EVAL_GLOBAL_BATCH_SIZE="${EVAL_GLOBAL_BATCH_SIZE:-}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-16384}"
if (( EVAL_MAX_PROMPT_TOKENS >= EVAL_MAX_CONTEXT_LEN )); then
  echo "EVAL_MAX_PROMPT_TOKENS must be smaller than EVAL_MAX_CONTEXT_LEN" >&2
  exit 1
fi
if (( EVAL_MAX_CONTEXT_LEN <= EVAL_MAX_PROMPT_TOKENS )); then
  echo "EVAL_MAX_CONTEXT_LEN must be larger than EVAL_MAX_PROMPT_TOKENS" >&2
  exit 1
fi
EVAL_RESPONSE_BUDGET=$((EVAL_MAX_CONTEXT_LEN - EVAL_MAX_PROMPT_TOKENS))
if (( EVAL_RESPONSE_BUDGET <= 0 )); then
  echo "EVAL_MAX_PROMPT_TOKENS must be smaller than EVAL_MAX_CONTEXT_LEN" >&2
  exit 1
fi
if (( EVAL_MAX_RESPONSE_LEN > EVAL_RESPONSE_BUDGET )); then
  echo "Adjusting EVAL_MAX_RESPONSE_LEN from ${EVAL_MAX_RESPONSE_LEN} to ${EVAL_RESPONSE_BUDGET} to reserve 1 token" >&2
  EVAL_MAX_RESPONSE_LEN="${EVAL_RESPONSE_BUDGET}"
fi
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://<JUDGE_HOST>:<JUDGE_PORT>/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-DeepSeek-V4-Flash-0731}"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-128}"
JUDGE_WEIGHTS="${JUDGE_WEIGHTS:-format:1,factuality:2,consistency:2,realism:2,quality:2,efficiency:1}"
JUDGE_ENABLED="${JUDGE_ENABLED:-1}"
# Shared defaults are judge-only (1.0/0.0), with similarity disabled.
# Model wrappers or the environment can override these: the 122B-A10B
# 160-GPU wrapper enables similarity with weights 0.9/0.1 by default.
JUDGE_WEIGHT="${JUDGE_WEIGHT:-1.0}"
SIMILARITY_ENABLED="${SIMILARITY_ENABLED:-0}"
SIMILARITY_WEIGHT="${SIMILARITY_WEIGHT:-0.0}"
SIMILARITY_URL="${SIMILARITY_URL:-http://<SIMILARITY_HOST>:<SIMILARITY_PORT>/similarity}"
SIMILARITY_TIMEOUT="${SIMILARITY_TIMEOUT:-60}"
SIMILARITY_CONCURRENCY="${SIMILARITY_CONCURRENCY:-128}"
THINK_MODE="${THINK_MODE:-0}"
JUDGE_PROMPT_DIR="${JUDGE_PROMPT_DIR:-${SCRIPT_DIR}/../prompts}"
SGLANG_QWEN3_5_TEXT_ONLY="${SGLANG_QWEN3_5_TEXT_ONLY:-1}"

if [[ -n "${WORLD_SIZE:-}" ]]; then
  ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-${WORLD_SIZE}}"
else
  ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-16}"
fi
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-${RESOURCE_GPU:-8}}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-${RESOURCE_GPU:-${ACTOR_NUM_GPUS_PER_NODE}}}"
HOSTFILE="${HOSTFILE:-}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-6379}"
SOCKET_IFNAME="${SOCKET_IFNAME:-}"
detect_socket_ifname() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import fcntl
import socket
import struct
import sys

target_ip = sys.argv[1]
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def interface_ip(name: str):
    try:
        ifreq = struct.pack('256s', name.encode())
        result = fcntl.ioctl(sock.fileno(), 0x8915, ifreq)
        return socket.inet_ntoa(result[20:24])
    except OSError:
        return None


interfaces = [name for _, name in socket.if_nameindex() if name != 'lo']
for name in interfaces:
    if interface_ip(name) == target_ip:
        print(name)
        raise SystemExit(0)

for name in interfaces:
    if interface_ip(name):
        print(name)
        raise SystemExit(0)

print('eth0')
PY
}
if [[ -n "${WORLD_SIZE:-}" && -n "${RANK:-}" && -z "${HOSTFILE}" && "${ACTOR_NUM_NODES}" != "1" ]]; then
  if [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
    echo "CPT-style multi-node mode requires MASTER_ADDR=<rank-0 IP/hostname>." >&2
    exit 1
  fi
fi

# miles' SGLang health check connects to the worker's own IP (not 127.0.0.1).
# Keep those host addresses out of the corporate proxy so engine startup does
# not hang.  Include the master, the local IP, every hostfile worker, and the
# RFC1918 ranges as a safety net.
LOCAL_IP="${LOCAL_IP:-}"
if [[ -z "${LOCAL_IP}" ]]; then
  LOCAL_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
fi
if [[ -z "${SOCKET_IFNAME}" ]]; then
  SOCKET_IFNAME="$(detect_socket_ifname "${LOCAL_IP}")"
  echo "[run_gspo][network] auto-detected SOCKET_IFNAME=${SOCKET_IFNAME} from LOCAL_IP=${LOCAL_IP}"
else
  echo "[run_gspo][network] using SOCKET_IFNAME=${SOCKET_IFNAME}"
fi
RUNTIME_NO_PROXY="${NO_PROXY:-${no_proxy:-}}"
append_no_proxy() {
  local value="$1"
  [[ -z "${value}" || "${RUNTIME_NO_PROXY}" == *"${value}"* ]] && return
  if [[ -z "${RUNTIME_NO_PROXY}" ]]; then
    RUNTIME_NO_PROXY="${value}"
  else
    RUNTIME_NO_PROXY="${RUNTIME_NO_PROXY},${value}"
  fi
}
for _ip in localhost 127.0.0.1 0.0.0.0 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16 "${MASTER_ADDR}" "${LOCAL_IP}"; do
  append_no_proxy "${_ip}"
done
if [[ -n "${HOSTFILE}" && -f "${HOSTFILE}" ]]; then
  while read -r _worker_ip; do
    [[ -z "${_worker_ip}" || "${_worker_ip}" == \#* ]] && continue
    append_no_proxy "${_worker_ip}"
  done < "${HOSTFILE}"
fi
unset -f append_no_proxy
export NO_PROXY="${RUNTIME_NO_PROXY}"
export no_proxy="${RUNTIME_NO_PROXY}"

ray_no_proxy() {
  env \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    NO_PROXY="${RUNTIME_NO_PROXY}" \
    no_proxy="${RUNTIME_NO_PROXY}" \
    "${RAY_BIN}" "$@"
}

python_no_proxy() {
  env \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    NO_PROXY="${RUNTIME_NO_PROXY}" \
    no_proxy="${RUNTIME_NO_PROXY}" \
    "${PYTHON_BIN}" "$@"
}

RL_DEBUG_LOG="${RL_DEBUG_LOG:-1}"
debug_log() {
  if [[ "${RL_DEBUG_LOG}" != "1" && "${RL_DEBUG_LOG}" != "true" ]]; then
    return 0
  fi
  echo "[run_gspo][debug] ================== environment debug =================="
  echo "[run_gspo][debug] hostname=$(hostname 2>/dev/null || echo unknown)"
  echo "[run_gspo][debug] RANK=${RANK:-<unset>} WORLD_SIZE=${WORLD_SIZE:-<unset>}"
  echo "[run_gspo][debug] RESOURCE_GPU=${RESOURCE_GPU:-<unset>} NPROC_PER_NODE=${NPROC_PER_NODE:-<unset>}"
  echo "[run_gspo][debug] MASTER_ADDR=${MASTER_ADDR:-<unset>} MASTER_PORT=${MASTER_PORT:-<unset>}"
  echo "[run_gspo][debug] LOCAL_IP=${LOCAL_IP} SOCKET_IFNAME=${SOCKET_IFNAME}"

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>&1 | while read -r line; do
      echo "[run_gspo][debug] gpu: ${line}"
    done || true
  else
    echo "[run_gspo][debug] nvidia-smi not found"
  fi

  if [[ -x "${PYTHON_BIN}" ]]; then
    "${PYTHON_BIN}" - <<'PY' || true
import sys
try:
    import torch
except Exception as exc:
    print("[run_gspo][debug] torch import failed:", exc)
    sys.exit(0)
print(
    "[run_gspo][debug] torch",
    torch.__version__,
    "cuda",
    torch.version.cuda,
    "cuda_available",
    torch.cuda.is_available(),
    "device_count",
    torch.cuda.device_count(),
)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print("[run_gspo][debug] device", i, torch.cuda.get_device_name(i))
PY
  else
    echo "[run_gspo][debug] python3 not found"
  fi

  if command -v nvcc >/dev/null 2>&1; then
    echo "[run_gspo][debug] nvcc=$(command -v nvcc)"
    nvcc --version 2>&1 | sed -n '1,2p' | while read -r line; do
      echo "[run_gspo][debug] nvcc: ${line}"
    done || true
  else
    echo "[run_gspo][debug] nvcc not found"
  fi
  echo "[run_gspo][debug] ========================================================="
}

if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" ]]; then
  debug_log
fi

wait_for_ray_cluster() {
  local expected_nodes="$1"
  local timeout_sec="${2:-300}"
  local poll_sec="${3:-5}"
  python_no_proxy - "$expected_nodes" "$timeout_sec" "$poll_sec" <<'PY'
import sys
import time

expected_nodes = int(sys.argv[1])
timeout_sec = int(sys.argv[2])
poll_sec = int(sys.argv[3])

import ray

ray.init(address="auto", logging_level="ERROR")
deadline = time.time() + timeout_sec
start_time = time.time()
last_alive = 0
node_summary = "<unavailable>"
while time.time() < deadline:
    nodes = ray.nodes()
    last_alive = sum(1 for node in nodes if node.get("Alive"))
    elapsed = int(time.time() - start_time)
    node_summary = ", ".join(
        f"{node.get('NodeManagerAddress', 'unknown')}:{node.get('NodeManagerPort', 'unknown')}:{'alive' if node.get('Alive') else 'dead'}"
        for node in nodes
    ) or "<empty>"
    print(
        f"[run_gspo][wait_ray_cluster] waiting for {expected_nodes} nodes, "
        f"alive={last_alive}, elapsed={elapsed}s, nodes=[{node_summary}]"
    )
    if last_alive >= expected_nodes:
        print(f"Ray cluster ready: {last_alive}/{expected_nodes} nodes")
        sys.exit(0)
    time.sleep(poll_sec)
print(
    f"[run_gspo][wait_ray_cluster] timeout: alive={last_alive}/{expected_nodes}, "
    f"elapsed={int(time.time() - start_time)}s, nodes=[{node_summary}]"
)
raise SystemExit(f"Timed out waiting for Ray cluster: {last_alive}/{expected_nodes} nodes")
PY
}

wait_for_ray_head() {
  local address="$1"
  local timeout_sec="${2:-600}"
  local poll_sec="${3:-5}"
  local deadline=$(( $(date +%s) + timeout_sec ))
  local attempts=0
  while (( $(date +%s) < deadline )); do
    attempts=$((attempts + 1))
    echo "[run_gspo][wait_ray_head] waiting for ${address} (attempt=${attempts}, timeout=${timeout_sec}s)"
    if ray_no_proxy health-check --address="${address}" >/dev/null 2>&1; then
      echo "Ray head ready at ${address}"
      return 0
    fi
    sleep "${poll_sec}"
  done
  echo "Timed out waiting for Ray head at ${address}" >&2
  return 1
}

MODEL_TAG="${MODEL_TAG:?MODEL_TAG must be set by the model-specific wrapper}"
MODEL_HF_DEFAULT="${MODEL_HF_DEFAULT:?MODEL_HF_DEFAULT must be set by the model-specific wrapper}"
SAVE_DEFAULT="${SAVE_DEFAULT:?SAVE_DEFAULT must be set by the model-specific wrapper}"
TP_DEFAULT="${TP_DEFAULT:?TP_DEFAULT must be set by the model-specific wrapper}"
PP_DEFAULT="${PP_DEFAULT:?PP_DEFAULT must be set by the model-specific wrapper}"
CP_DEFAULT="${CP_DEFAULT:?CP_DEFAULT must be set by the model-specific wrapper}"
EP_DEFAULT="${EP_DEFAULT:?EP_DEFAULT must be set by the model-specific wrapper}"
ETP_DEFAULT="${ETP_DEFAULT:?ETP_DEFAULT must be set by the model-specific wrapper}"
MAX_TOKENS_PER_GPU_DEFAULT="${MAX_TOKENS_PER_GPU_DEFAULT:?MAX_TOKENS_PER_GPU_DEFAULT must be set by the model-specific wrapper}"
LOG_PROBS_MAX_TOKENS_PER_GPU_DEFAULT="${LOG_PROBS_MAX_TOKENS_PER_GPU_DEFAULT:-${MAX_TOKENS_PER_GPU_DEFAULT}}"
ROLLOUT_GPUS_PER_ENGINE_DEFAULT="${ROLLOUT_GPUS_PER_ENGINE_DEFAULT:?ROLLOUT_GPUS_PER_ENGINE_DEFAULT must be set by the model-specific wrapper}"
DECODER_LAST_PIPELINE_NUM_LAYERS_DEFAULT="${DECODER_LAST_PIPELINE_NUM_LAYERS_DEFAULT:-}"
MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-${SCRIPT_DIR}/model_configs/${MODEL_TAG}.sh}"
if [[ ! -f "${MODEL_CONFIG_PATH}" ]]; then
  echo "Model config not found: ${MODEL_CONFIG_PATH}" >&2
  exit 1
fi

# Allow single-GPU debug and custom topology overrides without editing the
# per-model case above.
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${TP_DEFAULT}}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-${PP_DEFAULT}}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-${CP_DEFAULT}}"
EXPERT_PARALLEL_SIZE="${EXPERT_PARALLEL_SIZE:-${EP_DEFAULT}}"
EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-${ETP_DEFAULT}}"
if [[ -z "${DECODER_LAST_PIPELINE_NUM_LAYERS+x}" ]]; then
  DECODER_LAST_PIPELINE_NUM_LAYERS="${DECODER_LAST_PIPELINE_NUM_LAYERS_DEFAULT}"
fi
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-${MAX_TOKENS_PER_GPU_DEFAULT}}"
LOG_PROBS_MAX_TOKENS_PER_GPU="${LOG_PROBS_MAX_TOKENS_PER_GPU:-${LOG_PROBS_MAX_TOKENS_PER_GPU_DEFAULT}}"
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-128}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-${ROLLOUT_GPUS_PER_ENGINE_DEFAULT}}"
TRAIN_DRIVER="${TRAIN_DRIVER:-train.py}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-}"
ROLLOUT_FUNCTION_PATH="${ROLLOUT_FUNCTION_PATH:-world_model_rl.rollout.raw_train_rollout}"

MILES_LOAD="${MILES_LOAD:-${MODEL_HF_DEFAULT}}"
MILES_HF_CHECKPOINT="${MILES_HF_CHECKPOINT:-${MODEL_HF_DEFAULT}}"
MILES_SAVE="${MILES_SAVE:-${SAVE_DEFAULT}}"
ROLLOUT_OUTPUT_DIR="${ROLLOUT_OUTPUT_DIR:-${MILES_SAVE}/rollout_jsonl}"
MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-bridge}"
MILES_PROMPT_DATA="${MILES_PROMPT_DATA:-${PROJECT_ROOT}/data/rl/datasets/world_model_rl/train.jsonl}"
MILES_DATA_SOURCE_PATH="${MILES_DATA_SOURCE_PATH:-world_model_rl.data_source.WorldModelRLJsonlDataSource}"
CUSTOM_RM_PATH="${CUSTOM_RM_PATH-world_model_rl.reward.custom_rm}"
RM_TYPE="${RM_TYPE:-}"

source "${MODEL_CONFIG_PATH}"

if [[ ! -f "${MILES_HF_CHECKPOINT}/config.json" ]]; then
  echo "HF checkpoint not found: ${MILES_HF_CHECKPOINT}/config.json" >&2
  echo "Set MILES_HF_CHECKPOINT or MILES_LOAD to a valid HF checkpoint." >&2
  exit 1
fi
if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" && ! -f "${MILES_PROMPT_DATA}" ]]; then
  echo "RL prompt data not found: ${MILES_PROMPT_DATA}" >&2
  echo "Set MILES_PROMPT_DATA to the raw jsonl path." >&2
  exit 1
fi

CKPT_ARGS=(
  --hf-checkpoint "${MILES_HF_CHECKPOINT}"
  --load "${MILES_LOAD}"
  --save "${MILES_SAVE}"
  --save-interval "${SAVE_INTERVAL}"
  --megatron-to-hf-mode "${MEGATRON_TO_HF_MODE}"
)
if [[ -n "${REF_LOAD}" ]]; then
  CKPT_ARGS+=(--ref-load "${REF_LOAD}")
fi

ROLLOUT_ARGS=(
  --prompt-data "${MILES_PROMPT_DATA}"
  --data-source-path "${MILES_DATA_SOURCE_PATH}"
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --rollout-shuffle
  --rollout-seed 42
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --num-steps-per-rollout 1
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --rollout-max-prompt-len "${MAX_PROMPT_TOKENS}"
  --rollout-max-response-len "${MAX_RESPONSE_TOKENS}"
  --rollout-max-context-len "${SEQ_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE}"
  --rollout-top-p 1.0
  --balance-data
)

case "${THINK_MODE}" in
  1|true|TRUE|yes|YES|on|ON)
    ROLLOUT_ARGS+=(--think-mode)
    ;;
  *)
    ROLLOUT_ARGS+=(--no-think-mode)
    ;;
esac

if [[ -n "${ROLLOUT_STOP_TOKEN_IDS:-}" ]]; then
  ROLLOUT_ARGS+=(--rollout-stop-token-ids ${ROLLOUT_STOP_TOKEN_IDS})
fi

if [[ -n "${CUSTOM_RM_PATH}" ]]; then
  ROLLOUT_ARGS+=(--custom-rm-path "${CUSTOM_RM_PATH}")
fi
if [[ -n "${RM_TYPE}" ]]; then
  ROLLOUT_ARGS+=(--rm-type "${RM_TYPE}")
fi

if [[ -n "${NUM_ROLLOUT}" ]]; then
  ROLLOUT_ARGS+=(--num-rollout "${NUM_ROLLOUT}")
elif [[ -n "${NUM_EPOCH}" ]]; then
  ROLLOUT_ARGS+=(--num-epoch "${NUM_EPOCH}")
else
  echo "Either NUM_ROLLOUT or NUM_EPOCH must be set." >&2
  exit 1
fi

GSPO_ARGS=(
  --advantage-estimator "${ADVANTAGE_ESTIMATOR:-gspo}"
  --kl-loss-coef "${KL_LOSS_COEF}"
  --kl-loss-type low_var_kl
  --kl-coef 0.00
  --entropy-coef 0.00
  --eps-clip 3.5e-4
)
if [[ "${USE_KL_LOSS}" == "1" || "${USE_KL_LOSS}" == "true" ]]; then
  GSPO_ARGS+=(--use-kl-loss)
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LEARNING_RATE}"
  --lr-decay-style "${LR_DECAY_STYLE}"
  --min-lr "${MIN_LR}"
  --clip-grad "${CLIP_GRAD}"
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

PERF_ARGS=(
  --tensor-model-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --pipeline-model-parallel-size "${PIPELINE_PARALLEL_SIZE}"
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
  --expert-model-parallel-size "${EXPERT_PARALLEL_SIZE}"
  --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE}"
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --micro-batch-size 1
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-max-tokens-per-gpu "${LOG_PROBS_MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
)
if [[ "${QKV_FORMAT}" != "bshd" ]]; then
  PERF_ARGS+=(--use-dynamic-batch-size)
fi
if [[ "${SEQUENCE_PARALLEL}" == "1" || "${SEQUENCE_PARALLEL}" == "true" ]]; then
  PERF_ARGS+=(--sequence-parallel)
fi

if [[ -n "${DECODER_LAST_PIPELINE_NUM_LAYERS}" ]]; then
  PERF_ARGS+=(--decoder-last-pipeline-num-layers "${DECODER_LAST_PIPELINE_NUM_LAYERS}")
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
  --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
)
if [[ -n "${ROLLOUT_NUM_GPUS}" ]]; then
  SGLANG_ARGS+=(--rollout-num-gpus "${ROLLOUT_NUM_GPUS}")
fi
MISC_ARGS=(
  --rollout-function-path "${ROLLOUT_FUNCTION_PATH}"
  --log-interval "${LOG_INTERVAL}"
  --seq-length "${SEQ_LEN}"
  --max-position-embeddings "${SEQ_LEN}"
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --distributed-timeout-minutes 30
  "$@"
)
if [[ "${DEBUG_ROLLOUT_ONLY}" == "1" || "${DEBUG_ROLLOUT_ONLY}" == "true" ]]; then
  MISC_ARGS+=(--debug-rollout-only)
fi
if [[ "${DEBUG_TRAIN_ONLY}" == "1" || "${DEBUG_TRAIN_ONLY}" == "true" ]]; then
  MISC_ARGS+=(--debug-train-only)
fi
if [[ -n "${SAVE_DEBUG_ROLLOUT_DATA:-}" ]]; then
  MISC_ARGS+=(--save-debug-rollout-data "${SAVE_DEBUG_ROLLOUT_DATA}")
fi
if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA:-}" ]]; then
  MISC_ARGS+=(--load-debug-rollout-data "${LOAD_DEBUG_ROLLOUT_DATA}")
fi
if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE:-}" ]]; then
  MISC_ARGS+=(--load-debug-rollout-data-subsample "${LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE}")
fi
if [[ -n "${LR_DECAY_ITERS:-}" ]]; then
  MISC_ARGS+=(--lr-decay-iters "${LR_DECAY_ITERS}")
fi

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT:-world-model-rl}"
    --wandb-group "${WANDB_GROUP:-qwen35-world-model-gspo}"
  )
fi

# Do not silently disable evaluation because a configured dataset is missing.
if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" && -n "${EVAL_DATA:-}" && ! -f "${EVAL_DATA}" ]]; then
  echo "Configured EVAL_DATA not found: ${EVAL_DATA}. Supply a file or explicitly set EVAL_DATA= to disable RL evaluation." >&2
  exit 1
fi
EVAL_ARGS=()
if [[ -n "${EVAL_DATA:-}" && -f "${EVAL_DATA}" ]]; then
  EVAL_DATA_SPEC="${EVAL_DATA}"
  EVAL_DATA_LIMIT="${EVAL_DATA_LIMIT:-}"
  if [[ -n "${EVAL_DATA_LIMIT}" ]]; then
    if [[ ! "${EVAL_DATA_LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
      echo "EVAL_DATA_LIMIT must be a positive integer, got: ${EVAL_DATA_LIMIT}" >&2
      exit 1
    fi
    EVAL_DATA_SPEC="${EVAL_DATA}@[0:${EVAL_DATA_LIMIT}]"
  fi
  EVAL_ARGS=(
    --eval-interval "${EVAL_INTERVAL}"
    --eval-function-path "${EVAL_FUNCTION_PATH:-world_model_rl.eval_rollout.raw_eval_rollout}"
    --eval-prompt-data world_model_rl_eval "${EVAL_DATA_SPEC}"
    --eval-input-key "${EVAL_INPUT_KEY:-prompt}"
    --eval-label-key "${EVAL_LABEL_KEY:-ground_truth}"
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
    --eval-max-prompt-len "${EVAL_MAX_PROMPT_TOKENS}"
    --eval-max-context-len "${EVAL_MAX_CONTEXT_LEN}"
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
  )
fi

if [[ -n "${EVAL_GLOBAL_BATCH_SIZE}" ]]; then
  EVAL_ARGS+=(--eval-global-batch-size "${EVAL_GLOBAL_BATCH_SIZE}")
fi

RUNTIME_ENV_JSON="$(
  RL_CODE_ROOT="${RL_CODE_ROOT}" \
  MEGATRON_REPO="${MEGATRON_REPO}" \
  MILES_REPO="${MILES_REPO}" \
  SGLANG_REPO="${SGLANG_REPO}" \
  MILES_EXTRA_PYTHONPATH="${MILES_EXTRA_PYTHONPATH}" \
  RUNTIME_PYTHONPATH="${RUNTIME_PYTHONPATH}" \
  MILES_PROMPT_DATA_LIMIT="${MILES_PROMPT_DATA_LIMIT:-}" \
  WANDB_API_KEY="${WANDB_API_KEY:-}" \
  SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-}" \
  JUDGE_BASE_URL="${JUDGE_BASE_URL}" \
  JUDGE_MODEL="${JUDGE_MODEL}" \
  JUDGE_API_KEY="${JUDGE_API_KEY}" \
  JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS}" \
  JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY}" \
  JUDGE_WEIGHTS="${JUDGE_WEIGHTS}" \
  JUDGE_ENABLED="${JUDGE_ENABLED}" \
  JUDGE_WEIGHT="${JUDGE_WEIGHT}" \
  SIMILARITY_ENABLED="${SIMILARITY_ENABLED}" \
  SIMILARITY_WEIGHT="${SIMILARITY_WEIGHT}" \
  SIMILARITY_URL="${SIMILARITY_URL}" \
  SIMILARITY_TIMEOUT="${SIMILARITY_TIMEOUT}" \
  SIMILARITY_CONCURRENCY="${SIMILARITY_CONCURRENCY}" \
  THINK_MODE="${THINK_MODE}" \
  JUDGE_PROMPT_DIR="${JUDGE_PROMPT_DIR}" \
  SGLANG_QWEN3_5_TEXT_ONLY="${SGLANG_QWEN3_5_TEXT_ONLY}" \
  MILES_TORCH_PRELOAD="${MILES_TORCH_PRELOAD:-}" \
  MILES_NCCL_SO="${MILES_NCCL_SO:-}" \
  SGLANG_NCCL_SO_PATH="${SGLANG_NCCL_SO_PATH:-${MILES_NCCL_SO:-}}" \
  MILES_VENV="${MILES_VENV}" \
  VIRTUAL_ENV="${VIRTUAL_ENV:-${MILES_VENV}}" \
  PATH="${MILES_VENV}/bin:${PATH}" \
  PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}" \
  SOCKET_IFNAME="${SOCKET_IFNAME}" \
  MASTER_ADDR="${MASTER_ADDR}" \
  ROLLOUT_OUTPUT_DIR="${ROLLOUT_OUTPUT_DIR}" \
  MILES_HF_EXPORT_MODE="${MILES_HF_EXPORT_MODE:-}" \
  RUNTIME_NO_PROXY="${RUNTIME_NO_PROXY}" \
  "${PYTHON_BIN}" - <<'PY'
import json, os

def env(k, default=""):
    return os.environ.get(k, default)

env_vars = {
        "PYTHONPATH": env("RUNTIME_PYTHONPATH"),
        "PATH": env("PATH"),
        "VIRTUAL_ENV": env("VIRTUAL_ENV"),
        "PYTHONNOUSERSITE": env("PYTHONNOUSERSITE", "1"),
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "LD_PRELOAD": os.pathsep.join(filter(None, [
            env("MILES_TORCH_PRELOAD"),
            env("MILES_NCCL_SO"),
        ])),
        "MILES_TORCH_PRELOAD": env("MILES_TORCH_PRELOAD", ""),
        "MILES_NCCL_SO": env("MILES_NCCL_SO", ""),
        "SGLANG_NCCL_SO_PATH": env("SGLANG_NCCL_SO_PATH", ""),
        "PYTORCH_CUDA_ALLOC_CONF": env("PYTORCH_CUDA_ALLOC_CONF", ""),
        "GLOO_SOCKET_IFNAME": env("SOCKET_IFNAME", "eth0"),
        "TP_SOCKET_IFNAME": env("SOCKET_IFNAME", "eth0"),
        "NCCL_SOCKET_IFNAME": env("SOCKET_IFNAME", "eth0"),
        "MASTER_ADDR": env("MASTER_ADDR"),
        "NO_PROXY": env("RUNTIME_NO_PROXY"),
        "no_proxy": env("RUNTIME_NO_PROXY"),
        "WM_JUDGE_BASE_URL": env("JUDGE_BASE_URL"),
        "WM_JUDGE_MODEL": env("JUDGE_MODEL"),
        "WM_JUDGE_API_KEY": env("JUDGE_API_KEY"),
        "WM_JUDGE_MAX_TOKENS": env("JUDGE_MAX_TOKENS", "4096"),
        "WM_JUDGE_CONCURRENCY": env("JUDGE_CONCURRENCY", "128"),
        "WM_JUDGE_WEIGHTS": env("JUDGE_WEIGHTS"),
        "WM_JUDGE_ENABLED": env("JUDGE_ENABLED", "1"),
        "WM_JUDGE_WEIGHT": env("JUDGE_WEIGHT", "1.0"),
        "WM_SIMILARITY_ENABLED": env("SIMILARITY_ENABLED", "0"),
        "WM_SIMILARITY_WEIGHT": env("SIMILARITY_WEIGHT", "0.0"),
        "WM_SIMILARITY_URL": env("SIMILARITY_URL"),
        "WM_SIMILARITY_TIMEOUT": env("SIMILARITY_TIMEOUT", "60"),
        "WM_SIMILARITY_CONCURRENCY": env("SIMILARITY_CONCURRENCY", "128"),
        "WM_THINK_MODE": env("THINK_MODE", "0"),
        "WM_JUDGE_PROMPT_DIR": env("JUDGE_PROMPT_DIR"),
        "WM_EVAL_DATA_PATH": env("EVAL_DATA"),
        "WM_EVAL_DATA_LIMIT": env("WM_EVAL_DATA_LIMIT"),
        "WM_ROLLOUT_OUTPUT_DIR": env("ROLLOUT_OUTPUT_DIR"),
        "MILES_HF_EXPORT_MODE": env("MILES_HF_EXPORT_MODE", ""),
        "SGLANG_DISABLE_CUDNN_CHECK": env("SGLANG_DISABLE_CUDNN_CHECK", "1"),
        "SGLANG_QWEN3_5_TEXT_ONLY": env("SGLANG_QWEN3_5_TEXT_ONLY", "1"),
        "MILES_PROMPT_DATA_LIMIT": env("MILES_PROMPT_DATA_LIMIT"),
}
for name in (
    # Optional reward controls must reach all remote Ray workers as well.
    "WM_JUDGE_RETRIES", "WM_JUDGE_TIMEOUT", "WM_JUDGE_SKIP", "WM_SIMILARITY_RETRIES",
    "WANDB_API_KEY",
):
    value = env(name)
    if value:
        env_vars[name] = value

print(json.dumps({"env_vars": env_vars}))
PY
)"

print_args() {
  echo "Resolved command for MODEL_TAG=${MODEL_TAG}:"
  echo "  cd ${MILES_REPO}"
  echo "  ${RAY_BIN} job submit --address=http://127.0.0.1:8265 --runtime-env-json=<generated> -- env PYTHONPATH=${RUNTIME_PYTHONPATH} ${PYTHON_BIN} ${TRAIN_DRIVER}"
  printf '    %q\n' "${MODEL_ARGS[@]}"
  printf '    %q\n' "${CKPT_ARGS[@]}"
  printf '    %q\n' "${ROLLOUT_ARGS[@]}"
  printf '    %q\n' "${OPTIMIZER_ARGS[@]}"
  printf '    %q\n' "${GSPO_ARGS[@]}"
  printf '    %q\n' "${PERF_ARGS[@]}"
  printf '    %q\n' "${SGLANG_ARGS[@]}"
  printf '    %q\n' "${MISC_ARGS[@]}"
  printf '    %q\n' "${EVAL_ARGS[@]}"
  printf '    %q\n' "${WANDB_ARGS[@]}"
}

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
  print_args
  exit 0
fi

PYTHONPATH="${RUNTIME_PYTHONPATH}" "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_runtime.py" \
  --miles-root "${MILES_REPO}" --megatron-root "${MEGATRON_REPO}" --sglang-root "${SGLANG_REPO}" \
  --driver "${TRAIN_DRIVER}" --data-source "${MILES_DATA_SOURCE_PATH}" --reward "${CUSTOM_RM_PATH}" \
  --rollout "${ROLLOUT_FUNCTION_PATH}" --eval-rollout "${EVAL_FUNCTION_PATH:-world_model_rl.eval_rollout.raw_eval_rollout}" \
  --prompt-root "${JUDGE_PROMPT_DIR}" --judge-enabled "${JUDGE_ENABLED}" \
  --judge-base-url "${JUDGE_BASE_URL}" --judge-model "${JUDGE_MODEL}" \
  --similarity-enabled "${SIMILARITY_ENABLED}" --similarity-url "${SIMILARITY_URL}"


if [[ "${COLOCATE}" != "1" && "${COLOCATE}" != "true" ]]; then
  if [[ -z "${ROLLOUT_NUM_GPUS}" ]]; then
    echo "Non-colocate mode requires ROLLOUT_NUM_GPUS=<number of rollout GPUs>." >&2
    exit 1
  fi
fi
if [[ "${TRAIN_DRIVER}" == "train_async.py" && ( "${COLOCATE}" == "1" || "${COLOCATE}" == "true" ) ]]; then
  echo "train_async.py does not support --colocate; set COLOCATE=0." >&2
  exit 1
fi

if [[ ! -x "${RAY_BIN}" ]]; then
  echo "ray command not found" >&2
  exit 1
fi

if [[ "${ACTOR_NUM_NODES}" != "1" && "${MASTER_ADDR}" == "127.0.0.1" ]]; then
  echo "Multi-node mode requires MASTER_ADDR=<master-node-ip>." >&2
  exit 1
fi

NVLINK_COUNT="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)"
HAS_NVLINK=0
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then HAS_NVLINK=1; fi
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-${HAS_NVLINK}}"

export no_proxy="${RUNTIME_NO_PROXY}"
export NO_PROXY="${RUNTIME_NO_PROXY}"
mkdir -p "${MILES_SAVE}"

if [[ "${RAY_STOP_BEFORE_START}" == "1" || "${RAY_STOP_BEFORE_START}" == "true" ]]; then
  ray_no_proxy stop --force >/dev/null 2>&1 || true
fi

BOOTSTRAP_MODE="hostfile"
if [[ -n "${WORLD_SIZE:-}" && -n "${RANK:-}" && -z "${HOSTFILE}" ]]; then
  BOOTSTRAP_MODE="cloudml"
fi

if [[ "${BOOTSTRAP_MODE}" == "cloudml" ]]; then
  echo "[run_gspo][bootstrap] mode=${BOOTSTRAP_MODE} rank=${RANK} world_size=${WORLD_SIZE} actor_nodes=${ACTOR_NUM_NODES} actor_gpus_per_node=${ACTOR_NUM_GPUS_PER_NODE} master_addr=${MASTER_ADDR} master_port=${MASTER_PORT} local_ip=${LOCAL_IP} ray_head_timeout=${RAY_HEAD_READY_TIMEOUT_SEC:-600}s ray_cluster_timeout=${RAY_CLUSTER_READY_TIMEOUT_SEC:-300}s"
  if [[ "${RANK}" == "0" ]]; then
    echo "[run_gspo][head] stopping stale Ray runtime on rank0"
    ray_no_proxy stop --force >/dev/null 2>&1 || true
    echo "[run_gspo][head] starting Ray head on ${MASTER_ADDR}:${MASTER_PORT} with ${RAY_NUM_GPUS} GPUs"
    ray_no_proxy start --head --node-ip-address "${MASTER_ADDR}" \
      --port "${MASTER_PORT}" \
      --num-gpus "${RAY_NUM_GPUS}" \
      --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
    echo "[run_gspo][head] waiting for ${ACTOR_NUM_NODES} Ray nodes"
    wait_for_ray_cluster "${ACTOR_NUM_NODES}" "${RAY_CLUSTER_READY_TIMEOUT_SEC:-300}" "${RAY_CLUSTER_READY_POLL_SEC:-5}"
    echo "[run_gspo][head] cluster ready; submitting Ray job"
    cd "${MILES_REPO}"
    ray_no_proxy job submit --address="http://127.0.0.1:8265" \
      --working-dir "${MILES_REPO}" \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      -- env PYTHONPATH="${RUNTIME_PYTHONPATH}" PYTHONNOUSERSITE=1 "${PYTHON_BIN}" "${TRAIN_DRIVER}" \
      --train-backend megatron \
      --actor-num-nodes "${ACTOR_NUM_NODES}" \
      --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
      "${COLOCATE_ARGS[@]}" \
      "${MODEL_ARGS[@]}" \
      "${CKPT_ARGS[@]}" \
      "${ROLLOUT_ARGS[@]}" \
      "${OPTIMIZER_ARGS[@]}" \
      "${GSPO_ARGS[@]}" \
      "${PERF_ARGS[@]}" \
      "${SGLANG_ARGS[@]}" \
      "${MISC_ARGS[@]}" \
      "${EVAL_ARGS[@]}" \
      "${WANDB_ARGS[@]}"
  else
    echo "[run_gspo][worker] waiting for Ray head at ${MASTER_ADDR}:${MASTER_PORT}"
    wait_for_ray_head "${MASTER_ADDR}:${MASTER_PORT}" "${RAY_HEAD_READY_TIMEOUT_SEC:-600}"
    echo "[run_gspo][worker] stopping stale Ray runtime on rank${RANK}"
    ray_no_proxy stop --force >/dev/null 2>&1 || true
    echo "[run_gspo][worker] starting Ray worker node-ip=${LOCAL_IP} with ${RAY_NUM_GPUS} GPUs and block mode"
    ray_no_proxy start --block --address="${MASTER_ADDR}:${MASTER_PORT}" \
      --node-ip-address "${LOCAL_IP}" \
      --num-gpus "${RAY_NUM_GPUS}" \
      --disable-usage-stats
  fi
else
  if [[ "${ACTOR_NUM_NODES}" != "1" && -z "${HOSTFILE}" ]]; then
    echo "Multi-node mode requires HOSTFILE=<path to one worker IP per line>." >&2
    exit 1
  fi
  if [[ "${ACTOR_NUM_NODES}" == "1" ]]; then
    ray_no_proxy start --head --node-ip-address "${MASTER_ADDR}" \
      --port "${MASTER_PORT}" \
      --num-gpus "${RAY_NUM_GPUS}" \
      --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
  else
    ray_no_proxy start --head --node-ip-address "${MASTER_ADDR}" \
      --port "${MASTER_PORT}" \
      --num-gpus "${RAY_NUM_GPUS}" \
      --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

    while read -r worker_ip; do
      [[ -z "${worker_ip}" || "${worker_ip}" == "${MASTER_ADDR}" ]] && continue
      echo "Starting Ray worker on ${worker_ip}"
      ssh root@"${worker_ip}" \
        "${RAY_BIN} start --address=${MASTER_ADDR}:${MASTER_PORT} --num-gpus ${RAY_NUM_GPUS} --node-ip-address ${worker_ip} --disable-usage-stats" &
    done < "${HOSTFILE}"
    wait
  fi

  cd "${MILES_REPO}"
  ray_no_proxy job submit --address="http://127.0.0.1:8265" \
    --working-dir "${MILES_REPO}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- env PYTHONPATH="${RUNTIME_PYTHONPATH}" PYTHONNOUSERSITE=1 "${PYTHON_BIN}" "${TRAIN_DRIVER}" \
    --train-backend megatron \
    --actor-num-nodes "${ACTOR_NUM_NODES}" \
    --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
    "${COLOCATE_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GSPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${WANDB_ARGS[@]}"
fi
