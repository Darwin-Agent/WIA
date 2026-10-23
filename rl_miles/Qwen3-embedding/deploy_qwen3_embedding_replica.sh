#!/usr/bin/env bash
set -euo pipefail

# Launch one independent TP=1 Qwen3 embedding replica.
# The four-replica launcher sets GPU_ID and PORT for each process.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MODEL_PATH="${MODEL_PATH:-<EMBEDDING_MODEL_PATH>}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8200}"
GPU_ID="${GPU_ID:-0}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-8B-Embedding}"
VLLM_VERSION="${VLLM_VERSION:-0.28.0}"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_NUM_SCHEDULED_TOKENS="${MAX_NUM_SCHEDULED_TOKENS:-${MAX_NUM_BATCHED_TOKENS}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"

export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "error: model directory not found: ${MODEL_PATH}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [[ ! -x "${VENV_DIR}/bin/vllm" ]]; then
  install_lock="${VENV_DIR}.install.lock"
  while ! mkdir "${install_lock}" 2>/dev/null; do
    sleep 1
  done
  trap 'rmdir "${install_lock}" 2>/dev/null || true' EXIT
  if [[ ! -x "${VENV_DIR}/bin/vllm" ]]; then
    echo ">> Creating vLLM environment at ${VENV_DIR}"
    VENV_DIR="${VENV_DIR}" VLLM_VERSION="${VLLM_VERSION}" bash "${SCRIPT_DIR}/install_runtime.sh"
  fi
  rmdir "${install_lock}" 2>/dev/null || true
  trap - EXIT
fi

VLLM_ARGS=(
  serve "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --convert embed
  --runner pooling
  --dtype bfloat16
  --max-model-len "${MAX_MODEL_LEN}"
  --tensor-parallel-size 1
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --max-num-scheduled-tokens "${MAX_NUM_SCHEDULED_TOKENS}"
)

if [[ -n "${KV_CACHE_DTYPE}" ]]; then
  VLLM_ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi

if [[ "${ENABLE_PREFIX_CACHING}" == "1" || "${ENABLE_PREFIX_CACHING}" == "true" ]]; then
  VLLM_ARGS+=(--enable-prefix-caching)
else
  VLLM_ARGS+=(--no-enable-prefix-caching)
fi

if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" || "${ENABLE_CHUNKED_PREFILL}" == "true" ]]; then
  VLLM_ARGS+=(--enable-chunked-prefill)
else
  VLLM_ARGS+=(--no-enable-chunked-prefill)
fi

exec "${VENV_DIR}/bin/vllm" "${VLLM_ARGS[@]}" "$@"
