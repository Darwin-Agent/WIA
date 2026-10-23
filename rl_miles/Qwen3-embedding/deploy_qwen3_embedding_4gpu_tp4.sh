#!/usr/bin/env bash
set -euo pipefail

# Fallback single-instance deployment. For independent long requests, use
# start_qwen3_embedding_4replicas.sh instead; four TP=1 replicas avoid TP
# collectives between GPUs and have higher aggregate throughput in tests.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MODEL_PATH="${MODEL_PATH:-<EMBEDDING_MODEL_PATH>}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
MAX_NUM_SCHEDULED_TOKENS="${MAX_NUM_SCHEDULED_TOKENS:-${MAX_NUM_BATCHED_TOKENS}}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "error: model directory not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -x "${VENV_DIR}/bin/vllm" ]]; then
  VENV_DIR="${VENV_DIR}" VLLM_VERSION="${VLLM_VERSION:-0.28.0}" bash "${SCRIPT_DIR}/install_runtime.sh"
fi

args=(
  serve "${MODEL_PATH}" --host "${HOST}" --port "${PORT}"
  --served-model-name "${SERVED_MODEL_NAME:-Qwen3-8B-Embedding}"
  --convert embed --runner pooling --dtype bfloat16
  --max-model-len "${MAX_MODEL_LEN:-32768}"
  --tensor-parallel-size 4
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --max-num-scheduled-tokens "${MAX_NUM_SCHEDULED_TOKENS}"
)
[[ -n "${KV_CACHE_DTYPE}" ]] && args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
if [[ "${ENABLE_PREFIX_CACHING}" == "1" || "${ENABLE_PREFIX_CACHING}" == "true" ]]; then
  args+=(--enable-prefix-caching)
else
  args+=(--no-enable-prefix-caching)
fi
if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" || "${ENABLE_CHUNKED_PREFILL}" == "true" ]]; then
  args+=(--enable-chunked-prefill)
else
  args+=(--no-enable-chunked-prefill)
fi

exec "${VENV_DIR}/bin/vllm" "${args[@]}" "$@"
