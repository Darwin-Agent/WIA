#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-serve}"
if [[ $# -gt 0 ]]; then
  shift
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TP_SIZE="${TP_SIZE:-1}"
export PORT="${PORT:-30000}"
export HOST="${HOST:-0.0.0.0}"
export WARMUP_AFTER_READY="${WARMUP_AFTER_READY:-1}"
export MM_FEATURE_TRANSPORT="${MM_FEATURE_TRANSPORT:-cpu}"
export CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-12}"
export ENABLE_MEMORY_SAVER="${ENABLE_MEMORY_SAVER:-1}"
export CUDA_GRAPH_BACKEND_DECODE="${CUDA_GRAPH_BACKEND_DECODE:-full}"
export CUDA_GRAPH_BACKEND_PREFILL="${CUDA_GRAPH_BACKEND_PREFILL:-disabled}"

exec "${SCRIPT_DIR}/deploy_sglang_deepseek_v4_flash.sh" "${MODE}" "$@"
