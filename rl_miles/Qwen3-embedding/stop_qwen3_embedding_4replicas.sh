#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

IFS=',' read -r -a gpu_ids <<< "${GPU_IDS}"

for gpu in "${gpu_ids[@]}"; do
  pid_file="${LOG_DIR}/qwen3_embedding_gpu${gpu}.pid"
  if [[ ! -f "${pid_file}" ]]; then
    echo "GPU ${gpu}: no pid file"
    continue
  fi

  pid="$(cat "${pid_file}")"
  if kill -0 "${pid}" 2>/dev/null; then
    echo "Stopping GPU ${gpu} replica (pid ${pid})..."
    kill -INT "${pid}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${pid}" 2>/dev/null; then
      echo "GPU ${gpu} did not stop gracefully; sending SIGKILL."
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  else
    echo "GPU ${gpu}: pid ${pid} is not running"
  fi
  rm -f "${pid_file}"
done
