#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"

stop_pid_file() {
  local name="$1"
  local pid_file="${LOG_DIR}/${name}.pid"
  if [[ ! -f "${pid_file}" ]]; then
    echo "${name}: no pid file"
    return
  fi
  local pid
  pid="$(cat "${pid_file}")"
  if kill -0 "${pid}" 2>/dev/null; then
    echo "Stopping ${name} (pid ${pid})..."
    kill -INT "${pid}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  fi
  rm -f "${pid_file}"
}

stop_pid_file similarity_service
GPU_IDS="${GPU_IDS:-0,1,2,3}" LOG_DIR="${LOG_DIR}" \
  bash "${SCRIPT_DIR}/stop_qwen3_embedding_4replicas.sh"
