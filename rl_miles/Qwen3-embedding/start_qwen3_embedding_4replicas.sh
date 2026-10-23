#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
PORT_BASE="${PORT_BASE:-8200}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-360}"

mkdir -p "${LOG_DIR}"
IFS=',' read -r -a gpu_ids <<< "${GPU_IDS}"

if [[ "${#gpu_ids[@]}" -eq 0 ]]; then
  echo "error: GPU_IDS is empty" >&2
  exit 1
fi

for index in "${!gpu_ids[@]}"; do
  gpu="${gpu_ids[$index]}"
  port=$((PORT_BASE + index))
  pid_file="${LOG_DIR}/qwen3_embedding_gpu${gpu}.pid"
  log_file="${LOG_DIR}/qwen3_embedding_gpu${gpu}.log"

  if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    echo "GPU ${gpu} / port ${port} is already running (pid $(cat "${pid_file}"))."
    continue
  fi

  rm -f "${pid_file}"
  echo "Starting TP=1 replica: GPU ${gpu}, port ${port}"
  nohup setsid env \
    GPU_ID="${gpu}" \
    PORT="${port}" \
    bash "${SCRIPT_DIR}/deploy_qwen3_embedding_replica.sh" \
    </dev/null >"${log_file}" 2>&1 &
  echo "$!" >"${pid_file}"
done

for index in "${!gpu_ids[@]}"; do
  gpu="${gpu_ids[$index]}"
  port=$((PORT_BASE + index))
  pid_file="${LOG_DIR}/qwen3_embedding_gpu${gpu}.pid"
  url="http://127.0.0.1:${port}/v1/models"
  ready=0

  for _ in $(seq 1 "${WAIT_TIMEOUT}"); do
    if curl -fsS --noproxy '*' --max-time 2 "${url}" >/dev/null 2>&1; then
      echo "Replica GPU ${gpu} is ready at ${url}"
      ready=1
      break
    fi
    if [[ -f "${pid_file}" ]] && ! kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
      echo "Replica GPU ${gpu} exited; see ${LOG_DIR}/qwen3_embedding_gpu${gpu}.log" >&2
      exit 1
    fi
    sleep 1
  done

  if [[ "${ready}" != 1 ]]; then
    echo "Timed out waiting for replica GPU ${gpu} at ${url}" >&2
    exit 1
  fi
done

echo "All Qwen3 TP=1 replicas are ready."
