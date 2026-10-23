#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
SIMILARITY_HOST="${SIMILARITY_HOST:-0.0.0.0}"
SIMILARITY_PORT="${SIMILARITY_PORT:-8100}"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"
EMBEDDING_API_BASES="${EMBEDDING_API_BASES:-http://127.0.0.1:8200,http://127.0.0.1:8201,http://127.0.0.1:8202,http://127.0.0.1:8203}"

mkdir -p "${LOG_DIR}"
GPU_IDS="${GPU_IDS:-0,1,2,3}" LOG_DIR="${LOG_DIR}" \
  bash "${SCRIPT_DIR}/start_qwen3_embedding_4replicas.sh"

PID_FILE="${LOG_DIR}/similarity_service.pid"
LOG_FILE="${LOG_DIR}/similarity_service.log"
if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "Similarity service is already running (pid $(cat "${PID_FILE}"))."
  exit 0
fi

echo "Starting similarity service on ${SIMILARITY_HOST}:${SIMILARITY_PORT}"
nohup setsid env \
  EMBEDDING_API_BASES="${EMBEDDING_API_BASES}" \
  SIMILARITY_HOST="${SIMILARITY_HOST}" \
  SIMILARITY_PORT="${SIMILARITY_PORT}" \
  "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/similarity_service.py" \
  </dev/null >"${LOG_FILE}" 2>&1 &
echo "$!" >"${PID_FILE}"

url="http://127.0.0.1:${SIMILARITY_PORT}/health"
for _ in $(seq 1 60); do
  if curl -fsS --noproxy '*' --max-time 2 "${url}" >/dev/null 2>&1; then
    echo "Similarity service is ready at ${url}"
    exit 0
  fi
  if ! kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "Similarity service exited; see ${LOG_FILE}" >&2
    exit 1
  fi
  sleep 1
done

echo "Timed out waiting for ${url}; see ${LOG_FILE}" >&2
exit 1
