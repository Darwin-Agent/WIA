#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/preset-models}"
PORT="${PORT:-30000}"
HOST="${HOST:-0.0.0.0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TP_SIZE="${TP_SIZE:-8}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logging/single-node}"
ACCESS_URL_LOG="${ACCESS_URL_LOG:-${SCRIPT_DIR}/logging/lod.log}"

LAUNCH_LOG="${LAUNCH_LOG:-${LOG_DIR}/deploy-${RUN_ID}.log}"
LAUNCH_PID_FILE="${LAUNCH_PID_FILE:-${LOG_DIR}/deploy-${RUN_ID}.pid}"
SERVICE_LOG="${SERVICE_LOG:-${LOG_DIR}/sglang-deepseek-v4-flash-${RUN_ID}.log}"
SERVICE_PID_FILE="${SERVICE_PID_FILE:-${LOG_DIR}/sglang-deepseek-v4-flash-${RUN_ID}.pid}"

mkdir -p "${LOG_DIR}"

local_ip_address() {
  local local_ip=""

  if command -v ip >/dev/null 2>&1; then
    local_ip="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i == "src") {print $(i + 1); exit}}')"
  fi
  if [[ -z "${local_ip}" ]] && command -v hostname >/dev/null 2>&1; then
    local_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  fi

  printf '%s' "${local_ip:-127.0.0.1}"
}

if [[ -f "${SERVICE_PID_FILE}" ]] && kill -0 "$(cat "${SERVICE_PID_FILE}")" 2>/dev/null; then
  echo "检测到服务已存在：PID $(cat "${SERVICE_PID_FILE}")，PID 文件：${SERVICE_PID_FILE}" >&2
  echo "如需重启，请先执行：PID_FILE=${SERVICE_PID_FILE} ${SCRIPT_DIR}/deploy_sglang_deepseek_v4_flash.sh stop" >&2
  exit 1
fi

: >"${LAUNCH_LOG}"

nohup setsid env \
  MODEL_PATH="${MODEL_PATH}" \
  PORT="${PORT}" \
  HOST="${HOST}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  TP_SIZE="${TP_SIZE}" \
  LOG_FILE="${SERVICE_LOG}" \
  PID_FILE="${SERVICE_PID_FILE}" \
  KEEP_ENTRYPOINT_ALIVE="${KEEP_ENTRYPOINT_ALIVE:-0}" \
  WAIT_FOR_READY="${WAIT_FOR_READY:-1}" \
  WARMUP_AFTER_READY="${WARMUP_AFTER_READY:-1}" \
  SKIP_RUNTIME_CHECK="${SKIP_RUNTIME_CHECK:-0}" \
  "${SCRIPT_DIR}/deploy_sglang_deepseek_v4_flash.sh" serve \
    >"${LAUNCH_LOG}" 2>&1 </dev/null &

launcher_pid=$!
echo "${launcher_pid}" >"${LAUNCH_PID_FILE}"
disown "${launcher_pid}" 2>/dev/null || true

local_ip="$(local_ip_address)"
external_url="http://${local_ip}:${PORT}/v1"
models_url="http://${local_ip}:${PORT}/v1/models"
{
  echo "RUN_ID=${RUN_ID}"
  echo "HOST=${HOST}"
  echo "LOCAL_IP=${local_ip}"
  echo "PORT=${PORT}"
  echo "URL=${external_url}"
  echo "MODELS_URL=${models_url}"
} >>"${ACCESS_URL_LOG}"

echo "DeepSeek-V4-Flash 单节点 8 卡服务已在独立会话启动"
echo "启动器 PID：${launcher_pid}（PID 文件：${LAUNCH_PID_FILE}）"
echo "启动器日志：${LAUNCH_LOG}"
echo "服务日志：${SERVICE_LOG}"
echo "服务 PID 文件：${SERVICE_PID_FILE}"
echo "模型：${MODEL_PATH}"
echo "GPU：${CUDA_VISIBLE_DEVICES}，TP_SIZE：${TP_SIZE}"
echo "本机对外接口：${models_url}"
echo "访问 URL 已写入：${ACCESS_URL_LOG}"
echo ""
echo "后续查询请使用："
echo "  LOG_FILE=${SERVICE_LOG} PID_FILE=${SERVICE_PID_FILE} ${SCRIPT_DIR}/deploy_sglang_deepseek_v4_flash.sh status"
echo "  LOG_FILE=${SERVICE_LOG} PID_FILE=${SERVICE_PID_FILE} ${SCRIPT_DIR}/deploy_sglang_deepseek_v4_flash.sh logs"
echo "  LOG_FILE=${SERVICE_LOG} PID_FILE=${SERVICE_PID_FILE} ${SCRIPT_DIR}/deploy_sglang_deepseek_v4_flash.sh health"
echo "  LOG_FILE=${SERVICE_LOG} PID_FILE=${SERVICE_PID_FILE} ${SCRIPT_DIR}/deploy_sglang_deepseek_v4_flash.sh stop"
