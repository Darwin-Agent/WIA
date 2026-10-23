#!/usr/bin/env bash

unset http_proxy https_proxy all_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset no_proxy NO_PROXY

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-serve}"

MODEL_PATH="${MODEL_PATH:-/preset-models}"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"
SGLANG_SRC="${SGLANG_SRC:-${SCRIPT_DIR}/sglang-src}"
SGLANG_REPO="${SGLANG_REPO:-https://github.com/sgl-project/sglang.git}"
SGLANG_REF="${SGLANG_REF:-a5c9636}"
PORT="${PORT:-30000}"
HOST="${HOST:-0.0.0.0}"
TP_SIZE="${RESOURCE_GPU:-8}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-81920}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-8192}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-V4-Flash-0731}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-dsv4}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-marlin}"
CUDA_GRAPH_BACKEND_DECODE="${CUDA_GRAPH_BACKEND_DECODE:-full}"
CUDA_GRAPH_BACKEND_PREFILL="${CUDA_GRAPH_BACKEND_PREFILL:-disabled}"
ENABLE_TWO_BATCH_OVERLAP="${ENABLE_TWO_BATCH_OVERLAP:-0}"
ENABLE_MIXED_CHUNK="${ENABLE_MIXED_CHUNK:-0}"
ENABLE_DYNAMIC_CHUNKING="${ENABLE_DYNAMIC_CHUNKING:-0}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${SCRIPT_DIR}/chat_templates/deepseek_v4_flash.jinja}"
REASONING_PARSER="${REASONING_PARSER:-deepseek-v4}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-deepseekv4}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/logging/sglang-deepseek-v4-flash-rank${RANK:-unknown}.log}"
PID_FILE="${PID_FILE:-${SCRIPT_DIR}/sglang-deepseek-v4-flash-rank${RANK:-unknown}.pid}"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-${SCRIPT_DIR}/run_sglang_deepseek_v4_flash.py}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
HEALTH_INTERVAL="${HEALTH_INTERVAL:-10}"
WARMUP_AFTER_READY="${WARMUP_AFTER_READY:-1}"
WARMUP_TIMEOUT="${WARMUP_TIMEOUT:-600}"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"

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

print_network_info() {
  local local_ip api_host

  local_ip="$(local_ip_address)"
  api_host="${local_ip}"

  echo "[多节点独立部署] 本节点 IP: ${local_ip}"
  echo "[多节点独立部署] 节点 Rank: ${RANK:-未设置}/${WORLD_SIZE:-未设置}"
  echo "[多节点独立部署] 本节点 GPU: ${RESOURCE_GPU:-未设置}，TP: ${TP_SIZE}"
  echo "[多节点独立部署] 监听: http://${HOST}:${PORT}/v1"
  echo "[多节点独立部署] API: http://${api_host}:${PORT}/v1"
  echo "[多节点独立部署] Models: http://${api_host}:${PORT}/v1/models"
}

port_is_available() {
  python3 - "$1" "$2" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((host, port))
except OSError as exc:
    raise SystemExit(f"{host}:{port} -> {exc}")
finally:
    sock.close()
PY
}

resolve_port() {
  local start_port candidate_port port_check
  start_port="${PORT}"
  candidate_port="${start_port}"
  for (( port_check = 0; port_check < 100; port_check++ )); do
    if port_is_available "${HOST}" "${candidate_port}" >/dev/null 2>&1; then
      PORT="${candidate_port}"
      export PORT
      if [[ "${candidate_port}" != "${start_port}" ]]; then
        echo "[多节点独立部署] 端口 ${start_port} 已占用，自动切换到 ${candidate_port}"
      fi
      return 0
    fi
    candidate_port=$((candidate_port + 1))
  done

  echo "多机服务端口不可用：已从 ${start_port} 连续探测 100 个端口仍未找到可用端口" >&2
  exit 1
}

require_commands() {
  local command_name
  for command_name in "$@"; do
    command -v "${command_name}" >/dev/null 2>&1 || {
      echo "缺少系统命令：${command_name}" >&2
      exit 1
    }
  done
}

clear_invalid_proxy_vars() {
  local name value
  for name in HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; do
    value="${!name:-}"
    if [[ "$value" == *'**'* ]]; then
      unset "$name"
    fi
  done
}

activate_venv() {
  if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "虚拟环境不存在：${VENV_DIR}。请先运行：$0 install" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
}

validate_model() {
  [[ -f "${MODEL_PATH}/config.json" ]] || {
    echo "找不到模型配置：${MODEL_PATH}/config.json" >&2
    exit 1
  }
  [[ -f "${MODEL_PATH}/model.safetensors.index.json" ]] || {
    echo "找不到模型权重索引：${MODEL_PATH}/model.safetensors.index.json" >&2
    exit 1
  }
  python3 - "${MODEL_PATH}/config.json" <<'PY'
import json
import sys

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as handle:
    config = json.load(handle)

quant_config = config.get("quantization_config") or {}
if quant_config.get("quant_method") != "fp8":
    raise SystemExit(f"模型不是 FP8 量化：quantization_config={quant_config!r}")

if config.get("model_type") != "deepseek_v4":
    raise SystemExit(f"模型类型不是 deepseek_v4：model_type={config.get('model_type')!r}")
PY
}

validate_multi_node_env() {
  local required_vars=(RESOURCE_GPU WORLD_SIZE RANK)
  local var_name var_value
  for var_name in "${required_vars[@]}"; do
    var_value="${!var_name:-}"
    if [[ -z "${var_value}" ]]; then
      echo "缺少多机环境变量：${var_name}" >&2
      exit 1
    fi
  done

  if [[ ! "${WORLD_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "WORLD_SIZE 必须是正整数，当前值：${WORLD_SIZE}" >&2
    exit 1
  fi

  if [[ ! "${RESOURCE_GPU}" =~ ^[0-9]+$ ]]; then
    echo "RESOURCE_GPU 必须是每个节点可用 GPU 数量，当前值：${RESOURCE_GPU}" >&2
    exit 1
  fi

  if [[ ! "${RANK}" =~ ^[0-9]+$ ]] || (( RANK >= WORLD_SIZE )); then
    echo "RANK 必须是 0 到 WORLD_SIZE-1 之间的节点序号，当前值：${RANK}/${WORLD_SIZE}" >&2
    exit 1
  fi

  if [[ ! "${TP_SIZE}" =~ ^[0-9]+$ ]]; then
    echo "TP_SIZE 必须是数字，当前值：${TP_SIZE}" >&2
    exit 1
  fi
}

validate_gpu_count() {
  require_commands nvidia-smi
  command -v nvidia-smi >/dev/null || {
    echo "未找到 nvidia-smi。此脚本仅支持 NVIDIA GPU 部署。" >&2
    exit 1
  }
  local gpu_count
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    gpu_count="$(tr ',' '\n' <<<"${CUDA_VISIBLE_DEVICES}" | sed '/^$/d' | wc -l | tr -d ' ')"
  else
    gpu_count="$(nvidia-smi -L | wc -l | tr -d ' ')"
  fi
  if (( gpu_count < RESOURCE_GPU )); then
    echo "可见 GPU 数量为 ${gpu_count}，但 RESOURCE_GPU=${RESOURCE_GPU}。CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-未设置}" >&2
    exit 1
  fi
}

pid_is_service() {
  local pid="$1"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  [[ "$(tr '\0' ' ' <"/proc/${pid}/cmdline")" == *"${LAUNCH_SCRIPT}"* ]]
}

runtime_ready() {
  [[ -f "${VENV_DIR}/bin/activate" ]] || return 1
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  verify_runtime
}

verify_runtime() {
  local patch_file
  for patch_file in "${SCRIPT_DIR}"/patches/*.patch; do
    if ! git -C "${SGLANG_SRC}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
      echo "Missing/unverified DS runtime patch: ${patch_file}; run install first" >&2
      return 1
    fi
  done
  export WM_DS_SGLANG_SRC="${SGLANG_SRC}"
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  python - <<'PY'
import importlib.metadata as metadata
import os
from pathlib import Path
import importlib.util
import sglang

required_versions = {
    "sglang-kernel": "0.4.6.post1",
    "flashinfer_python": "0.6.13",
    "transformers": "5.12.1",
    "xgrammar": "0.2.1",
}

for package, expected in required_versions.items():
    installed = metadata.version(package)
    if installed != expected:
        raise SystemExit(f"{package}={installed}，需要 {expected}")

if importlib.util.find_spec("sgl_kernel") is None:
    raise SystemExit("没有安装 sgl_kernel CUDA 包")

import flashinfer.tllm_enums  # noqa: F401
from sglang.srt.models import deepseek_v4  # noqa: F401
import sglang
expected_root = Path(os.environ["WM_DS_SGLANG_SRC"]).resolve() / "python"
assert Path(sglang.__file__).resolve().is_relative_to(expected_root), "Importing an unpatched SGLang installation"
print(f"SGLang: {sglang.__file__}")
print("DeepSeek-V4 运行时依赖检查通过")
PY
}

install() {
  # The four-node entry must install the same patched runtime as the single-node entry.
  VENV_DIR="${VENV_DIR}" SGLANG_SRC="${SGLANG_SRC}" SGLANG_REPO="${SGLANG_REPO}" \
    SGLANG_REF="${SGLANG_REF}" bash "${SCRIPT_DIR}/deploy_sglang_deepseek_v4_flash.sh" install
}

service_pid() {
  [[ -f "${PID_FILE}" ]] || return 1
  cat "${PID_FILE}"
}

health_url() {
  printf 'http://127.0.0.1:%s/v1/models' "${PORT}"
}

health_check() {
  clear_invalid_proxy_vars
  curl --noproxy '*' --fail --silent --show-error --max-time 5 "$(health_url)"
}

wait_ready() {
  local pid="$1"
  local deadline now
  deadline=$(( $(date +%s) + STARTUP_TIMEOUT ))
  echo "等待服务就绪，最长 ${STARTUP_TIMEOUT}s：$(health_url)"
  while true; do
    if health_check >/tmp/dsv4_flash_health_rank${RANK}.json 2>/tmp/dsv4_flash_health_rank${RANK}.err; then
      echo "服务已就绪：$(cat /tmp/dsv4_flash_health_rank${RANK}.json)"
      return 0
    fi

    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "服务启动失败，最近日志如下：" >&2
      tail -n 120 "${LOG_FILE}" >&2 || true
      rm -f "${PID_FILE}"
      return 1
    fi

    now="$(date +%s)"
    if (( now >= deadline )); then
      echo "等待服务就绪超时，进程仍在运行（PID ${pid}）。最近日志如下：" >&2
      tail -n 120 "${LOG_FILE}" >&2 || true
      return 1
    fi

    sleep "${HEALTH_INTERVAL}"
  done
}

warmup() {
  clear_invalid_proxy_vars
  local payload
  payload=$(cat <<JSON
{"model":"${SERVED_MODEL_NAME}","messages":[{"role":"user","content":"你好"}],"max_tokens":1,"temperature":0}
JSON
)
  echo "执行首轮推理预热，最长 ${WARMUP_TIMEOUT}s。"
  if curl --noproxy '*' --fail --silent --show-error --max-time "${WARMUP_TIMEOUT}" \
      "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "${payload}" >/tmp/dsv4_flash_warmup_rank${RANK}.json 2>/tmp/dsv4_flash_warmup_rank${RANK}.err; then
    echo "预热完成：$(cat /tmp/dsv4_flash_warmup_rank${RANK}.json)"
    return 0
  fi
  echo "预热失败：$(cat /tmp/dsv4_flash_warmup_rank${RANK}.err)" >&2
  tail -n 120 "${LOG_FILE}" >&2 || true
  return 1
}

kill_process_tree() {
  local pid="$1"
  local child
  for child in $(pgrep -P "${pid}" 2>/dev/null || true); do
    kill_process_tree "${child}"
  done
  kill "${pid}" 2>/dev/null || true
}

wait_stopped() {
  local pid="$1"
  local deadline=$(( $(date +%s) + STOP_TIMEOUT ))
  while pid_is_service "${pid}"; do
    if (( $(date +%s) >= deadline )); then
      echo "服务未能在 ${STOP_TIMEOUT}s 内退出，PID ${pid} 仍存在。" >&2
      return 1
    fi
    sleep 1
  done
}

stop() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}")"
    if pid_is_service "${pid}"; then
      kill_process_tree "${pid}"
      wait_stopped "${pid}" || true
      echo "已停止服务（PID ${pid}）"
    elif kill -0 "${pid}" 2>/dev/null; then
      echo "PID 文件指向的进程不是本服务，跳过终止：PID ${pid}" >&2
    fi
    rm -f "${PID_FILE}"
  else
    echo "未找到 PID 文件：${PID_FILE}"
  fi
}

status() {
  local pid
  pid="$(service_pid || true)"
  if [[ -n "${pid}" ]] && pid_is_service "${pid}"; then
    echo "服务进程运行中：PID ${pid}"
  else
    echo "服务未运行"
    rm -f "${PID_FILE}"
    return 1
  fi
}

health() {
  health_check
  echo
}

logs() {
  tail -n 200 -f "${LOG_FILE}"
}

serve() {
  validate_multi_node_env
  require_commands bash curl date pgrep python3 setsid sleep tail tr
  resolve_port
  print_network_info
  validate_model
  validate_gpu_count
  if ! runtime_ready; then
    echo "运行时依赖不完整，开始自动安装 / 修复。"
    install
  fi
  [[ -f "${LAUNCH_SCRIPT}" ]] || {
    echo "找不到启动脚本：${LAUNCH_SCRIPT}" >&2
    exit 1
  }
  if [[ -n "${CHAT_TEMPLATE}" && ! -f "${CHAT_TEMPLATE}" ]]; then
    echo "找不到 chat template：${CHAT_TEMPLATE}" >&2
    exit 1
  fi

  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((RESOURCE_GPU - 1)))"
  fi

  if [[ -f "${PID_FILE}" ]] && pid_is_service "$(cat "${PID_FILE}")"; then
    local running_pid
    running_pid="$(cat "${PID_FILE}")"
    if health_check >/tmp/dsv4_flash_health_rank${RANK}.json 2>/dev/null; then
      echo "服务已运行且健康（PID ${running_pid}）：$(cat /tmp/dsv4_flash_health_rank${RANK}.json)"
      return 0
    fi
    echo "服务进程已存在但尚未健康（PID ${running_pid}），继续等待。"
    wait_ready "${running_pid}"
    if [[ "${WARMUP_AFTER_READY}" == "1" ]]; then
      warmup
    fi
    return $?
  fi
  if [[ -f "${PID_FILE}" ]]; then
    echo "清理失效 PID 文件：${PID_FILE}"
    rm -f "${PID_FILE}"
  fi

  mkdir -p "$(dirname -- "${LOG_FILE}")"
  : >"${LOG_FILE}"

  nohup setsid env \
    HTTP_PROXY= \
    HTTPS_PROXY= \
    ALL_PROXY= \
    http_proxy= \
    https_proxy= \
    all_proxy= \
    NO_PROXY="127.0.0.1,localhost,::1,${NO_PROXY:-}" \
    no_proxy="127.0.0.1,localhost,::1,${no_proxy:-}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    SGLANG_MP_START_METHOD="${SGLANG_MP_START_METHOD:-spawn}" \
    MODEL_PATH="${MODEL_PATH}" \
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
    HOST="${HOST}" \
    PORT="${PORT}" \
    TP_SIZE="${TP_SIZE}" \
    CONTEXT_LENGTH="${CONTEXT_LENGTH}" \
    ATTENTION_BACKEND="${ATTENTION_BACKEND}" \
    MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND}" \
    CUDA_GRAPH_BACKEND_DECODE="${CUDA_GRAPH_BACKEND_DECODE}" \
    CUDA_GRAPH_BACKEND_PREFILL="${CUDA_GRAPH_BACKEND_PREFILL}" \
    ENABLE_TWO_BATCH_OVERLAP="${ENABLE_TWO_BATCH_OVERLAP}" \
    ENABLE_MIXED_CHUNK="${ENABLE_MIXED_CHUNK}" \
    ENABLE_DYNAMIC_CHUNKING="${ENABLE_DYNAMIC_CHUNKING}" \
    CHAT_TEMPLATE="${CHAT_TEMPLATE}" \
    REASONING_PARSER="${REASONING_PARSER}" \
    TOOL_CALL_PARSER="${TOOL_CALL_PARSER}" \
    MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
    MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS}" \
    CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE}" \
    MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS}" \
    WORLD_SIZE="${WORLD_SIZE}" \
    RANK="${RANK}" \
    RESOURCE_GPU="${RESOURCE_GPU}" \
    ENABLE_DISTRIBUTED=0 \
    SGLANG_NUMA_BIND_V2=0 \
    SGLANG_AUTO_NUMA_BIND=0 \
    SGLANG_DEFAULT_THINKING=0 \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
    SGLANG_ENABLE_JIT_DEEPGEMM=0 \
    SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP=0 \
    SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 \
    SGLANG_OPT_USE_TILELANG_MHC_PRE=0 \
    SGLANG_OPT_USE_TILELANG_MHC_POST=0 \
    SGLANG_DSV4_MHC_PREWARM=0 \
    python -u "${LAUNCH_SCRIPT}" \
      >"${LOG_FILE}" 2>&1 </dev/null &

  echo $! >"${PID_FILE}"
  disown "$(cat "${PID_FILE}")" 2>/dev/null || true
  echo "服务启动中：PID $(cat "${PID_FILE}")"
  echo "日志：${LOG_FILE}"
  echo "接口：http://127.0.0.1:${PORT}/v1/models"
  if [[ "${WAIT_FOR_READY:-1}" == "1" ]]; then
    wait_ready "$(cat "${PID_FILE}")"
    if [[ "${WARMUP_AFTER_READY}" == "1" ]]; then
      warmup
    fi
  fi
}

main() {
  case "${MODE}" in
    install)
      install
      ;;
    serve)
      serve
      ;;
    all)
      install
      serve
      ;;
    restart)
      stop
      serve
      ;;
    stop)
      stop
      ;;
    status)
      status
      ;;
    health)
      health
      ;;
    wait)
      if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
        wait_ready "$(cat "${PID_FILE}")"
      else
        echo "服务未运行" >&2
        exit 1
      fi
      ;;
    warmup)
      warmup
      ;;
    logs)
      logs
      ;;
    *)
      echo "用法：$0 [install|serve|all|restart|stop|status|health|wait|warmup|logs]" >&2
      exit 1
      ;;
  esac
}

main "$@"
