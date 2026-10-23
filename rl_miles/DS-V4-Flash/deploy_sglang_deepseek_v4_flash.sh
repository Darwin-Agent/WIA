#!/usr/bin/env bash

unset http_proxy https_proxy all_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset no_proxy NO_PROXY

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-serve}"

MODEL_PATH="${MODEL_PATH:-/preset-models}"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"
USE_VENV="${USE_VENV:-1}"
SKIP_RUNTIME_CHECK="${SKIP_RUNTIME_CHECK:-0}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
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
ENABLE_DYNAMIC_CHUNKING="${ENABLE_DYNAMIC_CHUNKING:-1}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${SCRIPT_DIR}/chat_templates/deepseek_v4_flash.jinja}"
REASONING_PARSER="${REASONING_PARSER:-deepseek-v4}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-deepseekv4}"
if [[ -n "${RANK:-}" ]]; then
  LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/logging/sglang-deepseek-v4-flash-rank${RANK}.log}"
  PID_FILE="${PID_FILE:-${SCRIPT_DIR}/logging/sglang-deepseek-v4-flash-rank${RANK}.pid}"
else
  LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/logging/single-node/sglang-deepseek-v4-flash.log}"
  PID_FILE="${PID_FILE:-${SCRIPT_DIR}/logging/single-node/sglang-deepseek-v4-flash.pid}"
fi
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-${SCRIPT_DIR}/run_sglang_deepseek_v4_flash.py}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
HEALTH_INTERVAL="${HEALTH_INTERVAL:-10}"
WARMUP_AFTER_READY="${WARMUP_AFTER_READY:-1}"
WARMUP_TIMEOUT="${WARMUP_TIMEOUT:-600}"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"
KEEP_ENTRYPOINT_ALIVE="${KEEP_ENTRYPOINT_ALIVE:-1}"

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

validate_gpu_count() {
  require_commands nvidia-smi
  command -v nvidia-smi >/dev/null || {
    echo "未找到 nvidia-smi。此脚本仅支持 NVIDIA GPU 部署。" >&2
    exit 1
  }
  local gpu_count
  if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    gpu_count="$(tr ',' '\n' <<<"${CUDA_VISIBLE_DEVICES}" | sed '/^$/d' | wc -l | tr -d ' ')"
  else
    gpu_count="$(nvidia-smi -L | wc -l | tr -d ' ')"
  fi
  if (( gpu_count < TP_SIZE )); then
    echo "可见 GPU 数量为 ${gpu_count}，但 TP_SIZE=${TP_SIZE}。CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-未设置}" >&2
    exit 1
  fi
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
        echo "服务端口 ${start_port} 已占用，自动切换到 ${candidate_port}"
      fi
      return 0
    fi
    candidate_port=$((candidate_port + 1))
  done

  echo "服务端口不可用：已从 ${start_port} 连续探测 100 个端口仍未找到可用端口" >&2
  exit 1
}

pid_is_service() {
  local pid="$1"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  [[ "$(tr '\0' ' ' <"/proc/${pid}/cmdline")" == *"${LAUNCH_SCRIPT}"* ]]
}

runtime_ready() {
  if [[ "${SKIP_RUNTIME_CHECK}" == "1" ]]; then
    return 0
  fi
  if [[ "${USE_VENV}" != "1" ]]; then
    verify_runtime
    return
  fi
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
  clear_invalid_proxy_vars
  # Never move or remove packages in the base Python installation.
  require_commands git python3.12
  command -v git >/dev/null || { echo "未找到 git" >&2; exit 1; }
  command -v python3.12 >/dev/null || {
    echo "需要 Python 3.12（未找到 python3.12）" >&2
    exit 1
  }

  if [[ ! -d "${VENV_DIR}" ]]; then
    python3.12 -m venv --system-site-packages "${VENV_DIR}"
  fi
  activate_venv

  export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
  python -m pip install --upgrade pip setuptools wheel

  if [[ -d "${SGLANG_SRC}/.git" ]]; then
    if [[ "${SGLANG_UPDATE:-0}" == "1" || ! -f "${SGLANG_SRC}/python/sglang/srt/models/deepseek_v4.py" ]]; then
      git -C "${SGLANG_SRC}" fetch --tags origin
      if [[ -n "$(git -C "${SGLANG_SRC}" status --porcelain --untracked-files=no)" ]]; then
        echo "Refusing to overwrite modified SGLang sources; use a clean SGLANG_SRC directory" >&2; exit 1
      fi
      git -C "${SGLANG_SRC}" checkout --detach "${SGLANG_REF}"
    else
      echo "使用现有 SGLang 源码：${SGLANG_SRC}（SGLANG_UPDATE=1 可强制更新）"
    fi
  else
    mkdir -p "$(dirname -- "${SGLANG_SRC}")"
    git clone --recursive "${SGLANG_REPO}" "${SGLANG_SRC}"
    git -C "${SGLANG_SRC}" checkout --detach "${SGLANG_REF}"
  fi

  local expected_revision actual_revision
  expected_revision="$(git -C "${SGLANG_SRC}" rev-parse --verify "${SGLANG_REF}^{commit}")"
  actual_revision="$(git -C "${SGLANG_SRC}" rev-parse HEAD)"
  if [[ "${actual_revision}" != "${expected_revision}" ]]; then
    echo "SGLang revision mismatch; use a clean SGLANG_SRC at ${SGLANG_REF} or explicitly request SGLANG_UPDATE=1" >&2
    exit 1
  fi
  git -C "${SGLANG_SRC}" submodule update --init --recursive

  for patch_file in "${SCRIPT_DIR}"/patches/*.patch; do
    [[ -f "${patch_file}" ]] || continue
    if git -C "${SGLANG_SRC}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
      echo "补丁已应用：$(basename "${patch_file}")"
    elif git -C "${SGLANG_SRC}" apply --check "${patch_file}" >/dev/null 2>&1; then
      git -C "${SGLANG_SRC}" apply "${patch_file}"
      echo "已应用补丁：$(basename "${patch_file}")"
    else
      echo "补丁无法应用到当前 SGLang 源码：${patch_file}" >&2
      echo "当前可用部署环境通常无需重新 install；如遇新环境请检查 SGLang commit 与补丁。" >&2
      exit 1
    fi
  done

  if verify_runtime; then
    echo "运行时依赖已满足，跳过 SGLang 重装。"
  else
    # 从 SGLang 主线的 pyproject.toml 解析并安装完整、匹配的运行时依赖。
    python -m pip install --upgrade --pre "${SGLANG_SRC}/python"
    # These explicit overrides reproduce the original service's checked runtime.
    python -m pip install --no-deps 'sglang-kernel==0.4.6.post1' \
      'flashinfer_python==0.6.13' 'transformers==5.12.1' 'xgrammar==0.2.1'
  fi
  python -m pip install --no-deps -e "${SGLANG_SRC}/python"
  verify_runtime
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
    if health_check >/tmp/dsv4_flash_health.json 2>/tmp/dsv4_flash_health.err; then
      echo "服务已就绪：$(cat /tmp/dsv4_flash_health.json)"
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
      -d "${payload}" >/tmp/dsv4_flash_warmup.json 2>/tmp/dsv4_flash_warmup.err; then
    echo "预热完成：$(cat /tmp/dsv4_flash_warmup.json)"
    return 0
  fi
  echo "预热失败：$(cat /tmp/dsv4_flash_warmup.err)" >&2
  tail -n 120 "${LOG_FILE}" >&2 || true
  return 1
}

keep_entrypoint_alive() {
  local pid
  pid="$(cat "${PID_FILE}")"
  echo "服务保持运行中：PID ${pid}"
  trap 'stop; exit 0' INT TERM
  while pid_is_service "${pid}"; do
    sleep 30
  done
  echo "服务进程已退出：PID ${pid}"
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

serve() {
  clear_invalid_proxy_vars
  # Never move or remove packages in the base Python installation.
  require_commands bash curl date pgrep python3 setsid sleep tail tr
  validate_model
  validate_gpu_count
  if [[ "${USE_VENV}" == "1" && ! -x "${PYTHON_BIN}" ]]; then
    echo "找不到当前部署环境 Python：${PYTHON_BIN}" >&2
    exit 1
  fi
  if [[ "${USE_VENV}" == "1" && "${SKIP_RUNTIME_CHECK}" != "1" ]] && ! runtime_ready; then
    echo "运行时依赖不完整，开始自动安装 / 修复。"
    install
  elif [[ "${USE_VENV}" != "1" && "${SKIP_RUNTIME_CHECK}" != "1" ]]; then
    verify_runtime
  fi
  [[ -f "${LAUNCH_SCRIPT}" ]] || {
    echo "找不到启动脚本：${LAUNCH_SCRIPT}" >&2
    exit 1
  }
  if [[ -n "${CHAT_TEMPLATE}" && ! -f "${CHAT_TEMPLATE}" ]]; then
    echo "找不到 chat template：${CHAT_TEMPLATE}" >&2
    exit 1
  fi

  if [[ -f "${PID_FILE}" ]] && pid_is_service "$(cat "${PID_FILE}")"; then
    local running_pid
    running_pid="$(cat "${PID_FILE}")"
    if health_check >/tmp/dsv4_flash_health.json 2>/dev/null; then
      echo "服务已运行且健康（PID ${running_pid}）：$(cat /tmp/dsv4_flash_health.json)"
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

  resolve_port

  mkdir -p "$(dirname -- "${LOG_FILE}")"
  : >"${LOG_FILE}"

  nohup setsid env \
    HTTP_PROXY= \
    HTTPS_PROXY= \
    ALL_PROXY= \
    http_proxy= \
    https_proxy= \
    all_proxy= \
    PYTHONPATH= \
    VIRTUAL_ENV="${VENV_DIR}" \
    PATH="${VENV_DIR}/bin:${PATH}" \
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
    "${PYTHON_BIN}" -u "${LAUNCH_SCRIPT}" \
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

  if [[ "${KEEP_ENTRYPOINT_ALIVE}" == "1" ]]; then
    keep_entrypoint_alive
  fi
}

status() {
  if [[ -f "${PID_FILE}" ]] && pid_is_service "$(cat "${PID_FILE}")"; then
    echo "服务进程运行中：PID $(cat "${PID_FILE}")"
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
    echo "用法：$0 {serve|restart|install|all|stop|status|health|wait|warmup|logs}" >&2
    exit 2
    ;;
esac
