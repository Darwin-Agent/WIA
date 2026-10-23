#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'USAGE'
Usage: bash rl_miles/bootstrap_miles.sh [--check]

Prepare pinned Miles, Megatron, Megatron-Bridge and SGLang sources plus runtime patches.
准备固定版本框架源码并应用运行补丁；不安装 Python 环境或 GPU 依赖。
For first-time RL setup use: bash rl_miles/install_official_miles_env.sh
首次 RL 安装请运行上述安装器，它会自动调用本脚本；缺少框架和依赖不能训练。
Default source paths are under rl_miles/; overrides:
MILES_REPO, MILES_MEGATRON_ROOT, MILES_SGLANG_ROOT, PYTHON_BASE.
安装和训练时须使用一致的路径与版本。详见 ENVIRONMENT.md / ENVIRONMENT_zh.md。
--check only verifies existing revisions/patches; it does not download or install.
--check 仅检查已有源码和补丁；--help/-h 仅显示帮助，不下载、不安装。
USAGE
  exit 0
fi
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BASE="${PYTHON_BASE:-python3}"
PREPARE="${ROOT}/../runtime/prepare_sources.py"
"${PYTHON_BASE}" -S "${PREPARE}" --component miles --destination "${MILES_REPO:-${ROOT}/miles}" "$@"
"${PYTHON_BASE}" -S "${PREPARE}" --component megatron --destination "${MILES_MEGATRON_ROOT:-${ROOT}/Megatron-LM}" "$@"
"${PYTHON_BASE}" -S "${PREPARE}" --component megatron-bridge --destination "${ROOT}/Megatron-Bridge" "$@"
"${PYTHON_BASE}" -S "${PREPARE}" --component sglang-miles --destination "${MILES_SGLANG_ROOT:-${ROOT}/sglang-miles}" "$@"
