#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

export VLLM_BIN="${VLLM_BIN:-${SCRIPT_DIR}/.venv/bin/vllm}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://<JUDGE_HOST>:<JUDGE_PORT>/v1}"
export JUDGE_MODEL="${JUDGE_MODEL:-DeepSeek-V4-Flash-0731}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"
export JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-64}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export NO_PROXY="${NO_PROXY:-*},127.0.0.1,localhost"
export no_proxy="$NO_PROXY"

exec "$PYTHON_BIN" "$SCRIPT_DIR/run_eval.py" "$@"
