#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT}/.venv}"
PYTHON_BASE="${PYTHON_BASE:-python3.12}"
"${PYTHON_BASE}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
# API_ONLY=1 skips the local GPU inference engine, not the tokenizer dependency.
if [[ "${API_ONLY:-0}" != "1" ]]; then
  "${VENV_DIR}/bin/python" -m pip install "vllm==${VLLM_VERSION:-0.28.0}"
fi
"${VENV_DIR}/bin/python" -m pip install -r "${ROOT}/requirements.txt"
"${VENV_DIR}/bin/python" "${ROOT}/run_eval.py" --help >/dev/null
printf 'Eval runtime installed at %s; no inference or remote requests were run.\n' "${VENV_DIR}"
