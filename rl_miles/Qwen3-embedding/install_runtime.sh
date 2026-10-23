#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT}/.venv}"
PYTHON_BASE="${PYTHON_BASE:-python3.12}"
VLLM_VERSION="${VLLM_VERSION:-0.28.0}"
"${PYTHON_BASE}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install "vllm==${VLLM_VERSION}" orjson fastapi 'httpx==0.28.1' uvicorn
"${VENV_DIR}/bin/python" - <<'PY'
import importlib.metadata as md
import vllm, httpx, fastapi, uvicorn, orjson
print('Similarity runtime installed; vLLM', md.version('vllm'))
PY
