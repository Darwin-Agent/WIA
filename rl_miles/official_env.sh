#!/usr/bin/env bash
# Source this before running miles launchers:
#   source rl_miles/official_env.sh

_THIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export MILES_CODE_ROOT="${MILES_CODE_ROOT:-${_THIS_DIR}}"
export MILES_ROOT="${MILES_REPO:-${MILES_CODE_ROOT}/miles}"
export MILES_VENV="${MILES_VENV:-${MILES_CODE_ROOT}/.venv}"
if [[ -z "${MILES_MEGATRON_ROOT:-}" ]]; then
  export MILES_MEGATRON_ROOT="${MILES_CODE_ROOT}/Megatron-LM"
fi
if [[ -z "${MILES_SGLANG_ROOT:-}" ]]; then
  export MILES_SGLANG_ROOT="${MILES_CODE_ROOT}/sglang-miles"
fi
if [[ ! -d "${MILES_ROOT}" ]]; then
  echo "Local Miles checkout not found: ${MILES_ROOT}" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -d "${MILES_MEGATRON_ROOT}" ]]; then
  echo "Local Megatron checkout not found: ${MILES_MEGATRON_ROOT}" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -d "${MILES_SGLANG_ROOT}" ]]; then
  echo "Local SGLang checkout not found: ${MILES_SGLANG_ROOT}" >&2
  return 1 2>/dev/null || exit 1
fi
export MILES_REPO="${MILES_ROOT}"
export MEGATRON_REPO="${MILES_MEGATRON_ROOT}"
export SGLANG_REPO="${MILES_SGLANG_ROOT}"
export MILES_COW_COMPAT="${MILES_COW_COMPAT:-${MILES_CODE_ROOT}/wheels/libtorch_cow_compat.so}"

if [[ -x "${MILES_VENV}/bin/python" ]]; then
  export VIRTUAL_ENV="${MILES_VENV}"
  export PATH="${MILES_VENV}/bin:${PATH}"
  export PYTHON_BIN="${MILES_VENV}/bin/python"
  export RAY_BIN="${MILES_VENV}/bin/ray"
else
  echo "Local Miles Python environment not found: ${MILES_VENV}/bin/python" >&2
  return 1 2>/dev/null || exit 1
fi

# Resolve the NCCL shared object from the active venv instead of hard-coding
# /usr/local dist-packages. The official miles cu13 closure pins
# nvidia-nccl-cu13==2.30.7 and preloads this library for SGLang/Megatron.
if [[ -z "${MILES_NCCL_SO:-}" && -x "${MILES_VENV}/bin/python" ]]; then
  MILES_NCCL_SO="$(${MILES_VENV}/bin/python - <<'PY' 2>/dev/null || true
from pathlib import Path
import importlib.util
try:
    import nvidia.nccl
except Exception:
    spec = importlib.util.find_spec('nvidia.nccl')
    roots = list(spec.submodule_search_locations or []) if spec else []
else:
    roots = [Path(nvidia.nccl.__file__).resolve().parent] if getattr(nvidia.nccl, '__file__', None) else []
if not roots:
    spec = importlib.util.find_spec('nvidia.nccl')
    roots = list(spec.submodule_search_locations or []) if spec else []
for root in roots:
  for path in Path(root).glob('lib/libnccl.so*'):
    if path.name == 'libnccl.so.2':
        print(path)
        raise SystemExit(0)
PY
)"
  export MILES_NCCL_SO
fi

export SGLANG_NCCL_SO_PATH="${SGLANG_NCCL_SO_PATH:-${MILES_NCCL_SO:-}}"
if [[ -f "${MILES_COW_COMPAT}" ]]; then
  export MILES_TORCH_PRELOAD="${MILES_TORCH_PRELOAD:-${MILES_COW_COMPAT}}"
else
  export MILES_TORCH_PRELOAD="${MILES_TORCH_PRELOAD:-}"
fi

_preload_parts=()
[[ -f "${MILES_COW_COMPAT}" ]] && _preload_parts+=("${MILES_COW_COMPAT}")
[[ -n "${MILES_NCCL_SO:-}" && -f "${MILES_NCCL_SO}" ]] && _preload_parts+=("${MILES_NCCL_SO}")
[[ -n "${LD_PRELOAD:-}" ]] && _preload_parts+=("${LD_PRELOAD}")
if ((${#_preload_parts[@]})); then
  _old_ifs="${IFS}"
  IFS=:
  export LD_PRELOAD="${_preload_parts[*]}"
  IFS="${_old_ifs}"
  unset _old_ifs
fi
unset _preload_parts

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH="${MILES_MEGATRON_ROOT}:${MILES_SGLANG_ROOT}/python:${MILES_SGLANG_ROOT}/sgl-kernel/python:${MILES_ROOT}:${_THIS_DIR}${MILES_EXTRA_PYTHONPATH:+:${MILES_EXTRA_PYTHONPATH}}"
