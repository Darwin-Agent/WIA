#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h || "${2:-}" == --help || "${2:-}" == -h ]]; then
  cat <<'USAGE'
Usage: bash cpt/setup_runtime.sh {cpt|sft}
       bash sft_code/setup_runtime.sh

Required before training: install pinned ms-swift, framework patches and dependencies.
训练前必须安装固定版本的 ms-swift、框架补丁及依赖；源码发布包不能解压后直接训练。
Run from the repository root after preparing the GPU base described in ENVIRONMENT.md.
请先按 ENVIRONMENT_zh.md 准备兼容的 GPU 基础环境，再从仓库根目录执行。

Downloads/checks pinned sources and applies runtime/patches before installing packages.
自动下载/检查固定版本源码并应用补丁，无需手工 git clone 或安装任意最新版。
Default source paths: cpt/ms-swift, cpt/mcore-bridge;
SFT additionally uses rl_miles/Megatron-LM (CP-capable).
Default environments: CPT cpt/ms-swift-venv; SFT sft_code/.venv.
源码路径覆盖：SWIFT_ROOT, MCORE_BRIDGE_ROOT; SFT MEGATRON_ROOT.
Environment overrides: CPT SWIFT_VENV; SFT SFT_VENV for installation,
then SWIFT_VENV pointing to the same location when launching SFT.
SFT 自定义环境：安装时设置 SFT_VENV，训练时设置 SWIFT_VENV 为同一路径。

Reuse an existing environment only if revisions, patches and dependencies match.
仅当版本、补丁和依赖均匹配时可复用已有环境；模型、数据和 GPU 配置仍需自行准备。
--help/-h only displays this message; no downloads or installation.
USAGE
  exit 0
fi
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${1:-cpt}"
if [[ "${PROFILE}" != cpt && "${PROFILE}" != sft ]]; then
  echo "Usage: bash cpt/setup_runtime.sh {cpt|sft}; see runtime/README.md for GPU base-image requirements" >&2; exit 1
fi
PYTHON_BASE="${PYTHON_BASE:-python3.12}"
SWIFT_ROOT="${SWIFT_ROOT:-${ROOT}/ms-swift}"
SWIFT_VENV="${SWIFT_VENV:-${ROOT}/ms-swift-venv}"
if [[ "${PROFILE}" == sft ]]; then SWIFT_VENV="${SFT_VENV:-${ROOT}/../sft_code/.venv}"; fi
MCORE_BRIDGE_ROOT="${MCORE_BRIDGE_ROOT:-${ROOT}/mcore-bridge}"
PREPARE="${ROOT}/../runtime/prepare_sources.py"
printf '%s\n' 'Required training setup: preparing pinned ms-swift, patches and dependencies.' \
  '训练必需安装：将准备固定版本 ms-swift、补丁和依赖；详见 ENVIRONMENT_zh.md。'
# Original CPT uses a CUDA12.8/Torch2.9.1/TE2.10 base image with a venv overlay.
# Verify the base before creating an overlay; do not silently replace its CUDA ABI.
"${PYTHON_BASE}" - <<'PY'
import sys, torch
import transformer_engine.pytorch, flash_attn
assert sys.version_info[:2] == (3, 12), 'The reference environment uses Python 3.12'
assert torch.__version__.startswith('2.9.'), 'Use the documented Torch 2.9 / CUDA12 base image'
assert (torch.version.cuda or '').startswith('12.'), 'Expected CUDA12 runtime'
print('Base ABI:', torch.__version__, torch.version.cuda)
PY
"${PYTHON_BASE}" -S "${PREPARE}" --component swift --destination "${SWIFT_ROOT}"
"${PYTHON_BASE}" -S "${PREPARE}" --component mcore-bridge --destination "${MCORE_BRIDGE_ROOT}"
"${PYTHON_BASE}" -m venv --system-site-packages "${SWIFT_VENV}"
PY="${SWIFT_VENV}/bin/python"
"${PY}" -m pip install --upgrade pip 'setuptools<80' wheel ninja packaging pybind11
"${PY}" -m pip install -e "${SWIFT_ROOT}" -c "${ROOT}/runtime-constraints.txt"
"${PY}" -m pip install -e "${MCORE_BRIDGE_ROOT}" --no-deps --no-build-isolation
"${PY}" -m pip install 'flash-linear-attention==0.5.2' 'deepspeed==0.19.2' \
  'nvidia-resiliency-ext==0.6.0' -c "${ROOT}/runtime-constraints.txt"
if [[ -n "${CAUSAL_CONV1D_WHEEL:-}" ]]; then
  "${PY}" -m pip install --no-deps "${CAUSAL_CONV1D_WHEEL}"
else
  "${PY}" -m pip install --no-build-isolation --no-deps 'causal-conv1d==1.6.2.post1'
fi
if [[ "${PROFILE}" == sft ]]; then
  MEGATRON_ROOT="${MEGATRON_ROOT:-${ROOT}/../rl_miles/Megatron-LM}"
  "${PYTHON_BASE}" -S "${PREPARE}" --component megatron --destination "${MEGATRON_ROOT}"
  "${PY}" -m pip install -e "${MEGATRON_ROOT}" --no-deps --no-build-isolation
else
  "${PY}" -m pip install --no-deps 'megatron-core==0.17.1'
fi
PYTHONPATH="${SWIFT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" "${ROOT}/../runtime/validate_swift.py" --train-user-loss
if [[ "${PROFILE}" == sft ]]; then
  PYTHONPATH="${MEGATRON_ROOT}:${SWIFT_ROOT}" "${PY}" - <<'PY'
import sys
from pathlib import Path
# Import-only minimum CP checks; this is not a distributed/GPU smoke run.
sys.path.insert(0, str(Path.cwd()))
import mcore_bridge.model.mm_gpts.qwen3_5_gdn
from megatron.core.ssm import gated_delta_net as gdn
for name in ('tensor_a2a_cp2hp', 'tensor_a2a_hp2cp', 'get_parameter_local_cp'):
    assert callable(getattr(gdn, name, None)), f'Missing GDN CP helper: {name}'
print('GDN CP helpers present. Run the model-specific SFT preflight before training.')
PY
fi
printf 'Prepared %s profile at %s (GPU execution has not been tested by this installer).\n' "${PROFILE}" "${SWIFT_VENV}"
