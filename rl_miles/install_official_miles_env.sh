#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'USAGE'
Usage: bash rl_miles/install_official_miles_env.sh

Required before RL: install pinned Miles, Megatron, Megatron-Bridge, SGLang and dependencies.
RL 训练前必须安装固定版本框架及依赖；源码发布包不能解压后直接训练。
This installer calls bootstrap_miles.sh to download/check sources and apply runtime patches.
安装器自动调用 bootstrap_miles.sh 准备源码与补丁，无需先手工下载 Miles。
Default sources: rl_miles/miles, rl_miles/Megatron-LM,
rl_miles/Megatron-Bridge, rl_miles/sglang-miles.
Default environment: rl_miles/.venv (CUDA13 / Torch2.13 profile).
Overrides: MILES_REPO, MILES_MEGATRON_ROOT, MILES_SGLANG_ROOT, MILES_VENV, PYTHON_BASE.
自定义路径须在安装和启动时保持一致；已有环境仅在版本、补丁和依赖匹配时可复用。

bootstrap_miles.sh alone does not install the Python environment or GPU dependencies.
仅运行 bootstrap_miles.sh 不会安装 Python 环境或 GPU 依赖，不能据此直接训练。
Read ENVIRONMENT.md / ENVIRONMENT_zh.md for prerequisites and dependency completion.
还需准备权重、数据、GPU 集群和奖励服务；本脚本不会替你配置这些资源。
--help/-h only displays this message; no downloads or installation.
USAGE
  exit 0
fi

# Recreate the official miles CUDA 13 / torch 2.13 environment used by
# radixark/miles:dev for Qwen3.5 CP runs. This script intentionally installs
# the torch closure first, then force-overrides NCCL/cuDNN to the miles pins,
# matching miles/docker/Dockerfile.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VENV="${MILES_VENV:-${ROOT}/.venv}"
PYPI_INDEX="${PYPI_INDEX:-https://pypi.org/simple}"
PYTORCH_CU130_INDEX="${PYTORCH_CU130_INDEX:-https://download.pytorch.org/whl/cu130}"
MILES_WHEELS_TAG="${MILES_WHEELS_TAG:-cu130-torch213-x86_64}"
MILES_WHEELS_REPO="${MILES_WHEELS_REPO:-yueming-yuan/miles-wheels}"
DOWNLOAD_WHEELS="${DOWNLOAD_WHEELS:-1}"
INSTALL_OPTIONAL_GPU_WHEELS="${INSTALL_OPTIONAL_GPU_WHEELS:-1}"
INSTALL_SGLANG_RUNTIME_DEPS="${INSTALL_SGLANG_RUNTIME_DEPS:-1}"
INSTALL_SGLANG_EDITABLE="${INSTALL_SGLANG_EDITABLE:-1}"
MILES_ROOT="${MILES_REPO:-${ROOT}/miles}"
MEGATRON_ROOT="${MILES_MEGATRON_ROOT:-${ROOT}/Megatron-LM}"
SGLANG_ROOT="${MILES_SGLANG_ROOT:-${ROOT}/sglang-miles}"
printf '%s\n' 'Required RL setup: preparing pinned frameworks, patches and Python/GPU dependencies.' \
  'RL 必需安装：将准备固定版本框架、补丁及 Python/GPU 依赖；详见 ENVIRONMENT_zh.md。'

# The configured proxy on this node can make GitHub/PyTorch downloads time out.
# Set KEEP_PROXY=1 if your training node needs the proxy.
if [[ "${KEEP_PROXY:-0}" != "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
fi

PYTHON_BASE="${PYTHON_BASE:-python3.12}"
command -v "${PYTHON_BASE}" >/dev/null || { echo "Python 3.12 is required for the pinned wheels" >&2; exit 1; }
"${PYTHON_BASE}" -S -c 'import sys; assert sys.version_info[:2] == (3, 12), "CPython 3.12 required"'
bash "${ROOT}/bootstrap_miles.sh"
"${PYTHON_BASE}" -m venv "${VENV}"
PY="${VENV}/bin/python"
PIP=("${PY}" -m pip --disable-pip-version-check)
"${PY}" -m pip install --upgrade pip setuptools wheel packaging ninja cmake pybind11 \
  --index-url "${PYPI_INDEX}"

mkdir -p "${ROOT}/wheels/torch_cu130"

# Use a dedicated venv. Never delete user-supplied wheels or other environments.

install_official_cuda_runtime() {
  "${PIP[@]}" install --force-reinstall --no-deps \
    'nvidia-nccl-cu13==2.30.7' \
    'nvidia-cudnn-cu13==9.22.0.52' \
    'apache-tvm-ffi==0.1.9' \
    --index-url "${PYPI_INDEX}"
}

remove_conflicting_runtime_packages() {
  "${PIP[@]}" uninstall -y \
    flash-attn-4 \
    nvidia-cutlass-dsl-libs-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-nccl-cu12 \
    nvidia-nccl-cu129 \
    torch-cu129 \
    || true
}

install_sglang_runtime_deps() {
  # Mirror the CUDA-13 SGLang runtime pieces used by the official image, but do
  # not install flash-attn-4: miles' Megatron training path explicitly installs
  # flash_attn + flash_attn_3 above.
  "${PIP[@]}" install \
    'cuda-python>=13.0' \
    'cuda-tile==1.6.0rc5' \
    'flashinfer_python[cu13]==0.6.17' \
    'humming-kernels[cu13]==0.1.10' \
    'kernels>=0.14.1,<0.15' \
    'mistral_common>=1.11.5' \
    'numba==0.65.1' \
    'nvidia-cutlass-dsl[cu13]==4.6.2' \
    'nvidia-mathdx==25.6.0' \
    'openai-harmony==0.0.4' \
    'outlines==0.1.11' \
    'quack-kernels==0.6.4' \
    'sgl-deep-ep==0.1.0' \
    'sgl-deep-gemm==0.1.5.post3' \
    'sglang-kernel==0.4.6.post1' \
    'smg-grpc-servicer>=0.5.0' \
    'timm==1.0.16' \
    'tokenspeed_mla==0.1.8' \
    'blobfile==3.0.0' \
    'openai==2.6.1' \
    'soundfile==0.13.1' \
    --index-url "${PYPI_INDEX}"
  "${PIP[@]}" install 'tilelang==0.1.9' \
    -f https://tile-ai.github.io/whl/nightly/cu128/ \
    --index-url "${PYPI_INDEX}"
  "${PIP[@]}" install --no-deps 'tile_kernels==1.0.0' \
    --index-url "${PYPI_INDEX}"
  remove_conflicting_runtime_packages
  install_official_cuda_runtime
}

remove_conflicting_runtime_packages

if [[ "${DOWNLOAD_WHEELS}" == "1" ]]; then
  MILES_WHEELS_TAG="${MILES_WHEELS_TAG}" \
  MILES_WHEELS_REPO="${MILES_WHEELS_REPO}" \
  MILES_WHEELS_DEST="${ROOT}/wheels" \
    "${PYTHON_BASE}" "${ROOT}/wheels/download_official_cu130_wheels.py"
fi

# Stage torch's CUDA-13.0 runtime closure. Do not request miles' newer NCCL/cuDNN
# in the same resolver transaction: torch 2.13.0+cu130 metadata asks for
# NCCL 2.29.7 and cuDNN 9.20.0.48. Miles intentionally overrides them later.
"${PIP[@]}" install 'torch==2.13.0+cu130' \
  --index-url "${PYTORCH_CU130_INDEX}" \
  --extra-index-url "${PYPI_INDEX}"

# Official miles runtime overrides for Qwen3.5 Megatron/CP/all-to-all paths.
install_official_cuda_runtime

if [[ "${INSTALL_OPTIONAL_GPU_WHEELS}" == "1" ]]; then
  shopt -s nullglob
  "${PIP[@]}" install --force-reinstall --no-deps \
    "${ROOT}"/wheels/transformer_engine-2.17.0-py3-none-any.whl \
    "${ROOT}"/wheels/transformer_engine_cu13-2.17.0-py3-none-manylinux_2_28_*.whl \
    "${ROOT}"/wheels/transformer_engine_torch-2.17.0-cp312-cp312-linux_*.whl
  "${PIP[@]}" install einops onnx onnxscript pydantic nvdlfw-inspect --index-url "${PYPI_INDEX}"
  install_official_cuda_runtime
  "${PIP[@]}" install --force-reinstall --no-deps "${ROOT}"/wheels/flash_attn-*.whl
  "${PIP[@]}" install --force-reinstall --no-deps "${ROOT}"/wheels/flash_attn_3-*.whl
  "${PIP[@]}" install --force-reinstall --no-deps "${ROOT}"/wheels/apex-*.whl
  "${PIP[@]}" install --force-reinstall --no-deps "${ROOT}"/wheels/fast_hadamard_transform-*.whl || true
  "${PIP[@]}" install --force-reinstall --no-deps "${ROOT}"/wheels/causal_conv1d-*.whl "${ROOT}"/wheels/mamba_ssm-*.whl || true
  "${PIP[@]}" install --force-reinstall --no-deps "${ROOT}"/wheels/fake_int4_quant_cuda-*.whl || true
  "${PIP[@]}" install --force-reinstall --no-deps "${ROOT}"/wheels/mooncake_transfer_engine_cuda13-*.whl || true
  shopt -u nullglob
fi
install_official_cuda_runtime

"${PIP[@]}" install 'git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c' --no-deps

# Exact source revisions and runtime patches were prepared by bootstrap_miles.sh.
"${PIP[@]}" install -e "${MEGATRON_ROOT}" --no-deps --no-build-isolation
"${PIP[@]}" install -e "${ROOT}/Megatron-Bridge" --no-deps --no-build-isolation
  "${PIP[@]}" install megatron-energon multi-storage-client --no-deps --index-url "${PYPI_INDEX}"
  "${PIP[@]}" install \
    importlib-metadata \
    braceexpand 'filetype>=1.0.0' mfusepy 'rapidyaml>=0.10.0' webdataset \
    's3fs==2026.6.0' 'fsspec==2026.6.0' \
    'jmespath<2,>=1.0.1' 'prettytable<4,>=3.10' \
    'tzdata<2026,>=2024.2' 'wcmatch<11,>=10' 'xattr<2,>=1.1.4' \
    --index-url "${PYPI_INDEX}"
  "${PIP[@]}" install -r "${MILES_ROOT}/requirements.txt" --index-url "${PYPI_INDEX}"
install_official_cuda_runtime
"${PIP[@]}" install -e "${MILES_ROOT}" --no-deps

if [[ "${INSTALL_SGLANG_EDITABLE}" == "1" ]]; then

  SGLANG_BUILD_RUST_EXTS=none "${PIP[@]}" install -e "${SGLANG_ROOT}/python[all]" --no-deps
fi

if [[ "${INSTALL_SGLANG_RUNTIME_DEPS}" == "1" ]]; then
  install_sglang_runtime_deps
fi
remove_conflicting_runtime_packages
install_official_cuda_runtime

# Reproduce the source environment's optional ABI shim from source, never a private binary.
if [[ "${BUILD_TORCH_COMPAT:-1}" == "1" ]]; then
  "${CXX:-c++}" -std=c++17 -shared -fPIC -O2 "${ROOT}/wheels/torch_cow_compat.cpp" \
    -o "${ROOT}/wheels/libtorch_cow_compat.so" -ldl
fi
# Preserve the source environment's key Python package pins after framework requirements.
"${PIP[@]}" install 'transformers==5.12.1' 'flash-linear-attention==0.5.2' \
  'ray[default]==2.58.0' 'httpx==0.28.1' 'openai==2.6.1' --index-url "${PYPI_INDEX}"
install_official_cuda_runtime
export MILES_VENV="${VENV}"
source "${ROOT}/official_env.sh"
"${PY}" - <<'PY'
import importlib
import importlib.metadata as md
import torch
assert torch.__version__.startswith('2.13.'), 'Installer resolved an unexpected Torch ABI'
for module in ('transformer_engine.pytorch', 'fla', 'megatron.core.ssm.gated_delta_net',
               'miles.backends.megatron_utils.model_provider', 'sglang.srt.models.qwen3_5'):
    importlib.import_module(module)
print('Required RL kernel/backend imports passed; no GPU job was launched.')
print('torch', torch.__version__, 'cuda', torch.version.cuda)
for name in ['nvidia-nccl-cu13', 'nvidia-cudnn-cu13', 'transformer-engine', 'transformer-engine-cu13', 'transformer-engine-torch', 'flash-attn', 'flash-attn-3', 'mbridge', 'megatron-core', 'megatron-bridge', 'sglang']:
    try:
        print(name, md.version(name))
    except md.PackageNotFoundError:
        print(name, 'not installed')
PY
