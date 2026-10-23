# Installation

**English** | [简体中文](ENVIRONMENT_zh.md) · [Home](README.md) · [Runtime](runtime/README.md) · [Configuration](docs/CONFIGURATION.md)

Create separate environments for CPT/SFT, RL, the DeepSeek judge, embeddings and evaluation.
Their CUDA and kernel requirements differ; **do not combine them into one venv**.

## Installation and reuse

**Framework installation is mandatory for real training.** The release includes project code, pinned revision metadata
and patches, but not complete Swift/Miles checkouts or Python/GPU environments. Default paths are recommendations,
not a requirement to redownload a matching existing installation.

| Stage | First-time installer, from repository root | What it prepares |
| --- | --- | --- |
| CPT | `bash cpt/setup_runtime.sh cpt` | Pinned Swift + mcore-bridge sources/patches, CPT venv and training dependencies |
| SFT | `bash sft_code/setup_runtime.sh` | Shared Swift/bridge sources, separate SFT venv and CP-capable Megatron |
| RL | `bash rl_miles/install_official_miles_env.sh` | Calls bootstrap for pinned Miles/Megatron/Bridge/SGLang sources and patches, then installs the RL environment |

Read the prerequisites and stage-specific sections below **before** executing an installer. Internet access is needed
for default downloads. `--help`/`-h` is safe to run without a GPU base or network access.
`bootstrap_miles.sh` is source preparation only, not a replacement for the RL environment installer.
Do not substitute arbitrary `pip install ms-swift`/Miles versions: the pinned revisions and patches are part of the runtime contract.

### Custom paths and existing environments

| Resource | Installation variable | Launch configuration |
| --- | --- | --- |
| Swift source | `SWIFT_ROOT` | Same `SWIFT_ROOT` for CPT/SFT |
| mcore-bridge source | `MCORE_BRIDGE_ROOT` | Keep the editable source available to the installed environment |
| CPT venv | `SWIFT_VENV` | Same `SWIFT_VENV` |
| SFT venv | `SFT_VENV` | Set `SWIFT_VENV` to this same directory when launching SFT |
| SFT Megatron source | `MEGATRON_ROOT` | Same `MEGATRON_ROOT` |
| Miles source / venv | `MILES_REPO` / `MILES_VENV` | Same variables when sourcing `official_env.sh` and launching RL |
| RL Megatron / SGLang sources | `MILES_MEGATRON_ROOT` / `MILES_SGLANG_ROOT` | Same variables for installation, source checks and launch |

Default source locations are listed in [Runtime](runtime/README.md); environment locations are in the matrix below.
A matching existing installation need not be downloaded again. Verify pinned revisions/patches with
`runtime/prepare_sources.py --check` (use `--component` and `--destination` for a custom checkout), then verify imports
and run the stage's preflight. For RL, `bash rl_miles/bootstrap_miles.sh --check` checks all four source trees, **not** all dependencies.
Keep editable source trees available and use consistent paths on every node. Installation does not provide model weights,
datasets, working cluster credentials or running reward services. See the stage guides for those requirements.

## 1. Prerequisites

- Linux x86_64, Python 3.12, Git and Bash; curl or aria2c for wheel downloads.
- NVIDIA GPUs and a compatible driver; CUDA toolkit, C/C++ compiler, ninja and cmake when building extensions.
- Disk capacity for sources, environments, weights, cached data and checkpoints.
- Consistent paths/versions across nodes, reachable NCCL/Ray ports and appropriately configured shared storage.

Run commands from the repository root in a dedicated, recoverable GPU environment.
Installers download software and build extensions. No universal minimum driver or GPU-memory guarantee is provided;
validate your hardware against the selected model, sequence length, precision and topology.

## 2. Environment matrix

| Purpose | Default environment | Python / Torch / CUDA | Core dependencies |
| --- | --- | --- | --- |
| CPT | `cpt/ms-swift-venv` | 3.12 / 2.9.1 / 12.8 | TE 2.10.0, FA 2.7.4.post1, Megatron Core 0.17.1 |
| SFT | `sft_code/.venv` | Same as CPT | Pinned Swift/mcore-bridge and GDN CP-capable Megatron source |
| RL | `rl_miles/.venv` | 3.12 / 2.13.0+cu130 / 13.0 | TE 2.17.0, NCCL 2.30.7, cuDNN 9.22.0.52, Ray 2.58.0 |
| DeepSeek judge | `rl_miles/DS-V4-Flash/.venv` | 3.12; compatible GPU base required | Separate SGLang revision and three deployment patches |
| Embedding / similarity | `rl_miles/Qwen3-embedding/.venv` | 3.12; GPU dependencies from vLLM | vLLM 0.28.0 by default, FastAPI |
| Evaluation | `eval/.venv` | 3.12; vLLM for local generation | API-only installation can omit the local engine |

These are installer profiles, **not complete dependency lockfiles**. Some transitive dependencies remain unpinned.
Inspect actual import locations and versions after installation. Keep an environment inventory privately;
generated inventories may contain local filesystem paths.

## 3. CPT

CPT uses a `--system-site-packages` overlay and expects a CUDA12/Torch2.9 base ABI.
If you do not already have a matching GPU image, prepare these versions in a dedicated environment:

```bash
python3.12 -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python3.12 -m pip install packaging ninja cmake pybind11
python3.12 -m pip install 'transformer-engine[pytorch]==2.10.0' --no-build-isolation
python3.12 -m pip install flash-attn==2.7.4.post1 --no-build-isolation
bash cpt/setup_runtime.sh cpt
```

The installer checks Torch/TE/FlashAttention before creating the overlay; it does not repair arbitrary base images.
`causal-conv1d==1.6.2.post1` builds from source unless `CAUSAL_CONV1D_WHEEL=/path/to/compatible.whl` is supplied.
The wheel must match Python3.12, Torch2.9, CUDA12 and the CXX11 ABI.

Other key packages: Transformers 5.12.1, DeepSpeed 0.19.2, FLA 0.5.2 and nvidia-resiliency-ext 0.6.0.
[Constraints](cpt/runtime-constraints.txt) also pin datasets 4.5.0; CPT TileLang/TVM-FFI are not independently pinned.

```bash
python3 -S runtime/prepare_sources.py --component swift --check
cpt/ms-swift-venv/bin/python runtime/validate_swift.py --train-user-loss
```

Do not omit the Swift patches: a standard assistant-only installation does not provide the required user-loss behavior.

## 4. SFT

After preparing the same CUDA12 base:

```bash
bash sft_code/setup_runtime.sh
```

This creates `sft_code/.venv`, shares pinned Swift/mcore-bridge sources and installs CP-capable Megatron.
Do not use the CPT CP1 environment as a substitute. Installation and `validate_sft_runtime.py` check GDN CP helpers;
you must still test model head/layer divisibility, distributed execution and memory.
SFT uses the existing masked Swift objective; there is no separate SFT-only loss patch.

## 5. RL

### Install the Miles runtime

```bash
bash rl_miles/bootstrap_miles.sh
bash rl_miles/install_official_miles_env.sh
```

Pinned sources cover Miles, Megatron, Megatron-Bridge and SGLang-Miles with runtime patches.
The recipe uses Transformers 5.12.1, FLA 0.5.2, FlashInfer 0.6.17, SGLang kernel 0.4.6.post1,
TileLang 0.1.9, TVM-FFI 0.1.9 and tile_kernels 1.0.0. Keep the judge's SGLang in a separate environment.

### Complete conversion and memory-management dependencies

Some framework installations use `--no-deps`, so optional modules are not guaranteed to be installed automatically.
Ensure `nvidia-modelopt==0.46.0` and the pinned `torch_memory_saver` source build are available.
If the installer stops at its final import check because either is missing, complete the existing RL venv:

```bash
export PYTHON_BIN="${MILES_VENV:-$PWD/rl_miles/.venv}/bin/python"
"$PYTHON_BIN" -m pip install 'nvidia-modelopt[torch]==0.46.0' --no-build-isolation
TMS_CUDA_MAJOR=13 "$PYTHON_BIN" -m pip install --no-deps --no-build-isolation \
  'git+https://github.com/fzyzcjy/torch_memory_saver.git@f05a8754daf68238d54e4cf31cb3ba866684bbaf'
```

Check that dependency resolution has not replaced the GPU stack. The Miles recipe deliberately overrides Torch's
default NCCL/cuDNN dependency versions after installing Torch; do not blindly downgrade them.
Interpret `pip check` alongside these explicit overrides, without ignoring unrelated conflicts.

```bash
source rl_miles/official_env.sh
bash rl_miles/bootstrap_miles.sh --check
"$PYTHON_BIN" -c 'import torch, modelopt, torch_memory_saver; import transformer_engine.pytorch; import megatron.core.ssm.gated_delta_net; import miles.backends.megatron_utils.model_provider; import sglang.srt.models.qwen3_5; print(torch.__version__, torch.version.cuda)'
```

`official_env.sh` selects Python/Ray, source paths, the venv's NCCL and an available compatibility library via
`LD_PRELOAD`. Every training node needs the same environment.

| Installation variable | Meaning |
| --- | --- |
| `PYTHON_BASE` | `python3.12` by default |
| `MILES_VENV` | Must match the runtime environment location |
| `MILES_WHEELS_REPO` / `MILES_WHEELS_TAG` | Public wheel source; default tag `cu130-torch213-x86_64` |
| `DOWNLOAD_WHEELS=0` | Use prepared wheels; does not make all source/dependency installation offline |
| `INSTALL_OPTIONAL_GPU_WHEELS` | Enabled by default; prepare required TE/FA kernels yourself before disabling |
| `INSTALL_SGLANG_RUNTIME_DEPS` / `INSTALL_SGLANG_EDITABLE` | Normally leave enabled |
| `BUILD_TORCH_COMPAT` | Builds `wheels/torch_cow_compat.cpp` by default; disable only on a validated compatible stack |
| `KEEP_PROXY=1` | Preserve download proxy settings; unrelated to private model-service proxies |

No complete offline package bundle is shipped. Prepare compatible wheels, model files and extension ABIs yourself.

## 6. Services and independent evaluation

```bash
bash rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh install
bash rl_miles/Qwen3-embedding/install_runtime.sh
bash eval/install_runtime.sh
```

Install only the components you need. Installation does not configure weights or training endpoint URLs.

- [DeepSeek judge](rl_miles/DS-V4-Flash/README.md): dedicated GPU stack; venv inherits system packages.
- [Embedding/similarity](rl_miles/Qwen3-embedding/README.md): four TP1 replicas and an HTTP similarity API by default.
- [Evaluation](eval/README.md): `API_ONLY=1 bash eval/install_runtime.sh` omits local vLLM but still needs a tokenizer.

## 7. Troubleshooting

| Symptom | Check |
| --- | --- |
| Undefined symbol / CUDA extension import failure | Python, Torch, CUDA, CXX11 ABI and accidental cross-venv imports |
| Missing GDN CP helper | Use the SFT CP-capable Megatron source, not CPT's 0.17.1 CP1 stack |
| User loss not applied | Swift patches, switch, masks and regenerated caches |
| Missing modelopt / torch_memory_saver | Complete section 5; downloading a wheel is not installing it |
| Ray job waits for resources | Allocate actor + rollout resources: 16 or 20 total nodes, not just actor nodes |
| Scoring requests fail | Reachability, URL path, served model name and request fields from a training node |

Syntax, imports, source/patch checks and dry-runs do not validate GPU execution. Test loading, forward/backward,
NCCL communication, weight synchronization and real scorer requests before a full run.
