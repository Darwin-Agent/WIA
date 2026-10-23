# 环境安装

[English](ENVIRONMENT.md) | **简体中文**

[项目首页](README_zh.md) · [框架与补丁](runtime/README_zh.md) · [完整配置参考](docs/CONFIGURATION_zh.md)

按训练阶段创建环境。CPT/SFT、RL、DeepSeek judge、embedding 和 Eval 的 CUDA
与内核依赖不同，**不要安装到同一个虚拟环境**。

## 安装与环境复用

**真正训练前必须安装对应框架。** 发布包包含项目代码、固定版本信息与补丁，不包含完整 Swift/Miles
源码副本或 Python/GPU 运行环境。默认安装位置是推荐布局，不意味着已有匹配环境也必须重新下载。

| 阶段 | 首次安装命令，从仓库根目录执行 | 安装内容 |
| --- | --- | --- |
| CPT | `bash cpt/setup_runtime.sh cpt` | 固定版本 Swift、mcore-bridge 源码与补丁，CPT 虚拟环境及训练依赖 |
| SFT | `bash sft_code/setup_runtime.sh` | 共享 Swift/bridge 源码、独立 SFT 环境及支持 CP 的 Megatron |
| RL | `bash rl_miles/install_official_miles_env.sh` | 自动调用 bootstrap 准备 Miles/Megatron/Bridge/SGLang 源码与补丁，再安装 RL 环境 |

**执行前先阅读下方基础环境要求及对应阶段的完整安装步骤。** 默认下载需要联网；安装器的 `--help`/`-h`
无需 GPU 基础环境或网络即可查看。`bootstrap_miles.sh` 只准备源码，不能替代 RL 环境安装器。
不能用任意版本的 `pip install ms-swift` 或 Miles 替代这些步骤：固定提交与补丁也是运行要求的一部分。

### 自定义目录与已有环境

| 资源 | 安装时变量 | 启动时配置 |
| --- | --- | --- |
| Swift 源码 | `SWIFT_ROOT` | CPT/SFT 使用相同的 `SWIFT_ROOT` |
| mcore-bridge 源码 | `MCORE_BRIDGE_ROOT` | editable 安装后保留源码目录，供环境导入 |
| CPT 虚拟环境 | `SWIFT_VENV` | 使用相同的 `SWIFT_VENV` |
| SFT 虚拟环境 | `SFT_VENV` | SFT 启动时将 `SWIFT_VENV` 设置为同一路径 |
| SFT Megatron 源码 | `MEGATRON_ROOT` | 使用相同的 `MEGATRON_ROOT` |
| Miles 源码 / 环境 | `MILES_REPO` / `MILES_VENV` | 加载 `official_env.sh` 和启动 RL 时保持这些变量一致 |
| RL Megatron / SGLang 源码 | `MILES_MEGATRON_ROOT` / `MILES_SGLANG_ROOT` | 安装、源码检查和启动均使用相同变量 |

默认源码目录见[运行时指南](runtime/README_zh.md)，默认环境目录见下方表格。
已有匹配环境无需重复下载：先用 `runtime/prepare_sources.py --check` 检查固定版本与补丁
（自定义源码目录需指定 `--component` 和 `--destination`），再验证导入并执行对应阶段预检查。
RL 的 `bash rl_miles/bootstrap_miles.sh --check` 检查四套源码，**不代表全部依赖已安装**。
保留 editable 安装所引用的源码目录，多节点保持路径一致。安装器不提供权重、数据、实际集群认证信息或已启动的奖励服务；
这些资源仍须按各阶段指南准备。

## 1. 安装前准备

- Linux x86_64、Python 3.12、Git、Bash；下载 wheel 时需要 curl 或 aria2c。
- NVIDIA GPU、兼容所选 CUDA 的驱动；编译扩展时还需要 CUDA toolkit、C/C++ 编译器、ninja、cmake。
- 足够的磁盘空间保存源码、环境、权重、数据缓存及 checkpoint。
- 多节点使用相同路径和版本，配置节点间网络、Ray/NCCL 端口及共享存储。

以下命令从仓库根目录执行。安装脚本会下载软件和构建扩展，请在专用容器或可恢复的环境中运行。
仓库不提供通用最低显存/驱动承诺；请按模型大小、序列长度、精度与拓扑验证实际资源。

## 2. 环境一览

| 用途 | 默认环境目录 | Python / Torch / CUDA | 主要依赖 |
| --- | --- | --- | --- |
| CPT | `cpt/ms-swift-venv` | 3.12 / 2.9.1 / 12.8 | TE 2.10.0、FA 2.7.4.post1、Megatron Core 0.17.1 |
| SFT | `sft_code/.venv` | 同 CPT | 固定版本 Swift/mcore-bridge 与支持 GDN CP 的 Megatron 源码 |
| RL | `rl_miles/.venv` | 3.12 / 2.13.0+cu130 / 13.0 | TE 2.17.0、NCCL 2.30.7、cuDNN 9.22.0.52、Ray 2.58.0 |
| DeepSeek judge | `rl_miles/DS-V4-Flash/.venv` | 3.12；需兼容 GPU 基础栈 | 独立 SGLang 提交及三份部署补丁 |
| Embedding / similarity | `rl_miles/Qwen3-embedding/.venv` | 3.12；GPU 依赖随 vLLM 配方 | vLLM 默认 0.28.0、FastAPI |
| Eval | `eval/.venv` | 3.12；本地推理使用 vLLM | vLLM 默认 0.28.0；API-only 可不装本地引擎 |

这些是安装入口的配置，不是完整依赖锁文件。某些传递依赖没有固定版本；安装完成后
应检查实际导入路径和版本，保存自己的环境清单，但不要把包含本机路径的清单直接公开。

## 3. CPT

### 准备基础镜像

CPT venv 使用 `--system-site-packages`，依赖基础镜像已有的 CUDA12/Torch2.9 ABI。
如果基础镜像尚未准备，可在专用环境中按下面的版本配置：

```bash
python3.12 -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python3.12 -m pip install packaging ninja cmake pybind11
python3.12 -m pip install 'transformer-engine[pytorch]==2.10.0' --no-build-isolation
python3.12 -m pip install flash-attn==2.7.4.post1 --no-build-isolation
bash cpt/setup_runtime.sh cpt
```

安装器启动时就检查 Torch、TE 和 FlashAttention；它不负责自动修复任意系统镜像。
`causal-conv1d==1.6.2.post1` 默认从源码构建，也可使用
`CAUSAL_CONV1D_WHEEL=/path/to/compatible.whl`，要求匹配 Python 3.12、Torch2.9、CUDA12 与 CXX11 ABI。

关键 Python 依赖：Transformers 5.12.1、DeepSpeed 0.19.2、FLA 0.5.2、
nvidia-resiliency-ext 0.6.0；约束文件见 [runtime-constraints.txt](cpt/runtime-constraints.txt)。
`datasets==4.5.0` 由该文件约束。CPT 的 TileLang/TVM-FFI 没有分别锁定，需验证实际组合。

```bash
python3 -S runtime/prepare_sources.py --component swift --check
cpt/ms-swift-venv/bin/python runtime/validate_swift.py --train-user-loss
```

不要省略 Swift 补丁；普通 assistant-only Swift 安装不能替代 user-loss 运行时。

## 4. SFT

准备同样的 CUDA12 基础镜像后：

```bash
bash sft_code/setup_runtime.sh
```

安装器创建独立 `sft_code/.venv`，复用 Swift/mcore-bridge 源码，并安装支持 GatedDeltaNet
上下文并行的固定 Megatron 源码。不要直接使用 CPT 的 CP1 环境运行 CP4。

安装检查与 `sft_code/validate_sft_runtime.py` 会检查 GDN CP helper；这不是分布式训练验证。
启动前仍需验证实际模型的 TP/CP 头数整除、PP 层数切分和 GPU 内存。
SFT 不引入独立的 loss 目标补丁，继续使用按消息掩码控制的 Swift 行为。

## 5. RL

### 安装 Miles 运行时

```bash
bash rl_miles/bootstrap_miles.sh
bash rl_miles/install_official_miles_env.sh
```

安装器固定 Miles、Megatron、Megatron-Bridge、SGLang-Miles 的源码提交并应用补丁。
关键版本包括 Transformers 5.12.1、FLA 0.5.2、FlashInfer 0.6.17、
SGLang kernel 0.4.6.post1、TileLang 0.1.9、TVM-FFI 0.1.9、tile_kernels 1.0.0。
不要把 judge 的 SGLang 安装到这个环境。

### 补齐模型转换与显存管理依赖

当前安装器使用 `--no-deps` 安装部分框架，不能保证自动补齐全部可选模块。
至少确认 `nvidia-modelopt==0.46.0` 与固定源码的 `torch_memory_saver` 已安装。
如果安装器在最终 import 检查处因这两个模块缺失而停止，在已建立的 RL venv 内补齐：

```bash
export PYTHON_BIN="${MILES_VENV:-$PWD/rl_miles/.venv}/bin/python"
"$PYTHON_BIN" -m pip install 'nvidia-modelopt[torch]==0.46.0' --no-build-isolation
TMS_CUDA_MAJOR=13 "$PYTHON_BIN" -m pip install --no-deps --no-build-isolation \
  'git+https://github.com/fzyzcjy/torch_memory_saver.git@f05a8754daf68238d54e4cf31cb3ba866684bbaf'
```

补装后检查核心 GPU 栈没有被依赖解析器替换。Miles 配方先安装 Torch，随后显式覆盖
NCCL/cuDNN；不要把这两个版本随意降回 Torch 元数据中的默认值。
`pip check` 的结果需要结合这些显式覆盖判断，其他依赖冲突仍需处理。

```bash
source rl_miles/official_env.sh
bash rl_miles/bootstrap_miles.sh --check
"$PYTHON_BIN" -c 'import torch, modelopt, torch_memory_saver; import transformer_engine.pytorch; import megatron.core.ssm.gated_delta_net; import miles.backends.megatron_utils.model_provider; import sglang.srt.models.qwen3_5; print(torch.__version__, torch.version.cuda)'
```

`official_env.sh` 选择正确的 Python/Ray、源码路径和 NCCL 动态库，并在存在兼容库时
配置 `LD_PRELOAD`。训练节点都必须具备同样的环境。

### Wheel 与安装参数

| 变量 | 用途 |
| --- | --- |
| `PYTHON_BASE` | 默认 `python3.12` |
| `MILES_VENV` | RL venv 位置，安装与运行需一致 |
| `MILES_WHEELS_REPO` / `MILES_WHEELS_TAG` | 公共 wheel 发布源；默认 tag 为 `cu130-torch213-x86_64` |
| `DOWNLOAD_WHEELS=0` | 使用已准备的 wheel；不会自动把源码/其余依赖切成离线模式 |
| `INSTALL_OPTIONAL_GPU_WHEELS` | 默认启用；关闭前必须自行准备 TE/FA 等所需扩展 |
| `INSTALL_SGLANG_RUNTIME_DEPS` / `INSTALL_SGLANG_EDITABLE` | 控制 SGLang 依赖与源码安装；一般保留默认 |
| `BUILD_TORCH_COMPAT` | 默认编译 `wheels/torch_cow_compat.cpp`；仅在已验证无需兼容层时关闭 |
| `KEEP_PROXY=1` | 安装时保留下载代理；不是私有模型服务代理 |

安装器没有提供完整的离线依赖包。wheel、模型权重和扩展 ABI 需由部署者准备与核验。

## 6. 服务和独立评测

```bash
bash rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh install
bash rl_miles/Qwen3-embedding/install_runtime.sh
bash eval/install_runtime.sh
```

只安装需要的部分；这些命令不会替你填写权重路径或自动配置训练服务 URL。

- [DS judge](rl_miles/DS-V4-Flash/README_zh.md)：专用 GPU 基础栈；venv 继承系统包。
- [Embedding / similarity](rl_miles/Qwen3-embedding/README_zh.md)：4 个默认 TP1 副本与 HTTP 相似度 API。
- [Eval](eval/README_zh.md)：`API_ONLY=1 bash eval/install_runtime.sh` 可跳过本地 vLLM，仍需要 tokenizer。

## 7. 常见问题

| 现象 | 检查方法 |
| --- | --- |
| CUDA 扩展 undefined symbol / 导入失败 | 对齐 Python、Torch、CUDA、CXX11 ABI；检查是否混入另一个 venv 的包 |
| 找不到 GDN CP helper | SFT 必须使用 CP-capable Megatron，而非 CPT 的 0.17.1 CP1 栈 |
| user loss 不生效 | 检查 Swift 补丁、`TRAIN_USER_LOSS`、消息掩码；修改策略后重建数据缓存 |
| RL backend 导入缺 `modelopt` / `torch_memory_saver` | 按第 5 节补齐，不要仅确认 wheel 已下载 |
| Ray 集群有节点但作业一直等待资源 | 检查总 actor+rollout GPU；默认需要 16 或 20 节点，而不只是 actor 节点 |
| judge / similarity 调用失败 | 从训练节点验证 URL、模型名称、端口与请求字段；安装成功不等于服务可达 |

源码/补丁检查、语法检查、dry-run 和 import 检查均不能替代模型加载、前向反向、
NCCL 通信、权重同步与真实服务请求的验证。请先完成小规模 smoke，再启动完整训练。
