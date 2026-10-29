# World Model

[English](README.md) | **简体中文**

**从轨迹学习环境反馈：CPT → SFT → 异步 RL → Evaluation。**

本项目提供 Qwen3.5 系列语言世界模型的训练、奖励服务与评测工作流。
模型根据历史上下文预测下一条环境 Observation；训练覆盖持续预训练、
监督微调和基于 Miles 的异步 GSPO，评测使用独立 LLM judge 对照真实反馈评分。

<p align="center">
  <a href="https://darwin-agent.github.io/WIA/"><img alt="Homepage" src="https://img.shields.io/badge/Homepage-WIA-ff6900?style=flat"/></a>
</p>

[快速开始](#快速开始) · [环境安装](ENVIRONMENT_zh.md) · [并行配置](#模型与资源) ·
[完整参数参考](docs/CONFIGURATION_zh.md) · [服务部署](#需要哪些服务) · [文档导航](docs/README_zh.md)

## 训练前必须安装

**本项目是源码发行包，不是解压即用的运行环境。CPT/SFT 必须具备 ms-swift；RL 必须具备 Miles 及配套框架。缺少这些依赖不能启动训练。**
按需安装要运行的阶段，不必为了单个阶段安装所有训练框架。
先按[环境说明](ENVIRONMENT_zh.md)准备兼容的 GPU 基础环境，再从仓库根目录执行：

| 阶段 | 必需安装命令 | 默认框架源码位置 |
| --- | --- | --- |
| CPT | `bash cpt/setup_runtime.sh cpt` | `cpt/ms-swift`、`cpt/mcore-bridge` |
| SFT | `bash sft_code/setup_runtime.sh` | 复用上述 Swift/bridge 源码；CP 支持使用 `rl_miles/Megatron-LM` |
| RL | `bash rl_miles/install_official_miles_env.sh` | `rl_miles/miles`、`rl_miles/Megatron-LM`、`rl_miles/Megatron-Bridge`、`rl_miles/sglang-miles` |

安装器会下载固定版本源码、应用随项目提供的框架补丁并安装依赖，不需要手工克隆任意上游版本。
RL 安装器会自动调用 `bootstrap_miles.sh`；仅运行 bootstrap **只准备源码，不安装 Python/GPU 环境**。
这些安装入口的 `--help` 只显示帮助，不下载、不安装。
已有匹配环境可在版本和补丁检查通过后复用；自定义源码、环境路径必须在安装和启动时保持一致，
详见[安装与环境复用](ENVIRONMENT_zh.md#安装与环境复用)。模型、数据、集群配置及所需奖励服务仍需单独准备。

## 功能

- **CPT**：支持 assistant-only 和按消息掩码学习后续 user + assistant 两种目标。
- **SFT**：统一 CP4，提供 Dense/MoE 模型入口及 CPT checkpoint 衔接说明。
- **RL**：actor 与 rollout 分离，支持异步采样、权重同步、断点恢复和可配置奖励。
- **Reward Server**：DeepSeek-V4-Flash judge；可选 Qwen3 Embedding 相似度服务。
- **Evaluation**：本地 vLLM 或自建兼容服务，支持 CoT 开关、分领域评分与续跑。

仓库包含代码、启动器、提示词和框架补丁；**不包含权重、训练数据、虚拟环境、
已部署服务或访问凭据**。下载代码后，需要先完成环境、数据和服务配置。

## 工作流

```text
轨迹 / messages ── CPT ── HF 或兼容 Megatron checkpoint
                              │
                              ▼
                         SFT（CP4）
                              │ HF 导出
                              ▼
                    Miles 异步 GSPO
                    ├─ actor：Megatron 训练
                    ├─ rollout：SGLang 生成
                    └─ reward：LLM judge + 可选 similarity
                              │
                              ▼
                    独立 Evaluation / 报表
```

## 模型与资源

默认每节点 8 GPU。下表的 GPU 数**不包含独立部署的 judge 和 embedding 服务**。
TP/PP/CP 分别表示张量/流水线/上下文并行；EP/ETP 表示专家/专家张量并行。
DP 是 Dense 数据并行度，不把 EP 再作为除数。

### CPT

| 模型 | GPU / 节点 | 后端与并行配置 | Dense DP | 全局 batch |
| --- | --- | --- | ---: | ---: |
| 4B | 64 / 8 | HF Swift；SP4、ZeRO-1 | 16 | 1024 |
| 9B | 64 / 8 | Megatron；TP4 / PP1 / CP1 | 16 | 1024 |
| 27B | 128 / 16 | Megatron；TP8 / PP1 / CP1 | 16 | 1024 |
| 35B-A3B | 128 / 16 | Megatron；TP2 / PP8 / CP1 / EP2 / ETP1 / SP | 8 | 1024 |

4B 的 HF SP4 与 Megatron 的 `sequence_parallel=true` 不是同一机制。
35B 表格行对应 `cpt/train_qwen35_35b_a3b_cpt_128gpu.sh`，使用 GBS1024；EP 是专家路由拓扑，不能再次作为 Dense DP 的除数。
启动入口与缓存流程见 [CPT 指南](cpt/README_zh.md)。

### SFT

| 模型 | GPU / 节点 | TP / PP / CP | EP / ETP | Dense DP | 全局 batch |
| --- | --- | --- | --- | ---: | ---: |
| 4B | 8 / 1 | 2 / 1 / 4 | 1 / 1 | 1 | 128 |
| 9B | 8 / 1 | 2 / 1 / 4 | 1 / 1 | 1 | 128 |
| 27B | 64 / 8 | 4 / 4 / 4 | 1 / 1 | 1 | 128 |
| 35B-A3B | 32 / 4 | 2 / 4 / 4 | 8 / 1 | 1 | 128 |

默认 micro batch 为 1，累计 128 个 microbatch；需支持 GatedDeltaNet CP 的运行环境。
权重格式、分布式启动与安全限制见 [SFT 指南](sft_code/README_zh.md)。

### RL

| 模型 | 总 GPU | actor / rollout GPU | actor TP / PP / CP | EP / ETP | actor Dense DP | rollout GPU/引擎 × 引擎数 | 全局 batch |
| --- | ---: | --- | --- | --- | ---: | --- | ---: |
| 4B | 128 | 64 / 64 | 2 / 1 / 4 | 1 / 1 | 8 | 1 × 64 | 1024 |
| 9B | 128 | 64 / 64 | 2 / 2 / 4 | 1 / 1 | 4 | 2 × 32 | 2048 |
| 27B | 128 | 64 / 64 | 4 / 4 / 4 | 1 / 1 | 1 | 2 × 32 | 2048 |
| 35B-A3B | 128 | 64 / 64 | 2 / 4 / 4 | 8 / 1 | 2 | 2 × 32 | 2048 |
| 122B-A10B | 160 | 96 / 64 | 2 / 12 / 4 | 8 / 1 | 1 | 8 × 8 | 2048 |

122B-A10B 仅提供 160-GPU RL 入口，不提供 CPT/SFT 入口。
所有 RL 入口默认 `MAX_WEIGHT_STALENESS=1`、`ASYNC_DATA_BUFFER_CAPACITY_FACTOR=1.0`。
动态 token batching 下，microbatch 数由 token 预算决定，不应直接用静态梯度累计公式推算。
详见 [RL 指南](rl_miles/README_zh.md)。

## 需要哪些服务

| 阶段 | 本地/集群运行时 | 需要单独准备的服务 |
| --- | --- | --- |
| CPT / SFT | Swift、Megatron；4B/35B CPT 还使用 DeepSpeed | 无 |
| RL：4B / 9B / 27B / 35B-A3B | Ray + Megatron actor + SGLang rollout | LLM judge；similarity 默认关闭 |
| RL：122B-A10B | Ray + Megatron actor + SGLang rollout | LLM judge + similarity API + embedding 后端 |
| 独立 Eval | 本地 vLLM，或用户配置的兼容推理服务 | LLM judge；`--skip-judge` 可只生成 |

RL rollout 由 Miles 管理，**不是**单独部署的 DeepSeek judge。
部署文档：[DeepSeek judge](rl_miles/DS-V4-Flash/README_zh.md) ·
[Embedding / similarity](rl_miles/Qwen3-embedding/README_zh.md)。

| 模型 | CoT | judge / similarity 权重 | 相似度服务 |
| --- | --- | --- | --- |
| 4B / 9B / 27B / 35B-A3B | 关闭 | 1.0 / 0.0 | 默认不需要 |
| 122B-A10B | 开启 | 0.9 / 0.1 | 默认需要 |

RL 中 thinking **超过 10,000 字符**时，最终奖励乘以 0.9；独立 Eval 不应用此惩罚。
奖励失败降级与权重覆盖规则见 [RL 奖励配置](rl_miles/README_zh.md#奖励配置)。

## 消融范围

公开报告中的 4B、9B 和 35B-A3B 提供完整的 CPT→SFT→RL 阶段对比。27B 仅报告起始的 Qwen3.5-27B 检查点和最终的 WIA-WM-27B 检查点，不包含中间阶段结果。因此，27B 应解释为起始检查点到最终模型的端点对比，不能据此单独估计 CPT、SFT 和 RL 的贡献。

## 快速开始

### 1. 准备运行环境

从仓库根目录操作：

```bash
export PROJECT_ROOT="$PWD"
```

按 [环境安装指南](ENVIRONMENT_zh.md) 选择需要的阶段。CPT/SFT 使用 CUDA12 栈，
RL 使用独立 CUDA13 栈；不要混装。准备权重时保留完整 HF config、tokenizer 与权重文件。

### 2. 填写自己的配置

| 内容 | CPT / SFT | RL | Eval |
| --- | --- | --- | --- |
| 模型路径 | `MODEL_PATH`；SFT 可另设 `MCORE_MODEL` | `MILES_HF_CHECKPOINT`、`MILES_LOAD` | `--checkpoint`、可选 `--template-model` |
| 数据路径 | `DATASET_PATH`；CPT 可选 `CACHED_DATASET` | `MILES_PROMPT_DATA`、`EVAL_DATA` | `--eval-data` |
| 输出路径 | `OUTPUT_DIR` | `MILES_SAVE` | `--output-root`、`--run-name` |
| 节点设置 | `NNODES`、`NODE_RANK`、`MASTER_ADDR` | `MASTER_ADDR`；逐节点 `WORLD_SIZE/RANK` 或 `HOSTFILE` | 本地 `--num-gpus` 或 `--rollout-base-url` |
| judge | 不需要 | `JUDGE_BASE_URL`、`JUDGE_MODEL` | 同名环境变量或 CLI |
| similarity | 不需要 | `SIMILARITY_URL`、`SIMILARITY_ENABLED` | 主入口不调用 |

训练数据格式见 [CPT](cpt/README_zh.md#数据与-loss-mask)、[SFT](sft_code/README_zh.md#数据与权重)、
[RL](rl_miles/README_zh.md#数据与权重)。示例中的 `/path/to/...` 和 `*.example.invalid`
必须换成自己的资源；真实密钥仅通过运行环境注入，不写入仓库。

### 3. 选择阶段并启动

- [CPT：预处理、缓存与逐节点启动](cpt/README_zh.md#快速开始)
- [SFT：加载 CPT 权重并运行 CP4 微调](sft_code/README_zh.md#快速开始)
- [RL：部署奖励服务、组建 Ray 集群并训练](rl_miles/README_zh.md#快速开始)
- [Eval：生成、评分、续跑与汇总](eval/README_zh.md#快速开始)

先使用入口的 `--help` / `DRY_RUN=1` 检查参数，再验证小规模模型加载、通信和服务请求。
RL dry-run 仍需可读取的 HF `config.json`；任何 dry-run 都不证明完整 GPU 训练可运行。

## 参数覆盖

环境变量的默认值、模型包装器的覆盖以及 CLI 支持范围各不相同。
优先使用各阶段 README 中公开的环境变量，不能假定任意 `--flag` 都会被转发。
完整索引见 [配置参考](docs/CONFIGURATION_zh.md)，包含默认表达式、适用入口和源码定位。

## 目录导航

| 目录 | 用途 |
| --- | --- |
| [cpt](cpt/README_zh.md) | loss mask、缓存导出、4 个 CPT 入口 |
| [sft_code](sft_code/README_zh.md) | 4 个 SFT 入口、CP/拓扑检查 |
| [rl_miles](rl_miles/README_zh.md) | 5 个异步 GSPO 入口、数据加载和奖励 |
| [eval](eval/README_zh.md) | 独立评测、双 judge、重新评分与汇总 |
| [prompts](prompts/README_zh.md) | 七个领域的 judge 提示词 |
| [runtime](runtime/README_zh.md) | 固定框架版本、补丁应用与启动检查 |
| [docs](docs/README_zh.md) | 配置索引与使用导航 |

## 使用边界

- GPU 数是默认分配，不是对任意显卡型号和显存的适配保证；部署前检查驱动、CUDA、ABI 与通信。
- 多机环境使用相同源码、依赖、模型和数据路径；不要把已有虚拟环境目录直接搬到另一种基础镜像。
- 服务端口只向可信网络开放；Ray dashboard、judge 与 similarity API 不应直接暴露到公网。
- 不提交模型、数据、日志、请求响应、机器清单或凭据。忽略规则不能替代发布前的隐私检查。
- 第三方框架、模型和数据分别受其授权条款约束；使用前自行确认许可范围。
