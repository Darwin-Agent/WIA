# SFT：监督微调

[English](README.md) | **简体中文** · [项目首页](../README_zh.md)

从 CPT 权重继续训练，使用 Megatron-SWIFT 和 CP4 处理长上下文。
SFT 不需要外部奖励服务；数据通常设置为 assistant-only。

## 必需框架安装

**SFT 必须具备 ms-swift、mcore-bridge 和支持 CP 的 Megatron，不能仅解压发布包就启动训练。**
先按[环境指南](../ENVIRONMENT_zh.md)准备基础环境，再从仓库根目录执行：

```bash
bash sft_code/setup_runtime.sh
```

此入口调用 `cpt/setup_runtime.sh sft`，准备 `cpt/` 下固定版本的 Swift/bridge 源码、共享 Swift 补丁，
以及 `rl_miles/Megatron-LM` 下支持 CP 的 Megatron，并安装独立的 `sft_code/.venv`。
Megatron 源码放在该目录**不代表 SFT 需要安装 Miles RL 环境或部署奖励服务**。
无需先运行 CPT 安装器；不能把 CPT 的 CP1 环境直接当作此 SFT 环境。
自定义环境时，安装阶段设置 `SFT_VENV`，训练阶段设置 `SWIFT_VENV` 为同一路径；
`SWIFT_ROOT`、`MEGATRON_ROOT` 也要保持一致。只有版本、补丁与导入检查通过的已有环境才可复用。
`bash sft_code/setup_runtime.sh --help` 只显示帮助，详见[安装与环境复用](../ENVIRONMENT_zh.md#安装与环境复用)。

## 模型与默认配置

| 模型 | 启动脚本 | 节点 × GPU | TP / PP / CP | EP / ETP | Dense DP | 全局 batch |
| --- | --- | --- | --- | --- | ---: | ---: |
| 4B | `train_qwen35_4b_sft_8gpu.sh` | 1 × 8 | 2 / 1 / 4 | 1 / 1 | 1 | 128 |
| 9B | `train_qwen35_9b_sft_8gpu.sh` | 1 × 8 | 2 / 1 / 4 | 1 / 1 | 1 | 128 |
| 27B | `train_qwen35_27b_sft_64gpu.sh` | 8 × 8 | 4 / 4 / 4 | 1 / 1 | 1 | 128 |
| 35B-A3B | `train_qwen35_35b_a3b_sft_32gpu.sh` | 4 × 8 | 2 / 4 / 4 | 8 / 1 | 1 | 128 |

共同默认：epoch 2、micro batch 1、累计 128、长度 65536、学习率 `1e-5`、cosine 调度、
CPU optimizer offload。35B-A3B 的 expert DP = 32 / (ETP1 × EP8 × PP4) = 1；这里不再乘 CP。
Megatron 布尔 SP 在 TP 组内运行，不增加 GPU 维度。

## 数据与权重

数据为 `messages` JSONL；详细掩码契约见 [CPT 数据说明](../cpt/README_zh.md)。
assistant-only 样例：

```json
{"messages":[{"role":"user","content":"Predict the next observation.","loss_mask":false},{"role":"assistant","content":"The action succeeds.","loss_mask":true}]}
```

`TRAIN_USER_LOSS=true` 只允许显式启用的后续 user；当 user 掩码全部为 false 时，仍只学习 assistant。
没有独立的 SFT-only loss 补丁。

两种加载方式：

1. **完整 HF CPT 导出**：设置 `MODEL_PATH` 指向 config、tokenizer、weights 完整目录。
2. **native Megatron checkpoint**：另设 `MCORE_MODEL` 指向权重，`MODEL_PATH` 仍需对应的 HF 元数据/tokenizer。

从 CPT 转到 SFT 时 TP/PP/CP 会变化；native checkpoint 是否支持重分片取决于保存格式与运行时。
不兼容时先导出 HF，不能只改目录名或把普通 checkpoint 链接成 HF 模型。
入口固定 finetune、`no_load_optim=true`、`no_load_rng=true`，这是阶段初始化，不是完整优化器续训入口。

## 快速开始

按 [环境指南](../ENVIRONMENT_zh.md) 创建 CP-capable SFT 环境；以下从仓库根目录运行。
以 27B 的 8 节点训练为例：

```bash
export MODEL_PATH=/path/to/cpt-27b-hf
export DATASET_PATH=/path/to/sft.jsonl
export OUTPUT_DIR=/path/to/checkpoints/sft-27b
export NNODES=8 NODE_RANK=0 RESOURCE_GPU=8 NPROC_PER_NODE=8
export MASTER_ADDR=rank0.example.invalid MASTER_PORT=29503
DRY_RUN=1 bash sft_code/train_qwen35_27b_sft_64gpu.sh
```

替换主机地址，每节点执行一次，只改变 `NODE_RANK`。确认后移除 `DRY_RUN=1`。
4B/9B 单节点可以省略 master 地址，默认回环地址；35B 使用 4 节点和其对应入口。
不要在外层重复 torchrun。`WORLD_SIZE/RANK` 是可选的节点数/节点编号别名。

## 参数覆盖

入口只接受 `--help`，其他设置通过环境变量传入。四个包装器固定模型尺度；
直接调用 `train_sft.sh` 时可设置 `SFT_MODEL_SIZE=4b|9b|27b|35b_a3b`。
完整列表见 [配置参考](../docs/CONFIGURATION_zh.md)。

| 类别 | 参数与默认值 / 用途 |
| --- | --- |
| 路径 | `MODEL_PATH`、可选 `MCORE_MODEL`、`DATASET_PATH`、`OUTPUT_DIR`、`CACHE_ROOT` |
| 环境 | `SWIFT_ROOT`、`SWIFT_VENV`（默认 `sft_code/.venv`）、可选 `MEGATRON_ROOT`；安装时自定义 venv 使用 `SFT_VENV` |
| 模型模板 | `MODEL_TYPE` 按模型选择 `qwen3_5` / `qwen3_5_moe`；`TEMPLATE=qwen3_5`，必须与 checkpoint 相符 |
| 节点 | `NNODES/NODE_RANK`、`RESOURCE_GPU/NPROC_PER_NODE`、`CUDA_VISIBLE_DEVICES`、`MASTER_ADDR/MASTER_PORT` |
| 并行 | `TENSOR_MODEL_PARALLEL_SIZE`、`PIPELINE_MODEL_PARALLEL_SIZE`、`CONTEXT_PARALLEL_SIZE=4`、`EXPERT_MODEL_PARALLEL_SIZE`、`EXPERT_TENSOR_PARALLEL_SIZE=1` |
| CP / SP | `CP_COMM_TYPE=p2p`；`USE_MCORE_GDN=1`；TP>1 时 `SEQUENCE_PARALLEL=true` |
| Batch | `MICRO_BATCH_SIZE=1`（可从 `PER_DEVICE_TRAIN_BATCH_SIZE` 取值）、`GLOBAL_BATCH_SIZE=128` |
| 累计 | 由全局 batch/(DP×micro batch) 推导；不要设置冲突的 `GRADIENT_ACCUMULATION_STEPS` |
| 时长 | `NUM_TRAIN_EPOCHS=2`；非空 `TRAIN_ITERS` 优先按步数训练 |
| 长度 | `MAX_LENGTH=65536`、`TRUNCATION_STRATEGY=right` |
| 优化 | `LEARNING_RATE=1e-5`、`MIN_LR=0`、`WARMUP_RATIO=0.03`、`WEIGHT_DECAY=0.1`、`MAX_GRAD_NORM=1.0` |
| 数据目标 | `TRAIN_USER_LOSS=true`、`LOSS_SCALE=default`；`PACKING=false`、`PADDING_FREE=false` |
| 数据读取 | `DATASET_NUM_PROC=8`、`DATALOADER_NUM_WORKERS=4`、`LOAD_FROM_CACHE_FILE=false`、`DATASET_SHUFFLE=true` |
| 保存 | `SAVE_STEPS=10`、`SAVE_TOTAL_LIMIT=20`、`LOGGING_STEPS=1`、`SAVE_SAFETENSORS=true` |
| 冻结 | `FREEZE_VIT=true`、`FREEZE_ALIGNER=true`；LLM 全参微调 |
| Offload | `OPTIMIZER_CPU_OFFLOAD=true`、`OPTIMIZER_OFFLOAD_FRACTION=1`、`USE_PRECISION_AWARE_OPTIMIZER=true` |

## 更改拓扑时的约束

- GPU 上限：27B 为 64，其余尺度为 32；这些是入口安全限制，不能用环境变量直接解除。
- 总 GPU 必须能被 TP×PP×CP 整除；全局 batch 必须能被 Dense DP×micro batch 整除。
- 模型层数按 PP 切分，hidden size/attention heads 按 TP 切分。
- GDN key/value heads 必须能被 TP×CP 整除；因此不能把 27B CPT 的 TP8 原样用于 CP4。
- `MAX_LENGTH` 必须能被 2×CP 整除；CP>1 需要 MCore GDN；当前仅接受 `CP_COMM_TYPE=p2p`。
- MoE 还检查专家数、ETP 与 expert rank grid。通过算术检查不代表一定有足够显存。

## 输出与下一步

检查保存目录是否包含下一阶段需要的完整 config、tokenizer 和权重；
`SAVE_SAFETENSORS=true` 不应被当作所有 checkpoint 都已完成通用 HF 导出的保证。
准备好完整 HF 模型后进入 [RL 指南](../rl_miles/README_zh.md)。
`register_qwen3_5_text.py` 是可选 HF 注册辅助，不由默认 Megatron SFT 入口加载。
