# CPT：持续预训练

[English](README.md) | **简体中文** · [项目首页](../README_zh.md)

学习轨迹中的动作与环境反馈，使用消息掩码选择训练 assistant，或同时训练后续 user 与 assistant。
本阶段不需要 judge 或 similarity 服务。

## 必需框架安装

**未安装 ms-swift 及匹配训练依赖，不能执行 CPT 训练。** 源码发行包不包含这些完整框架和环境。
按[环境指南](../ENVIRONMENT_zh.md)准备 GPU 基础环境后，从仓库根目录执行：

```bash
bash cpt/setup_runtime.sh cpt
```

安装器将固定版本 Swift 和 mcore-bridge 下载至 `cpt/ms-swift`、`cpt/mcore-bridge`，应用随项目提供的
Swift 补丁，并在 `cpt/ms-swift-venv` 安装训练依赖，不必手工克隆最新版 Swift。
这些补丁用于保证项目的 loss mask 语义；任意上游安装不能视为等价替代。
`bash cpt/setup_runtime.sh --help` 可在不安装的情况下查看路径说明。已有匹配环境可通过 `SWIFT_ROOT`
和 `SWIFT_VENV` 复用，安装与启动保持一致；复用前检查版本、补丁及导入。
模型、数据和多节点资源仍需自行准备，详见[安装与环境复用](../ENVIRONMENT_zh.md#安装与环境复用)。

## 模型与默认配置

| 模型 | 启动脚本 | 节点 × GPU | 并行配置 | Dense DP | micro batch | 全局 batch | 累计步数 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 4B | `train_qwen35_4b_cpt_64gpu.sh` | 8 × 8 | HF SP4、ZeRO-1 | 16 | 1 | 512 | 32 |
| 9B | `train_qwen35_9b_cpt_64gpu.sh` | 8 × 8 | TP4 / PP1 / CP1 | 16 | 1 | 1024 | 64 |
| 27B | `train_qwen35_27b_cpt_128gpu.sh` | 16 × 8 | TP8 / PP1 / CP1 | 16 | 1 | 1024 | 64 |
| 35B-A3B | `train_qwen35_35b_a3b_cpt_128gpu.sh` | 16 × 8 | TP2 / PP8 / CP1 / EP8 / ETP1 | 8 | 1 | 16 | 2 |

共同默认：`NUM_TRAIN_EPOCHS=1`、`MAX_LENGTH=32768`、`LEARNING_RATE=2e-5`，左侧截断。
Megatron 的 Dense DP = 总 GPU / (TP × PP × CP)，累计步数 = 全局 batch / (DP × micro batch)。
EP 不应再次除入 Dense DP。4B 的 HF SP 是独立切分机制；Megatron 的布尔 SP 在 TP 组内运行。
35B-A3B 使用 PP8，每个 stage 5 层，不需要特殊尾层配置。

## 数据与 loss mask

输入为 JSONL，每行一条 `messages`。下面是合成样例：

```json
{"messages":[{"role":"system","content":"Predict environment feedback.","loss_mask":false},{"role":"user","content":"Initial state.","loss_mask":false},{"role":"assistant","content":"Perform an action.","loss_mask":true},{"role":"user","content":"The state changes.","loss_mask":true},{"role":"assistant","content":"Next action.","loss_mask":true}]}
```

- **user + assistant**：首个 user/system 设为 false，后续正文按掩码决定是否计算 loss。
- **assistant-only**：所有 user/system 设为 false，需要学习的 assistant 设为 true。
- 接受消息级 `loss` 或 `loss_mask`（布尔值、0/1），以及顶层每消息一个值的 `loss_mask` 列表。
- 缺省掩码只启用 assistant；首 user/system 显式 true、冲突字段和 token 级数组不被接受。
- `TRAIN_USER_LOSS=true` 允许显式启用后续 user，并不会自动学习所有 user。使用小写 `true/false`。
  false 时恢复 assistant-only；未参与 loss 的消息仍作为上下文。
- user 角色/模板分隔符不会因该开关一起训练；assistant 模板/EOS 保留框架行为。

建议先校验数据，避免框架的非严格预处理静默过滤非法行：

```bash
python3 -S cpt/prepare_loss_masks.py --input /path/to/raw.jsonl --output /path/to/canonical.jsonl
```

## 快速开始

先完成 [CPT 环境安装](../ENVIRONMENT_zh.md)。以下命令从仓库根目录运行；以 9B 为例：

```bash
export MODEL_PATH=/path/to/qwen35-9b-hf
export DATASET_PATH='/path/to/cpt/*.jsonl'
export OUTPUT_DIR=/path/to/checkpoints/cpt-9b
export NNODES=8 NODE_RANK=0 RESOURCE_GPU=8 NPROC_PER_NODE=8
export MASTER_ADDR=rank0.example.invalid MASTER_PORT=29500
export USE_CACHED_DATASET=false
DRY_RUN=1 bash cpt/train_qwen35_9b_cpt_64gpu.sh
```

将主机占位符替换为真实可达地址。每个节点各执行一次相同入口，只改变 `NODE_RANK`
为 `0..NNODES-1`。确认后去掉 `DRY_RUN=1` 正式启动。不要在外层再次使用 torchrun。
平台 `WORLD_SIZE/RANK` 在这些入口中是**节点数/节点编号**，不是 GPU worker 数；
若同时设置 `NNODES/NODE_RANK`，两者必须一致。

### 缓存预处理

4B/9B/27B 默认读取缓存。如果还没有缓存，使用上面的 `USE_CACHED_DATASET=false`。
也可先进行单进程导出，`OUTPUT_DIR` 在导出时表示缓存目录，而不是训练 checkpoint 目录：

```bash
MODEL_PATH=/path/to/qwen35-9b-hf DATASET_PATH='/path/to/cpt/*.jsonl' \
OUTPUT_DIR=/path/to/cache/cpt-9b TRAIN_USER_LOSS=true \
bash cpt/preprocess_cpt_cache.sh
```

随后设置 `USE_CACHED_DATASET=true CACHED_DATASET=/path/to/cache/cpt-9b/train`。
预处理需要完整的本地模型/tokenizer，脚本使用离线加载。35B-A3B 当前入口直接使用
`DATASET_PATH`，不提供与 Dense 入口相同的缓存选择接口。
**修改模型模板、长度、loss mask、补丁或 TRAIN_USER_LOSS 后重新导出缓存**，旧 labels 不会自动改变。

## 参数覆盖

通过 `export VAR=value` 或 `VAR=value bash ...` 覆盖；完整默认表达式见
[配置参考](../docs/CONFIGURATION_zh.md)。不要把未经入口支持的任意 CLI 参数当成可用覆盖。

| 类别 | 参数与默认值 / 用途 |
| --- | --- |
| 模型与数据 | `MODEL_PATH`：完整 HF 模型；`DATASET_PATH`：JSONL/匹配模式，必须加引号避免提前展开 |
| 输出与缓存 | `OUTPUT_DIR`、`CACHE_ROOT`；4B/9B/27B 的 `USE_CACHED_DATASET`、`CACHED_DATASET`、`CACHED_DATASET_ROOT` |
| 环境路径 | `PROJECT_ROOT`、`CODE_ROOT`、`SWIFT_ROOT`、`SWIFT_VENV`；默认按仓库目录定位 |
| 节点 | `NNODES`、`NODE_RANK`、`RESOURCE_GPU`、`NPROC_PER_NODE`、`CUDA_VISIBLE_DEVICES`、`MASTER_ADDR/MASTER_PORT` |
| 数据目标 | `TRAIN_USER_LOSS=true`；4B `LOSS_SCALE=default`，不要在不了解 loss scale 交互时更改 |
| 长度 | `MAX_LENGTH=32768`、`TRUNCATION_STRATEGY=left` |
| 训练时长 | `NUM_TRAIN_EPOCHS=1`；Megatron `TRAIN_ITERS` 非空时优先按步数训练 |
| 4B batch | `PER_DEVICE_TRAIN_BATCH_SIZE=1`、`GRADIENT_ACCUMULATION_STEPS=32`、`SEQUENCE_PARALLEL_SIZE=4`；不是通过 `GLOBAL_BATCH_SIZE` 设置 |
| Megatron batch | `MICRO_BATCH_SIZE=1`、`GLOBAL_BATCH_SIZE` 按表；总 batch 必须能被 DP × micro batch 整除 |
| Megatron 拓扑 | `TENSOR_MODEL_PARALLEL_SIZE`、`PIPELINE_MODEL_PARALLEL_SIZE`、`CONTEXT_PARALLEL_SIZE`；MoE 另有 `EXPERT_MODEL_PARALLEL_SIZE`、`EXPERT_TENSOR_PARALLEL_SIZE` |
| Dense 拓扑检查 | 9B/27B 默认验证目标节点数；改变默认网格需 `ALLOW_TOPOLOGY_OVERRIDE=true`，不会绕过其他可整除性要求 |
| 学习率 | `LEARNING_RATE=2e-5`；Megatron `MIN_LR=2e-6`、`LR_WARMUP_FRACTION=0.03`；4B `WARMUP_RATIO=0.03` |
| 保存 | `SAVE_STEPS`：4B 100，9B/27B 500，35B 200；4B 另有 `SAVE_TOTAL_LIMIT=100` |
| 数据线程 | `DATASET_NUM_PROC`、`DATALOADER_NUM_WORKERS`；导出默认 16 个数据进程，训练默认 8 |
| 重用缓存 | `LOAD_FROM_CACHE_FILE=false`（训练）；不是 `USE_CACHED_DATASET` 的同义词 |
| 4B 性能 | `DEEPSPEED=zero1`、`ATTN_IMPL=flash_attn`、`PACKING=false`、`PADDING_FREE=false`、`GROUP_BY_LENGTH=false` |
| Megatron offload | `OPTIMIZER_CPU_OFFLOAD`：9B/27B false，35B true；`OPTIMIZER_OFFLOAD_FRACTION=1`、`USE_PRECISION_AWARE_OPTIMIZER=true` |
| Dense 验证 | `NCCL_PREFLIGHT_ONLY=true` 运行 GPU/通信检查；只有同时 dry-run 才不启动 worker |

示例：9B 使用不同全局 batch 和训练时长：

```bash
GLOBAL_BATCH_SIZE=512 NUM_TRAIN_EPOCHS=1 TRAIN_ITERS= \
bash cpt/train_qwen35_9b_cpt_64gpu.sh
```

## 输出与下一步

Megatron CPT 入口设置 `save_safetensors=false`，不要把输出目录默认当作完整 HF 模型。
进入 SFT 时可使用兼容的 native checkpoint，或先导出包含 config/tokenizer/weights 的 HF 目录。
权重格式和拓扑转换说明见 [SFT 指南](../sft_code/README_zh.md)。

兼容别名 `train_cpt_swift.sh`、`train_cpt_megatron_dense.sh`、`train_cpt_megatron_moe.sh`
分别指向 4B、27B、35B 的多机入口，不是单节点 smoke 脚本。
