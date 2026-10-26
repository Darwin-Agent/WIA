# RL：Miles 异步 GSPO

[English](README.md) | **简体中文** · [项目首页](../README_zh.md)

使用独立 actor/rollout GPU 池训练语言世界模型：SGLang 生成 Observation，奖励服务评分，
Megatron 更新策略，Miles 管理异步采样与权重同步。

## 必需框架安装

**RL 必须具备 Miles、Megatron、Megatron-Bridge、SGLang 及其 Python/GPU 依赖，缺少这些不能训练。**
源码 ZIP 不包含这些完整框架和环境。先阅读[环境安装](../ENVIRONMENT_zh.md)的基础环境要求，
再从仓库根目录执行：

```bash
bash rl_miles/install_official_miles_env.sh
```

安装器自动调用 `bootstrap_miles.sh` 下载/检查固定版本源码并应用运行补丁，然后安装 RL 环境。
默认源码位置为 `rl_miles/miles`、`rl_miles/Megatron-LM`、`rl_miles/Megatron-Bridge`、
`rl_miles/sglang-miles`，默认虚拟环境为 `rl_miles/.venv`，无需手工克隆任意上游版本。
**仅运行 bootstrap 不等于安装完成。** 按环境指南补齐所列额外依赖并验证导入；训练启动器不会自动安装缺失框架。
自定义路径时，`MILES_REPO`、`MILES_VENV`、`MILES_MEGATRON_ROOT`、`MILES_SGLANG_ROOT` 必须在安装与启动时一致。
已有环境仅在版本、补丁和依赖均匹配时可复用。
`bash rl_miles/install_official_miles_env.sh --help` 只查看帮助；`bash rl_miles/bootstrap_miles.sh --check`
只验证源码版本和补丁，不能证明 GPU 可运行。权重、数据、完整集群及所需评分服务仍须单独准备，
详见[安装与环境复用](../ENVIRONMENT_zh.md#安装与环境复用)。

## 模型与并行配置

| 模型 | 启动脚本 | 总节点 / GPU | actor 节点 / GPU | TP / PP / CP | EP / ETP | Dense DP | rollout 引擎 |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| 4B | `run_qwen35_4b_gspo_128gpu_async.sh` | 16 / 128 | 8 / 64 | 2 / 1 / 4 | 1 / 1 | 8 | 64 × 1 GPU |
| 9B | `run_qwen35_9b_gspo_128gpu_async.sh` | 16 / 128 | 8 / 64 | 2 / 2 / 4 | 1 / 1 | 4 | 32 × 2 GPU |
| 27B | `run_qwen35_27b_gspo_128gpu_async.sh` | 16 / 128 | 8 / 64 | 4 / 4 / 4 | 1 / 1 | 1 | 32 × 2 GPU |
| 35B-A3B | `run_qwen35_35b_a3b_gspo_128gpu_async.sh` | 16 / 128 | 8 / 64 | 2 / 4 / 4 | 8 / 1 | 2 | 32 × 2 GPU |
| 122B-A10B | `run_qwen35_122b_a10b_gspo_160gpu_async.sh` | 20 / 160 | 12 / 96 | 2 / 12 / 4 | 8 / 1 | 1 | 8 × 8 GPU |

每节点 8 GPU，rollout 均使用 64 GPU；服务 GPU 另算。Dense DP = actor GPU/(TP×PP×CP)。
MoE expert DP 在当前 MCore rank grid 下为 actor GPU/(ETP×EP×PP)：35B 为 2，122B 为 1。
异步包装器固定 `train_async.py`，不支持 `COLOCATE=1`。

| 模型 | rollout batch（prompt 数） | 每 prompt 样本数 | 全局 batch | prompt / response token | 总长度 |
| --- | ---: | ---: | ---: | --- | ---: |
| 4B | 128 | 8 | 1024 | 32768 / 8192 | 40960 |
| 9B / 27B / 35B-A3B | 256 | 8 | 2048 | 32768 / 8192 | 40960 |
| 122B-A10B | 256 | 8 | 2048 | 32768 / 10240 | 43008 |

默认 micro batch 参数为 1；`QKV_FORMAT=thd` 启用动态 batch，实际 microbatch 划分受 token
预算与长度影响。不要把 rollout batch 当作单条序列的 token 数，也不要用静态累计公式推算调度。

## 需要部署的服务

1. **LLM judge**：全部模型默认启用。可使用 [DeepSeek-V4-Flash 部署](DS-V4-Flash/README_zh.md)。
   配置 `JUDGE_BASE_URL`（带 `/v1`）、`JUDGE_MODEL`；需要认证时通过 `JUDGE_API_KEY` 注入。
2. **Similarity API**：122B 默认启用，其余默认不需要。部署 [embedding 后端与相似度服务](Qwen3-embedding/README_zh.md)，
   配置完整 `SIMILARITY_URL`，例如 `http://similarity.example.invalid:8100/similarity`。
3. **Ray / SGLang rollout**：由训练入口在 actor+rollout 节点上管理，不要与 judge 进程共用已分配 GPU。

服务启动顺序：embedding → similarity（若启用），judge → 健康检查 → RL。
跨节点不能把 `127.0.0.1` 当作另一台机器的服务地址；用每个训练节点都能访问的主机名。
仅向可信网络开放端口，不提供私有代理模型服务。

## 数据与权重

推荐原始 JSONL，每行至少包含 `prompt` 与 `ground_truth`，可另含 `system/domain/dataset/source`：

```json
{"system":"Predict environment feedback.","prompt":"An action was performed.\n###[Observation]\n","ground_truth":"The action succeeds.","domain":"web","dataset":"ID"}
```

也支持兼容的 `messages` 和已渲染 `prompt/label`。原始数据由 loader 按训练时 CoT 设定渲染。
已渲染 prompt 不会自动重写；需要预处理时：

```bash
python rl_miles/prepare_rl_data.py --input /path/to/raw.jsonl --output /path/to/prepared.jsonl \
  --template-model /path/to/hf-model --think-mode off --max-prompt-tokens 32768
```

122B 使用 `--think-mode on`。带有模式元数据的 prepared prompt 与运行模式不符时会拒绝；
旧 prepared 数据没有模式信息时须自行确认。切换 CoT 后重新准备数据。

`MILES_HF_CHECKPOINT` 必须是完整 HF 模型；`MILES_LOAD` 用于初始化/恢复，`MILES_SAVE` 为保存目录。
默认包装器在 `SAVE_DEFAULT` 指向的目录存在 `latest_checkpointed_iteration.txt` 时选择恢复，否则从 HF 初始化。
自定义恢复路径时显式设置 `MILES_LOAD/MILES_SAVE`，不要只改 `MILES_SAVE` 后假定恢复检查也已改变。
换模型或开始新实验时使用新的保存目录；`REF_LOAD` 可指定 KL reference 权重。

## 快速开始

先完成 [RL 环境安装](../ENVIRONMENT_zh.md) 并在所有节点准备相同路径。
以下以 4B 的 16 节点为例，从仓库根目录运行：

```bash
export MODEL_HF_DEFAULT=/path/to/sft-4b-hf
export MILES_HF_CHECKPOINT="$MODEL_HF_DEFAULT"
export MILES_PROMPT_DATA=/path/to/train.jsonl
export EVAL_DATA=/path/to/test.jsonl
export MILES_SAVE=/path/to/checkpoints/rl-4b
export JUDGE_BASE_URL=http://judge.example.invalid:30000/v1
export JUDGE_MODEL=DeepSeek-V4-Flash-0731
export MASTER_ADDR=rank0.example.invalid MASTER_PORT=6379
export WORLD_SIZE=16 RANK=0 RESOURCE_GPU=8
export HOSTFILE=
DRY_RUN=1 bash rl_miles/run_qwen35_4b_gspo_128gpu_async.sh
```

替换示例路径和主机名。每个节点执行一次，`RANK` 分别为 0..15；122B 设置 `WORLD_SIZE=20`，
使用对应 160-GPU 入口，`RANK` 为 0..19，同时填写 `SIMILARITY_URL`。
确认后移除 `DRY_RUN=1`。**dry-run 仍检查 HF `config.json` 是否存在**。

两种集群启动方式：

- **逐节点启动（推荐）**：设置 `WORLD_SIZE/RANK`，保持 `HOSTFILE` 为空；head 提交作业，其余节点加入并阻塞。
  `WORLD_SIZE` 是总节点数；`ACTOR_NUM_NODES` 仍是 8 或 12，不能误设成 16 或 20。
  当前 head 的就绪等待只覆盖 actor 节点数量，不表示全部 rollout GPU 已就绪；必须启动所有节点。
- **Hostfile**：只在 head 执行，`HOSTFILE` 每行一个 worker 地址、不含 head；16 节点写 15 行、20 节点写 19 行。
  当前脚本用 `ssh root@worker` 启动远端 Ray，要求相应访问权限和相同可执行文件路径；
  不具备此权限时选择逐节点方式。机器清单保存在仓库外，不提交到 Git。

训练入口会启动/停止指定 Ray runtime；勿与其他作业共享同一 Ray 实例。
Ray dashboard 默认端口 8265，只对可信网络开放。

## 奖励配置

| 模型 | `THINK_MODE` | `JUDGE_WEIGHT` | `SIMILARITY_WEIGHT` | `SIMILARITY_ENABLED` |
| --- | ---: | ---: | ---: | ---: |
| 4B / 9B / 27B / 35B-A3B | 0 | 1.0 | 0.0 | 0 |
| 122B-A10B | 1 | 0.9 | 0.1 | 1 |

judge 给出六个 1–5 分维度，按 `format:1,factuality:2,consistency:2,realism:2,quality:2,efficiency:1`
加权平均，再用 `(score-1)/4` 限制到 [0,1]。Similarity 为限制到 [0,1] 的 embedding cosine。

- 两个 scorer 成功：`JUDGE_WEIGHT × judge_reward + SIMILARITY_WEIGHT × similarity_score`。
- 一个成功：使用它的**未加权**得分；两个都失败：本地规则奖励。相似度关闭时 judge 失败也走规则降级。
- `JUDGE_ENABLED=0` 选择本地规则路径，**不是 similarity-only 开关**。
- 权重各自限制在 [0,1]，不会自动归一化为和为 1。
- thinking 超过 **10,000 字符**时，在融合/降级后乘以 0.9；等于 10,000 不惩罚。
  CoT 关闭的普通长 Observation 不视为 thinking；显式 think 标签仍被识别。
- 评分前剥离 thinking；judge 自身保持 non-thinking，与策略 CoT 独立。

| 参数 | 默认 / 用途 |
| --- | --- |
| `JUDGE_BASE_URL` | 必填可达 `/v1` 地址，支持逗号分隔多个 endpoint |
| `JUDGE_MODEL` | 默认 `DeepSeek-V4-Flash-0731`，必须等于服务模型名 |
| `JUDGE_API_KEY` | 环境注入；默认 `EMPTY` 为无认证 SDK 占位，不是真实密钥 |
| `JUDGE_MAX_TOKENS` / `JUDGE_CONCURRENCY` | 4096 / 128 |
| `JUDGE_WEIGHTS` | 六维权重字符串；与 judge/similarity 混合权重不同 |
| `JUDGE_PROMPT_DIR` | 默认项目顶层 `prompts` |
| `SIMILARITY_URL` | 完整 `/similarity` 地址；可配置多个 endpoint |
| `SIMILARITY_TIMEOUT` / `SIMILARITY_CONCURRENCY` | 60 秒 / 128 |
| `WM_JUDGE_TIMEOUT` / `WM_JUDGE_RETRIES` | worker 侧 300 秒 / 3 次；可直接设置并转发 |
| `WM_SIMILARITY_RETRIES` | worker 侧 3 次 |
| `WM_JUDGE_SKIP` | 调试用，跳过远端 judge；改变奖励路径，勿无意开启 |

启动器把 `JUDGE_*`、`SIMILARITY_*`、`THINK_MODE` 映射成 Ray worker 的 `WM_*`。
通过启动脚本训练时优先覆盖非 WM 变量；同名 WM 值可能被映射覆盖。
直接调用 reward 模块时使用 WM 变量。仅上述重试/超时/skip 例外直接透传。

## 训练与资源覆盖

| 类别 | 参数与默认值 / 用途 |
| --- | --- |
| Batch / 时长 | `ROLLOUT_BATCH_SIZE`、`N_SAMPLES_PER_PROMPT`、`GLOBAL_BATCH_SIZE` 按表；非空 `NUM_ROLLOUT` 优先 |
| 长度 | `MAX_PROMPT_TOKENS`、`MAX_RESPONSE_TOKENS`、`SEQ_LEN` 按表；响应预算受总长度限制 |
| 优化 | `LEARNING_RATE=1e-6`、`CLIP_GRAD=1`；前四模型 `LR_DECAY_STYLE=constant`、`MIN_LR=0`，122B 为 cosine / `1e-7` |
| KL | 前四模型 `USE_KL_LOSS=0`、`KL_LOSS_COEF=0`；122B 为 1 / 0.01；可设 `REF_LOAD` |
| 算法 | `ADVANTAGE_ESTIMATOR=gspo`；clip epsilon 固定 `3.5e-4`、entropy coefficient 固定 0，不能靠未声明变量覆盖 |
| actor 资源 | `ACTOR_NUM_NODES`、`ACTOR_NUM_GPUS_PER_NODE`、`RESOURCE_GPU`；Ray 单节点资源为 `RAY_NUM_GPUS` |
| 最终并行值 | `TENSOR_PARALLEL_SIZE`、`PIPELINE_PARALLEL_SIZE`、`CONTEXT_PARALLEL_SIZE`、`EXPERT_PARALLEL_SIZE`、`EXPERT_TENSOR_PARALLEL_SIZE` |
| 并行缺省 | 上述最终值优先于包装器 `TP_DEFAULT/PP_DEFAULT/CP_DEFAULT/EP_DEFAULT/ETP_DEFAULT` |
| Token batching | `MAX_TOKENS_PER_GPU` / `LOG_PROBS_MAX_TOKENS_PER_GPU`：前四模型 8192，122B 4096；`LOG_PROBS_CHUNK_SIZE`：前四模型 4096，122B 2048 |
| 其他并行 | `SEQUENCE_PARALLEL=1`、`QKV_FORMAT=thd`、可选 `DECODER_LAST_PIPELINE_NUM_LAYERS`；层数/头数/专家必须可整除 |
| rollout 资源 | `ROLLOUT_NUM_GPUS=64`、`ROLLOUT_GPUS_PER_ENGINE` 按表；不与 actor GPU 重叠 |
| rollout 性能 | `SGLANG_MEM_FRACTION_STATIC=0.8`、`SGLANG_MAX_RUNNING_REQUESTS=32`、`SGLANG_SERVER_CONCURRENCY=32`、`ROLLOUT_TEMPERATURE=1.0` |
| Async | `MAX_WEIGHT_STALENESS=1`、`ASYNC_DATA_BUFFER_CAPACITY_FACTOR=1.0`；通过包装器转发到 Miles |
| 保存/评测 | `SAVE_INTERVAL=20`、`EVAL_INTERVAL=20`、`LOG_INTERVAL=1`；`EVAL_DATA=` 显式关闭评测，错误路径会报错 |
| Eval 长度 | `EVAL_MAX_PROMPT_TOKENS=65536`、`EVAL_MAX_RESPONSE_LEN=16384`、`EVAL_MAX_CONTEXT_LEN=81920` |
| Eval batch | `N_SAMPLES_PER_EVAL_PROMPT=1`、可选 `EVAL_GLOBAL_BATCH_SIZE`、`EVAL_DATA_LIMIT` |
| 导出/输出 | `MEGATRON_TO_HF_MODE=bridge`、`ROLLOUT_OUTPUT_DIR` 默认保存目录下 `rollout_jsonl` |
| 源码/环境 | `MILES_VENV`、`MILES_REPO`、`MILES_MEGATRON_ROOT`、`MILES_SGLANG_ROOT`、`MILES_EXTRA_PYTHONPATH` |
| 网络 | `MASTER_ADDR/MASTER_PORT`、`HOSTFILE`、`LOCAL_IP`、`SOCKET_IFNAME`、Ray head/cluster timeout/poll 变量 |
| 日志/调试 | `RL_DEBUG_LOG`、`RAY_STOP_BEFORE_START`、`DEBUG_ROLLOUT_ONLY`、`DEBUG_TRAIN_ONLY`；后两者不能同时启用 |
| W&B | 设置 `WANDB_API_KEY` 时启用；`WANDB_PROJECT/WANDB_GROUP` 控制分组，密钥不作为可打印训练参数传入 |

完整变量、安装/服务参数与 CLI 见 [配置参考](../docs/CONFIGURATION_zh.md)。
包装器接受额外 Miles CLI，但不要重复传入已生成的参数；wrapper 尾部仍可能追加固定参数，
不能保证“最后一个 CLI 值一定优先”。`TRAIN_DRIVER` 在包装器中固定为 async，不是通用可覆盖开关。

## 输出与扩展

训练 checkpoint 位于 `MILES_SAVE`，rollout JSONL 位于 `ROLLOUT_OUTPUT_DIR`。
JSONL 可能包含业务输入/模型响应，只保存在受控位置。

- [自定义数据加载与 reward](world_model_rl/README_zh.md)
- [模型结构配置](model_configs/README_zh.md)
- [Wheel 与兼容层](wheels/README_zh.md)
- [独立评测](../eval/README_zh.md)：其分数不自动包含 RL 混合奖励与超长惩罚。
