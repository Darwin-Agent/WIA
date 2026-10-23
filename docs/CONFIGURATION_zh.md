# 完整配置参考

[English](CONFIGURATION.md) | **简体中文** · [项目首页](../README_zh.md) · [文档导航](README_zh.md)

推荐配置见 [CPT](../cpt/README_zh.md)、[SFT](../sft_code/README_zh.md)、[RL](../rl_miles/README_zh.md)、
[Eval](../eval/README_zh.md)。本索引列出项目启动器、安装器、验证工具和服务读取的配置表达式及 CLI 声明，
不试图列出外部框架的所有参数。

## 参数覆盖规则

1. 常见 `${NAME:-default}` 在变量未设置或为空时采用默认值；`${NAME-default}` 保留显式空值。
2. `${NAME:?message}` 要求提供值；`${NAME:+...}` 是条件展开，不是默认值。
3. 单纯 `${NAME}` 是读取，不保证可由用户覆盖；其中一些是推导值或脚本输出。
   同时列出固定 `export NAME=value`，便于识别会被包装器覆盖的值。
4. 模型包装器与公共启动器必须一起看。RL 最终 `*_PARALLEL_SIZE` 优先于 `*_DEFAULT`；
   包装器中的 `TRAIN_DRIVER=train_async.py` 和 SFT 尺度选择器是固定值，不能只看公共默认值判断最终配置。
5. 非空 `TRAIN_ITERS` 优先于 CPT/SFT epoch；RL 非空 `NUM_ROLLOUT` 优先于 `NUM_EPOCH`。
   4B HF CPT 的全局 batch 由 DP×每设备 batch×累计步数得到，不通过 `GLOBAL_BATCH_SIZE` 设置。
6. SFT 仅接受 `--help`，训练使用环境变量；RL 可追加 CLI，但包装器仍会在用户参数后追加部分固定选项，避免重复。
   Eval 的显式 CLI 优先于对应环境默认值。
7. `EVAL_DATA=` 显式关闭 RL 评测；`CUSTOM_RM_PATH=` 清空自定义奖励 hook，不会自动安装替代奖励。
   `TRAIN_USER_LOSS` 使用小写 `true/false`；服务开关遵循各自文档。
8. 训练入口将奖励配置映射到 worker `WM_*`，启动时优先设 `JUDGE_*`、`SIMILARITY_*`、`THINK_MODE`；
   `WM_JUDGE_TIMEOUT/WM_JUDGE_RETRIES/WM_SIMILARITY_RETRIES/WM_JUDGE_SKIP` 是直接透传例外。
9. 多节点使用一致的绝对资源路径，但实际路径、hostfile 和凭据应保存在仓库外。
   `EMPTY` 只是无认证 SDK 占位；真实认证信息仅在运行时注入。

## 必填资源与 URL 格式

| 资源 | 配置 | 格式要求 |
| --- | --- | --- |
| CPT/SFT 模型 | `MODEL_PATH` | 完整本地 HF 文件；SFT native 权重另设 `MCORE_MODEL` |
| RL 模型/续训 | `MILES_HF_CHECKPOINT`、`MILES_LOAD`、`MILES_SAVE` | HF 元数据/权重、显式初始化或 native 恢复目录、输出目录 |
| 训练数据 | `DATASET_PATH` 或 `MILES_PROMPT_DATA` | JSONL；遵循 CPT 掩码或 RL Observation 数据契约 |
| Judge | `JUDGE_BASE_URL`、`JUDGE_MODEL` | 可达 `/v1` base 与准确的服务模型名 |
| Similarity | `SIMILARITY_URL` | 完整 `/similarity` endpoint |
| Embedding 后端 | `EMBEDDING_API_BASES` | 逗号分隔根地址，不带 `/v1`；模型名与副本一致 |
| Eval 远程推理 | `--rollout-base-url`、`--rollout-model-name` | 兼容 API base 和模型名，仍需本地 tokenizer |

## 阅读方式

下表保留源码表达式，不执行安装器来猜测最终值。`env(...)` / `_env(...)` 指链接源码中的 helper；
不同条件分支可能给出不同表达式。源码行号便于定位，变量存在不代表任意拓扑覆盖都安全。
调整 batch、CP、内存和服务前先阅读阶段指南。此处不填写任何真实凭据。

## 入口索引

- [cpt/dense_multinode_preflight.py](#entry-1)
- [cpt/prepare_loss_masks.py](#entry-2)
- [cpt/preprocess_cpt_cache.sh](#entry-3)
- [cpt/setup_runtime.sh](#entry-4)
- [cpt/train_cpt_megatron_dense.sh](#entry-5)
- [cpt/train_cpt_megatron_moe.sh](#entry-6)
- [cpt/train_cpt_swift.sh](#entry-7)
- [cpt/train_qwen35_27b_cpt_128gpu.sh](#entry-8)
- [cpt/train_qwen35_35b_a3b_cpt_128gpu.sh](#entry-9)
- [cpt/train_qwen35_4b_cpt_64gpu.sh](#entry-10)
- [cpt/train_qwen35_9b_cpt_64gpu.sh](#entry-11)
- [eval/convert_moe_for_vllm.py](#entry-12)
- [eval/eval_config.py](#entry-13)
- [eval/install_runtime.sh](#entry-14)
- [eval/judge_plus.py](#entry-15)
- [eval/run.sh](#entry-16)
- [eval/run_eval.py](#entry-17)
- [eval/run_streaming_eval_worker.py](#entry-18)
- [eval/summarize_rollout_eval.py](#entry-19)
- [rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh](#entry-20)
- [rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh](#entry-21)
- [rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_single.sh](#entry-22)
- [rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py](#entry-23)
- [rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash.sh](#entry-24)
- [rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_4node.sh](#entry-25)
- [rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh](#entry-26)
- [rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_node0.sh](#entry-27)
- [rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_node1.sh](#entry-28)
- [rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_node2.sh](#entry-29)
- [rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_node3.sh](#entry-30)
- [rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh](#entry-31)
- [rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh](#entry-32)
- [rl_miles/Qwen3-embedding/install_runtime.sh](#entry-33)
- [rl_miles/Qwen3-embedding/similarity_service.py](#entry-34)
- [rl_miles/Qwen3-embedding/start_qwen3_embedding_4replicas.sh](#entry-35)
- [rl_miles/Qwen3-embedding/start_similarity_services.sh](#entry-36)
- [rl_miles/Qwen3-embedding/stop_qwen3_embedding_4replicas.sh](#entry-37)
- [rl_miles/Qwen3-embedding/stop_similarity_services.sh](#entry-38)
- [rl_miles/bootstrap_miles.sh](#entry-39)
- [rl_miles/install_official_miles_env.sh](#entry-40)
- [rl_miles/official_env.sh](#entry-41)
- [rl_miles/prepare_rl_data.py](#entry-42)
- [rl_miles/run_gspo.sh](#entry-43)
- [rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh](#entry-44)
- [rl_miles/run_qwen35_27b_gspo_128gpu_async.sh](#entry-45)
- [rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh](#entry-46)
- [rl_miles/run_qwen35_4b_gspo_128gpu_async.sh](#entry-47)
- [rl_miles/run_qwen35_9b_gspo_128gpu_async.sh](#entry-48)
- [rl_miles/validate_runtime.py](#entry-49)
- [rl_miles/wheels/download_official_cu130_wheels.py](#entry-50)
- [rl_miles/world_model_rl/data_source.py](#entry-51)
- [rl_miles/world_model_rl/dataloader.py](#entry-52)
- [rl_miles/world_model_rl/eval_rollout.py](#entry-53)
- [rl_miles/world_model_rl/fully_async_rollout.py](#entry-54)
- [rl_miles/world_model_rl/jsonl_writer.py](#entry-55)
- [rl_miles/world_model_rl/reward.py](#entry-56)
- [runtime/prepare_sources.py](#entry-57)
- [runtime/validate_swift.py](#entry-58)
- [sft_code/setup_runtime.sh](#entry-59)
- [sft_code/train_qwen35_27b_sft_64gpu.sh](#entry-60)
- [sft_code/train_qwen35_35b_a3b_sft_32gpu.sh](#entry-61)
- [sft_code/train_qwen35_4b_sft_8gpu.sh](#entry-62)
- [sft_code/train_qwen35_9b_sft_8gpu.sh](#entry-63)
- [sft_code/train_sft.sh](#entry-64)
- [sft_code/validate_sft_runtime.py](#entry-65)

<a id="entry-1"></a>

## cpt/dense_multinode_preflight.py

[源码](../cpt/dense_multinode_preflight.py)


### CLI

| 参数 | 声明 | 位置 |
| --- | --- | --- |
| `--expected-world-size` | `type=int; default=None; help='Fail if torchrun reports a different global process count.'` | [L30](../cpt/dense_multinode_preflight.py#L30) |
| `--expected-local-world-size` | `type=int; default=None; help='Fail if torchrun reports a different local process count.'` | [L36](../cpt/dense_multinode_preflight.py#L36) |
| `--tensor-mib` | `type=int; default=16; help='GPU tensor size for the second all-reduce (default: 16 MiB).'` | [L42](../cpt/dense_multinode_preflight.py#L42) |
| `--timeout-sec` | `type=int; default=180; help='Process-group initialization and collective timeout.'` | [L48](../cpt/dense_multinode_preflight.py#L48) |

<a id="entry-2"></a>

## cpt/prepare_loss_masks.py

[源码](../cpt/prepare_loss_masks.py)


### CLI

| 参数 | 声明 | 位置 |
| --- | --- | --- |
| `--input` | `type=Path; required=True` | [L11](../cpt/prepare_loss_masks.py#L11) |
| `--output` | `type=Path; help='write canonical Swift messages; omit for validation only'` | [L12](../cpt/prepare_loss_masks.py#L12) |

<a id="entry-3"></a>

## cpt/preprocess_cpt_cache.sh

[源码](../cpt/preprocess_cpt_cache.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `LINENO` | `${LINENO}` | [L3](../cpt/preprocess_cpt_cache.sh#L3) |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L5](../cpt/preprocess_cpt_cache.sh#L5) |
| `PROJECT_ROOT` | `${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}` | [L6](../cpt/preprocess_cpt_cache.sh#L6) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L6](../cpt/preprocess_cpt_cache.sh#L6) |
| `CODE_ROOT` | `${CODE_ROOT:-${SCRIPT_DIR}}` | [L7](../cpt/preprocess_cpt_cache.sh#L7) |
| `CPT_ROOT` | `${CPT_ROOT:-${PROJECT_ROOT}/models}` | [L8](../cpt/preprocess_cpt_cache.sh#L8) |
| `PROJECT_ROOT` | `${PROJECT_ROOT}` | [L8](../cpt/preprocess_cpt_cache.sh#L8) |
| `CODE_ROOT` | `${CODE_ROOT}` | [L9](../cpt/preprocess_cpt_cache.sh#L9) |
| `SWIFT_ROOT` | `${SWIFT_ROOT:-${CODE_ROOT}/ms-swift}` | [L9](../cpt/preprocess_cpt_cache.sh#L9) |
| `SWIFT_VENV` | `${SWIFT_VENV:-${CODE_ROOT}/ms-swift-venv}` | [L10](../cpt/preprocess_cpt_cache.sh#L10) |
| `CACHE_ROOT` | `${CACHE_ROOT:-${PROJECT_ROOT}/cache/cpt}` | [L11](../cpt/preprocess_cpt_cache.sh#L11) |
| `CPT_ROOT` | `${CPT_ROOT}` | [L12](../cpt/preprocess_cpt_cache.sh#L12) |
| `MODEL_PATH` | `${MODEL_PATH:-${CPT_ROOT}/Qwen3.5-4B-Base}` | [L12](../cpt/preprocess_cpt_cache.sh#L12) |
| `DATASET_PATH` | `${DATASET_PATH:-${PROJECT_ROOT}/data/cpt_data*.jsonl}` | [L13](../cpt/preprocess_cpt_cache.sh#L13) |
| `CACHE_ROOT` | `${CACHE_ROOT}` | [L15](../cpt/preprocess_cpt_cache.sh#L15) |
| `OUTPUT_DIR` | `${OUTPUT_DIR:-${CACHE_ROOT}/qwen35_4b/cpt_data_cache}` | [L15](../cpt/preprocess_cpt_cache.sh#L15) |
| `MAX_LENGTH` | `${MAX_LENGTH:-32768}` | [L17](../cpt/preprocess_cpt_cache.sh#L17) |
| `TRUNCATION_STRATEGY` | `${TRUNCATION_STRATEGY:-left}` | [L18](../cpt/preprocess_cpt_cache.sh#L18) |
| `DATASET_NUM_PROC` | `${DATASET_NUM_PROC:-16}` | [L19](../cpt/preprocess_cpt_cache.sh#L19) |
| `LOAD_FROM_CACHE_FILE` | `${LOAD_FROM_CACHE_FILE:-true}` | [L20](../cpt/preprocess_cpt_cache.sh#L20) |
| `DATASET_SHUFFLE` | `${DATASET_SHUFFLE:-true}` | [L21](../cpt/preprocess_cpt_cache.sh#L21) |
| `ATTN_IMPL` | `${ATTN_IMPL:-flash_attn}` | [L22](../cpt/preprocess_cpt_cache.sh#L22) |
| `TRAIN_USER_LOSS` | `${TRAIN_USER_LOSS:-true}` | [L23](../cpt/preprocess_cpt_cache.sh#L23) |
| `TRAIN_USER_LOSS` | `${TRAIN_USER_LOSS}` | [L24](../cpt/preprocess_cpt_cache.sh#L24) |
| `SWIFT_TRAIN_USER_LOSS` | `export SWIFT_TRAIN_USER_LOSS=1` | [L25](../cpt/preprocess_cpt_cache.sh#L25) |
| `PYTHONPATH` | `${PYTHONPATH:+:${PYTHONPATH}}` | [L29](../cpt/preprocess_cpt_cache.sh#L29) |
| `PYTHONPATH` | `${PYTHONPATH}` | [L29](../cpt/preprocess_cpt_cache.sh#L29) |
| `SWIFT_ROOT` | `${SWIFT_ROOT}` | [L29](../cpt/preprocess_cpt_cache.sh#L29) |
| `DATASET_PATH` | `${DATASET_PATH}` | [L33](../cpt/preprocess_cpt_cache.sh#L33) |
| `OUTPUT_DIR` | `${OUTPUT_DIR}` | [L40](../cpt/preprocess_cpt_cache.sh#L40) |
| `SWIFT_VENV` | `${SWIFT_VENV}` | [L41](../cpt/preprocess_cpt_cache.sh#L41) |
| `HF_HUB_OFFLINE` | `export HF_HUB_OFFLINE=1` | [L48](../cpt/preprocess_cpt_cache.sh#L48) |
| `TRANSFORMERS_OFFLINE` | `export TRANSFORMERS_OFFLINE=1` | [L49](../cpt/preprocess_cpt_cache.sh#L49) |
| `HF_DATASETS_OFFLINE` | `export HF_DATASETS_OFFLINE=1` | [L50](../cpt/preprocess_cpt_cache.sh#L50) |
| `MODELSCOPE_OFFLINE` | `export MODELSCOPE_OFFLINE=1` | [L51](../cpt/preprocess_cpt_cache.sh#L51) |
| `TOKENIZERS_PARALLELISM` | `${TOKENIZERS_PARALLELISM:-false}` | [L58](../cpt/preprocess_cpt_cache.sh#L58) |
| `IMAGE_MAX_TOKEN_NUM` | `${IMAGE_MAX_TOKEN_NUM:-1024}` | [L59](../cpt/preprocess_cpt_cache.sh#L59) |
| `VIDEO_MAX_TOKEN_NUM` | `${VIDEO_MAX_TOKEN_NUM:-128}` | [L60](../cpt/preprocess_cpt_cache.sh#L60) |
| `FPS_MAX_FRAMES` | `${FPS_MAX_FRAMES:-12}` | [L61](../cpt/preprocess_cpt_cache.sh#L61) |
| `MODEL_PATH` | `${MODEL_PATH}` | [L64](../cpt/preprocess_cpt_cache.sh#L64) |
| `MAX_LENGTH` | `${MAX_LENGTH}` | [L68](../cpt/preprocess_cpt_cache.sh#L68) |
| `TRUNCATION_STRATEGY` | `${TRUNCATION_STRATEGY}` | [L69](../cpt/preprocess_cpt_cache.sh#L69) |
| `DATASET_NUM_PROC` | `${DATASET_NUM_PROC}` | [L70](../cpt/preprocess_cpt_cache.sh#L70) |
| `DATASET_SHUFFLE` | `${DATASET_SHUFFLE}` | [L71](../cpt/preprocess_cpt_cache.sh#L71) |
| `ATTN_IMPL` | `${ATTN_IMPL}` | [L72](../cpt/preprocess_cpt_cache.sh#L72) |
| `SWIFT_TRAIN_USER_LOSS` | `${SWIFT_TRAIN_USER_LOSS:-0}` | [L76](../cpt/preprocess_cpt_cache.sh#L76) |
| `LOSS_CHECK_ARGS` | `${LOSS_CHECK_ARGS[@]}` | [L77](../cpt/preprocess_cpt_cache.sh#L77) |
| `DATASET_FILES` | `${DATASET_FILES[@]}` | [L82](../cpt/preprocess_cpt_cache.sh#L82) |
| `LOAD_FROM_CACHE_FILE` | `${LOAD_FROM_CACHE_FILE}` | [L90](../cpt/preprocess_cpt_cache.sh#L90) |

<a id="entry-4"></a>

## cpt/setup_runtime.sh

[源码](../cpt/setup_runtime.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L29](../cpt/setup_runtime.sh#L29) |
| `PROFILE` | `${PROFILE}` | [L31](../cpt/setup_runtime.sh#L31) |
| `PYTHON_BASE` | `${PYTHON_BASE:-python3.12}` | [L34](../cpt/setup_runtime.sh#L34) |
| `ROOT` | `${ROOT}` | [L35](../cpt/setup_runtime.sh#L35) |
| `SWIFT_ROOT` | `${SWIFT_ROOT:-${ROOT}/ms-swift}` | [L35](../cpt/setup_runtime.sh#L35) |
| `SWIFT_VENV` | `${SWIFT_VENV:-${ROOT}/ms-swift-venv}` | [L36](../cpt/setup_runtime.sh#L36) |
| `SFT_VENV` | `${SFT_VENV:-${ROOT}/../sft_code/.venv}` | [L37](../cpt/setup_runtime.sh#L37) |
| `MCORE_BRIDGE_ROOT` | `${MCORE_BRIDGE_ROOT:-${ROOT}/mcore-bridge}` | [L38](../cpt/setup_runtime.sh#L38) |
| `PYTHON_BASE` | `${PYTHON_BASE}` | [L44](../cpt/setup_runtime.sh#L44) |
| `PREPARE` | `${PREPARE}` | [L52](../cpt/setup_runtime.sh#L52) |
| `SWIFT_ROOT` | `${SWIFT_ROOT}` | [L52](../cpt/setup_runtime.sh#L52) |
| `MCORE_BRIDGE_ROOT` | `${MCORE_BRIDGE_ROOT}` | [L53](../cpt/setup_runtime.sh#L53) |
| `SWIFT_VENV` | `${SWIFT_VENV}` | [L54](../cpt/setup_runtime.sh#L54) |
| `PY` | `${PY}` | [L56](../cpt/setup_runtime.sh#L56) |
| `CAUSAL_CONV1D_WHEEL` | `${CAUSAL_CONV1D_WHEEL:-}` | [L61](../cpt/setup_runtime.sh#L61) |
| `CAUSAL_CONV1D_WHEEL` | `${CAUSAL_CONV1D_WHEEL}` | [L62](../cpt/setup_runtime.sh#L62) |
| `MEGATRON_ROOT` | `${MEGATRON_ROOT:-${ROOT}/../rl_miles/Megatron-LM}` | [L67](../cpt/setup_runtime.sh#L67) |
| `MEGATRON_ROOT` | `${MEGATRON_ROOT}` | [L68](../cpt/setup_runtime.sh#L68) |
| `PYTHONPATH` | `${PYTHONPATH:+:${PYTHONPATH}}` | [L73](../cpt/setup_runtime.sh#L73) |
| `PYTHONPATH` | `${PYTHONPATH}` | [L73](../cpt/setup_runtime.sh#L73) |

<a id="entry-5"></a>

## cpt/train_cpt_megatron_dense.sh

[源码](../cpt/train_cpt_megatron_dense.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L4](../cpt/train_cpt_megatron_dense.sh#L4) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L5](../cpt/train_cpt_megatron_dense.sh#L5) |

<a id="entry-6"></a>

## cpt/train_cpt_megatron_moe.sh

[源码](../cpt/train_cpt_megatron_moe.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L4](../cpt/train_cpt_megatron_moe.sh#L4) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L5](../cpt/train_cpt_megatron_moe.sh#L5) |

<a id="entry-7"></a>

## cpt/train_cpt_swift.sh

[源码](../cpt/train_cpt_swift.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L4](../cpt/train_cpt_swift.sh#L4) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L5](../cpt/train_cpt_swift.sh#L5) |

<a id="entry-8"></a>

## cpt/train_qwen35_27b_cpt_128gpu.sh

[源码](../cpt/train_qwen35_27b_cpt_128gpu.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L17](../cpt/train_qwen35_27b_cpt_128gpu.sh#L17) |
| `PROJECT_ROOT` | `${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}` | [L18](../cpt/train_qwen35_27b_cpt_128gpu.sh#L18) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L18](../cpt/train_qwen35_27b_cpt_128gpu.sh#L18) |
| `CODE_ROOT` | `${CODE_ROOT:-${SCRIPT_DIR}}` | [L19](../cpt/train_qwen35_27b_cpt_128gpu.sh#L19) |
| `CPT_ROOT` | `${CPT_ROOT:-${PROJECT_ROOT}/models}` | [L20](../cpt/train_qwen35_27b_cpt_128gpu.sh#L20) |
| `PROJECT_ROOT` | `${PROJECT_ROOT}` | [L20](../cpt/train_qwen35_27b_cpt_128gpu.sh#L20) |
| `CODE_ROOT` | `${CODE_ROOT}` | [L21](../cpt/train_qwen35_27b_cpt_128gpu.sh#L21) |
| `SWIFT_ROOT` | `${SWIFT_ROOT:-${CODE_ROOT}/ms-swift}` | [L21](../cpt/train_qwen35_27b_cpt_128gpu.sh#L21) |
| `SWIFT_VENV` | `${SWIFT_VENV:-${CODE_ROOT}/ms-swift-venv}` | [L22](../cpt/train_qwen35_27b_cpt_128gpu.sh#L22) |
| `CPT_ROOT` | `${CPT_ROOT}` | [L26](../cpt/train_qwen35_27b_cpt_128gpu.sh#L26) |
| `MODEL_PATH` | `${MODEL_PATH:-${CPT_ROOT}/Qwen3.5-27B}` | [L26](../cpt/train_qwen35_27b_cpt_128gpu.sh#L26) |
| `DATASET_PATH` | `${DATASET_PATH:-${PROJECT_ROOT}/data/cpt_data*.jsonl}` | [L27](../cpt/train_qwen35_27b_cpt_128gpu.sh#L27) |
| `OUTPUT_DIR` | `${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/cpt/qwen35_27b}` | [L28](../cpt/train_qwen35_27b_cpt_128gpu.sh#L28) |
| `CACHE_ROOT` | `${CACHE_ROOT:-${PROJECT_ROOT}/cache/cpt}` | [L29](../cpt/train_qwen35_27b_cpt_128gpu.sh#L29) |
| `CACHED_DATASET_ROOT` | `${CACHED_DATASET_ROOT:-${CACHE_ROOT}/qwen35_27b/cpt_data_cache}` | [L30](../cpt/train_qwen35_27b_cpt_128gpu.sh#L30) |
| `CACHE_ROOT` | `${CACHE_ROOT}` | [L30](../cpt/train_qwen35_27b_cpt_128gpu.sh#L30) |
| `CACHED_DATASET` | `${CACHED_DATASET:-${CACHED_DATASET_ROOT}/train}` | [L31](../cpt/train_qwen35_27b_cpt_128gpu.sh#L31) |
| `CACHED_DATASET_ROOT` | `${CACHED_DATASET_ROOT}` | [L31](../cpt/train_qwen35_27b_cpt_128gpu.sh#L31) |
| `USE_CACHED_DATASET` | `${USE_CACHED_DATASET:-true}` | [L32](../cpt/train_qwen35_27b_cpt_128gpu.sh#L32) |
| `NPROC_PER_NODE` | `${NPROC_PER_NODE:-8}` | [L37](../cpt/train_qwen35_27b_cpt_128gpu.sh#L37) |
| `RESOURCE_GPU` | `${RESOURCE_GPU:-${NPROC_PER_NODE:-8}}` | [L37](../cpt/train_qwen35_27b_cpt_128gpu.sh#L37) |
| `WORLD_SIZE` | `${WORLD_SIZE:-}` | [L38](../cpt/train_qwen35_27b_cpt_128gpu.sh#L38) |
| `NNODES` | `${NNODES:-}` | [L39](../cpt/train_qwen35_27b_cpt_128gpu.sh#L39) |
| `NNODES` | `${NNODES}` | [L39](../cpt/train_qwen35_27b_cpt_128gpu.sh#L39) |
| `WORLD_SIZE` | `${WORLD_SIZE}` | [L39](../cpt/train_qwen35_27b_cpt_128gpu.sh#L39) |
| `RANK` | `${RANK:-}` | [L50](../cpt/train_qwen35_27b_cpt_128gpu.sh#L50) |
| `NODE_RANK` | `${NODE_RANK:-}` | [L51](../cpt/train_qwen35_27b_cpt_128gpu.sh#L51) |
| `NODE_RANK` | `${NODE_RANK}` | [L51](../cpt/train_qwen35_27b_cpt_128gpu.sh#L51) |
| `RANK` | `${RANK}` | [L51](../cpt/train_qwen35_27b_cpt_128gpu.sh#L51) |
| `NPROC_PER_NODE` | `${NPROC_PER_NODE:-${RESOURCE_GPU}}` | [L62](../cpt/train_qwen35_27b_cpt_128gpu.sh#L62) |
| `RESOURCE_GPU` | `${RESOURCE_GPU}` | [L62](../cpt/train_qwen35_27b_cpt_128gpu.sh#L62) |
| `MASTER_ADDR` | `${MASTER_ADDR:-}` | [L63](../cpt/train_qwen35_27b_cpt_128gpu.sh#L63) |
| `MASTER_PORT` | `${MASTER_PORT:-29500}` | [L64](../cpt/train_qwen35_27b_cpt_128gpu.sh#L64) |
| `CUDA_VISIBLE_DEVICES` | `${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}` | [L65](../cpt/train_qwen35_27b_cpt_128gpu.sh#L65) |
| `MAX_LENGTH` | `${MAX_LENGTH:-32768}` | [L75](../cpt/train_qwen35_27b_cpt_128gpu.sh#L75) |
| `TRUNCATION_STRATEGY` | `${TRUNCATION_STRATEGY:-left}` | [L76](../cpt/train_qwen35_27b_cpt_128gpu.sh#L76) |
| `NUM_TRAIN_EPOCHS` | `${NUM_TRAIN_EPOCHS:-1}` | [L77](../cpt/train_qwen35_27b_cpt_128gpu.sh#L77) |
| `TRAIN_ITERS` | `${TRAIN_ITERS-}` | [L80](../cpt/train_qwen35_27b_cpt_128gpu.sh#L80) |
| `MICRO_BATCH_SIZE` | `${MICRO_BATCH_SIZE:-1}` | [L81](../cpt/train_qwen35_27b_cpt_128gpu.sh#L81) |
| `GLOBAL_BATCH_SIZE` | `${GLOBAL_BATCH_SIZE:-1024}` | [L82](../cpt/train_qwen35_27b_cpt_128gpu.sh#L82) |
| `NPROC_PER_NODE` | `${NPROC_PER_NODE}` | [L83](../cpt/train_qwen35_27b_cpt_128gpu.sh#L83) |
| `TENSOR_MODEL_PARALLEL_SIZE` | `${TENSOR_MODEL_PARALLEL_SIZE:-${NPROC_PER_NODE}}` | [L83](../cpt/train_qwen35_27b_cpt_128gpu.sh#L83) |
| `PIPELINE_MODEL_PARALLEL_SIZE` | `${PIPELINE_MODEL_PARALLEL_SIZE:-1}` | [L84](../cpt/train_qwen35_27b_cpt_128gpu.sh#L84) |
| `CONTEXT_PARALLEL_SIZE` | `${CONTEXT_PARALLEL_SIZE:-1}` | [L85](../cpt/train_qwen35_27b_cpt_128gpu.sh#L85) |
| `DATASET_NUM_PROC` | `${DATASET_NUM_PROC:-8}` | [L86](../cpt/train_qwen35_27b_cpt_128gpu.sh#L86) |
| `DATALOADER_NUM_WORKERS` | `${DATALOADER_NUM_WORKERS:-4}` | [L87](../cpt/train_qwen35_27b_cpt_128gpu.sh#L87) |
| `LEARNING_RATE` | `${LEARNING_RATE:-2e-5}` | [L88](../cpt/train_qwen35_27b_cpt_128gpu.sh#L88) |
| `MIN_LR` | `${MIN_LR:-2e-6}` | [L89](../cpt/train_qwen35_27b_cpt_128gpu.sh#L89) |
| `LR_WARMUP_FRACTION` | `${LR_WARMUP_FRACTION:-0.03}` | [L90](../cpt/train_qwen35_27b_cpt_128gpu.sh#L90) |
| `SAVE_STEPS` | `${SAVE_STEPS:-500}` | [L91](../cpt/train_qwen35_27b_cpt_128gpu.sh#L91) |
| `LOGGING_STEPS` | `${LOGGING_STEPS:-1}` | [L92](../cpt/train_qwen35_27b_cpt_128gpu.sh#L92) |
| `TRAIN_USER_LOSS` | `${TRAIN_USER_LOSS:-true}` | [L93](../cpt/train_qwen35_27b_cpt_128gpu.sh#L93) |
| `OPTIMIZER_CPU_OFFLOAD` | `${OPTIMIZER_CPU_OFFLOAD:-false}` | [L94](../cpt/train_qwen35_27b_cpt_128gpu.sh#L94) |
| `OPTIMIZER_OFFLOAD_FRACTION` | `${OPTIMIZER_OFFLOAD_FRACTION:-1}` | [L95](../cpt/train_qwen35_27b_cpt_128gpu.sh#L95) |
| `USE_PRECISION_AWARE_OPTIMIZER` | `${USE_PRECISION_AWARE_OPTIMIZER:-true}` | [L96](../cpt/train_qwen35_27b_cpt_128gpu.sh#L96) |
| `MAIN_GRADS_DTYPE` | `${MAIN_GRADS_DTYPE:-bf16}` | [L97](../cpt/train_qwen35_27b_cpt_128gpu.sh#L97) |
| `MAIN_PARAMS_DTYPE` | `${MAIN_PARAMS_DTYPE:-fp32}` | [L98](../cpt/train_qwen35_27b_cpt_128gpu.sh#L98) |
| `EXP_AVG_DTYPE` | `${EXP_AVG_DTYPE:-fp32}` | [L99](../cpt/train_qwen35_27b_cpt_128gpu.sh#L99) |
| `EXP_AVG_SQ_DTYPE` | `${EXP_AVG_SQ_DTYPE:-fp32}` | [L100](../cpt/train_qwen35_27b_cpt_128gpu.sh#L100) |
| `USE_CPU_INITIALIZATION` | `${USE_CPU_INITIALIZATION:-true}` | [L101](../cpt/train_qwen35_27b_cpt_128gpu.sh#L101) |
| `LOAD_FROM_CACHE_FILE` | `${LOAD_FROM_CACHE_FILE:-false}` | [L102](../cpt/train_qwen35_27b_cpt_128gpu.sh#L102) |
| `NO_SAVE_OPTIM` | `${NO_SAVE_OPTIM:-false}` | [L103](../cpt/train_qwen35_27b_cpt_128gpu.sh#L103) |
| `NO_SAVE_RNG` | `${NO_SAVE_RNG:-false}` | [L104](../cpt/train_qwen35_27b_cpt_128gpu.sh#L104) |
| `DRY_RUN` | `${DRY_RUN:-false}` | [L105](../cpt/train_qwen35_27b_cpt_128gpu.sh#L105) |
| `NCCL_PREFLIGHT_ONLY` | `${NCCL_PREFLIGHT_ONLY:-false}` | [L106](../cpt/train_qwen35_27b_cpt_128gpu.sh#L106) |
| `NCCL_PREFLIGHT_TENSOR_MIB` | `${NCCL_PREFLIGHT_TENSOR_MIB:-16}` | [L107](../cpt/train_qwen35_27b_cpt_128gpu.sh#L107) |
| `NCCL_PREFLIGHT_TIMEOUT_SEC` | `${NCCL_PREFLIGHT_TIMEOUT_SEC:-180}` | [L108](../cpt/train_qwen35_27b_cpt_128gpu.sh#L108) |
| `ALLOW_TOPOLOGY_OVERRIDE` | `${ALLOW_TOPOLOGY_OVERRIDE:-false}` | [L109](../cpt/train_qwen35_27b_cpt_128gpu.sh#L109) |
| `EXPECTED_NNODES` | `${EXPECTED_NNODES:-16}` | [L110](../cpt/train_qwen35_27b_cpt_128gpu.sh#L110) |
| `EXPECTED_GPUS_PER_NODE` | `${EXPECTED_GPUS_PER_NODE:-8}` | [L111](../cpt/train_qwen35_27b_cpt_128gpu.sh#L111) |
| `LOCAL_CACHE_BASE` | `${LOCAL_CACHE_BASE:-/tmp/world_model_cpt}` | [L112](../cpt/train_qwen35_27b_cpt_128gpu.sh#L112) |
| `LOCAL_CACHE_BASE` | `${LOCAL_CACHE_BASE}` | [L113](../cpt/train_qwen35_27b_cpt_128gpu.sh#L113) |
| `LOCAL_TMP_ROOT` | `${LOCAL_TMP_ROOT:-${LOCAL_CACHE_BASE}/qwen35_27b_rank${NODE_RANK}}` | [L113](../cpt/train_qwen35_27b_cpt_128gpu.sh#L113) |
| `SHORT_TMPDIR` | `${SHORT_TMPDIR:-/tmp/q27b_r${NODE_RANK}}` | [L114](../cpt/train_qwen35_27b_cpt_128gpu.sh#L114) |
| `EXPECTED_NNODES` | `${EXPECTED_NNODES}` | [L128](../cpt/train_qwen35_27b_cpt_128gpu.sh#L128) |
| `EXPECTED_GPUS_PER_NODE` | `${EXPECTED_GPUS_PER_NODE}` | [L129](../cpt/train_qwen35_27b_cpt_128gpu.sh#L129) |
| `TENSOR_MODEL_PARALLEL_SIZE` | `${TENSOR_MODEL_PARALLEL_SIZE}` | [L130](../cpt/train_qwen35_27b_cpt_128gpu.sh#L130) |
| `PIPELINE_MODEL_PARALLEL_SIZE` | `${PIPELINE_MODEL_PARALLEL_SIZE}` | [L131](../cpt/train_qwen35_27b_cpt_128gpu.sh#L131) |
| `CONTEXT_PARALLEL_SIZE` | `${CONTEXT_PARALLEL_SIZE}` | [L132](../cpt/train_qwen35_27b_cpt_128gpu.sh#L132) |
| `MASTER_PORT` | `${MASTER_PORT}` | [L144](../cpt/train_qwen35_27b_cpt_128gpu.sh#L144) |
| `MASTER_ADDR` | `${MASTER_ADDR}` | [L157](../cpt/train_qwen35_27b_cpt_128gpu.sh#L157) |
| `ALLOW_TOPOLOGY_OVERRIDE` | `${ALLOW_TOPOLOGY_OVERRIDE}` | [L170](../cpt/train_qwen35_27b_cpt_128gpu.sh#L170) |
| `CUDA_VISIBLE_DEVICES` | `${CUDA_VISIBLE_DEVICES}` | [L185](../cpt/train_qwen35_27b_cpt_128gpu.sh#L185) |
| `MODEL_PARALLEL_SIZE` | `${MODEL_PARALLEL_SIZE}` | [L196](../cpt/train_qwen35_27b_cpt_128gpu.sh#L196) |
| `WORLD_SIZE_TOTAL` | `${WORLD_SIZE_TOTAL}` | [L196](../cpt/train_qwen35_27b_cpt_128gpu.sh#L196) |
| `GLOBAL_BATCH_SIZE` | `${GLOBAL_BATCH_SIZE}` | [L200](../cpt/train_qwen35_27b_cpt_128gpu.sh#L200) |
| `MICRO_BATCH_SIZE` | `${MICRO_BATCH_SIZE}` | [L200](../cpt/train_qwen35_27b_cpt_128gpu.sh#L200) |
| `STEP_BATCH_SIZE` | `${STEP_BATCH_SIZE}` | [L206](../cpt/train_qwen35_27b_cpt_128gpu.sh#L206) |
| `DATASET_PATH` | `${DATASET_PATH}` | [L211](../cpt/train_qwen35_27b_cpt_128gpu.sh#L211) |
| `DRY_RUN` | `${DRY_RUN}` | [L212](../cpt/train_qwen35_27b_cpt_128gpu.sh#L212) |
| `SWIFT_ROOT` | `${SWIFT_ROOT}` | [L217](../cpt/train_qwen35_27b_cpt_128gpu.sh#L217) |
| `SWIFT_VENV` | `${SWIFT_VENV}` | [L221](../cpt/train_qwen35_27b_cpt_128gpu.sh#L221) |
| `MODEL_PATH` | `${MODEL_PATH}` | [L225](../cpt/train_qwen35_27b_cpt_128gpu.sh#L225) |
| `USE_CACHED_DATASET` | `${USE_CACHED_DATASET}` | [L232](../cpt/train_qwen35_27b_cpt_128gpu.sh#L232) |
| `CACHED_DATASET` | `${CACHED_DATASET}` | [L233](../cpt/train_qwen35_27b_cpt_128gpu.sh#L233) |
| `OUTPUT_DIR` | `${OUTPUT_DIR}` | [L315](../cpt/train_qwen35_27b_cpt_128gpu.sh#L315) |
| `LOCAL_TMP_ROOT` | `${LOCAL_TMP_ROOT}` | [L316](../cpt/train_qwen35_27b_cpt_128gpu.sh#L316) |
| `SHORT_TMPDIR` | `${SHORT_TMPDIR}` | [L317](../cpt/train_qwen35_27b_cpt_128gpu.sh#L317) |
| `PATH` | `${PATH}` | [L320](../cpt/train_qwen35_27b_cpt_128gpu.sh#L320) |
| `PYTHONPATH` | `${PYTHONPATH:+:${PYTHONPATH}}` | [L322](../cpt/train_qwen35_27b_cpt_128gpu.sh#L322) |
| `PYTHONPATH` | `${PYTHONPATH}` | [L322](../cpt/train_qwen35_27b_cpt_128gpu.sh#L322) |
| `NCCL_PREFLIGHT_ONLY` | `${NCCL_PREFLIGHT_ONLY}` | [L397](../cpt/train_qwen35_27b_cpt_128gpu.sh#L397) |
| `NCCL_PREFLIGHT_TENSOR_MIB` | `${NCCL_PREFLIGHT_TENSOR_MIB}` | [L420](../cpt/train_qwen35_27b_cpt_128gpu.sh#L420) |
| `NCCL_PREFLIGHT_TIMEOUT_SEC` | `${NCCL_PREFLIGHT_TIMEOUT_SEC}` | [L421](../cpt/train_qwen35_27b_cpt_128gpu.sh#L421) |
| `HF_HUB_OFFLINE` | `export HF_HUB_OFFLINE=1` | [L429](../cpt/train_qwen35_27b_cpt_128gpu.sh#L429) |
| `TRANSFORMERS_OFFLINE` | `export TRANSFORMERS_OFFLINE=1` | [L430](../cpt/train_qwen35_27b_cpt_128gpu.sh#L430) |
| `HF_DATASETS_OFFLINE` | `export HF_DATASETS_OFFLINE=1` | [L431](../cpt/train_qwen35_27b_cpt_128gpu.sh#L431) |
| `MODELSCOPE_OFFLINE` | `export MODELSCOPE_OFFLINE=1` | [L432](../cpt/train_qwen35_27b_cpt_128gpu.sh#L432) |
| `PYTORCH_CUDA_ALLOC_CONF` | `${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}` | [L444](../cpt/train_qwen35_27b_cpt_128gpu.sh#L444) |
| `PYTORCH_ALLOC_CONF` | `${PYTORCH_ALLOC_CONF:-expandable_segments:True}` | [L445](../cpt/train_qwen35_27b_cpt_128gpu.sh#L445) |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `${CUDA_DEVICE_MAX_CONNECTIONS:-1}` | [L446](../cpt/train_qwen35_27b_cpt_128gpu.sh#L446) |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | `${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}` | [L447](../cpt/train_qwen35_27b_cpt_128gpu.sh#L447) |
| `TORCH_NCCL_ENABLE_MONITORING` | `${TORCH_NCCL_ENABLE_MONITORING:-0}` | [L448](../cpt/train_qwen35_27b_cpt_128gpu.sh#L448) |
| `TORCH_NCCL_BLOCKING_WAIT` | `${TORCH_NCCL_BLOCKING_WAIT:-0}` | [L449](../cpt/train_qwen35_27b_cpt_128gpu.sh#L449) |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING` | `${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}` | [L450](../cpt/train_qwen35_27b_cpt_128gpu.sh#L450) |
| `NCCL_DEBUG` | `${NCCL_DEBUG:-WARN}` | [L451](../cpt/train_qwen35_27b_cpt_128gpu.sh#L451) |
| `TOKENIZERS_PARALLELISM` | `${TOKENIZERS_PARALLELISM:-false}` | [L452](../cpt/train_qwen35_27b_cpt_128gpu.sh#L452) |
| `OMP_NUM_THREADS` | `${OMP_NUM_THREADS:-1}` | [L453](../cpt/train_qwen35_27b_cpt_128gpu.sh#L453) |
| `IMAGE_MAX_TOKEN_NUM` | `${IMAGE_MAX_TOKEN_NUM:-1024}` | [L455](../cpt/train_qwen35_27b_cpt_128gpu.sh#L455) |
| `VIDEO_MAX_TOKEN_NUM` | `${VIDEO_MAX_TOKEN_NUM:-128}` | [L456](../cpt/train_qwen35_27b_cpt_128gpu.sh#L456) |
| `FPS_MAX_FRAMES` | `${FPS_MAX_FRAMES:-12}` | [L457](../cpt/train_qwen35_27b_cpt_128gpu.sh#L457) |
| `USE_MCORE_GDN` | `${USE_MCORE_GDN:-1}` | [L458](../cpt/train_qwen35_27b_cpt_128gpu.sh#L458) |
| `FLA_TILELANG` | `${FLA_TILELANG:-1}` | [L459](../cpt/train_qwen35_27b_cpt_128gpu.sh#L459) |
| `TRAIN_USER_LOSS` | `${TRAIN_USER_LOSS}` | [L460](../cpt/train_qwen35_27b_cpt_128gpu.sh#L460) |
| `SWIFT_TRAIN_USER_LOSS` | `export SWIFT_TRAIN_USER_LOSS=1` | [L461](../cpt/train_qwen35_27b_cpt_128gpu.sh#L461) |
| `DRY_RUN` | `${DRY_RUN:-0}` | [L465](../cpt/train_qwen35_27b_cpt_128gpu.sh#L465) |
| `SWIFT_TRAIN_USER_LOSS` | `${SWIFT_TRAIN_USER_LOSS:-0}` | [L467](../cpt/train_qwen35_27b_cpt_128gpu.sh#L467) |
| `LOSS_CHECK_ARGS` | `${LOSS_CHECK_ARGS[@]}` | [L469](../cpt/train_qwen35_27b_cpt_128gpu.sh#L469) |
| `LOAD_FROM_CACHE_FILE` | `${LOAD_FROM_CACHE_FILE}` | [L476](../cpt/train_qwen35_27b_cpt_128gpu.sh#L476) |
| `USE_CPU_INITIALIZATION` | `${USE_CPU_INITIALIZATION}` | [L491](../cpt/train_qwen35_27b_cpt_128gpu.sh#L491) |
| `LEARNING_RATE` | `${LEARNING_RATE}` | [L495](../cpt/train_qwen35_27b_cpt_128gpu.sh#L495) |
| `LR_WARMUP_FRACTION` | `${LR_WARMUP_FRACTION}` | [L496](../cpt/train_qwen35_27b_cpt_128gpu.sh#L496) |
| `MIN_LR` | `${MIN_LR}` | [L497](../cpt/train_qwen35_27b_cpt_128gpu.sh#L497) |
| `SAVE_STEPS` | `${SAVE_STEPS}` | [L499](../cpt/train_qwen35_27b_cpt_128gpu.sh#L499) |
| `LOGGING_STEPS` | `${LOGGING_STEPS}` | [L500](../cpt/train_qwen35_27b_cpt_128gpu.sh#L500) |
| `MAX_LENGTH` | `${MAX_LENGTH}` | [L501](../cpt/train_qwen35_27b_cpt_128gpu.sh#L501) |
| `TRUNCATION_STRATEGY` | `${TRUNCATION_STRATEGY}` | [L502](../cpt/train_qwen35_27b_cpt_128gpu.sh#L502) |
| `DATALOADER_NUM_WORKERS` | `${DATALOADER_NUM_WORKERS}` | [L505](../cpt/train_qwen35_27b_cpt_128gpu.sh#L505) |
| `DATASET_NUM_PROC` | `${DATASET_NUM_PROC}` | [L506](../cpt/train_qwen35_27b_cpt_128gpu.sh#L506) |
| `NO_SAVE_OPTIM` | `${NO_SAVE_OPTIM}` | [L507](../cpt/train_qwen35_27b_cpt_128gpu.sh#L507) |
| `NO_SAVE_RNG` | `${NO_SAVE_RNG}` | [L508](../cpt/train_qwen35_27b_cpt_128gpu.sh#L508) |
| `USE_PRECISION_AWARE_OPTIMIZER` | `${USE_PRECISION_AWARE_OPTIMIZER}` | [L509](../cpt/train_qwen35_27b_cpt_128gpu.sh#L509) |
| `MAIN_GRADS_DTYPE` | `${MAIN_GRADS_DTYPE}` | [L510](../cpt/train_qwen35_27b_cpt_128gpu.sh#L510) |
| `MAIN_PARAMS_DTYPE` | `${MAIN_PARAMS_DTYPE}` | [L511](../cpt/train_qwen35_27b_cpt_128gpu.sh#L511) |
| `EXP_AVG_DTYPE` | `${EXP_AVG_DTYPE}` | [L512](../cpt/train_qwen35_27b_cpt_128gpu.sh#L512) |
| `EXP_AVG_SQ_DTYPE` | `${EXP_AVG_SQ_DTYPE}` | [L513](../cpt/train_qwen35_27b_cpt_128gpu.sh#L513) |
| `OPTIMIZER_CPU_OFFLOAD` | `${OPTIMIZER_CPU_OFFLOAD}` | [L514](../cpt/train_qwen35_27b_cpt_128gpu.sh#L514) |
| `OPTIMIZER_OFFLOAD_FRACTION` | `${OPTIMIZER_OFFLOAD_FRACTION}` | [L515](../cpt/train_qwen35_27b_cpt_128gpu.sh#L515) |
| `DATASET_FILES` | `${DATASET_FILES[@]}` | [L522](../cpt/train_qwen35_27b_cpt_128gpu.sh#L522) |
| `TRAIN_ITERS` | `${TRAIN_ITERS}` | [L525](../cpt/train_qwen35_27b_cpt_128gpu.sh#L525) |
| `NUM_TRAIN_EPOCHS` | `${NUM_TRAIN_EPOCHS}` | [L528](../cpt/train_qwen35_27b_cpt_128gpu.sh#L528) |
| `DATA_PARALLEL_SIZE` | `${DATA_PARALLEL_SIZE}` | [L543](../cpt/train_qwen35_27b_cpt_128gpu.sh#L543) |
| `GRAD_ACCUM_MICROBATCHES` | `${GRAD_ACCUM_MICROBATCHES}` | [L548](../cpt/train_qwen35_27b_cpt_128gpu.sh#L548) |
| `TRAIN_ITERS` | `${TRAIN_ITERS:-auto}` | [L549](../cpt/train_qwen35_27b_cpt_128gpu.sh#L549) |
| `NCCL_SOCKET_IFNAME` | `${NCCL_SOCKET_IFNAME:-platform-default}` | [L551](../cpt/train_qwen35_27b_cpt_128gpu.sh#L551) |
| `NCCL_IB_HCA` | `${NCCL_IB_HCA:-platform-default}` | [L552](../cpt/train_qwen35_27b_cpt_128gpu.sh#L552) |
| `FLA_TILELANG` | `${FLA_TILELANG}` | [L553](../cpt/train_qwen35_27b_cpt_128gpu.sh#L553) |
| `TMPDIR` | `${TMPDIR}` | [L555](../cpt/train_qwen35_27b_cpt_128gpu.sh#L555) |
| `HF_HOME` | `${HF_HOME}` | [L556](../cpt/train_qwen35_27b_cpt_128gpu.sh#L556) |
| `MODELSCOPE_CACHE` | `${MODELSCOPE_CACHE}` | [L557](../cpt/train_qwen35_27b_cpt_128gpu.sh#L557) |
| `ARGS` | `${ARGS[@]}` | [L561](../cpt/train_qwen35_27b_cpt_128gpu.sh#L561) |

<a id="entry-9"></a>

## cpt/train_qwen35_35b_a3b_cpt_128gpu.sh

[源码](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L17](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L17) |
| `PROJECT_ROOT` | `${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}` | [L18](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L18) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L18](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L18) |
| `CODE_ROOT` | `${CODE_ROOT:-${SCRIPT_DIR}}` | [L19](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L19) |
| `CPT_ROOT` | `${CPT_ROOT:-${PROJECT_ROOT}/models}` | [L20](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L20) |
| `PROJECT_ROOT` | `${PROJECT_ROOT}` | [L20](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L20) |
| `CODE_ROOT` | `${CODE_ROOT}` | [L21](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L21) |
| `SWIFT_ROOT` | `${SWIFT_ROOT:-${CODE_ROOT}/ms-swift}` | [L21](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L21) |
| `SWIFT_VENV` | `${SWIFT_VENV:-${CODE_ROOT}/ms-swift-venv}` | [L22](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L22) |
| `CPT_ROOT` | `${CPT_ROOT}` | [L26](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L26) |
| `MODEL_PATH` | `${MODEL_PATH:-${CPT_ROOT}/Qwen3.5-35B-A3B-Base}` | [L26](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L26) |
| `DATASET_PATH` | `${DATASET_PATH:-${PROJECT_ROOT}/data/cpt_data*.jsonl}` | [L27](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L27) |
| `OUTPUT_DIR` | `${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/cpt/qwen35_35b_a3b}` | [L28](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L28) |
| `CACHE_ROOT` | `${CACHE_ROOT:-${PROJECT_ROOT}/cache/cpt}` | [L29](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L29) |
| `LOCAL_RANK` | `${LOCAL_RANK:-}` | [L34](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L34) |
| `NPROC_PER_NODE` | `${NPROC_PER_NODE:-8}` | [L38](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L38) |
| `RESOURCE_GPU` | `${RESOURCE_GPU:-${NPROC_PER_NODE:-8}}` | [L38](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L38) |
| `NPROC_PER_NODE` | `${NPROC_PER_NODE:-${RESOURCE_GPU}}` | [L39](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L39) |
| `RESOURCE_GPU` | `${RESOURCE_GPU}` | [L39](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L39) |
| `NNODES` | `${NNODES:-}` | [L40](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L40) |
| `NNODES` | `${NNODES}` | [L40](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L40) |
| `WORLD_SIZE` | `${WORLD_SIZE:-}` | [L40](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L40) |
| `WORLD_SIZE` | `${WORLD_SIZE}` | [L40](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L40) |
| `NODE_RANK` | `${NODE_RANK:-}` | [L43](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L43) |
| `NODE_RANK` | `${NODE_RANK}` | [L43](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L43) |
| `RANK` | `${RANK:-}` | [L43](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L43) |
| `RANK` | `${RANK}` | [L43](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L43) |
| `NNODES` | `${NNODES:-${WORLD_SIZE:-16}}` | [L46](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L46) |
| `WORLD_SIZE` | `${WORLD_SIZE:-16}` | [L46](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L46) |
| `NODE_RANK` | `${NODE_RANK:-${RANK:-0}}` | [L47](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L47) |
| `RANK` | `${RANK:-0}` | [L47](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L47) |
| `MASTER_ADDR` | `${MASTER_ADDR:-}` | [L48](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L48) |
| `MASTER_PORT` | `${MASTER_PORT:-29500}` | [L49](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L49) |
| `CUDA_VISIBLE_DEVICES` | `${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}` | [L50](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L50) |
| `MASTER_ADDR` | `${MASTER_ADDR:-127.0.0.1}` | [L65](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L65) |
| `MASTER_ADDR` | `${MASTER_ADDR}` | [L66](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L66) |
| `CUDA_VISIBLE_DEVICES` | `${CUDA_VISIBLE_DEVICES}` | [L69](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L69) |
| `MAX_LENGTH` | `${MAX_LENGTH:-32768}` | [L74](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L74) |
| `TRUNCATION_STRATEGY` | `${TRUNCATION_STRATEGY:-left}` | [L75](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L75) |
| `NUM_TRAIN_EPOCHS` | `${NUM_TRAIN_EPOCHS:-1}` | [L76](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L76) |
| `TRAIN_ITERS` | `${TRAIN_ITERS:-}` | [L77](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L77) |
| `MICRO_BATCH_SIZE` | `${MICRO_BATCH_SIZE:-1}` | [L78](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L78) |
| `GLOBAL_BATCH_SIZE` | `${GLOBAL_BATCH_SIZE:-16}` | [L79](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L79) |
| `TENSOR_MODEL_PARALLEL_SIZE` | `${TENSOR_MODEL_PARALLEL_SIZE:-2}` | [L80](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L80) |
| `PIPELINE_MODEL_PARALLEL_SIZE` | `${PIPELINE_MODEL_PARALLEL_SIZE:-8}` | [L82](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L82) |
| `EXPERT_MODEL_PARALLEL_SIZE` | `${EXPERT_MODEL_PARALLEL_SIZE:-8}` | [L83](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L83) |
| `EXPERT_TENSOR_PARALLEL_SIZE` | `${EXPERT_TENSOR_PARALLEL_SIZE:-1}` | [L84](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L84) |
| `DECODER_LAST_PIPELINE_NUM_LAYERS` | `${DECODER_LAST_PIPELINE_NUM_LAYERS:-}` | [L86](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L86) |
| `CONTEXT_PARALLEL_SIZE` | `${CONTEXT_PARALLEL_SIZE:-1}` | [L87](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L87) |
| `DATASET_NUM_PROC` | `${DATASET_NUM_PROC:-8}` | [L88](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L88) |
| `DATALOADER_NUM_WORKERS` | `${DATALOADER_NUM_WORKERS:-4}` | [L89](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L89) |
| `LEARNING_RATE` | `${LEARNING_RATE:-2e-5}` | [L90](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L90) |
| `MIN_LR` | `${MIN_LR:-2e-6}` | [L91](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L91) |
| `LR_WARMUP_FRACTION` | `${LR_WARMUP_FRACTION:-0.03}` | [L92](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L92) |
| `SAVE_STEPS` | `${SAVE_STEPS:-200}` | [L93](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L93) |
| `TRAIN_USER_LOSS` | `${TRAIN_USER_LOSS:-true}` | [L94](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L94) |
| `OPTIMIZER_CPU_OFFLOAD` | `${OPTIMIZER_CPU_OFFLOAD:-true}` | [L95](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L95) |
| `OPTIMIZER_OFFLOAD_FRACTION` | `${OPTIMIZER_OFFLOAD_FRACTION:-1}` | [L96](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L96) |
| `USE_PRECISION_AWARE_OPTIMIZER` | `${USE_PRECISION_AWARE_OPTIMIZER:-true}` | [L97](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L97) |
| `LOAD_FROM_CACHE_FILE` | `${LOAD_FROM_CACHE_FILE:-false}` | [L98](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L98) |
| `DRY_RUN` | `${DRY_RUN:-false}` | [L99](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L99) |
| `DECODER_LAST_PIPELINE_NUM_LAYERS` | `${DECODER_LAST_PIPELINE_NUM_LAYERS}` | [L110](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L110) |
| `DATASET_PATH` | `${DATASET_PATH}` | [L121](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L121) |
| `DRY_RUN` | `${DRY_RUN}` | [L123](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L123) |
| `SWIFT_ROOT` | `${SWIFT_ROOT}` | [L124](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L124) |
| `SWIFT_VENV` | `${SWIFT_VENV}` | [L128](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L128) |
| `MODEL_PATH` | `${MODEL_PATH}` | [L132](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L132) |
| `MODEL_PARALLEL_SIZE` | `${MODEL_PARALLEL_SIZE}` | [L150](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L150) |
| `WORLD_SIZE_TOTAL` | `${WORLD_SIZE_TOTAL}` | [L150](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L150) |
| `EXPERT_PARALLEL_PRODUCT` | `${EXPERT_PARALLEL_PRODUCT}` | [L154](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L154) |
| `GLOBAL_BATCH_SIZE` | `${GLOBAL_BATCH_SIZE}` | [L160](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L160) |
| `STEP_BATCH_SIZE` | `${STEP_BATCH_SIZE}` | [L160](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L160) |
| `CACHE_ROOT` | `${CACHE_ROOT}` | [L169](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L169) |
| `OUTPUT_DIR` | `${OUTPUT_DIR}` | [L169](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L169) |
| `HF_HUB_OFFLINE` | `export HF_HUB_OFFLINE=1` | [L187](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L187) |
| `TRANSFORMERS_OFFLINE` | `export TRANSFORMERS_OFFLINE=1` | [L188](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L188) |
| `HF_DATASETS_OFFLINE` | `export HF_DATASETS_OFFLINE=1` | [L189](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L189) |
| `MODELSCOPE_OFFLINE` | `export MODELSCOPE_OFFLINE=1` | [L190](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L190) |
| `PYTORCH_CUDA_ALLOC_CONF` | `${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}` | [L199](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L199) |
| `PYTORCH_ALLOC_CONF` | `${PYTORCH_ALLOC_CONF:-expandable_segments:True}` | [L200](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L200) |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | `${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}` | [L201](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L201) |
| `TORCH_NCCL_ENABLE_MONITORING` | `${TORCH_NCCL_ENABLE_MONITORING:-0}` | [L202](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L202) |
| `TORCH_NCCL_BLOCKING_WAIT` | `${TORCH_NCCL_BLOCKING_WAIT:-1}` | [L203](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L203) |
| `NCCL_DEBUG` | `${NCCL_DEBUG:-WARN}` | [L204](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L204) |
| `TOKENIZERS_PARALLELISM` | `${TOKENIZERS_PARALLELISM:-false}` | [L205](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L205) |
| `IMAGE_MAX_TOKEN_NUM` | `${IMAGE_MAX_TOKEN_NUM:-1024}` | [L207](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L207) |
| `VIDEO_MAX_TOKEN_NUM` | `${VIDEO_MAX_TOKEN_NUM:-128}` | [L208](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L208) |
| `FPS_MAX_FRAMES` | `${FPS_MAX_FRAMES:-12}` | [L209](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L209) |
| `USE_MCORE_GDN` | `${USE_MCORE_GDN:-0}` | [L210](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L210) |
| `TRAIN_USER_LOSS` | `${TRAIN_USER_LOSS}` | [L211](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L211) |
| `SWIFT_TRAIN_USER_LOSS` | `export SWIFT_TRAIN_USER_LOSS=1` | [L212](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L212) |
| `DRY_RUN` | `${DRY_RUN:-0}` | [L216](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L216) |
| `SWIFT_TRAIN_USER_LOSS` | `${SWIFT_TRAIN_USER_LOSS:-0}` | [L218](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L218) |
| `PYTHONPATH` | `${PYTHONPATH:+:${PYTHONPATH}}` | [L219](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L219) |
| `PYTHONPATH` | `${PYTHONPATH}` | [L219](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L219) |
| `LOSS_CHECK_ARGS` | `${LOSS_CHECK_ARGS[@]}` | [L220](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L220) |
| `LOAD_FROM_CACHE_FILE` | `${LOAD_FROM_CACHE_FILE}` | [L228](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L228) |
| `TENSOR_MODEL_PARALLEL_SIZE` | `${TENSOR_MODEL_PARALLEL_SIZE}` | [L233](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L233) |
| `PIPELINE_MODEL_PARALLEL_SIZE` | `${PIPELINE_MODEL_PARALLEL_SIZE}` | [L234](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L234) |
| `EXPERT_MODEL_PARALLEL_SIZE` | `${EXPERT_MODEL_PARALLEL_SIZE}` | [L235](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L235) |
| `EXPERT_TENSOR_PARALLEL_SIZE` | `${EXPERT_TENSOR_PARALLEL_SIZE}` | [L236](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L236) |
| `CONTEXT_PARALLEL_SIZE` | `${CONTEXT_PARALLEL_SIZE}` | [L242](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L242) |
| `MICRO_BATCH_SIZE` | `${MICRO_BATCH_SIZE}` | [L246](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L246) |
| `LEARNING_RATE` | `${LEARNING_RATE}` | [L252](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L252) |
| `LR_WARMUP_FRACTION` | `${LR_WARMUP_FRACTION}` | [L253](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L253) |
| `MIN_LR` | `${MIN_LR}` | [L254](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L254) |
| `SAVE_STEPS` | `${SAVE_STEPS}` | [L256](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L256) |
| `MAX_LENGTH` | `${MAX_LENGTH}` | [L257](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L257) |
| `TRUNCATION_STRATEGY` | `${TRUNCATION_STRATEGY}` | [L258](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L258) |
| `DATALOADER_NUM_WORKERS` | `${DATALOADER_NUM_WORKERS}` | [L260](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L260) |
| `DATASET_NUM_PROC` | `${DATASET_NUM_PROC}` | [L261](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L261) |
| `USE_PRECISION_AWARE_OPTIMIZER` | `${USE_PRECISION_AWARE_OPTIMIZER}` | [L264](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L264) |
| `OPTIMIZER_CPU_OFFLOAD` | `${OPTIMIZER_CPU_OFFLOAD}` | [L265](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L265) |
| `OPTIMIZER_OFFLOAD_FRACTION` | `${OPTIMIZER_OFFLOAD_FRACTION}` | [L266](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L266) |
| `TRAIN_ITERS` | `${TRAIN_ITERS}` | [L274](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L274) |
| `NUM_TRAIN_EPOCHS` | `${NUM_TRAIN_EPOCHS}` | [L277](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L277) |
| `NPROC_PER_NODE` | `${NPROC_PER_NODE}` | [L286](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L286) |
| `MASTER_PORT` | `${MASTER_PORT}` | [L287](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L287) |
| `DATA_PARALLEL_SIZE` | `${DATA_PARALLEL_SIZE}` | [L288](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L288) |
| `DECODER_LAST_PIPELINE_NUM_LAYERS` | `${DECODER_LAST_PIPELINE_NUM_LAYERS:-even partition}` | [L291](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L291) |
| `TRAIN_ITERS` | `${TRAIN_ITERS:-auto}` | [L295](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L295) |
| `ARGS` | `${ARGS[@]}` | [L300](../cpt/train_qwen35_35b_a3b_cpt_128gpu.sh#L300) |

<a id="entry-10"></a>

## cpt/train_qwen35_4b_cpt_64gpu.sh

[源码](../cpt/train_qwen35_4b_cpt_64gpu.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L17](../cpt/train_qwen35_4b_cpt_64gpu.sh#L17) |
| `PROJECT_ROOT` | `${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}` | [L18](../cpt/train_qwen35_4b_cpt_64gpu.sh#L18) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L18](../cpt/train_qwen35_4b_cpt_64gpu.sh#L18) |
| `CODE_ROOT` | `${CODE_ROOT:-${SCRIPT_DIR}}` | [L19](../cpt/train_qwen35_4b_cpt_64gpu.sh#L19) |
| `CPT_ROOT` | `${CPT_ROOT:-${PROJECT_ROOT}/models}` | [L20](../cpt/train_qwen35_4b_cpt_64gpu.sh#L20) |
| `PROJECT_ROOT` | `${PROJECT_ROOT}` | [L20](../cpt/train_qwen35_4b_cpt_64gpu.sh#L20) |
| `CODE_ROOT` | `${CODE_ROOT}` | [L21](../cpt/train_qwen35_4b_cpt_64gpu.sh#L21) |
| `SWIFT_ROOT` | `${SWIFT_ROOT:-${CODE_ROOT}/ms-swift}` | [L21](../cpt/train_qwen35_4b_cpt_64gpu.sh#L21) |
| `SWIFT_VENV` | `${SWIFT_VENV:-${CODE_ROOT}/ms-swift-venv}` | [L22](../cpt/train_qwen35_4b_cpt_64gpu.sh#L22) |
| `CPT_ROOT` | `${CPT_ROOT}` | [L26](../cpt/train_qwen35_4b_cpt_64gpu.sh#L26) |
| `MODEL_PATH` | `${MODEL_PATH:-${CPT_ROOT}/Qwen3.5-4B-Base}` | [L26](../cpt/train_qwen35_4b_cpt_64gpu.sh#L26) |
| `DATASET_PATH` | `${DATASET_PATH:-${PROJECT_ROOT}/data/cpt_data*.jsonl}` | [L27](../cpt/train_qwen35_4b_cpt_64gpu.sh#L27) |
| `OUTPUT_DIR` | `${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/cpt/qwen35_4b}` | [L28](../cpt/train_qwen35_4b_cpt_64gpu.sh#L28) |
| `CACHE_ROOT` | `${CACHE_ROOT:-${PROJECT_ROOT}/cache/cpt}` | [L29](../cpt/train_qwen35_4b_cpt_64gpu.sh#L29) |
| `CACHED_DATASET_ROOT` | `${CACHED_DATASET_ROOT:-${CACHE_ROOT}/qwen35_4b/cpt_data_cache}` | [L30](../cpt/train_qwen35_4b_cpt_64gpu.sh#L30) |
| `CACHE_ROOT` | `${CACHE_ROOT}` | [L30](../cpt/train_qwen35_4b_cpt_64gpu.sh#L30) |
| `CACHED_DATASET` | `${CACHED_DATASET:-${CACHED_DATASET_ROOT}/train}` | [L31](../cpt/train_qwen35_4b_cpt_64gpu.sh#L31) |
| `CACHED_DATASET_ROOT` | `${CACHED_DATASET_ROOT}` | [L31](../cpt/train_qwen35_4b_cpt_64gpu.sh#L31) |
| `USE_CACHED_DATASET` | `${USE_CACHED_DATASET:-true}` | [L32](../cpt/train_qwen35_4b_cpt_64gpu.sh#L32) |
| `LOCAL_RANK` | `${LOCAL_RANK:-}` | [L36](../cpt/train_qwen35_4b_cpt_64gpu.sh#L36) |
| `NPROC_PER_NODE` | `${NPROC_PER_NODE:-8}` | [L40](../cpt/train_qwen35_4b_cpt_64gpu.sh#L40) |
| `RESOURCE_GPU` | `${RESOURCE_GPU:-${NPROC_PER_NODE:-8}}` | [L40](../cpt/train_qwen35_4b_cpt_64gpu.sh#L40) |
| `NPROC_PER_NODE` | `${NPROC_PER_NODE:-${RESOURCE_GPU}}` | [L41](../cpt/train_qwen35_4b_cpt_64gpu.sh#L41) |
| `RESOURCE_GPU` | `${RESOURCE_GPU}` | [L41](../cpt/train_qwen35_4b_cpt_64gpu.sh#L41) |
| `NNODES` | `${NNODES:-}` | [L42](../cpt/train_qwen35_4b_cpt_64gpu.sh#L42) |
| `NNODES` | `${NNODES}` | [L42](../cpt/train_qwen35_4b_cpt_64gpu.sh#L42) |
| `WORLD_SIZE` | `${WORLD_SIZE:-}` | [L42](../cpt/train_qwen35_4b_cpt_64gpu.sh#L42) |
| `WORLD_SIZE` | `${WORLD_SIZE}` | [L42](../cpt/train_qwen35_4b_cpt_64gpu.sh#L42) |
| `NODE_RANK` | `${NODE_RANK:-}` | [L45](../cpt/train_qwen35_4b_cpt_64gpu.sh#L45) |
| `NODE_RANK` | `${NODE_RANK}` | [L45](../cpt/train_qwen35_4b_cpt_64gpu.sh#L45) |
| `RANK` | `${RANK:-}` | [L45](../cpt/train_qwen35_4b_cpt_64gpu.sh#L45) |
| `RANK` | `${RANK}` | [L45](../cpt/train_qwen35_4b_cpt_64gpu.sh#L45) |
| `NNODES` | `${NNODES:-${WORLD_SIZE:-8}}` | [L48](../cpt/train_qwen35_4b_cpt_64gpu.sh#L48) |
| `WORLD_SIZE` | `${WORLD_SIZE:-8}` | [L48](../cpt/train_qwen35_4b_cpt_64gpu.sh#L48) |
| `NODE_RANK` | `${NODE_RANK:-${RANK:-0}}` | [L49](../cpt/train_qwen35_4b_cpt_64gpu.sh#L49) |
| `RANK` | `${RANK:-0}` | [L49](../cpt/train_qwen35_4b_cpt_64gpu.sh#L49) |
| `MASTER_ADDR` | `${MASTER_ADDR:-}` | [L50](../cpt/train_qwen35_4b_cpt_64gpu.sh#L50) |
| `MASTER_PORT` | `${MASTER_PORT:-29500}` | [L51](../cpt/train_qwen35_4b_cpt_64gpu.sh#L51) |
| `CUDA_VISIBLE_DEVICES` | `${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}` | [L52](../cpt/train_qwen35_4b_cpt_64gpu.sh#L52) |
| `MASTER_ADDR` | `${MASTER_ADDR:-127.0.0.1}` | [L67](../cpt/train_qwen35_4b_cpt_64gpu.sh#L67) |
| `MASTER_ADDR` | `${MASTER_ADDR}` | [L68](../cpt/train_qwen35_4b_cpt_64gpu.sh#L68) |
| `CUDA_VISIBLE_DEVICES` | `${CUDA_VISIBLE_DEVICES}` | [L71](../cpt/train_qwen35_4b_cpt_64gpu.sh#L71) |
| `MAX_LENGTH` | `${MAX_LENGTH:-32768}` | [L76](../cpt/train_qwen35_4b_cpt_64gpu.sh#L76) |
| `TRUNCATION_STRATEGY` | `${TRUNCATION_STRATEGY:-left}` | [L77](../cpt/train_qwen35_4b_cpt_64gpu.sh#L77) |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `${PER_DEVICE_TRAIN_BATCH_SIZE:-1}` | [L78](../cpt/train_qwen35_4b_cpt_64gpu.sh#L78) |
| `PER_DEVICE_EVAL_BATCH_SIZE` | `${PER_DEVICE_EVAL_BATCH_SIZE:-1}` | [L79](../cpt/train_qwen35_4b_cpt_64gpu.sh#L79) |
| `GRADIENT_ACCUMULATION_STEPS` | `${GRADIENT_ACCUMULATION_STEPS:-32}` | [L80](../cpt/train_qwen35_4b_cpt_64gpu.sh#L80) |
| `LEARNING_RATE` | `${LEARNING_RATE:-2e-5}` | [L81](../cpt/train_qwen35_4b_cpt_64gpu.sh#L81) |
| `OPTIM` | `${OPTIM:-adamw_torch_fused}` | [L82](../cpt/train_qwen35_4b_cpt_64gpu.sh#L82) |
| `LOSS_SCALE` | `${LOSS_SCALE:-default}` | [L83](../cpt/train_qwen35_4b_cpt_64gpu.sh#L83) |
| `TRAIN_USER_LOSS` | `${TRAIN_USER_LOSS:-true}` | [L84](../cpt/train_qwen35_4b_cpt_64gpu.sh#L84) |
| `ATTN_IMPL` | `${ATTN_IMPL:-flash_attn}` | [L85](../cpt/train_qwen35_4b_cpt_64gpu.sh#L85) |
| `PACKING` | `${PACKING:-false}` | [L86](../cpt/train_qwen35_4b_cpt_64gpu.sh#L86) |
| `PADDING_FREE` | `${PADDING_FREE:-false}` | [L87](../cpt/train_qwen35_4b_cpt_64gpu.sh#L87) |
| `SEQUENCE_PARALLEL_SIZE` | `${SEQUENCE_PARALLEL_SIZE:-4}` | [L88](../cpt/train_qwen35_4b_cpt_64gpu.sh#L88) |
| `WARMUP_RATIO` | `${WARMUP_RATIO:-0.03}` | [L89](../cpt/train_qwen35_4b_cpt_64gpu.sh#L89) |
| `WEIGHT_DECAY` | `${WEIGHT_DECAY:-0.1}` | [L90](../cpt/train_qwen35_4b_cpt_64gpu.sh#L90) |
| `MAX_GRAD_NORM` | `${MAX_GRAD_NORM:-1.0}` | [L91](../cpt/train_qwen35_4b_cpt_64gpu.sh#L91) |
| `NUM_TRAIN_EPOCHS` | `${NUM_TRAIN_EPOCHS:-1}` | [L92](../cpt/train_qwen35_4b_cpt_64gpu.sh#L92) |
| `DRY_RUN` | `${DRY_RUN:-false}` | [L93](../cpt/train_qwen35_4b_cpt_64gpu.sh#L93) |
| `SAVE_STEPS` | `${SAVE_STEPS:-100}` | [L94](../cpt/train_qwen35_4b_cpt_64gpu.sh#L94) |
| `LOGGING_STEPS` | `${LOGGING_STEPS:-1}` | [L95](../cpt/train_qwen35_4b_cpt_64gpu.sh#L95) |
| `SAVE_TOTAL_LIMIT` | `${SAVE_TOTAL_LIMIT:-100}` | [L96](../cpt/train_qwen35_4b_cpt_64gpu.sh#L96) |
| `SPLIT_DATASET_RATIO` | `${SPLIT_DATASET_RATIO:-0}` | [L97](../cpt/train_qwen35_4b_cpt_64gpu.sh#L97) |
| `DATASET_NUM_PROC` | `${DATASET_NUM_PROC:-8}` | [L98](../cpt/train_qwen35_4b_cpt_64gpu.sh#L98) |
| `DATALOADER_NUM_WORKERS` | `${DATALOADER_NUM_WORKERS:-8}` | [L99](../cpt/train_qwen35_4b_cpt_64gpu.sh#L99) |
| `GROUP_BY_LENGTH` | `${GROUP_BY_LENGTH:-false}` | [L101](../cpt/train_qwen35_4b_cpt_64gpu.sh#L101) |
| `DDP_TIMEOUT` | `${DDP_TIMEOUT:-18000000}` | [L102](../cpt/train_qwen35_4b_cpt_64gpu.sh#L102) |
| `DEEPSPEED` | `${DEEPSPEED:-zero1}` | [L103](../cpt/train_qwen35_4b_cpt_64gpu.sh#L103) |
| `LOAD_FROM_CACHE_FILE` | `${LOAD_FROM_CACHE_FILE:-false}` | [L104](../cpt/train_qwen35_4b_cpt_64gpu.sh#L104) |
| `DATASET_SHUFFLE` | `${DATASET_SHUFFLE:-true}` | [L105](../cpt/train_qwen35_4b_cpt_64gpu.sh#L105) |
| `DATASET_PATH` | `${DATASET_PATH}` | [L117](../cpt/train_qwen35_4b_cpt_64gpu.sh#L117) |
| `DRY_RUN` | `${DRY_RUN}` | [L119](../cpt/train_qwen35_4b_cpt_64gpu.sh#L119) |
| `SWIFT_ROOT` | `${SWIFT_ROOT}` | [L120](../cpt/train_qwen35_4b_cpt_64gpu.sh#L120) |
| `SWIFT_VENV` | `${SWIFT_VENV}` | [L124](../cpt/train_qwen35_4b_cpt_64gpu.sh#L124) |
| `MODEL_PATH` | `${MODEL_PATH}` | [L128](../cpt/train_qwen35_4b_cpt_64gpu.sh#L128) |
| `USE_CACHED_DATASET` | `${USE_CACHED_DATASET}` | [L133](../cpt/train_qwen35_4b_cpt_64gpu.sh#L133) |
| `CACHED_DATASET` | `${CACHED_DATASET}` | [L134](../cpt/train_qwen35_4b_cpt_64gpu.sh#L134) |
| `OUTPUT_DIR` | `${OUTPUT_DIR}` | [L148](../cpt/train_qwen35_4b_cpt_64gpu.sh#L148) |
| `HF_HUB_OFFLINE` | `export HF_HUB_OFFLINE=1` | [L157](../cpt/train_qwen35_4b_cpt_64gpu.sh#L157) |
| `TRANSFORMERS_OFFLINE` | `export TRANSFORMERS_OFFLINE=1` | [L158](../cpt/train_qwen35_4b_cpt_64gpu.sh#L158) |
| `HF_DATASETS_OFFLINE` | `export HF_DATASETS_OFFLINE=1` | [L159](../cpt/train_qwen35_4b_cpt_64gpu.sh#L159) |
| `MODELSCOPE_OFFLINE` | `export MODELSCOPE_OFFLINE=1` | [L160](../cpt/train_qwen35_4b_cpt_64gpu.sh#L160) |
| `DEEPSPEED` | `${DEEPSPEED}` | [L168](../cpt/train_qwen35_4b_cpt_64gpu.sh#L168) |
| `PYTORCH_ALLOC_CONF` | `${PYTORCH_ALLOC_CONF:-expandable_segments:True}` | [L190](../cpt/train_qwen35_4b_cpt_64gpu.sh#L190) |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | `${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}` | [L191](../cpt/train_qwen35_4b_cpt_64gpu.sh#L191) |
| `TORCH_NCCL_ENABLE_MONITORING` | `${TORCH_NCCL_ENABLE_MONITORING:-0}` | [L192](../cpt/train_qwen35_4b_cpt_64gpu.sh#L192) |
| `TORCH_NCCL_BLOCKING_WAIT` | `${TORCH_NCCL_BLOCKING_WAIT:-1}` | [L193](../cpt/train_qwen35_4b_cpt_64gpu.sh#L193) |
| `NCCL_ASYNC_ERROR_HANDLING` | `${NCCL_ASYNC_ERROR_HANDLING:-1}` | [L194](../cpt/train_qwen35_4b_cpt_64gpu.sh#L194) |
| `NCCL_BLOCKING_WAIT` | `${NCCL_BLOCKING_WAIT:-1}` | [L195](../cpt/train_qwen35_4b_cpt_64gpu.sh#L195) |
| `NCCL_DEBUG` | `${NCCL_DEBUG:-WARN}` | [L196](../cpt/train_qwen35_4b_cpt_64gpu.sh#L196) |
| `TOKENIZERS_PARALLELISM` | `${TOKENIZERS_PARALLELISM:-false}` | [L197](../cpt/train_qwen35_4b_cpt_64gpu.sh#L197) |
| `TRAIN_USER_LOSS` | `${TRAIN_USER_LOSS}` | [L198](../cpt/train_qwen35_4b_cpt_64gpu.sh#L198) |
| `SWIFT_TRAIN_USER_LOSS` | `export SWIFT_TRAIN_USER_LOSS=1` | [L199](../cpt/train_qwen35_4b_cpt_64gpu.sh#L199) |
| `DRY_RUN` | `${DRY_RUN:-0}` | [L203](../cpt/train_qwen35_4b_cpt_64gpu.sh#L203) |
| `SWIFT_TRAIN_USER_LOSS` | `${SWIFT_TRAIN_USER_LOSS:-0}` | [L205](../cpt/train_qwen35_4b_cpt_64gpu.sh#L205) |
| `PYTHONPATH` | `${PYTHONPATH:+:${PYTHONPATH}}` | [L206](../cpt/train_qwen35_4b_cpt_64gpu.sh#L206) |
| `PYTHONPATH` | `${PYTHONPATH}` | [L206](../cpt/train_qwen35_4b_cpt_64gpu.sh#L206) |
| `LOSS_CHECK_ARGS` | `${LOSS_CHECK_ARGS[@]}` | [L207](../cpt/train_qwen35_4b_cpt_64gpu.sh#L207) |
| `SWIFT_SKIP_ZERO3_PARAM_INFO` | `${SWIFT_SKIP_ZERO3_PARAM_INFO:-1}` | [L209](../cpt/train_qwen35_4b_cpt_64gpu.sh#L209) |
| `IMAGE_MAX_TOKEN_NUM` | `${IMAGE_MAX_TOKEN_NUM:-1024}` | [L210](../cpt/train_qwen35_4b_cpt_64gpu.sh#L210) |
| `VIDEO_MAX_TOKEN_NUM` | `${VIDEO_MAX_TOKEN_NUM:-128}` | [L211](../cpt/train_qwen35_4b_cpt_64gpu.sh#L211) |
| `FPS_MAX_FRAMES` | `${FPS_MAX_FRAMES:-12}` | [L212](../cpt/train_qwen35_4b_cpt_64gpu.sh#L212) |
| `NUM_TRAIN_EPOCHS` | `${NUM_TRAIN_EPOCHS}` | [L219](../cpt/train_qwen35_4b_cpt_64gpu.sh#L219) |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `${PER_DEVICE_TRAIN_BATCH_SIZE}` | [L220](../cpt/train_qwen35_4b_cpt_64gpu.sh#L220) |
| `PER_DEVICE_EVAL_BATCH_SIZE` | `${PER_DEVICE_EVAL_BATCH_SIZE}` | [L221](../cpt/train_qwen35_4b_cpt_64gpu.sh#L221) |
| `GRADIENT_ACCUMULATION_STEPS` | `${GRADIENT_ACCUMULATION_STEPS}` | [L222](../cpt/train_qwen35_4b_cpt_64gpu.sh#L222) |
| `LEARNING_RATE` | `${LEARNING_RATE}` | [L223](../cpt/train_qwen35_4b_cpt_64gpu.sh#L223) |
| `OPTIM` | `${OPTIM}` | [L224](../cpt/train_qwen35_4b_cpt_64gpu.sh#L224) |
| `WARMUP_RATIO` | `${WARMUP_RATIO}` | [L225](../cpt/train_qwen35_4b_cpt_64gpu.sh#L225) |
| `WEIGHT_DECAY` | `${WEIGHT_DECAY}` | [L226](../cpt/train_qwen35_4b_cpt_64gpu.sh#L226) |
| `MAX_GRAD_NORM` | `${MAX_GRAD_NORM}` | [L227](../cpt/train_qwen35_4b_cpt_64gpu.sh#L227) |
| `LOSS_SCALE` | `${LOSS_SCALE}` | [L230](../cpt/train_qwen35_4b_cpt_64gpu.sh#L230) |
| `ATTN_IMPL` | `${ATTN_IMPL}` | [L231](../cpt/train_qwen35_4b_cpt_64gpu.sh#L231) |
| `MAX_LENGTH` | `${MAX_LENGTH}` | [L232](../cpt/train_qwen35_4b_cpt_64gpu.sh#L232) |
| `TRUNCATION_STRATEGY` | `${TRUNCATION_STRATEGY}` | [L233](../cpt/train_qwen35_4b_cpt_64gpu.sh#L233) |
| `PACKING` | `${PACKING}` | [L234](../cpt/train_qwen35_4b_cpt_64gpu.sh#L234) |
| `PADDING_FREE` | `${PADDING_FREE}` | [L235](../cpt/train_qwen35_4b_cpt_64gpu.sh#L235) |
| `LOAD_FROM_CACHE_FILE` | `${LOAD_FROM_CACHE_FILE}` | [L237](../cpt/train_qwen35_4b_cpt_64gpu.sh#L237) |
| `DATASET_SHUFFLE` | `${DATASET_SHUFFLE}` | [L238](../cpt/train_qwen35_4b_cpt_64gpu.sh#L238) |
| `SPLIT_DATASET_RATIO` | `${SPLIT_DATASET_RATIO}` | [L239](../cpt/train_qwen35_4b_cpt_64gpu.sh#L239) |
| `DATASET_NUM_PROC` | `${DATASET_NUM_PROC}` | [L240](../cpt/train_qwen35_4b_cpt_64gpu.sh#L240) |
| `DATALOADER_NUM_WORKERS` | `${DATALOADER_NUM_WORKERS}` | [L241](../cpt/train_qwen35_4b_cpt_64gpu.sh#L241) |
| `GROUP_BY_LENGTH` | `${GROUP_BY_LENGTH}` | [L242](../cpt/train_qwen35_4b_cpt_64gpu.sh#L242) |
| `DDP_TIMEOUT` | `${DDP_TIMEOUT}` | [L243](../cpt/train_qwen35_4b_cpt_64gpu.sh#L243) |
| `SAVE_STEPS` | `${SAVE_STEPS}` | [L244](../cpt/train_qwen35_4b_cpt_64gpu.sh#L244) |
| `LOGGING_STEPS` | `${LOGGING_STEPS}` | [L245](../cpt/train_qwen35_4b_cpt_64gpu.sh#L245) |
| `SAVE_TOTAL_LIMIT` | `${SAVE_TOTAL_LIMIT}` | [L246](../cpt/train_qwen35_4b_cpt_64gpu.sh#L246) |
| `DATASET_FILES` | `${DATASET_FILES[@]}` | [L257](../cpt/train_qwen35_4b_cpt_64gpu.sh#L257) |
| `SEQUENCE_PARALLEL_SIZE` | `${SEQUENCE_PARALLEL_SIZE}` | [L264](../cpt/train_qwen35_4b_cpt_64gpu.sh#L264) |
| `DATA_PARALLEL_SIZE` | `${DATA_PARALLEL_SIZE}` | [L269](../cpt/train_qwen35_4b_cpt_64gpu.sh#L269) |
| `EFFECTIVE_BATCH_SIZE` | `${EFFECTIVE_BATCH_SIZE}` | [L269](../cpt/train_qwen35_4b_cpt_64gpu.sh#L269) |
| `WORLD_SIZE_TOTAL` | `${WORLD_SIZE_TOTAL}` | [L269](../cpt/train_qwen35_4b_cpt_64gpu.sh#L269) |
| `MASTER_PORT` | `${MASTER_PORT}` | [L282](../cpt/train_qwen35_4b_cpt_64gpu.sh#L282) |
| `SWIFT_SKIP_ZERO3_PARAM_INFO` | `${SWIFT_SKIP_ZERO3_PARAM_INFO}` | [L294](../cpt/train_qwen35_4b_cpt_64gpu.sh#L294) |
| `ARGS` | `${ARGS[@]}` | [L304](../cpt/train_qwen35_4b_cpt_64gpu.sh#L304) |

<a id="entry-11"></a>

## cpt/train_qwen35_9b_cpt_64gpu.sh

[源码](../cpt/train_qwen35_9b_cpt_64gpu.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L17](../cpt/train_qwen35_9b_cpt_64gpu.sh#L17) |
| `PROJECT_ROOT` | `${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}` | [L18](../cpt/train_qwen35_9b_cpt_64gpu.sh#L18) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L18](../cpt/train_qwen35_9b_cpt_64gpu.sh#L18) |
| `CODE_ROOT` | `${CODE_ROOT:-${SCRIPT_DIR}}` | [L19](../cpt/train_qwen35_9b_cpt_64gpu.sh#L19) |
| `CPT_ROOT` | `${CPT_ROOT:-${PROJECT_ROOT}/models}` | [L20](../cpt/train_qwen35_9b_cpt_64gpu.sh#L20) |
| `PROJECT_ROOT` | `${PROJECT_ROOT}` | [L20](../cpt/train_qwen35_9b_cpt_64gpu.sh#L20) |
| `CODE_ROOT` | `${CODE_ROOT}` | [L21](../cpt/train_qwen35_9b_cpt_64gpu.sh#L21) |
| `SWIFT_ROOT` | `${SWIFT_ROOT:-${CODE_ROOT}/ms-swift}` | [L21](../cpt/train_qwen35_9b_cpt_64gpu.sh#L21) |
| `SWIFT_VENV` | `${SWIFT_VENV:-${CODE_ROOT}/ms-swift-venv}` | [L22](../cpt/train_qwen35_9b_cpt_64gpu.sh#L22) |
| `CPT_ROOT` | `${CPT_ROOT}` | [L26](../cpt/train_qwen35_9b_cpt_64gpu.sh#L26) |
| `MODEL_PATH` | `${MODEL_PATH:-${CPT_ROOT}/Qwen3.5-9B-Base}` | [L26](../cpt/train_qwen35_9b_cpt_64gpu.sh#L26) |
| `DATASET_PATH` | `${DATASET_PATH:-${PROJECT_ROOT}/data/cpt_data*.jsonl}` | [L27](../cpt/train_qwen35_9b_cpt_64gpu.sh#L27) |
| `OUTPUT_DIR` | `${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/cpt/qwen35_9b}` | [L28](../cpt/train_qwen35_9b_cpt_64gpu.sh#L28) |
| `CACHE_ROOT` | `${CACHE_ROOT:-${PROJECT_ROOT}/cache/cpt}` | [L29](../cpt/train_qwen35_9b_cpt_64gpu.sh#L29) |
| `CACHED_DATASET_ROOT` | `${CACHED_DATASET_ROOT:-${CACHE_ROOT}/qwen35_9b/cpt_data_cache}` | [L30](../cpt/train_qwen35_9b_cpt_64gpu.sh#L30) |
| `CACHE_ROOT` | `${CACHE_ROOT}` | [L30](../cpt/train_qwen35_9b_cpt_64gpu.sh#L30) |
| `CACHED_DATASET` | `${CACHED_DATASET:-${CACHED_DATASET_ROOT}/train}` | [L31](../cpt/train_qwen35_9b_cpt_64gpu.sh#L31) |
| `CACHED_DATASET_ROOT` | `${CACHED_DATASET_ROOT}` | [L31](../cpt/train_qwen35_9b_cpt_64gpu.sh#L31) |
| `USE_CACHED_DATASET` | `${USE_CACHED_DATASET:-true}` | [L32](../cpt/train_qwen35_9b_cpt_64gpu.sh#L32) |
| `NPROC_PER_NODE` | `${NPROC_PER_NODE:-8}` | [L37](../cpt/train_qwen35_9b_cpt_64gpu.sh#L37) |
| `RESOURCE_GPU` | `${RESOURCE_GPU:-${NPROC_PER_NODE:-8}}` | [L37](../cpt/train_qwen35_9b_cpt_64gpu.sh#L37) |
| `WORLD_SIZE` | `${WORLD_SIZE:-}` | [L38](../cpt/train_qwen35_9b_cpt_64gpu.sh#L38) |
| `NNODES` | `${NNODES:-}` | [L39](../cpt/train_qwen35_9b_cpt_64gpu.sh#L39) |
| `NNODES` | `${NNODES}` | [L39](../cpt/train_qwen35_9b_cpt_64gpu.sh#L39) |
| `WORLD_SIZE` | `${WORLD_SIZE}` | [L39](../cpt/train_qwen35_9b_cpt_64gpu.sh#L39) |
| `RANK` | `${RANK:-}` | [L50](../cpt/train_qwen35_9b_cpt_64gpu.sh#L50) |
| `NODE_RANK` | `${NODE_RANK:-}` | [L51](../cpt/train_qwen35_9b_cpt_64gpu.sh#L51) |
| `NODE_RANK` | `${NODE_RANK}` | [L51](../cpt/train_qwen35_9b_cpt_64gpu.sh#L51) |
| `RANK` | `${RANK}` | [L51](../cpt/train_qwen35_9b_cpt_64gpu.sh#L51) |
| `NPROC_PER_NODE` | `${NPROC_PER_NODE:-${RESOURCE_GPU}}` | [L62](../cpt/train_qwen35_9b_cpt_64gpu.sh#L62) |
| `RESOURCE_GPU` | `${RESOURCE_GPU}` | [L62](../cpt/train_qwen35_9b_cpt_64gpu.sh#L62) |
| `MASTER_ADDR` | `${MASTER_ADDR:-}` | [L63](../cpt/train_qwen35_9b_cpt_64gpu.sh#L63) |
| `MASTER_PORT` | `${MASTER_PORT:-29500}` | [L64](../cpt/train_qwen35_9b_cpt_64gpu.sh#L64) |
| `CUDA_VISIBLE_DEVICES` | `${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}` | [L65](../cpt/train_qwen35_9b_cpt_64gpu.sh#L65) |
| `MAX_LENGTH` | `${MAX_LENGTH:-32768}` | [L75](../cpt/train_qwen35_9b_cpt_64gpu.sh#L75) |
| `TRUNCATION_STRATEGY` | `${TRUNCATION_STRATEGY:-left}` | [L76](../cpt/train_qwen35_9b_cpt_64gpu.sh#L76) |
| `NUM_TRAIN_EPOCHS` | `${NUM_TRAIN_EPOCHS:-1}` | [L77](../cpt/train_qwen35_9b_cpt_64gpu.sh#L77) |
| `TRAIN_ITERS` | `${TRAIN_ITERS-}` | [L80](../cpt/train_qwen35_9b_cpt_64gpu.sh#L80) |
| `MICRO_BATCH_SIZE` | `${MICRO_BATCH_SIZE:-1}` | [L81](../cpt/train_qwen35_9b_cpt_64gpu.sh#L81) |
| `GLOBAL_BATCH_SIZE` | `${GLOBAL_BATCH_SIZE:-1024}` | [L82](../cpt/train_qwen35_9b_cpt_64gpu.sh#L82) |
| `TENSOR_MODEL_PARALLEL_SIZE` | `${TENSOR_MODEL_PARALLEL_SIZE:-4}` | [L83](../cpt/train_qwen35_9b_cpt_64gpu.sh#L83) |
| `PIPELINE_MODEL_PARALLEL_SIZE` | `${PIPELINE_MODEL_PARALLEL_SIZE:-1}` | [L84](../cpt/train_qwen35_9b_cpt_64gpu.sh#L84) |
| `CONTEXT_PARALLEL_SIZE` | `${CONTEXT_PARALLEL_SIZE:-1}` | [L85](../cpt/train_qwen35_9b_cpt_64gpu.sh#L85) |
| `DATASET_NUM_PROC` | `${DATASET_NUM_PROC:-8}` | [L86](../cpt/train_qwen35_9b_cpt_64gpu.sh#L86) |
| `DATALOADER_NUM_WORKERS` | `${DATALOADER_NUM_WORKERS:-4}` | [L87](../cpt/train_qwen35_9b_cpt_64gpu.sh#L87) |
| `LEARNING_RATE` | `${LEARNING_RATE:-2e-5}` | [L88](../cpt/train_qwen35_9b_cpt_64gpu.sh#L88) |
| `MIN_LR` | `${MIN_LR:-2e-6}` | [L89](../cpt/train_qwen35_9b_cpt_64gpu.sh#L89) |
| `LR_WARMUP_FRACTION` | `${LR_WARMUP_FRACTION:-0.03}` | [L90](../cpt/train_qwen35_9b_cpt_64gpu.sh#L90) |
| `SAVE_STEPS` | `${SAVE_STEPS:-500}` | [L91](../cpt/train_qwen35_9b_cpt_64gpu.sh#L91) |
| `LOGGING_STEPS` | `${LOGGING_STEPS:-1}` | [L92](../cpt/train_qwen35_9b_cpt_64gpu.sh#L92) |
| `TRAIN_USER_LOSS` | `${TRAIN_USER_LOSS:-true}` | [L93](../cpt/train_qwen35_9b_cpt_64gpu.sh#L93) |
| `OPTIMIZER_CPU_OFFLOAD` | `${OPTIMIZER_CPU_OFFLOAD:-false}` | [L94](../cpt/train_qwen35_9b_cpt_64gpu.sh#L94) |
| `OPTIMIZER_OFFLOAD_FRACTION` | `${OPTIMIZER_OFFLOAD_FRACTION:-1}` | [L95](../cpt/train_qwen35_9b_cpt_64gpu.sh#L95) |
| `USE_PRECISION_AWARE_OPTIMIZER` | `${USE_PRECISION_AWARE_OPTIMIZER:-true}` | [L96](../cpt/train_qwen35_9b_cpt_64gpu.sh#L96) |
| `MAIN_GRADS_DTYPE` | `${MAIN_GRADS_DTYPE:-bf16}` | [L97](../cpt/train_qwen35_9b_cpt_64gpu.sh#L97) |
| `MAIN_PARAMS_DTYPE` | `${MAIN_PARAMS_DTYPE:-fp32}` | [L98](../cpt/train_qwen35_9b_cpt_64gpu.sh#L98) |
| `EXP_AVG_DTYPE` | `${EXP_AVG_DTYPE:-fp32}` | [L99](../cpt/train_qwen35_9b_cpt_64gpu.sh#L99) |
| `EXP_AVG_SQ_DTYPE` | `${EXP_AVG_SQ_DTYPE:-fp32}` | [L100](../cpt/train_qwen35_9b_cpt_64gpu.sh#L100) |
| `USE_CPU_INITIALIZATION` | `${USE_CPU_INITIALIZATION:-true}` | [L101](../cpt/train_qwen35_9b_cpt_64gpu.sh#L101) |
| `LOAD_FROM_CACHE_FILE` | `${LOAD_FROM_CACHE_FILE:-false}` | [L102](../cpt/train_qwen35_9b_cpt_64gpu.sh#L102) |
| `NO_SAVE_OPTIM` | `${NO_SAVE_OPTIM:-false}` | [L103](../cpt/train_qwen35_9b_cpt_64gpu.sh#L103) |
| `NO_SAVE_RNG` | `${NO_SAVE_RNG:-false}` | [L104](../cpt/train_qwen35_9b_cpt_64gpu.sh#L104) |
| `DRY_RUN` | `${DRY_RUN:-false}` | [L105](../cpt/train_qwen35_9b_cpt_64gpu.sh#L105) |
| `NCCL_PREFLIGHT_ONLY` | `${NCCL_PREFLIGHT_ONLY:-false}` | [L106](../cpt/train_qwen35_9b_cpt_64gpu.sh#L106) |
| `NCCL_PREFLIGHT_TENSOR_MIB` | `${NCCL_PREFLIGHT_TENSOR_MIB:-16}` | [L107](../cpt/train_qwen35_9b_cpt_64gpu.sh#L107) |
| `NCCL_PREFLIGHT_TIMEOUT_SEC` | `${NCCL_PREFLIGHT_TIMEOUT_SEC:-180}` | [L108](../cpt/train_qwen35_9b_cpt_64gpu.sh#L108) |
| `ALLOW_TOPOLOGY_OVERRIDE` | `${ALLOW_TOPOLOGY_OVERRIDE:-false}` | [L109](../cpt/train_qwen35_9b_cpt_64gpu.sh#L109) |
| `EXPECTED_NNODES` | `${EXPECTED_NNODES:-8}` | [L110](../cpt/train_qwen35_9b_cpt_64gpu.sh#L110) |
| `EXPECTED_GPUS_PER_NODE` | `${EXPECTED_GPUS_PER_NODE:-8}` | [L111](../cpt/train_qwen35_9b_cpt_64gpu.sh#L111) |
| `LOCAL_CACHE_BASE` | `${LOCAL_CACHE_BASE:-/tmp/world_model_cpt}` | [L112](../cpt/train_qwen35_9b_cpt_64gpu.sh#L112) |
| `LOCAL_CACHE_BASE` | `${LOCAL_CACHE_BASE}` | [L113](../cpt/train_qwen35_9b_cpt_64gpu.sh#L113) |
| `LOCAL_TMP_ROOT` | `${LOCAL_TMP_ROOT:-${LOCAL_CACHE_BASE}/qwen35_9b_rank${NODE_RANK}}` | [L113](../cpt/train_qwen35_9b_cpt_64gpu.sh#L113) |
| `SHORT_TMPDIR` | `${SHORT_TMPDIR:-/tmp/q9b_r${NODE_RANK}}` | [L114](../cpt/train_qwen35_9b_cpt_64gpu.sh#L114) |
| `NPROC_PER_NODE` | `${NPROC_PER_NODE}` | [L126](../cpt/train_qwen35_9b_cpt_64gpu.sh#L126) |
| `EXPECTED_NNODES` | `${EXPECTED_NNODES}` | [L128](../cpt/train_qwen35_9b_cpt_64gpu.sh#L128) |
| `EXPECTED_GPUS_PER_NODE` | `${EXPECTED_GPUS_PER_NODE}` | [L129](../cpt/train_qwen35_9b_cpt_64gpu.sh#L129) |
| `TENSOR_MODEL_PARALLEL_SIZE` | `${TENSOR_MODEL_PARALLEL_SIZE}` | [L130](../cpt/train_qwen35_9b_cpt_64gpu.sh#L130) |
| `PIPELINE_MODEL_PARALLEL_SIZE` | `${PIPELINE_MODEL_PARALLEL_SIZE}` | [L131](../cpt/train_qwen35_9b_cpt_64gpu.sh#L131) |
| `CONTEXT_PARALLEL_SIZE` | `${CONTEXT_PARALLEL_SIZE}` | [L132](../cpt/train_qwen35_9b_cpt_64gpu.sh#L132) |
| `MASTER_PORT` | `${MASTER_PORT}` | [L144](../cpt/train_qwen35_9b_cpt_64gpu.sh#L144) |
| `MASTER_ADDR` | `${MASTER_ADDR}` | [L157](../cpt/train_qwen35_9b_cpt_64gpu.sh#L157) |
| `ALLOW_TOPOLOGY_OVERRIDE` | `${ALLOW_TOPOLOGY_OVERRIDE}` | [L170](../cpt/train_qwen35_9b_cpt_64gpu.sh#L170) |
| `CUDA_VISIBLE_DEVICES` | `${CUDA_VISIBLE_DEVICES}` | [L185](../cpt/train_qwen35_9b_cpt_64gpu.sh#L185) |
| `MODEL_PARALLEL_SIZE` | `${MODEL_PARALLEL_SIZE}` | [L196](../cpt/train_qwen35_9b_cpt_64gpu.sh#L196) |
| `WORLD_SIZE_TOTAL` | `${WORLD_SIZE_TOTAL}` | [L196](../cpt/train_qwen35_9b_cpt_64gpu.sh#L196) |
| `GLOBAL_BATCH_SIZE` | `${GLOBAL_BATCH_SIZE}` | [L200](../cpt/train_qwen35_9b_cpt_64gpu.sh#L200) |
| `MICRO_BATCH_SIZE` | `${MICRO_BATCH_SIZE}` | [L200](../cpt/train_qwen35_9b_cpt_64gpu.sh#L200) |
| `STEP_BATCH_SIZE` | `${STEP_BATCH_SIZE}` | [L206](../cpt/train_qwen35_9b_cpt_64gpu.sh#L206) |
| `DATASET_PATH` | `${DATASET_PATH}` | [L211](../cpt/train_qwen35_9b_cpt_64gpu.sh#L211) |
| `DRY_RUN` | `${DRY_RUN}` | [L212](../cpt/train_qwen35_9b_cpt_64gpu.sh#L212) |
| `SWIFT_ROOT` | `${SWIFT_ROOT}` | [L217](../cpt/train_qwen35_9b_cpt_64gpu.sh#L217) |
| `SWIFT_VENV` | `${SWIFT_VENV}` | [L221](../cpt/train_qwen35_9b_cpt_64gpu.sh#L221) |
| `MODEL_PATH` | `${MODEL_PATH}` | [L225](../cpt/train_qwen35_9b_cpt_64gpu.sh#L225) |
| `USE_CACHED_DATASET` | `${USE_CACHED_DATASET}` | [L232](../cpt/train_qwen35_9b_cpt_64gpu.sh#L232) |
| `CACHED_DATASET` | `${CACHED_DATASET}` | [L233](../cpt/train_qwen35_9b_cpt_64gpu.sh#L233) |
| `OUTPUT_DIR` | `${OUTPUT_DIR}` | [L315](../cpt/train_qwen35_9b_cpt_64gpu.sh#L315) |
| `LOCAL_TMP_ROOT` | `${LOCAL_TMP_ROOT}` | [L316](../cpt/train_qwen35_9b_cpt_64gpu.sh#L316) |
| `SHORT_TMPDIR` | `${SHORT_TMPDIR}` | [L317](../cpt/train_qwen35_9b_cpt_64gpu.sh#L317) |
| `PATH` | `${PATH}` | [L320](../cpt/train_qwen35_9b_cpt_64gpu.sh#L320) |
| `PYTHONPATH` | `${PYTHONPATH:+:${PYTHONPATH}}` | [L322](../cpt/train_qwen35_9b_cpt_64gpu.sh#L322) |
| `PYTHONPATH` | `${PYTHONPATH}` | [L322](../cpt/train_qwen35_9b_cpt_64gpu.sh#L322) |
| `NCCL_PREFLIGHT_ONLY` | `${NCCL_PREFLIGHT_ONLY}` | [L397](../cpt/train_qwen35_9b_cpt_64gpu.sh#L397) |
| `NCCL_PREFLIGHT_TENSOR_MIB` | `${NCCL_PREFLIGHT_TENSOR_MIB}` | [L420](../cpt/train_qwen35_9b_cpt_64gpu.sh#L420) |
| `NCCL_PREFLIGHT_TIMEOUT_SEC` | `${NCCL_PREFLIGHT_TIMEOUT_SEC}` | [L421](../cpt/train_qwen35_9b_cpt_64gpu.sh#L421) |
| `HF_HUB_OFFLINE` | `export HF_HUB_OFFLINE=1` | [L429](../cpt/train_qwen35_9b_cpt_64gpu.sh#L429) |
| `TRANSFORMERS_OFFLINE` | `export TRANSFORMERS_OFFLINE=1` | [L430](../cpt/train_qwen35_9b_cpt_64gpu.sh#L430) |
| `HF_DATASETS_OFFLINE` | `export HF_DATASETS_OFFLINE=1` | [L431](../cpt/train_qwen35_9b_cpt_64gpu.sh#L431) |
| `MODELSCOPE_OFFLINE` | `export MODELSCOPE_OFFLINE=1` | [L432](../cpt/train_qwen35_9b_cpt_64gpu.sh#L432) |
| `PYTORCH_CUDA_ALLOC_CONF` | `${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}` | [L444](../cpt/train_qwen35_9b_cpt_64gpu.sh#L444) |
| `PYTORCH_ALLOC_CONF` | `${PYTORCH_ALLOC_CONF:-expandable_segments:True}` | [L445](../cpt/train_qwen35_9b_cpt_64gpu.sh#L445) |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `${CUDA_DEVICE_MAX_CONNECTIONS:-1}` | [L446](../cpt/train_qwen35_9b_cpt_64gpu.sh#L446) |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | `${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}` | [L447](../cpt/train_qwen35_9b_cpt_64gpu.sh#L447) |
| `TORCH_NCCL_ENABLE_MONITORING` | `${TORCH_NCCL_ENABLE_MONITORING:-0}` | [L448](../cpt/train_qwen35_9b_cpt_64gpu.sh#L448) |
| `TORCH_NCCL_BLOCKING_WAIT` | `${TORCH_NCCL_BLOCKING_WAIT:-0}` | [L449](../cpt/train_qwen35_9b_cpt_64gpu.sh#L449) |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING` | `${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}` | [L450](../cpt/train_qwen35_9b_cpt_64gpu.sh#L450) |
| `NCCL_DEBUG` | `${NCCL_DEBUG:-WARN}` | [L451](../cpt/train_qwen35_9b_cpt_64gpu.sh#L451) |
| `TOKENIZERS_PARALLELISM` | `${TOKENIZERS_PARALLELISM:-false}` | [L452](../cpt/train_qwen35_9b_cpt_64gpu.sh#L452) |
| `OMP_NUM_THREADS` | `${OMP_NUM_THREADS:-1}` | [L453](../cpt/train_qwen35_9b_cpt_64gpu.sh#L453) |
| `IMAGE_MAX_TOKEN_NUM` | `${IMAGE_MAX_TOKEN_NUM:-1024}` | [L455](../cpt/train_qwen35_9b_cpt_64gpu.sh#L455) |
| `VIDEO_MAX_TOKEN_NUM` | `${VIDEO_MAX_TOKEN_NUM:-128}` | [L456](../cpt/train_qwen35_9b_cpt_64gpu.sh#L456) |
| `FPS_MAX_FRAMES` | `${FPS_MAX_FRAMES:-12}` | [L457](../cpt/train_qwen35_9b_cpt_64gpu.sh#L457) |
| `USE_MCORE_GDN` | `${USE_MCORE_GDN:-1}` | [L458](../cpt/train_qwen35_9b_cpt_64gpu.sh#L458) |
| `FLA_TILELANG` | `${FLA_TILELANG:-1}` | [L459](../cpt/train_qwen35_9b_cpt_64gpu.sh#L459) |
| `TRAIN_USER_LOSS` | `${TRAIN_USER_LOSS}` | [L460](../cpt/train_qwen35_9b_cpt_64gpu.sh#L460) |
| `SWIFT_TRAIN_USER_LOSS` | `export SWIFT_TRAIN_USER_LOSS=1` | [L461](../cpt/train_qwen35_9b_cpt_64gpu.sh#L461) |
| `DRY_RUN` | `${DRY_RUN:-0}` | [L465](../cpt/train_qwen35_9b_cpt_64gpu.sh#L465) |
| `SWIFT_TRAIN_USER_LOSS` | `${SWIFT_TRAIN_USER_LOSS:-0}` | [L467](../cpt/train_qwen35_9b_cpt_64gpu.sh#L467) |
| `LOSS_CHECK_ARGS` | `${LOSS_CHECK_ARGS[@]}` | [L469](../cpt/train_qwen35_9b_cpt_64gpu.sh#L469) |
| `LOAD_FROM_CACHE_FILE` | `${LOAD_FROM_CACHE_FILE}` | [L476](../cpt/train_qwen35_9b_cpt_64gpu.sh#L476) |
| `USE_CPU_INITIALIZATION` | `${USE_CPU_INITIALIZATION}` | [L491](../cpt/train_qwen35_9b_cpt_64gpu.sh#L491) |
| `LEARNING_RATE` | `${LEARNING_RATE}` | [L495](../cpt/train_qwen35_9b_cpt_64gpu.sh#L495) |
| `LR_WARMUP_FRACTION` | `${LR_WARMUP_FRACTION}` | [L496](../cpt/train_qwen35_9b_cpt_64gpu.sh#L496) |
| `MIN_LR` | `${MIN_LR}` | [L497](../cpt/train_qwen35_9b_cpt_64gpu.sh#L497) |
| `SAVE_STEPS` | `${SAVE_STEPS}` | [L499](../cpt/train_qwen35_9b_cpt_64gpu.sh#L499) |
| `LOGGING_STEPS` | `${LOGGING_STEPS}` | [L500](../cpt/train_qwen35_9b_cpt_64gpu.sh#L500) |
| `MAX_LENGTH` | `${MAX_LENGTH}` | [L501](../cpt/train_qwen35_9b_cpt_64gpu.sh#L501) |
| `TRUNCATION_STRATEGY` | `${TRUNCATION_STRATEGY}` | [L502](../cpt/train_qwen35_9b_cpt_64gpu.sh#L502) |
| `DATALOADER_NUM_WORKERS` | `${DATALOADER_NUM_WORKERS}` | [L505](../cpt/train_qwen35_9b_cpt_64gpu.sh#L505) |
| `DATASET_NUM_PROC` | `${DATASET_NUM_PROC}` | [L506](../cpt/train_qwen35_9b_cpt_64gpu.sh#L506) |
| `NO_SAVE_OPTIM` | `${NO_SAVE_OPTIM}` | [L507](../cpt/train_qwen35_9b_cpt_64gpu.sh#L507) |
| `NO_SAVE_RNG` | `${NO_SAVE_RNG}` | [L508](../cpt/train_qwen35_9b_cpt_64gpu.sh#L508) |
| `USE_PRECISION_AWARE_OPTIMIZER` | `${USE_PRECISION_AWARE_OPTIMIZER}` | [L509](../cpt/train_qwen35_9b_cpt_64gpu.sh#L509) |
| `MAIN_GRADS_DTYPE` | `${MAIN_GRADS_DTYPE}` | [L510](../cpt/train_qwen35_9b_cpt_64gpu.sh#L510) |
| `MAIN_PARAMS_DTYPE` | `${MAIN_PARAMS_DTYPE}` | [L511](../cpt/train_qwen35_9b_cpt_64gpu.sh#L511) |
| `EXP_AVG_DTYPE` | `${EXP_AVG_DTYPE}` | [L512](../cpt/train_qwen35_9b_cpt_64gpu.sh#L512) |
| `EXP_AVG_SQ_DTYPE` | `${EXP_AVG_SQ_DTYPE}` | [L513](../cpt/train_qwen35_9b_cpt_64gpu.sh#L513) |
| `OPTIMIZER_CPU_OFFLOAD` | `${OPTIMIZER_CPU_OFFLOAD}` | [L514](../cpt/train_qwen35_9b_cpt_64gpu.sh#L514) |
| `OPTIMIZER_OFFLOAD_FRACTION` | `${OPTIMIZER_OFFLOAD_FRACTION}` | [L515](../cpt/train_qwen35_9b_cpt_64gpu.sh#L515) |
| `DATASET_FILES` | `${DATASET_FILES[@]}` | [L522](../cpt/train_qwen35_9b_cpt_64gpu.sh#L522) |
| `TRAIN_ITERS` | `${TRAIN_ITERS}` | [L525](../cpt/train_qwen35_9b_cpt_64gpu.sh#L525) |
| `NUM_TRAIN_EPOCHS` | `${NUM_TRAIN_EPOCHS}` | [L528](../cpt/train_qwen35_9b_cpt_64gpu.sh#L528) |
| `DATA_PARALLEL_SIZE` | `${DATA_PARALLEL_SIZE}` | [L543](../cpt/train_qwen35_9b_cpt_64gpu.sh#L543) |
| `GRAD_ACCUM_MICROBATCHES` | `${GRAD_ACCUM_MICROBATCHES}` | [L548](../cpt/train_qwen35_9b_cpt_64gpu.sh#L548) |
| `TRAIN_ITERS` | `${TRAIN_ITERS:-auto}` | [L549](../cpt/train_qwen35_9b_cpt_64gpu.sh#L549) |
| `NCCL_SOCKET_IFNAME` | `${NCCL_SOCKET_IFNAME:-platform-default}` | [L551](../cpt/train_qwen35_9b_cpt_64gpu.sh#L551) |
| `NCCL_IB_HCA` | `${NCCL_IB_HCA:-platform-default}` | [L552](../cpt/train_qwen35_9b_cpt_64gpu.sh#L552) |
| `FLA_TILELANG` | `${FLA_TILELANG}` | [L553](../cpt/train_qwen35_9b_cpt_64gpu.sh#L553) |
| `TMPDIR` | `${TMPDIR}` | [L555](../cpt/train_qwen35_9b_cpt_64gpu.sh#L555) |
| `HF_HOME` | `${HF_HOME}` | [L556](../cpt/train_qwen35_9b_cpt_64gpu.sh#L556) |
| `MODELSCOPE_CACHE` | `${MODELSCOPE_CACHE}` | [L557](../cpt/train_qwen35_9b_cpt_64gpu.sh#L557) |
| `ARGS` | `${ARGS[@]}` | [L561](../cpt/train_qwen35_9b_cpt_64gpu.sh#L561) |

<a id="entry-12"></a>

## eval/convert_moe_for_vllm.py

[源码](../eval/convert_moe_for_vllm.py)


### CLI

| 参数 | 声明 | 位置 |
| --- | --- | --- |
| `--source` | `type=Path; default=PROJECT_ROOT / 'models' / 'Qwen3.5-35B-A3B'` | [L42](../eval/convert_moe_for_vllm.py#L42) |
| `--dest` | `type=Path; default=PROJECT_ROOT / 'models' / 'Qwen3.5-35B-A3B-vllm'` | [L47](../eval/convert_moe_for_vllm.py#L47) |
| `--force` | `action='store_true'` | [L52](../eval/convert_moe_for_vllm.py#L52) |

<a id="entry-13"></a>

## eval/eval_config.py

[源码](../eval/eval_config.py)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `PROJECT_ROOT` | `os.environ.get('PROJECT_ROOT', str(EVAL_ROOT.parent))` | [L10](../eval/eval_config.py#L10) |
| `JUDGE_PROMPT_DIR` | `os.environ.get('JUDGE_PROMPT_DIR', str(EVAL_ROOT.parent / 'prompts'))` | [L12](../eval/eval_config.py#L12) |

<a id="entry-14"></a>

## eval/install_runtime.sh

[源码](../eval/install_runtime.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L3](../eval/install_runtime.sh#L3) |
| `ROOT` | `${ROOT}` | [L4](../eval/install_runtime.sh#L4) |
| `VENV_DIR` | `${VENV_DIR:-${ROOT}/.venv}` | [L4](../eval/install_runtime.sh#L4) |
| `PYTHON_BASE` | `${PYTHON_BASE:-python3.12}` | [L5](../eval/install_runtime.sh#L5) |
| `PYTHON_BASE` | `${PYTHON_BASE}` | [L6](../eval/install_runtime.sh#L6) |
| `VENV_DIR` | `${VENV_DIR}` | [L6](../eval/install_runtime.sh#L6) |
| `API_ONLY` | `${API_ONLY:-0}` | [L9](../eval/install_runtime.sh#L9) |
| `VLLM_VERSION` | `${VLLM_VERSION:-0.28.0}` | [L10](../eval/install_runtime.sh#L10) |

<a id="entry-15"></a>

## eval/judge_plus.py

[源码](../eval/judge_plus.py)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `JUDGE_PLUS_API_BASE_URL` | `os.environ.get('JUDGE_PLUS_API_BASE_URL', '')` | [L60](../eval/judge_plus.py#L60) |
| `JUDGE_PLUS_MODEL` | `os.environ.get('JUDGE_PLUS_MODEL', DEFAULT_JUDGE_MODEL)` | [L61](../eval/judge_plus.py#L61) |
| `JUDGE_PLUS_API_KEY` | `os.environ.get('JUDGE_PLUS_API_KEY', 'EMPTY')` | [L278](../eval/judge_plus.py#L278) |

### CLI

| 参数 | 声明 | 位置 |
| --- | --- | --- |
| `--eval-root` | `type=Path; default=EVAL_ROOT` | [L57](../eval/judge_plus.py#L57) |
| `--eval-data` | `type=Path; default=DEFAULT_EVAL_DATA` | [L58](../eval/judge_plus.py#L58) |
| `--tokenizer` | `type=Path; default=DEFAULT_TOKENIZER` | [L59](../eval/judge_plus.py#L59) |
| `--api-base-url` | `default=os.environ.get('JUDGE_PLUS_API_BASE_URL', '')` | [L60](../eval/judge_plus.py#L60) |
| `--judge-model` | `default=os.environ.get('JUDGE_PLUS_MODEL', DEFAULT_JUDGE_MODEL)` | [L61](../eval/judge_plus.py#L61) |
| `--concurrency` | `type=int; default=DEFAULT_CONCURRENCY` | [L62](../eval/judge_plus.py#L62) |
| `--judge-max-tokens` | `type=int; default=DEFAULT_JUDGE_MAX_TOKENS` | [L63](../eval/judge_plus.py#L63) |
| `--max-judge-tokens` | `type=int; default=DEFAULT_MAX_JUDGE_TOKENS` | [L64](../eval/judge_plus.py#L64) |
| `--dirs` | `nargs='*'; default=None; help='optional run_dir names to process instead of auto-discovering'` | [L65](../eval/judge_plus.py#L65) |
| `--limit` | `type=int; default=None; help='smoke-test: judge only first N predictions per directory'` | [L66](../eval/judge_plus.py#L66) |
| `--output-name` | `default='judge-plus.jsonl'` | [L67](../eval/judge_plus.py#L67) |
| `--shard-dir-name` | `default='judge_plus_shards'` | [L68](../eval/judge_plus.py#L68) |

<a id="entry-16"></a>

## eval/run.sh

[源码](../eval/run.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L4](../eval/run.sh#L4) |
| `PYTHON_BIN` | `${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}` | [L7](../eval/run.sh#L7) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L7](../eval/run.sh#L7) |
| `VLLM_BIN` | `${VLLM_BIN:-${SCRIPT_DIR}/.venv/bin/vllm}` | [L12](../eval/run.sh#L12) |
| `JUDGE_BASE_URL` | `${JUDGE_BASE_URL:-http://<JUDGE_HOST>:<JUDGE_PORT>/v1}` | [L13](../eval/run.sh#L13) |
| `JUDGE_MODEL` | `${JUDGE_MODEL:-DeepSeek-V4-Flash-0731}` | [L14](../eval/run.sh#L14) |
| `JUDGE_API_KEY` | `${JUDGE_API_KEY:-EMPTY}` | [L15](../eval/run.sh#L15) |
| `JUDGE_MAX_TOKENS` | `${JUDGE_MAX_TOKENS:-4096}` | [L16](../eval/run.sh#L16) |
| `JUDGE_CONCURRENCY` | `${JUDGE_CONCURRENCY:-64}` | [L17](../eval/run.sh#L17) |
| `PYTHONNOUSERSITE` | `${PYTHONNOUSERSITE:-1}` | [L18](../eval/run.sh#L18) |
| `NO_PROXY` | `${NO_PROXY:-*}` | [L19](../eval/run.sh#L19) |

<a id="entry-17"></a>

## eval/run_eval.py

[源码](../eval/run_eval.py)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `ROLLOUT_BASE_URL` | `os.environ.get('ROLLOUT_BASE_URL', '')` | [L127](../eval/run_eval.py#L127) |
| `ROLLOUT_MODEL_NAME` | `os.environ.get('ROLLOUT_MODEL_NAME', '')` | [L128](../eval/run_eval.py#L128) |
| `JUDGE_BASE_URL` | `os.environ.get('JUDGE_BASE_URL', DEFAULT_JUDGE_BASE_URL)` | [L129](../eval/run_eval.py#L129) |
| `JUDGE_MODEL` | `os.environ.get('JUDGE_MODEL', DEFAULT_JUDGE_MODEL)` | [L130](../eval/run_eval.py#L130) |
| `JUDGE_API_KEY` | `os.environ.get('JUDGE_API_KEY', DEFAULT_JUDGE_API_KEY)` | [L131](../eval/run_eval.py#L131) |
| `JUDGE_MAX_TOKENS` | `os.environ.get('JUDGE_MAX_TOKENS', DEFAULT_JUDGE_MAX_TOKENS)` | [L132](../eval/run_eval.py#L132) |
| `JUDGE_CONCURRENCY` | `os.environ.get('JUDGE_CONCURRENCY', DEFAULT_JUDGE_CONCURRENCY)` | [L133](../eval/run_eval.py#L133) |
| `VLLM_BIN` | `os.environ.get('VLLM_BIN', str(DEFAULT_VLLM_BIN))` | [L134](../eval/run_eval.py#L134) |
| `THINK_MODE` | `os.environ.get('THINK_MODE', 'auto')` | [L140](../eval/run_eval.py#L140) |
| `NO_PROXY` | `os.environ['NO_PROXY']` | [L150](../eval/run_eval.py#L150) |
| `no_proxy` | `os.environ['no_proxy']` | [L151](../eval/run_eval.py#L151) |

### CLI

| 参数 | 声明 | 位置 |
| --- | --- | --- |
| `--model` | `choices=sorted(MODELS); default='Qwen3.5-4B'` | [L106](../eval/run_eval.py#L106) |
| `--checkpoint` | `type=Path; default=None; help='override model checkpoint/HF directory'` | [L107](../eval/run_eval.py#L107) |
| `--template-model` | `type=Path; default=None; help='override tokenizer/template model directory'` | [L108](../eval/run_eval.py#L108) |
| `--eval-data` | `type=Path; default=DEFAULT_EVAL_DATA` | [L109](../eval/run_eval.py#L109) |
| `--output-root` | `type=Path; default=OUTPUT_ROOT` | [L110](../eval/run_eval.py#L110) |
| `--num-gpus` | `type=int; default=None` | [L111](../eval/run_eval.py#L111) |
| `--base-port` | `type=int; default=38000` | [L112](../eval/run_eval.py#L112) |
| `--max-model-len` | `type=int; default=None` | [L113](../eval/run_eval.py#L113) |
| `--max-prompt-tokens` | `type=int; default=DEFAULT_MAX_PROMPT_TOKENS` | [L114](../eval/run_eval.py#L114) |
| `--max-response-tokens` | `type=int; default=DEFAULT_MAX_RESPONSE_TOKENS` | [L115](../eval/run_eval.py#L115) |
| `--prompt-safety-margin` | `type=int; default=DEFAULT_PROMPT_SAFETY_MARGIN` | [L116](../eval/run_eval.py#L116) |
| `--gpu-memory-utilization` | `type=float; default=0.8` | [L117](../eval/run_eval.py#L117) |
| `--dtype` | `default=None; help='optional vLLM dtype override, e.g. auto, half, bfloat16'` | [L118](../eval/run_eval.py#L118) |
| `--kv-cache-dtype` | `default=None; help='optional vLLM KV cache dtype override, e.g. fp8'` | [L119](../eval/run_eval.py#L119) |
| `--quantization` | `default=None; help='optional vLLM quantization override, e.g. fp8'` | [L120](../eval/run_eval.py#L120) |
| `--temperature` | `type=float; default=DEFAULT_TEMPERATURE` | [L121](../eval/run_eval.py#L121) |
| `--top-p` | `type=float; default=DEFAULT_TOP_P` | [L122](../eval/run_eval.py#L122) |
| `--top-k` | `type=int; default=DEFAULT_TOP_K` | [L123](../eval/run_eval.py#L123) |
| `--repetition-penalty` | `type=float; default=1.0` | [L124](../eval/run_eval.py#L124) |
| `--request-concurrency` | `type=int; default=DEFAULT_REQUEST_CONCURRENCY` | [L125](../eval/run_eval.py#L125) |
| `--server-timeout` | `type=int; default=3600` | [L126](../eval/run_eval.py#L126) |
| `--rollout-base-url` | `default=os.environ.get('ROLLOUT_BASE_URL', ''); help='OpenAI-compatible rollout server base URL; if set, skip local server launch'` | [L127](../eval/run_eval.py#L127) |
| `--rollout-model-name` | `default=os.environ.get('ROLLOUT_MODEL_NAME', ''); help='model name sent to the rollout server'` | [L128](../eval/run_eval.py#L128) |
| `--judge-base-url` | `default=os.environ.get('JUDGE_BASE_URL', DEFAULT_JUDGE_BASE_URL)` | [L129](../eval/run_eval.py#L129) |
| `--judge-model` | `default=os.environ.get('JUDGE_MODEL', DEFAULT_JUDGE_MODEL)` | [L130](../eval/run_eval.py#L130) |
| `--judge-api-key` | `default=os.environ.get('JUDGE_API_KEY', DEFAULT_JUDGE_API_KEY)` | [L131](../eval/run_eval.py#L131) |
| `--judge-max-tokens` | `type=int; default=int(os.environ.get('JUDGE_MAX_TOKENS', DEFAULT_JUDGE_MAX_TOKENS))` | [L132](../eval/run_eval.py#L132) |
| `--judge-concurrency` | `type=int; default=int(os.environ.get('JUDGE_CONCURRENCY', DEFAULT_JUDGE_CONCURRENCY))` | [L133](../eval/run_eval.py#L133) |
| `--vllm-bin` | `default=os.environ.get('VLLM_BIN', str(DEFAULT_VLLM_BIN))` | [L134](../eval/run_eval.py#L134) |
| `--eval-limit` | `type=int; default=None` | [L135](../eval/run_eval.py#L135) |
| `--skip-judge` | `action='store_true'` | [L136](../eval/run_eval.py#L136) |
| `--dry-run` | `action='store_true'` | [L137](../eval/run_eval.py#L137) |
| `--resume` | `action='store_true'; help='keep existing per-sample shards and skip completed indices'` | [L138](../eval/run_eval.py#L138) |
| `--run-name` | `default=None; help='override the output subdirectory name'` | [L139](../eval/run_eval.py#L139) |
| `--think-mode` | `type=parse_think_mode; choices=('auto', 'on', 'off'); default=os.environ.get('THINK_MODE', 'auto'); help='CoT generation: on enables thinking, off disables it, auto uses model defaults; independent of judge thinking'` | [L140](../eval/run_eval.py#L140) |
| `--inline-judge` | `action='store_true'; help='judge each sample immediately after generation'` | [L142](../eval/run_eval.py#L142) |
| `--no-inline-judge` | `action='store_false'; help='disable per-sample inline judging'` | [L143](../eval/run_eval.py#L143) |

<a id="entry-18"></a>

## eval/run_streaming_eval_worker.py

[源码](../eval/run_streaming_eval_worker.py)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `THINK_MODE` | `os.environ.get('THINK_MODE', 'off')` | [L73](../eval/run_streaming_eval_worker.py#L73) |
| `JUDGE_API_KEY` | `os.environ.get('JUDGE_API_KEY', 'EMPTY')` | [L76](../eval/run_streaming_eval_worker.py#L76) |
| `NO_PROXY` | `os.environ['NO_PROXY']` | [L92](../eval/run_streaming_eval_worker.py#L92) |
| `no_proxy` | `os.environ['no_proxy']` | [L93](../eval/run_streaming_eval_worker.py#L93) |
| `JUDGE_PLUS_API_KEY` | `os.environ.get('JUDGE_PLUS_API_KEY', 'EMPTY')` | [L217](../eval/run_streaming_eval_worker.py#L217) |

### CLI

| 参数 | 声明 | 位置 |
| --- | --- | --- |
| `--model` | `choices=sorted(MODELS); default='Qwen3.5-4B'` | [L57](../eval/run_streaming_eval_worker.py#L57) |
| `--checkpoint` | `type=Path; required=True` | [L58](../eval/run_streaming_eval_worker.py#L58) |
| `--template-model` | `type=Path; required=True` | [L59](../eval/run_streaming_eval_worker.py#L59) |
| `--eval-data` | `type=Path; default=DEFAULT_EVAL_DATA` | [L60](../eval/run_streaming_eval_worker.py#L60) |
| `--output-root` | `type=Path; required=True` | [L61](../eval/run_streaming_eval_worker.py#L61) |
| `--run-name` | `required=True` | [L62](../eval/run_streaming_eval_worker.py#L62) |
| `--rollout-base-url` | `required=True` | [L63](../eval/run_streaming_eval_worker.py#L63) |
| `--rollout-model-name` | `default=None` | [L64](../eval/run_streaming_eval_worker.py#L64) |
| `--max-model-len` | `type=int; default=81920` | [L65](../eval/run_streaming_eval_worker.py#L65) |
| `--max-prompt-tokens` | `type=int; default=65536` | [L66](../eval/run_streaming_eval_worker.py#L66) |
| `--max-response-tokens` | `type=int; default=16384` | [L67](../eval/run_streaming_eval_worker.py#L67) |
| `--temperature` | `type=float; default=0.6` | [L68](../eval/run_streaming_eval_worker.py#L68) |
| `--top-p` | `type=float; default=0.95` | [L69](../eval/run_streaming_eval_worker.py#L69) |
| `--top-k` | `type=int; default=-1` | [L70](../eval/run_streaming_eval_worker.py#L70) |
| `--repetition-penalty` | `type=float; default=1.1` | [L71](../eval/run_streaming_eval_worker.py#L71) |
| `--request-concurrency` | `type=int; default=64` | [L72](../eval/run_streaming_eval_worker.py#L72) |
| `--think-mode` | `type=parse_think_mode; choices=('auto', 'on', 'off'); default=os.environ.get('THINK_MODE', 'off'); help='CoT generation: on enables thinking, off disables it, auto uses model defaults; independent of judge thinking'` | [L73](../eval/run_streaming_eval_worker.py#L73) |
| `--judge-base-url` | `required=True` | [L74](../eval/run_streaming_eval_worker.py#L74) |
| `--judge-model` | `required=True` | [L75](../eval/run_streaming_eval_worker.py#L75) |
| `--judge-api-key` | `default=os.environ.get('JUDGE_API_KEY', 'EMPTY')` | [L76](../eval/run_streaming_eval_worker.py#L76) |
| `--judge-max-tokens` | `type=int; default=4096` | [L77](../eval/run_streaming_eval_worker.py#L77) |
| `--judge-concurrency` | `type=int; default=64` | [L78](../eval/run_streaming_eval_worker.py#L78) |
| `--judge-plus-base-url` | `required=True` | [L79](../eval/run_streaming_eval_worker.py#L79) |
| `--judge-plus-model` | `required=True` | [L80](../eval/run_streaming_eval_worker.py#L80) |
| `--judge-plus-max-tokens` | `type=int; default=4096` | [L81](../eval/run_streaming_eval_worker.py#L81) |
| `--judge-plus-concurrency` | `type=int; default=64` | [L82](../eval/run_streaming_eval_worker.py#L82) |
| `--max-judge-input-tokens` | `type=int; default=16384` | [L83](../eval/run_streaming_eval_worker.py#L83) |
| `--resume` | `action='store_true'` | [L84](../eval/run_streaming_eval_worker.py#L84) |
| `--limit` | `type=int; default=None` | [L85](../eval/run_streaming_eval_worker.py#L85) |

<a id="entry-19"></a>

## eval/summarize_rollout_eval.py

[源码](../eval/summarize_rollout_eval.py)


### CLI

| 参数 | 声明 | 位置 |
| --- | --- | --- |
| `root` | `type=Path; help='Directory containing eval_test_*.jsonl files'` | [L372](../eval/summarize_rollout_eval.py#L372) |
| `--split` | `choices=('both', 'ID', 'OOD'); default='both'; help='Which split(s) to report: both, ID, or OOD (default: both).'` | [L373](../eval/summarize_rollout_eval.py#L373) |

<a id="entry-20"></a>

## rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh

[源码](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L8](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L8) |
| `MODEL_PATH` | `${MODEL_PATH:-/preset-models}` | [L11](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L11) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L12](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L12) |
| `VENV_DIR` | `${VENV_DIR:-${SCRIPT_DIR}/.venv}` | [L12](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L12) |
| `USE_VENV` | `${USE_VENV:-1}` | [L13](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L13) |
| `SKIP_RUNTIME_CHECK` | `${SKIP_RUNTIME_CHECK:-0}` | [L14](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L14) |
| `PYTHON_BIN` | `${PYTHON_BIN:-${VENV_DIR}/bin/python}` | [L15](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L15) |
| `VENV_DIR` | `${VENV_DIR}` | [L15](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L15) |
| `SGLANG_SRC` | `${SGLANG_SRC:-${SCRIPT_DIR}/sglang-src}` | [L16](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L16) |
| `SGLANG_REPO` | `${SGLANG_REPO:-https://github.com/sgl-project/sglang.git}` | [L17](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L17) |
| `SGLANG_REF` | `${SGLANG_REF:-a5c9636}` | [L18](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L18) |
| `PORT` | `${PORT:-30000}` | [L19](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L19) |
| `HOST` | `${HOST:-0.0.0.0}` | [L20](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L20) |
| `RESOURCE_GPU` | `${RESOURCE_GPU:-8}` | [L21](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L21) |
| `CONTEXT_LENGTH` | `${CONTEXT_LENGTH:-81920}` | [L22](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L22) |
| `MEM_FRACTION_STATIC` | `${MEM_FRACTION_STATIC:-0.80}` | [L23](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L23) |
| `MAX_RUNNING_REQUESTS` | `${MAX_RUNNING_REQUESTS:-128}` | [L24](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L24) |
| `CHUNKED_PREFILL_SIZE` | `${CHUNKED_PREFILL_SIZE:-4096}` | [L25](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L25) |
| `MAX_PREFILL_TOKENS` | `${MAX_PREFILL_TOKENS:-8192}` | [L26](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L26) |
| `SERVED_MODEL_NAME` | `${SERVED_MODEL_NAME:-DeepSeek-V4-Flash-0731}` | [L27](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L27) |
| `ATTENTION_BACKEND` | `${ATTENTION_BACKEND:-dsv4}` | [L28](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L28) |
| `MOE_RUNNER_BACKEND` | `${MOE_RUNNER_BACKEND:-marlin}` | [L29](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L29) |
| `CUDA_GRAPH_BACKEND_DECODE` | `${CUDA_GRAPH_BACKEND_DECODE:-full}` | [L30](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L30) |
| `CUDA_GRAPH_BACKEND_PREFILL` | `${CUDA_GRAPH_BACKEND_PREFILL:-disabled}` | [L31](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L31) |
| `ENABLE_TWO_BATCH_OVERLAP` | `${ENABLE_TWO_BATCH_OVERLAP:-0}` | [L32](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L32) |
| `ENABLE_MIXED_CHUNK` | `${ENABLE_MIXED_CHUNK:-0}` | [L33](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L33) |
| `ENABLE_DYNAMIC_CHUNKING` | `${ENABLE_DYNAMIC_CHUNKING:-1}` | [L34](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L34) |
| `CHAT_TEMPLATE` | `${CHAT_TEMPLATE:-${SCRIPT_DIR}/chat_templates/deepseek_v4_flash.jinja}` | [L35](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L35) |
| `REASONING_PARSER` | `${REASONING_PARSER:-deepseek-v4}` | [L36](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L36) |
| `TOOL_CALL_PARSER` | `${TOOL_CALL_PARSER:-deepseekv4}` | [L37](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L37) |
| `RANK` | `${RANK:-}` | [L38](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L38) |
| `LOG_FILE` | `${LOG_FILE:-${SCRIPT_DIR}/logging/sglang-deepseek-v4-flash-rank${RANK}.log}` | [L39](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L39) |
| `RANK` | `${RANK}` | [L39](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L39) |
| `PID_FILE` | `${PID_FILE:-${SCRIPT_DIR}/logging/sglang-deepseek-v4-flash-rank${RANK}.pid}` | [L40](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L40) |
| `LOG_FILE` | `${LOG_FILE:-${SCRIPT_DIR}/logging/single-node/sglang-deepseek-v4-flash.log}` | [L42](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L42) |
| `PID_FILE` | `${PID_FILE:-${SCRIPT_DIR}/logging/single-node/sglang-deepseek-v4-flash.pid}` | [L43](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L43) |
| `LAUNCH_SCRIPT` | `${LAUNCH_SCRIPT:-${SCRIPT_DIR}/run_sglang_deepseek_v4_flash.py}` | [L45](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L45) |
| `CUDA_VISIBLE_DEVICES` | `${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}` | [L46](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L46) |
| `STARTUP_TIMEOUT` | `${STARTUP_TIMEOUT:-1800}` | [L47](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L47) |
| `HEALTH_INTERVAL` | `${HEALTH_INTERVAL:-10}` | [L48](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L48) |
| `WARMUP_AFTER_READY` | `${WARMUP_AFTER_READY:-1}` | [L49](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L49) |
| `WARMUP_TIMEOUT` | `${WARMUP_TIMEOUT:-600}` | [L50](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L50) |
| `STOP_TIMEOUT` | `${STOP_TIMEOUT:-30}` | [L51](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L51) |
| `KEEP_ENTRYPOINT_ALIVE` | `${KEEP_ENTRYPOINT_ALIVE:-1}` | [L52](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L52) |
| `MODEL_PATH` | `${MODEL_PATH}` | [L85](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L85) |
| `CUDA_VISIBLE_DEVICES` | `${CUDA_VISIBLE_DEVICES}` | [L117](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L117) |
| `CUDA_VISIBLE_DEVICES` | `${CUDA_VISIBLE_DEVICES:-未设置}` | [L123](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L123) |
| `TP_SIZE` | `${TP_SIZE}` | [L123](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L123) |
| `PORT` | `${PORT}` | [L147](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L147) |
| `HOST` | `${HOST}` | [L150](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L150) |
| `LAUNCH_SCRIPT` | `${LAUNCH_SCRIPT}` | [L170](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L170) |
| `SKIP_RUNTIME_CHECK` | `${SKIP_RUNTIME_CHECK}` | [L174](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L174) |
| `USE_VENV` | `${USE_VENV}` | [L177](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L177) |
| `SGLANG_SRC` | `${SGLANG_SRC}` | [L190](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L190) |
| `FLASHINFER_DISABLE_VERSION_CHECK` | `export FLASHINFER_DISABLE_VERSION_CHECK=1` | [L196](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L196) |
| `PIP_INDEX_URL` | `${PIP_INDEX_URL:-https://pypi.org/simple}` | [L244](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L244) |
| `SGLANG_UPDATE` | `${SGLANG_UPDATE:-0}` | [L248](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L248) |
| `SGLANG_REF` | `${SGLANG_REF}` | [L253](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L253) |
| `SGLANG_REPO` | `${SGLANG_REPO}` | [L259](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L259) |
| `STARTUP_TIMEOUT` | `${STARTUP_TIMEOUT}` | [L312](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L312) |
| `LOG_FILE` | `${LOG_FILE}` | [L321](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L321) |
| `PID_FILE` | `${PID_FILE}` | [L322](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L322) |
| `HEALTH_INTERVAL` | `${HEALTH_INTERVAL}` | [L333](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L333) |
| `SERVED_MODEL_NAME` | `${SERVED_MODEL_NAME}` | [L341](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L341) |
| `WARMUP_TIMEOUT` | `${WARMUP_TIMEOUT}` | [L344](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L344) |
| `STOP_TIMEOUT` | `${STOP_TIMEOUT}` | [L383](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L383) |
| `PYTHON_BIN` | `${PYTHON_BIN}` | [L413](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L413) |
| `CHAT_TEMPLATE` | `${CHAT_TEMPLATE}` | [L427](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L427) |
| `WARMUP_AFTER_READY` | `${WARMUP_AFTER_READY}` | [L441](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L441) |
| `PATH` | `${PATH}` | [L465](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L465) |
| `NO_PROXY` | `${NO_PROXY:-}` | [L466](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L466) |
| `SGLANG_MP_START_METHOD` | `${SGLANG_MP_START_METHOD:-spawn}` | [L470](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L470) |
| `CONTEXT_LENGTH` | `${CONTEXT_LENGTH}` | [L476](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L476) |
| `ATTENTION_BACKEND` | `${ATTENTION_BACKEND}` | [L477](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L477) |
| `MOE_RUNNER_BACKEND` | `${MOE_RUNNER_BACKEND}` | [L478](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L478) |
| `CUDA_GRAPH_BACKEND_DECODE` | `${CUDA_GRAPH_BACKEND_DECODE}` | [L479](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L479) |
| `CUDA_GRAPH_BACKEND_PREFILL` | `${CUDA_GRAPH_BACKEND_PREFILL}` | [L480](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L480) |
| `ENABLE_TWO_BATCH_OVERLAP` | `${ENABLE_TWO_BATCH_OVERLAP}` | [L481](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L481) |
| `ENABLE_MIXED_CHUNK` | `${ENABLE_MIXED_CHUNK}` | [L482](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L482) |
| `ENABLE_DYNAMIC_CHUNKING` | `${ENABLE_DYNAMIC_CHUNKING}` | [L483](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L483) |
| `REASONING_PARSER` | `${REASONING_PARSER}` | [L485](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L485) |
| `TOOL_CALL_PARSER` | `${TOOL_CALL_PARSER}` | [L486](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L486) |
| `MEM_FRACTION_STATIC` | `${MEM_FRACTION_STATIC}` | [L487](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L487) |
| `MAX_RUNNING_REQUESTS` | `${MAX_RUNNING_REQUESTS}` | [L488](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L488) |
| `CHUNKED_PREFILL_SIZE` | `${CHUNKED_PREFILL_SIZE}` | [L489](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L489) |
| `MAX_PREFILL_TOKENS` | `${MAX_PREFILL_TOKENS}` | [L490](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L490) |
| `WAIT_FOR_READY` | `${WAIT_FOR_READY:-1}` | [L509](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L509) |
| `KEEP_ENTRYPOINT_ALIVE` | `${KEEP_ENTRYPOINT_ALIVE}` | [L516](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L516) |
| `MODE` | `${MODE}` | [L540](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh#L540) |

<a id="entry-21"></a>

## rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh

[源码](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L8](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L8) |
| `MODEL_PATH` | `${MODEL_PATH:-/preset-models}` | [L11](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L11) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L12](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L12) |
| `VENV_DIR` | `${VENV_DIR:-${SCRIPT_DIR}/.venv}` | [L12](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L12) |
| `SGLANG_SRC` | `${SGLANG_SRC:-${SCRIPT_DIR}/sglang-src}` | [L13](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L13) |
| `SGLANG_REPO` | `${SGLANG_REPO:-https://github.com/sgl-project/sglang.git}` | [L14](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L14) |
| `SGLANG_REF` | `${SGLANG_REF:-a5c9636}` | [L15](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L15) |
| `PORT` | `${PORT:-30000}` | [L16](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L16) |
| `HOST` | `${HOST:-0.0.0.0}` | [L17](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L17) |
| `RESOURCE_GPU` | `${RESOURCE_GPU:-8}` | [L18](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L18) |
| `CONTEXT_LENGTH` | `${CONTEXT_LENGTH:-81920}` | [L19](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L19) |
| `MEM_FRACTION_STATIC` | `${MEM_FRACTION_STATIC:-0.80}` | [L20](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L20) |
| `MAX_RUNNING_REQUESTS` | `${MAX_RUNNING_REQUESTS:-128}` | [L21](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L21) |
| `CHUNKED_PREFILL_SIZE` | `${CHUNKED_PREFILL_SIZE:-4096}` | [L22](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L22) |
| `MAX_PREFILL_TOKENS` | `${MAX_PREFILL_TOKENS:-8192}` | [L23](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L23) |
| `SERVED_MODEL_NAME` | `${SERVED_MODEL_NAME:-DeepSeek-V4-Flash-0731}` | [L24](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L24) |
| `ATTENTION_BACKEND` | `${ATTENTION_BACKEND:-dsv4}` | [L25](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L25) |
| `MOE_RUNNER_BACKEND` | `${MOE_RUNNER_BACKEND:-marlin}` | [L26](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L26) |
| `CUDA_GRAPH_BACKEND_DECODE` | `${CUDA_GRAPH_BACKEND_DECODE:-full}` | [L27](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L27) |
| `CUDA_GRAPH_BACKEND_PREFILL` | `${CUDA_GRAPH_BACKEND_PREFILL:-disabled}` | [L28](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L28) |
| `ENABLE_TWO_BATCH_OVERLAP` | `${ENABLE_TWO_BATCH_OVERLAP:-0}` | [L29](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L29) |
| `ENABLE_MIXED_CHUNK` | `${ENABLE_MIXED_CHUNK:-0}` | [L30](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L30) |
| `ENABLE_DYNAMIC_CHUNKING` | `${ENABLE_DYNAMIC_CHUNKING:-0}` | [L31](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L31) |
| `CHAT_TEMPLATE` | `${CHAT_TEMPLATE:-${SCRIPT_DIR}/chat_templates/deepseek_v4_flash.jinja}` | [L32](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L32) |
| `REASONING_PARSER` | `${REASONING_PARSER:-deepseek-v4}` | [L33](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L33) |
| `TOOL_CALL_PARSER` | `${TOOL_CALL_PARSER:-deepseekv4}` | [L34](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L34) |
| `LOG_FILE` | `${LOG_FILE:-${SCRIPT_DIR}/logging/sglang-deepseek-v4-flash-rank${RANK:-unknown}.log}` | [L35](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L35) |
| `RANK` | `${RANK:-unknown}` | [L35](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L35) |
| `PID_FILE` | `${PID_FILE:-${SCRIPT_DIR}/sglang-deepseek-v4-flash-rank${RANK:-unknown}.pid}` | [L36](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L36) |
| `LAUNCH_SCRIPT` | `${LAUNCH_SCRIPT:-${SCRIPT_DIR}/run_sglang_deepseek_v4_flash.py}` | [L37](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L37) |
| `STARTUP_TIMEOUT` | `${STARTUP_TIMEOUT:-1800}` | [L38](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L38) |
| `HEALTH_INTERVAL` | `${HEALTH_INTERVAL:-10}` | [L39](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L39) |
| `WARMUP_AFTER_READY` | `${WARMUP_AFTER_READY:-1}` | [L40](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L40) |
| `WARMUP_TIMEOUT` | `${WARMUP_TIMEOUT:-600}` | [L41](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L41) |
| `STOP_TIMEOUT` | `${STOP_TIMEOUT:-30}` | [L42](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L42) |
| `RANK` | `${RANK:-未设置}` | [L64](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L64) |
| `WORLD_SIZE` | `${WORLD_SIZE:-未设置}` | [L64](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L64) |
| `RESOURCE_GPU` | `${RESOURCE_GPU:-未设置}` | [L65](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L65) |
| `TP_SIZE` | `${TP_SIZE}` | [L65](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L65) |
| `HOST` | `${HOST}` | [L66](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L66) |
| `PORT` | `${PORT}` | [L66](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L66) |
| `VENV_DIR` | `${VENV_DIR}` | [L129](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L129) |
| `MODEL_PATH` | `${MODEL_PATH}` | [L138](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L138) |
| `WORLD_SIZE` | `${WORLD_SIZE}` | [L174](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L174) |
| `RESOURCE_GPU` | `${RESOURCE_GPU}` | [L179](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L179) |
| `RANK` | `${RANK}` | [L184](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L184) |
| `CUDA_VISIBLE_DEVICES` | `${CUDA_VISIBLE_DEVICES:-}` | [L202](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L202) |
| `CUDA_VISIBLE_DEVICES` | `${CUDA_VISIBLE_DEVICES}` | [L203](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L203) |
| `CUDA_VISIBLE_DEVICES` | `${CUDA_VISIBLE_DEVICES:-未设置}` | [L208](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L208) |
| `LAUNCH_SCRIPT` | `${LAUNCH_SCRIPT}` | [L218](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L218) |
| `SGLANG_SRC` | `${SGLANG_SRC}` | [L231](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L231) |
| `FLASHINFER_DISABLE_VERSION_CHECK` | `export FLASHINFER_DISABLE_VERSION_CHECK=1` | [L237](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L237) |
| `SGLANG_REPO` | `${SGLANG_REPO}` | [L272](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L272) |
| `SGLANG_REF` | `${SGLANG_REF}` | [L273](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L273) |
| `PID_FILE` | `${PID_FILE}` | [L277](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L277) |
| `STARTUP_TIMEOUT` | `${STARTUP_TIMEOUT}` | [L294](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L294) |
| `LOG_FILE` | `${LOG_FILE}` | [L303](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L303) |
| `HEALTH_INTERVAL` | `${HEALTH_INTERVAL}` | [L315](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L315) |
| `SERVED_MODEL_NAME` | `${SERVED_MODEL_NAME}` | [L323](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L323) |
| `WARMUP_TIMEOUT` | `${WARMUP_TIMEOUT}` | [L326](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L326) |
| `STOP_TIMEOUT` | `${STOP_TIMEOUT}` | [L353](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L353) |
| `CHAT_TEMPLATE` | `${CHAT_TEMPLATE}` | [L413](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L413) |
| `WARMUP_AFTER_READY` | `${WARMUP_AFTER_READY}` | [L431](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L431) |
| `NO_PROXY` | `${NO_PROXY:-}` | [L451](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L451) |
| `SGLANG_MP_START_METHOD` | `${SGLANG_MP_START_METHOD:-spawn}` | [L455](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L455) |
| `CONTEXT_LENGTH` | `${CONTEXT_LENGTH}` | [L461](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L461) |
| `ATTENTION_BACKEND` | `${ATTENTION_BACKEND}` | [L462](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L462) |
| `MOE_RUNNER_BACKEND` | `${MOE_RUNNER_BACKEND}` | [L463](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L463) |
| `CUDA_GRAPH_BACKEND_DECODE` | `${CUDA_GRAPH_BACKEND_DECODE}` | [L464](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L464) |
| `CUDA_GRAPH_BACKEND_PREFILL` | `${CUDA_GRAPH_BACKEND_PREFILL}` | [L465](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L465) |
| `ENABLE_TWO_BATCH_OVERLAP` | `${ENABLE_TWO_BATCH_OVERLAP}` | [L466](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L466) |
| `ENABLE_MIXED_CHUNK` | `${ENABLE_MIXED_CHUNK}` | [L467](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L467) |
| `ENABLE_DYNAMIC_CHUNKING` | `${ENABLE_DYNAMIC_CHUNKING}` | [L468](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L468) |
| `REASONING_PARSER` | `${REASONING_PARSER}` | [L470](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L470) |
| `TOOL_CALL_PARSER` | `${TOOL_CALL_PARSER}` | [L471](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L471) |
| `MEM_FRACTION_STATIC` | `${MEM_FRACTION_STATIC}` | [L472](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L472) |
| `MAX_RUNNING_REQUESTS` | `${MAX_RUNNING_REQUESTS}` | [L473](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L473) |
| `CHUNKED_PREFILL_SIZE` | `${CHUNKED_PREFILL_SIZE}` | [L474](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L474) |
| `MAX_PREFILL_TOKENS` | `${MAX_PREFILL_TOKENS}` | [L475](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L475) |
| `WAIT_FOR_READY` | `${WAIT_FOR_READY:-1}` | [L498](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L498) |
| `MODE` | `${MODE}` | [L507](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_4node.sh#L507) |

<a id="entry-22"></a>

## rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_single.sh

[源码](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_single.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L4](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_single.sh#L4) |
| `CUDA_VISIBLE_DEVICES` | `${CUDA_VISIBLE_DEVICES:-0}` | [L10](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_single.sh#L10) |
| `TP_SIZE` | `${TP_SIZE:-1}` | [L11](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_single.sh#L11) |
| `PORT` | `${PORT:-30000}` | [L12](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_single.sh#L12) |
| `HOST` | `${HOST:-0.0.0.0}` | [L13](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_single.sh#L13) |
| `WARMUP_AFTER_READY` | `${WARMUP_AFTER_READY:-1}` | [L14](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_single.sh#L14) |
| `MM_FEATURE_TRANSPORT` | `${MM_FEATURE_TRANSPORT:-cpu}` | [L15](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_single.sh#L15) |
| `CPU_OFFLOAD_GB` | `${CPU_OFFLOAD_GB:-12}` | [L16](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_single.sh#L16) |
| `ENABLE_MEMORY_SAVER` | `${ENABLE_MEMORY_SAVER:-1}` | [L17](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_single.sh#L17) |
| `CUDA_GRAPH_BACKEND_DECODE` | `${CUDA_GRAPH_BACKEND_DECODE:-full}` | [L18](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_single.sh#L18) |
| `CUDA_GRAPH_BACKEND_PREFILL` | `${CUDA_GRAPH_BACKEND_PREFILL:-disabled}` | [L19](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_single.sh#L19) |
| `MODE` | `${MODE}` | [L21](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_single.sh#L21) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L21](../rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash_single.sh#L21) |

<a id="entry-23"></a>

## rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py

[源码](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `CHAT_TEMPLATE` | `_env('CHAT_TEMPLATE', str(script_dir / 'chat_templates' / 'deepseek_v4_flash.jinja'))` | [L25](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L25) |
| `REASONING_PARSER` | `_env('REASONING_PARSER', 'deepseek-v4')` | [L29](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L29) |
| `TOOL_CALL_PARSER` | `_env('TOOL_CALL_PARSER', 'deepseekv4')` | [L30](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L30) |
| `DIST_INIT_ADDR` | `_env('DIST_INIT_ADDR')` | [L31](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L31) |
| `MASTER_ADDR` | `_env('MASTER_ADDR')` | [L33](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L33) |
| `MASTER_PORT` | `_env('MASTER_PORT')` | [L34](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L34) |
| `MODEL_PATH` | `_env('MODEL_PATH', '/preset-models')` | [L40](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L40) |
| `SERVED_MODEL_NAME` | `_env('SERVED_MODEL_NAME', 'DeepSeek-V4-Flash-0731')` | [L41](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L41) |
| `HOST` | `_env('HOST', '0.0.0.0')` | [L42](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L42) |
| `PORT` | `_env('PORT', '30000')` | [L43](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L43) |
| `RESOURCE_GPU` | `_env('RESOURCE_GPU', '8')` | [L44](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L44) |
| `TP_SIZE` | `_env('TP_SIZE', _env('RESOURCE_GPU', '8'))` | [L44](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L44) |
| `CONTEXT_LENGTH` | `_env('CONTEXT_LENGTH', '81920')` | [L45](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L45) |
| `ATTENTION_BACKEND` | `_env('ATTENTION_BACKEND', 'dsv4')` | [L46](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L46) |
| `MOE_RUNNER_BACKEND` | `_env('MOE_RUNNER_BACKEND', 'marlin')` | [L47](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L47) |
| `MEM_FRACTION_STATIC` | `_env('MEM_FRACTION_STATIC', '0.80')` | [L52](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L52) |
| `MAX_RUNNING_REQUESTS` | `_env('MAX_RUNNING_REQUESTS', '128')` | [L53](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L53) |
| `ENABLE_DISTRIBUTED` | `_enabled('ENABLE_DISTRIBUTED')` | [L57](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L57) |
| `NNODES` | `_env('NNODES', _env('WORLD_SIZE'))` | [L59](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L59) |
| `WORLD_SIZE` | `_env('WORLD_SIZE')` | [L59](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L59) |
| `NODE_RANK` | `_env('NODE_RANK', _env('RANK'))` | [L60](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L60) |
| `RANK` | `_env('RANK')` | [L60](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L60) |
| `MM_FEATURE_TRANSPORT` | `_env('MM_FEATURE_TRANSPORT')` | [L61](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L61) |
| `CPU_OFFLOAD_GB` | `_env('CPU_OFFLOAD_GB')` | [L62](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L62) |
| `ENABLE_MEMORY_SAVER` | `_env('ENABLE_MEMORY_SAVER')` | [L63](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L63) |
| `ENABLE_TWO_BATCH_OVERLAP` | `_enabled('ENABLE_TWO_BATCH_OVERLAP')` | [L65](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L65) |
| `ENABLE_MIXED_CHUNK` | `_enabled('ENABLE_MIXED_CHUNK')` | [L67](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L67) |
| `ENABLE_DYNAMIC_CHUNKING` | `_enabled('ENABLE_DYNAMIC_CHUNKING')` | [L69](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L69) |
| `CHUNKED_PREFILL_SIZE` | `_env('CHUNKED_PREFILL_SIZE', '4096')` | [L74](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L74) |
| `MAX_PREFILL_TOKENS` | `_env('MAX_PREFILL_TOKENS', '8192')` | [L79](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L79) |
| `CUDA_GRAPH_BACKEND_DECODE` | `_env('CUDA_GRAPH_BACKEND_DECODE', 'full')` | [L84](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L84) |
| `CUDA_GRAPH_BACKEND_PREFILL` | `_env('CUDA_GRAPH_BACKEND_PREFILL', 'disabled')` | [L89](../rl_miles/DS-V4-Flash/run_sglang_deepseek_v4_flash.py#L89) |

<a id="entry-24"></a>

## rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash.sh

[源码](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L4](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash.sh#L4) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L6](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash.sh#L6) |

<a id="entry-25"></a>

## rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_4node.sh

[源码](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_4node.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L4](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_4node.sh#L4) |
| `RESOURCE_GPU` | `${RESOURCE_GPU:?请设置 RESOURCE_GPU=8}` | [L11](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_4node.sh#L11) |
| `WORLD_SIZE` | `${WORLD_SIZE:?请设置 WORLD_SIZE（节点总数）}` | [L12](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_4node.sh#L12) |
| `RANK` | `${RANK:?请设置 RANK=0 到 WORLD_SIZE-1（节点序号）}` | [L13](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_4node.sh#L13) |
| `ENABLE_DISTRIBUTED` | `export ENABLE_DISTRIBUTED=0` | [L17](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_4node.sh#L17) |
| `RANK` | `${RANK}` | [L19](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_4node.sh#L19) |
| `RESOURCE_GPU` | `${RESOURCE_GPU}` | [L19](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_4node.sh#L19) |
| `WORLD_SIZE` | `${WORLD_SIZE}` | [L19](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_4node.sh#L19) |
| `MODE` | `${MODE}` | [L20](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_4node.sh#L20) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L20](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_4node.sh#L20) |

<a id="entry-26"></a>

## rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh

[源码](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L5](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L5) |
| `MODEL_PATH` | `${MODEL_PATH:-/preset-models}` | [L7](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L7) |
| `PORT` | `${PORT:-30000}` | [L8](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L8) |
| `HOST` | `${HOST:-0.0.0.0}` | [L9](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L9) |
| `CUDA_VISIBLE_DEVICES` | `${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}` | [L10](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L10) |
| `TP_SIZE` | `${TP_SIZE:-8}` | [L11](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L11) |
| `RUN_ID` | `${RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}` | [L12](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L12) |
| `LOG_DIR` | `${LOG_DIR:-${SCRIPT_DIR}/logging/single-node}` | [L13](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L13) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L13](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L13) |
| `ACCESS_URL_LOG` | `${ACCESS_URL_LOG:-${SCRIPT_DIR}/logging/lod.log}` | [L14](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L14) |
| `LAUNCH_LOG` | `${LAUNCH_LOG:-${LOG_DIR}/deploy-${RUN_ID}.log}` | [L16](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L16) |
| `LOG_DIR` | `${LOG_DIR}` | [L16](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L16) |
| `RUN_ID` | `${RUN_ID}` | [L16](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L16) |
| `LAUNCH_PID_FILE` | `${LAUNCH_PID_FILE:-${LOG_DIR}/deploy-${RUN_ID}.pid}` | [L17](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L17) |
| `SERVICE_LOG` | `${SERVICE_LOG:-${LOG_DIR}/sglang-deepseek-v4-flash-${RUN_ID}.log}` | [L18](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L18) |
| `SERVICE_PID_FILE` | `${SERVICE_PID_FILE:-${LOG_DIR}/sglang-deepseek-v4-flash-${RUN_ID}.pid}` | [L19](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L19) |
| `SERVICE_PID_FILE` | `${SERVICE_PID_FILE}` | [L36](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L36) |
| `LAUNCH_LOG` | `${LAUNCH_LOG}` | [L42](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L42) |
| `MODEL_PATH` | `${MODEL_PATH}` | [L45](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L45) |
| `PORT` | `${PORT}` | [L46](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L46) |
| `HOST` | `${HOST}` | [L47](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L47) |
| `CUDA_VISIBLE_DEVICES` | `${CUDA_VISIBLE_DEVICES}` | [L48](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L48) |
| `TP_SIZE` | `${TP_SIZE}` | [L49](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L49) |
| `SERVICE_LOG` | `${SERVICE_LOG}` | [L50](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L50) |
| `KEEP_ENTRYPOINT_ALIVE` | `${KEEP_ENTRYPOINT_ALIVE:-0}` | [L52](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L52) |
| `WAIT_FOR_READY` | `${WAIT_FOR_READY:-1}` | [L53](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L53) |
| `WARMUP_AFTER_READY` | `${WARMUP_AFTER_READY:-1}` | [L54](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L54) |
| `SKIP_RUNTIME_CHECK` | `${SKIP_RUNTIME_CHECK:-0}` | [L55](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L55) |
| `LAUNCH_PID_FILE` | `${LAUNCH_PID_FILE}` | [L60](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L60) |
| `ACCESS_URL_LOG` | `${ACCESS_URL_LOG}` | [L73](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_8gpu_detached.sh#L73) |

<a id="entry-27"></a>

## rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_node0.sh

[源码](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_node0.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L3](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_node0.sh#L3) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L4](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_node0.sh#L4) |

<a id="entry-28"></a>

## rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_node1.sh

[源码](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_node1.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L3](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_node1.sh#L3) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L4](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_node1.sh#L4) |

<a id="entry-29"></a>

## rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_node2.sh

[源码](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_node2.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L3](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_node2.sh#L3) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L4](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_node2.sh#L4) |

<a id="entry-30"></a>

## rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_node3.sh

[源码](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_node3.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L3](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_node3.sh#L3) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L4](../rl_miles/DS-V4-Flash/start_sglang_deepseek_v4_flash_node3.sh#L4) |

<a id="entry-31"></a>

## rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh

[源码](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L8](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L8) |
| `MODEL_PATH` | `${MODEL_PATH:-<EMBEDDING_MODEL_PATH>}` | [L9](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L9) |
| `HOST` | `${HOST:-0.0.0.0}` | [L10](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L10) |
| `PORT` | `${PORT:-8000}` | [L11](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L11) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L12](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L12) |
| `VENV_DIR` | `${VENV_DIR:-${SCRIPT_DIR}/.venv}` | [L12](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L12) |
| `GPU_MEMORY_UTILIZATION` | `${GPU_MEMORY_UTILIZATION:-0.95}` | [L13](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L13) |
| `MAX_NUM_SEQS` | `${MAX_NUM_SEQS:-64}` | [L14](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L14) |
| `MAX_NUM_BATCHED_TOKENS` | `${MAX_NUM_BATCHED_TOKENS:-32768}` | [L15](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L15) |
| `MAX_NUM_BATCHED_TOKENS` | `${MAX_NUM_BATCHED_TOKENS}` | [L16](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L16) |
| `MAX_NUM_SCHEDULED_TOKENS` | `${MAX_NUM_SCHEDULED_TOKENS:-${MAX_NUM_BATCHED_TOKENS}}` | [L16](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L16) |
| `ENABLE_PREFIX_CACHING` | `${ENABLE_PREFIX_CACHING:-1}` | [L17](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L17) |
| `ENABLE_CHUNKED_PREFILL` | `${ENABLE_CHUNKED_PREFILL:-1}` | [L18](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L18) |
| `KV_CACHE_DTYPE` | `${KV_CACHE_DTYPE:-fp8}` | [L19](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L19) |
| `CUDA_VISIBLE_DEVICES` | `${CUDA_VISIBLE_DEVICES:-0,1,2,3}` | [L21](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L21) |
| `PIP_INDEX_URL` | `${PIP_INDEX_URL:-https://pypi.org/simple}` | [L22](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L22) |
| `MODEL_PATH` | `${MODEL_PATH}` | [L24](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L24) |
| `VENV_DIR` | `${VENV_DIR}` | [L28](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L28) |
| `VLLM_VERSION` | `${VLLM_VERSION:-0.28.0}` | [L29](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L29) |
| `HOST` | `${HOST}` | [L33](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L33) |
| `PORT` | `${PORT}` | [L33](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L33) |
| `SERVED_MODEL_NAME` | `${SERVED_MODEL_NAME:-Qwen3-8B-Embedding}` | [L34](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L34) |
| `MAX_MODEL_LEN` | `${MAX_MODEL_LEN:-32768}` | [L36](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L36) |
| `GPU_MEMORY_UTILIZATION` | `${GPU_MEMORY_UTILIZATION}` | [L38](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L38) |
| `MAX_NUM_SEQS` | `${MAX_NUM_SEQS}` | [L39](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L39) |
| `MAX_NUM_SCHEDULED_TOKENS` | `${MAX_NUM_SCHEDULED_TOKENS}` | [L41](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L41) |
| `KV_CACHE_DTYPE` | `${KV_CACHE_DTYPE}` | [L43](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L43) |
| `ENABLE_PREFIX_CACHING` | `${ENABLE_PREFIX_CACHING}` | [L44](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L44) |
| `ENABLE_CHUNKED_PREFILL` | `${ENABLE_CHUNKED_PREFILL}` | [L49](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_4gpu_tp4.sh#L49) |

<a id="entry-32"></a>

## rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh

[源码](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L7](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L7) |
| `MODEL_PATH` | `${MODEL_PATH:-<EMBEDDING_MODEL_PATH>}` | [L8](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L8) |
| `HOST` | `${HOST:-127.0.0.1}` | [L9](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L9) |
| `PORT` | `${PORT:-8200}` | [L10](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L10) |
| `GPU_ID` | `${GPU_ID:-0}` | [L11](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L11) |
| `SERVED_MODEL_NAME` | `${SERVED_MODEL_NAME:-Qwen3-8B-Embedding}` | [L12](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L12) |
| `VLLM_VERSION` | `${VLLM_VERSION:-0.28.0}` | [L13](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L13) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L14](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L14) |
| `VENV_DIR` | `${VENV_DIR:-${SCRIPT_DIR}/.venv}` | [L14](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L14) |
| `GPU_MEMORY_UTILIZATION` | `${GPU_MEMORY_UTILIZATION:-0.90}` | [L15](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L15) |
| `MAX_NUM_SEQS` | `${MAX_NUM_SEQS:-8}` | [L16](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L16) |
| `MAX_NUM_BATCHED_TOKENS` | `${MAX_NUM_BATCHED_TOKENS:-65536}` | [L17](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L17) |
| `MAX_NUM_BATCHED_TOKENS` | `${MAX_NUM_BATCHED_TOKENS}` | [L18](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L18) |
| `MAX_NUM_SCHEDULED_TOKENS` | `${MAX_NUM_SCHEDULED_TOKENS:-${MAX_NUM_BATCHED_TOKENS}}` | [L18](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L18) |
| `MAX_MODEL_LEN` | `${MAX_MODEL_LEN:-32768}` | [L19](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L19) |
| `KV_CACHE_DTYPE` | `${KV_CACHE_DTYPE:-fp8}` | [L20](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L20) |
| `ENABLE_PREFIX_CACHING` | `${ENABLE_PREFIX_CACHING:-1}` | [L21](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L21) |
| `ENABLE_CHUNKED_PREFILL` | `${ENABLE_CHUNKED_PREFILL:-1}` | [L22](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L22) |
| `PIP_INDEX_URL` | `${PIP_INDEX_URL:-https://pypi.org/simple}` | [L24](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L24) |
| `MODEL_PATH` | `${MODEL_PATH}` | [L26](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L26) |
| `GPU_ID` | `${GPU_ID}` | [L31](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L31) |
| `VENV_DIR` | `${VENV_DIR}` | [L33](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L33) |
| `VLLM_VERSION` | `${VLLM_VERSION}` | [L41](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L41) |
| `HOST` | `${HOST}` | [L49](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L49) |
| `PORT` | `${PORT}` | [L50](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L50) |
| `SERVED_MODEL_NAME` | `${SERVED_MODEL_NAME}` | [L51](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L51) |
| `MAX_MODEL_LEN` | `${MAX_MODEL_LEN}` | [L55](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L55) |
| `GPU_MEMORY_UTILIZATION` | `${GPU_MEMORY_UTILIZATION}` | [L57](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L57) |
| `MAX_NUM_SEQS` | `${MAX_NUM_SEQS}` | [L58](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L58) |
| `MAX_NUM_SCHEDULED_TOKENS` | `${MAX_NUM_SCHEDULED_TOKENS}` | [L60](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L60) |
| `KV_CACHE_DTYPE` | `${KV_CACHE_DTYPE}` | [L63](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L63) |
| `ENABLE_PREFIX_CACHING` | `${ENABLE_PREFIX_CACHING}` | [L67](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L67) |
| `ENABLE_CHUNKED_PREFILL` | `${ENABLE_CHUNKED_PREFILL}` | [L73](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L73) |
| `VLLM_ARGS` | `${VLLM_ARGS[@]}` | [L79](../rl_miles/Qwen3-embedding/deploy_qwen3_embedding_replica.sh#L79) |

<a id="entry-33"></a>

## rl_miles/Qwen3-embedding/install_runtime.sh

[源码](../rl_miles/Qwen3-embedding/install_runtime.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L3](../rl_miles/Qwen3-embedding/install_runtime.sh#L3) |
| `ROOT` | `${ROOT}` | [L4](../rl_miles/Qwen3-embedding/install_runtime.sh#L4) |
| `VENV_DIR` | `${VENV_DIR:-${ROOT}/.venv}` | [L4](../rl_miles/Qwen3-embedding/install_runtime.sh#L4) |
| `PYTHON_BASE` | `${PYTHON_BASE:-python3.12}` | [L5](../rl_miles/Qwen3-embedding/install_runtime.sh#L5) |
| `VLLM_VERSION` | `${VLLM_VERSION:-0.28.0}` | [L6](../rl_miles/Qwen3-embedding/install_runtime.sh#L6) |
| `PYTHON_BASE` | `${PYTHON_BASE}` | [L7](../rl_miles/Qwen3-embedding/install_runtime.sh#L7) |
| `VENV_DIR` | `${VENV_DIR}` | [L7](../rl_miles/Qwen3-embedding/install_runtime.sh#L7) |
| `VLLM_VERSION` | `${VLLM_VERSION}` | [L9](../rl_miles/Qwen3-embedding/install_runtime.sh#L9) |

<a id="entry-34"></a>

## rl_miles/Qwen3-embedding/similarity_service.py

[源码](../rl_miles/Qwen3-embedding/similarity_service.py)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `NO_PROXY` | `os.environ['NO_PROXY']` | [L38](../rl_miles/Qwen3-embedding/similarity_service.py#L38) |
| `no_proxy` | `os.environ['no_proxy']` | [L39](../rl_miles/Qwen3-embedding/similarity_service.py#L39) |
| `EMBEDDING_API_BASES` | `os.environ.get('EMBEDDING_API_BASES', '')` | [L43](../rl_miles/Qwen3-embedding/similarity_service.py#L43) |
| `EMBEDDING_API_BASE` | `os.environ.get('EMBEDDING_API_BASE', 'http://127.0.0.1:8200,http://127.0.0.1:8201,http://127.0.0.1:8202,http://127.0.0.1:8203')` | [L45](../rl_miles/Qwen3-embedding/similarity_service.py#L45) |
| `EMBEDDING_MODEL_NAME` | `os.environ.get('EMBEDDING_MODEL_NAME', 'Qwen3-8B-Embedding')` | [L56](../rl_miles/Qwen3-embedding/similarity_service.py#L56) |
| `SIMILARITY_MAX_BATCH` | `os.environ.get('SIMILARITY_MAX_BATCH', '128')` | [L57](../rl_miles/Qwen3-embedding/similarity_service.py#L57) |
| `SIMILARITY_BATCH_TIMEOUT` | `os.environ.get('SIMILARITY_BATCH_TIMEOUT', '0.005')` | [L58](../rl_miles/Qwen3-embedding/similarity_service.py#L58) |
| `SIMILARITY_QUEUE_LIMIT` | `os.environ.get('SIMILARITY_QUEUE_LIMIT', '5000')` | [L59](../rl_miles/Qwen3-embedding/similarity_service.py#L59) |
| `SIMILARITY_REQUEST_TIMEOUT` | `os.environ.get('SIMILARITY_REQUEST_TIMEOUT', '1200')` | [L60](../rl_miles/Qwen3-embedding/similarity_service.py#L60) |
| `TRUNCATE_PROMPT_TOKENS` | `os.environ.get('TRUNCATE_PROMPT_TOKENS', '32768')` | [L61](../rl_miles/Qwen3-embedding/similarity_service.py#L61) |
| `TRUNCATION_SIDE` | `os.environ.get('TRUNCATION_SIDE', 'right')` | [L62](../rl_miles/Qwen3-embedding/similarity_service.py#L62) |
| `SIMILARITY_HOST` | `os.environ.get('SIMILARITY_HOST', '0.0.0.0')` | [L263](../rl_miles/Qwen3-embedding/similarity_service.py#L263) |
| `SIMILARITY_PORT` | `os.environ.get('SIMILARITY_PORT', '8100')` | [L265](../rl_miles/Qwen3-embedding/similarity_service.py#L265) |

### CLI

| 参数 | 声明 | 位置 |
| --- | --- | --- |
| `--host` | `default=os.environ.get('SIMILARITY_HOST', '0.0.0.0')` | [L263](../rl_miles/Qwen3-embedding/similarity_service.py#L263) |
| `--port` | `type=int; default=int(os.environ.get('SIMILARITY_PORT', '8100'))` | [L264](../rl_miles/Qwen3-embedding/similarity_service.py#L264) |

<a id="entry-35"></a>

## rl_miles/Qwen3-embedding/start_qwen3_embedding_4replicas.sh

[源码](../rl_miles/Qwen3-embedding/start_qwen3_embedding_4replicas.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L4](../rl_miles/Qwen3-embedding/start_qwen3_embedding_4replicas.sh#L4) |
| `LOG_DIR` | `${LOG_DIR:-${SCRIPT_DIR}/logs}` | [L5](../rl_miles/Qwen3-embedding/start_qwen3_embedding_4replicas.sh#L5) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L5](../rl_miles/Qwen3-embedding/start_qwen3_embedding_4replicas.sh#L5) |
| `GPU_IDS` | `${GPU_IDS:-0,1,2,3}` | [L6](../rl_miles/Qwen3-embedding/start_qwen3_embedding_4replicas.sh#L6) |
| `PORT_BASE` | `${PORT_BASE:-8200}` | [L7](../rl_miles/Qwen3-embedding/start_qwen3_embedding_4replicas.sh#L7) |
| `WAIT_TIMEOUT` | `${WAIT_TIMEOUT:-360}` | [L8](../rl_miles/Qwen3-embedding/start_qwen3_embedding_4replicas.sh#L8) |
| `LOG_DIR` | `${LOG_DIR}` | [L10](../rl_miles/Qwen3-embedding/start_qwen3_embedding_4replicas.sh#L10) |
| `GPU_IDS` | `${GPU_IDS}` | [L11](../rl_miles/Qwen3-embedding/start_qwen3_embedding_4replicas.sh#L11) |
| `WAIT_TIMEOUT` | `${WAIT_TIMEOUT}` | [L46](../rl_miles/Qwen3-embedding/start_qwen3_embedding_4replicas.sh#L46) |

<a id="entry-36"></a>

## rl_miles/Qwen3-embedding/start_similarity_services.sh

[源码](../rl_miles/Qwen3-embedding/start_similarity_services.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L4](../rl_miles/Qwen3-embedding/start_similarity_services.sh#L4) |
| `LOG_DIR` | `${LOG_DIR:-${SCRIPT_DIR}/logs}` | [L5](../rl_miles/Qwen3-embedding/start_similarity_services.sh#L5) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L5](../rl_miles/Qwen3-embedding/start_similarity_services.sh#L5) |
| `SIMILARITY_HOST` | `${SIMILARITY_HOST:-0.0.0.0}` | [L6](../rl_miles/Qwen3-embedding/start_similarity_services.sh#L6) |
| `SIMILARITY_PORT` | `${SIMILARITY_PORT:-8100}` | [L7](../rl_miles/Qwen3-embedding/start_similarity_services.sh#L7) |
| `VENV_DIR` | `${VENV_DIR:-${SCRIPT_DIR}/.venv}` | [L8](../rl_miles/Qwen3-embedding/start_similarity_services.sh#L8) |
| `EMBEDDING_API_BASES` | `${EMBEDDING_API_BASES:-http://127.0.0.1:8200,http://127.0.0.1:8201,http://127.0.0.1:8202,http://127.0.0.1:8203}` | [L9](../rl_miles/Qwen3-embedding/start_similarity_services.sh#L9) |
| `LOG_DIR` | `${LOG_DIR}` | [L11](../rl_miles/Qwen3-embedding/start_similarity_services.sh#L11) |
| `GPU_IDS` | `${GPU_IDS:-0,1,2,3}` | [L12](../rl_miles/Qwen3-embedding/start_similarity_services.sh#L12) |
| `PID_FILE` | `${PID_FILE}` | [L17](../rl_miles/Qwen3-embedding/start_similarity_services.sh#L17) |
| `SIMILARITY_HOST` | `${SIMILARITY_HOST}` | [L22](../rl_miles/Qwen3-embedding/start_similarity_services.sh#L22) |
| `SIMILARITY_PORT` | `${SIMILARITY_PORT}` | [L22](../rl_miles/Qwen3-embedding/start_similarity_services.sh#L22) |
| `EMBEDDING_API_BASES` | `${EMBEDDING_API_BASES}` | [L24](../rl_miles/Qwen3-embedding/start_similarity_services.sh#L24) |
| `VENV_DIR` | `${VENV_DIR}` | [L27](../rl_miles/Qwen3-embedding/start_similarity_services.sh#L27) |
| `LOG_FILE` | `${LOG_FILE}` | [L28](../rl_miles/Qwen3-embedding/start_similarity_services.sh#L28) |

<a id="entry-37"></a>

## rl_miles/Qwen3-embedding/stop_qwen3_embedding_4replicas.sh

[源码](../rl_miles/Qwen3-embedding/stop_qwen3_embedding_4replicas.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L4](../rl_miles/Qwen3-embedding/stop_qwen3_embedding_4replicas.sh#L4) |
| `LOG_DIR` | `${LOG_DIR:-${SCRIPT_DIR}/logs}` | [L5](../rl_miles/Qwen3-embedding/stop_qwen3_embedding_4replicas.sh#L5) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L5](../rl_miles/Qwen3-embedding/stop_qwen3_embedding_4replicas.sh#L5) |
| `GPU_IDS` | `${GPU_IDS:-0,1,2,3}` | [L6](../rl_miles/Qwen3-embedding/stop_qwen3_embedding_4replicas.sh#L6) |
| `GPU_IDS` | `${GPU_IDS}` | [L8](../rl_miles/Qwen3-embedding/stop_qwen3_embedding_4replicas.sh#L8) |
| `LOG_DIR` | `${LOG_DIR}` | [L11](../rl_miles/Qwen3-embedding/stop_qwen3_embedding_4replicas.sh#L11) |

<a id="entry-38"></a>

## rl_miles/Qwen3-embedding/stop_similarity_services.sh

[源码](../rl_miles/Qwen3-embedding/stop_similarity_services.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L4](../rl_miles/Qwen3-embedding/stop_similarity_services.sh#L4) |
| `LOG_DIR` | `${LOG_DIR:-${SCRIPT_DIR}/logs}` | [L5](../rl_miles/Qwen3-embedding/stop_similarity_services.sh#L5) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L5](../rl_miles/Qwen3-embedding/stop_similarity_services.sh#L5) |
| `LOG_DIR` | `${LOG_DIR}` | [L9](../rl_miles/Qwen3-embedding/stop_similarity_services.sh#L9) |
| `GPU_IDS` | `${GPU_IDS:-0,1,2,3}` | [L31](../rl_miles/Qwen3-embedding/stop_similarity_services.sh#L31) |

<a id="entry-39"></a>

## rl_miles/bootstrap_miles.sh

[源码](../rl_miles/bootstrap_miles.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L19](../rl_miles/bootstrap_miles.sh#L19) |
| `PYTHON_BASE` | `${PYTHON_BASE:-python3}` | [L20](../rl_miles/bootstrap_miles.sh#L20) |
| `ROOT` | `${ROOT}` | [L21](../rl_miles/bootstrap_miles.sh#L21) |
| `MILES_REPO` | `${MILES_REPO:-${ROOT}/miles}` | [L22](../rl_miles/bootstrap_miles.sh#L22) |
| `PREPARE` | `${PREPARE}` | [L22](../rl_miles/bootstrap_miles.sh#L22) |
| `PYTHON_BASE` | `${PYTHON_BASE}` | [L22](../rl_miles/bootstrap_miles.sh#L22) |
| `MILES_MEGATRON_ROOT` | `${MILES_MEGATRON_ROOT:-${ROOT}/Megatron-LM}` | [L23](../rl_miles/bootstrap_miles.sh#L23) |
| `MILES_SGLANG_ROOT` | `${MILES_SGLANG_ROOT:-${ROOT}/sglang-miles}` | [L25](../rl_miles/bootstrap_miles.sh#L25) |

<a id="entry-40"></a>

## rl_miles/install_official_miles_env.sh

[源码](../rl_miles/install_official_miles_env.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L31](../rl_miles/install_official_miles_env.sh#L31) |
| `MILES_VENV` | `${MILES_VENV:-${ROOT}/.venv}` | [L32](../rl_miles/install_official_miles_env.sh#L32) |
| `ROOT` | `${ROOT}` | [L32](../rl_miles/install_official_miles_env.sh#L32) |
| `PYPI_INDEX` | `${PYPI_INDEX:-https://pypi.org/simple}` | [L33](../rl_miles/install_official_miles_env.sh#L33) |
| `PYTORCH_CU130_INDEX` | `${PYTORCH_CU130_INDEX:-https://download.pytorch.org/whl/cu130}` | [L34](../rl_miles/install_official_miles_env.sh#L34) |
| `MILES_WHEELS_TAG` | `${MILES_WHEELS_TAG:-cu130-torch213-x86_64}` | [L35](../rl_miles/install_official_miles_env.sh#L35) |
| `MILES_WHEELS_REPO` | `${MILES_WHEELS_REPO:-yueming-yuan/miles-wheels}` | [L36](../rl_miles/install_official_miles_env.sh#L36) |
| `DOWNLOAD_WHEELS` | `${DOWNLOAD_WHEELS:-1}` | [L37](../rl_miles/install_official_miles_env.sh#L37) |
| `INSTALL_OPTIONAL_GPU_WHEELS` | `${INSTALL_OPTIONAL_GPU_WHEELS:-1}` | [L38](../rl_miles/install_official_miles_env.sh#L38) |
| `INSTALL_SGLANG_RUNTIME_DEPS` | `${INSTALL_SGLANG_RUNTIME_DEPS:-1}` | [L39](../rl_miles/install_official_miles_env.sh#L39) |
| `INSTALL_SGLANG_EDITABLE` | `${INSTALL_SGLANG_EDITABLE:-1}` | [L40](../rl_miles/install_official_miles_env.sh#L40) |
| `MILES_REPO` | `${MILES_REPO:-${ROOT}/miles}` | [L41](../rl_miles/install_official_miles_env.sh#L41) |
| `MILES_MEGATRON_ROOT` | `${MILES_MEGATRON_ROOT:-${ROOT}/Megatron-LM}` | [L42](../rl_miles/install_official_miles_env.sh#L42) |
| `MILES_SGLANG_ROOT` | `${MILES_SGLANG_ROOT:-${ROOT}/sglang-miles}` | [L43](../rl_miles/install_official_miles_env.sh#L43) |
| `KEEP_PROXY` | `${KEEP_PROXY:-0}` | [L49](../rl_miles/install_official_miles_env.sh#L49) |
| `PYTHON_BASE` | `${PYTHON_BASE:-python3.12}` | [L53](../rl_miles/install_official_miles_env.sh#L53) |
| `PYTHON_BASE` | `${PYTHON_BASE}` | [L54](../rl_miles/install_official_miles_env.sh#L54) |
| `VENV` | `${VENV}` | [L57](../rl_miles/install_official_miles_env.sh#L57) |
| `PY` | `${PY}` | [L59](../rl_miles/install_official_miles_env.sh#L59) |
| `PYPI_INDEX` | `${PYPI_INDEX}` | [L61](../rl_miles/install_official_miles_env.sh#L61) |
| `PIP` | `${PIP[@]}` | [L68](../rl_miles/install_official_miles_env.sh#L68) |
| `DOWNLOAD_WHEELS` | `${DOWNLOAD_WHEELS}` | [L124](../rl_miles/install_official_miles_env.sh#L124) |
| `MILES_WHEELS_TAG` | `${MILES_WHEELS_TAG}` | [L125](../rl_miles/install_official_miles_env.sh#L125) |
| `MILES_WHEELS_REPO` | `${MILES_WHEELS_REPO}` | [L126](../rl_miles/install_official_miles_env.sh#L126) |
| `PYTORCH_CU130_INDEX` | `${PYTORCH_CU130_INDEX}` | [L135](../rl_miles/install_official_miles_env.sh#L135) |
| `INSTALL_OPTIONAL_GPU_WHEELS` | `${INSTALL_OPTIONAL_GPU_WHEELS}` | [L141](../rl_miles/install_official_miles_env.sh#L141) |
| `MEGATRON_ROOT` | `${MEGATRON_ROOT}` | [L163](../rl_miles/install_official_miles_env.sh#L163) |
| `MILES_ROOT` | `${MILES_ROOT}` | [L173](../rl_miles/install_official_miles_env.sh#L173) |
| `INSTALL_SGLANG_EDITABLE` | `${INSTALL_SGLANG_EDITABLE}` | [L177](../rl_miles/install_official_miles_env.sh#L177) |
| `SGLANG_ROOT` | `${SGLANG_ROOT}` | [L179](../rl_miles/install_official_miles_env.sh#L179) |
| `INSTALL_SGLANG_RUNTIME_DEPS` | `${INSTALL_SGLANG_RUNTIME_DEPS}` | [L182](../rl_miles/install_official_miles_env.sh#L182) |
| `BUILD_TORCH_COMPAT` | `${BUILD_TORCH_COMPAT:-1}` | [L189](../rl_miles/install_official_miles_env.sh#L189) |
| `CXX` | `${CXX:-c++}` | [L190](../rl_miles/install_official_miles_env.sh#L190) |

<a id="entry-41"></a>

## rl_miles/official_env.sh

[源码](../rl_miles/official_env.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L5](../rl_miles/official_env.sh#L5) |
| `MILES_CODE_ROOT` | `${MILES_CODE_ROOT:-${_THIS_DIR}}` | [L6](../rl_miles/official_env.sh#L6) |
| `MILES_CODE_ROOT` | `${MILES_CODE_ROOT}` | [L7](../rl_miles/official_env.sh#L7) |
| `MILES_REPO` | `${MILES_REPO:-${MILES_CODE_ROOT}/miles}` | [L7](../rl_miles/official_env.sh#L7) |
| `MILES_VENV` | `${MILES_VENV:-${MILES_CODE_ROOT}/.venv}` | [L8](../rl_miles/official_env.sh#L8) |
| `MILES_MEGATRON_ROOT` | `${MILES_MEGATRON_ROOT:-}` | [L9](../rl_miles/official_env.sh#L9) |
| `MILES_SGLANG_ROOT` | `${MILES_SGLANG_ROOT:-}` | [L12](../rl_miles/official_env.sh#L12) |
| `MILES_ROOT` | `${MILES_ROOT}` | [L15](../rl_miles/official_env.sh#L15) |
| `MILES_MEGATRON_ROOT` | `${MILES_MEGATRON_ROOT}` | [L19](../rl_miles/official_env.sh#L19) |
| `MILES_SGLANG_ROOT` | `${MILES_SGLANG_ROOT}` | [L23](../rl_miles/official_env.sh#L23) |
| `MILES_COW_COMPAT` | `${MILES_COW_COMPAT:-${MILES_CODE_ROOT}/wheels/libtorch_cow_compat.so}` | [L30](../rl_miles/official_env.sh#L30) |
| `MILES_VENV` | `${MILES_VENV}` | [L32](../rl_miles/official_env.sh#L32) |
| `PATH` | `${PATH}` | [L34](../rl_miles/official_env.sh#L34) |
| `MILES_NCCL_SO` | `${MILES_NCCL_SO:-}` | [L45](../rl_miles/official_env.sh#L45) |
| `SGLANG_NCCL_SO_PATH` | `${SGLANG_NCCL_SO_PATH:-${MILES_NCCL_SO:-}}` | [L69](../rl_miles/official_env.sh#L69) |
| `MILES_COW_COMPAT` | `${MILES_COW_COMPAT}` | [L70](../rl_miles/official_env.sh#L70) |
| `MILES_TORCH_PRELOAD` | `${MILES_TORCH_PRELOAD:-${MILES_COW_COMPAT}}` | [L71](../rl_miles/official_env.sh#L71) |
| `MILES_TORCH_PRELOAD` | `${MILES_TORCH_PRELOAD:-}` | [L73](../rl_miles/official_env.sh#L73) |
| `MILES_NCCL_SO` | `${MILES_NCCL_SO}` | [L78](../rl_miles/official_env.sh#L78) |
| `LD_PRELOAD` | `${LD_PRELOAD:-}` | [L79](../rl_miles/official_env.sh#L79) |
| `LD_PRELOAD` | `${LD_PRELOAD}` | [L79](../rl_miles/official_env.sh#L79) |
| `IFS` | `${IFS}` | [L81](../rl_miles/official_env.sh#L81) |
| `PYTHONNOUSERSITE` | `export PYTHONNOUSERSITE=1` | [L89](../rl_miles/official_env.sh#L89) |
| `MILES_EXTRA_PYTHONPATH` | `${MILES_EXTRA_PYTHONPATH:+:${MILES_EXTRA_PYTHONPATH}}` | [L91](../rl_miles/official_env.sh#L91) |
| `MILES_EXTRA_PYTHONPATH` | `${MILES_EXTRA_PYTHONPATH}` | [L91](../rl_miles/official_env.sh#L91) |

<a id="entry-42"></a>

## rl_miles/prepare_rl_data.py

[源码](../rl_miles/prepare_rl_data.py)


### CLI

| 参数 | 声明 | 位置 |
| --- | --- | --- |
| `--input` | `required=True; help='RL/SFT jsonl input file.'` | [L32](../rl_miles/prepare_rl_data.py#L32) |
| `--output` | `required=True; help='Output RL prompt jsonl.'` | [L33](../rl_miles/prepare_rl_data.py#L33) |
| `--template-model` | `required=True; help='Local HF tokenizer/chat-template directory'` | [L34](../rl_miles/prepare_rl_data.py#L34) |
| `--think-mode` | `choices=('on', 'off'); default='off'; help='Bake the selected CoT mode into prepared prompts; regenerate to change it'` | [L35](../rl_miles/prepare_rl_data.py#L35) |
| `--max-prompt-tokens` | `type=int; default=32768` | [L36](../rl_miles/prepare_rl_data.py#L36) |
| `--limit` | `type=int; default=None; help='Only process the first N records.'` | [L37](../rl_miles/prepare_rl_data.py#L37) |

<a id="entry-43"></a>

## rl_miles/run_gspo.sh

[源码](../rl_miles/run_gspo.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L40](../rl_miles/run_gspo.sh#L40) |
| `PROJECT_ROOT` | `${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}` | [L41](../rl_miles/run_gspo.sh#L41) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L41](../rl_miles/run_gspo.sh#L41) |
| `MILES_CODE_ROOT` | `${MILES_CODE_ROOT:-${SCRIPT_DIR}}` | [L42](../rl_miles/run_gspo.sh#L42) |
| `MILES_REPO` | `${MILES_REPO:-${RL_CODE_ROOT}/miles}` | [L43](../rl_miles/run_gspo.sh#L43) |
| `RL_CODE_ROOT` | `${RL_CODE_ROOT}` | [L43](../rl_miles/run_gspo.sh#L43) |
| `MEGATRON_REPO` | `${MEGATRON_REPO:-${RL_CODE_ROOT}/Megatron-LM}` | [L44](../rl_miles/run_gspo.sh#L44) |
| `MILES_MEGATRON_ROOT` | `${MILES_MEGATRON_ROOT:-${MEGATRON_REPO:-${RL_CODE_ROOT}/Megatron-LM}}` | [L44](../rl_miles/run_gspo.sh#L44) |
| `MILES_SGLANG_ROOT` | `${MILES_SGLANG_ROOT:-${SGLANG_REPO:-${RL_CODE_ROOT}/sglang-miles}}` | [L45](../rl_miles/run_gspo.sh#L45) |
| `SGLANG_REPO` | `${SGLANG_REPO:-${RL_CODE_ROOT}/sglang-miles}` | [L45](../rl_miles/run_gspo.sh#L45) |
| `MILES_VENV` | `${MILES_VENV:-${RL_CODE_ROOT}/.venv}` | [L46](../rl_miles/run_gspo.sh#L46) |
| `MILES_VENV` | `${MILES_VENV}` | [L47](../rl_miles/run_gspo.sh#L47) |
| `PYTHON_BIN` | `${PYTHON_BIN:-$(command -v python3)}` | [L51](../rl_miles/run_gspo.sh#L51) |
| `RAY_BIN` | `${RAY_BIN:-$(command -v ray &#124;&#124; true)}` | [L52](../rl_miles/run_gspo.sh#L52) |
| `RAY_BIN` | `${RAY_BIN:-ray}` | [L53](../rl_miles/run_gspo.sh#L53) |
| `MILES_EXTRA_PYTHONPATH` | `${MILES_EXTRA_PYTHONPATH:-}` | [L55](../rl_miles/run_gspo.sh#L55) |
| `MEGATRON_REPO` | `${MEGATRON_REPO}` | [L56](../rl_miles/run_gspo.sh#L56) |
| `MILES_EXTRA_PYTHONPATH` | `${MILES_EXTRA_PYTHONPATH:+:${MILES_EXTRA_PYTHONPATH}}` | [L56](../rl_miles/run_gspo.sh#L56) |
| `MILES_EXTRA_PYTHONPATH` | `${MILES_EXTRA_PYTHONPATH}` | [L56](../rl_miles/run_gspo.sh#L56) |
| `MILES_REPO` | `${MILES_REPO}` | [L56](../rl_miles/run_gspo.sh#L56) |
| `SGLANG_REPO` | `${SGLANG_REPO}` | [L56](../rl_miles/run_gspo.sh#L56) |
| `DRY_RUN` | `${DRY_RUN:-0}` | [L57](../rl_miles/run_gspo.sh#L57) |
| `RAY_STOP_BEFORE_START` | `${RAY_STOP_BEFORE_START:-0}` | [L58](../rl_miles/run_gspo.sh#L58) |
| `DEBUG_ROLLOUT_ONLY` | `${DEBUG_ROLLOUT_ONLY:-0}` | [L59](../rl_miles/run_gspo.sh#L59) |
| `DEBUG_TRAIN_ONLY` | `${DEBUG_TRAIN_ONLY:-0}` | [L60](../rl_miles/run_gspo.sh#L60) |
| `SEQUENCE_PARALLEL` | `${SEQUENCE_PARALLEL:-1}` | [L61](../rl_miles/run_gspo.sh#L61) |
| `COLOCATE` | `${COLOCATE:-1}` | [L62](../rl_miles/run_gspo.sh#L62) |
| `DEBUG_ROLLOUT_ONLY` | `${DEBUG_ROLLOUT_ONLY}` | [L63](../rl_miles/run_gspo.sh#L63) |
| `DEBUG_TRAIN_ONLY` | `${DEBUG_TRAIN_ONLY}` | [L64](../rl_miles/run_gspo.sh#L64) |
| `LEARNING_RATE` | `${LEARNING_RATE:-1e-6}` | [L71](../rl_miles/run_gspo.sh#L71) |
| `LR_DECAY_STYLE` | `${LR_DECAY_STYLE:-constant}` | [L72](../rl_miles/run_gspo.sh#L72) |
| `MIN_LR` | `${MIN_LR:-0.0}` | [L73](../rl_miles/run_gspo.sh#L73) |
| `CLIP_GRAD` | `${CLIP_GRAD:-1.0}` | [L74](../rl_miles/run_gspo.sh#L74) |
| `KL_LOSS_COEF` | `${KL_LOSS_COEF:-0.00}` | [L75](../rl_miles/run_gspo.sh#L75) |
| `USE_KL_LOSS` | `${USE_KL_LOSS:-0}` | [L76](../rl_miles/run_gspo.sh#L76) |
| `REF_LOAD` | `${REF_LOAD:-}` | [L77](../rl_miles/run_gspo.sh#L77) |
| `ROLLOUT_BATCH_SIZE` | `${ROLLOUT_BATCH_SIZE:-128}` | [L78](../rl_miles/run_gspo.sh#L78) |
| `N_SAMPLES_PER_PROMPT` | `${N_SAMPLES_PER_PROMPT:-8}` | [L79](../rl_miles/run_gspo.sh#L79) |
| `GLOBAL_BATCH_SIZE` | `${GLOBAL_BATCH_SIZE:-1024}` | [L80](../rl_miles/run_gspo.sh#L80) |
| `NUM_ROLLOUT` | `${NUM_ROLLOUT:-}` | [L81](../rl_miles/run_gspo.sh#L81) |
| `NUM_EPOCH` | `${NUM_EPOCH:-}` | [L82](../rl_miles/run_gspo.sh#L82) |
| `MAX_PROMPT_TOKENS` | `${MAX_PROMPT_TOKENS:-32768}` | [L83](../rl_miles/run_gspo.sh#L83) |
| `MAX_RESPONSE_TOKENS` | `${MAX_RESPONSE_TOKENS:-8192}` | [L84](../rl_miles/run_gspo.sh#L84) |
| `SEQ_LEN` | `${SEQ_LEN:-40960}` | [L85](../rl_miles/run_gspo.sh#L85) |
| `EVAL_MAX_PROMPT_TOKENS` | `${EVAL_MAX_PROMPT_TOKENS:-65536}` | [L86](../rl_miles/run_gspo.sh#L86) |
| `EVAL_MAX_CONTEXT_LEN` | `${EVAL_MAX_CONTEXT_LEN:-81920}` | [L87](../rl_miles/run_gspo.sh#L87) |
| `QKV_FORMAT` | `${QKV_FORMAT:-thd}` | [L88](../rl_miles/run_gspo.sh#L88) |
| `ROLLOUT_STOP_TOKEN_IDS` | `${ROLLOUT_STOP_TOKEN_IDS:-}` | [L90](../rl_miles/run_gspo.sh#L90) |
| `DEBUG_TRAIN_NUM_ROLLOUT` | `${DEBUG_TRAIN_NUM_ROLLOUT:-0}` | [L95](../rl_miles/run_gspo.sh#L95) |
| `COLOCATE` | `${COLOCATE}` | [L99](../rl_miles/run_gspo.sh#L99) |
| `MAX_RESPONSE_TOKENS` | `${MAX_RESPONSE_TOKENS}` | [L110](../rl_miles/run_gspo.sh#L110) |
| `TRAIN_RESPONSE_BUDGET` | `${TRAIN_RESPONSE_BUDGET}` | [L110](../rl_miles/run_gspo.sh#L110) |
| `SAVE_INTERVAL` | `${SAVE_INTERVAL:-20}` | [L115](../rl_miles/run_gspo.sh#L115) |
| `LOG_INTERVAL` | `${LOG_INTERVAL:-1}` | [L116](../rl_miles/run_gspo.sh#L116) |
| `ROLLOUT_TEMPERATURE` | `${ROLLOUT_TEMPERATURE:-1.0}` | [L117](../rl_miles/run_gspo.sh#L117) |
| `SGLANG_MEM_FRACTION_STATIC` | `${SGLANG_MEM_FRACTION_STATIC:-0.75}` | [L118](../rl_miles/run_gspo.sh#L118) |
| `SGLANG_SERVER_CONCURRENCY` | `${SGLANG_SERVER_CONCURRENCY:-64}` | [L119](../rl_miles/run_gspo.sh#L119) |
| `SGLANG_MAX_RUNNING_REQUESTS` | `${SGLANG_MAX_RUNNING_REQUESTS:-64}` | [L120](../rl_miles/run_gspo.sh#L120) |
| `PROJECT_ROOT` | `${PROJECT_ROOT}` | [L122](../rl_miles/run_gspo.sh#L122) |
| `EVAL_INTERVAL` | `${EVAL_INTERVAL:-20}` | [L124](../rl_miles/run_gspo.sh#L124) |
| `EVAL_GLOBAL_BATCH_SIZE` | `${EVAL_GLOBAL_BATCH_SIZE:-}` | [L125](../rl_miles/run_gspo.sh#L125) |
| `N_SAMPLES_PER_EVAL_PROMPT` | `${N_SAMPLES_PER_EVAL_PROMPT:-1}` | [L126](../rl_miles/run_gspo.sh#L126) |
| `EVAL_MAX_RESPONSE_LEN` | `${EVAL_MAX_RESPONSE_LEN:-16384}` | [L127](../rl_miles/run_gspo.sh#L127) |
| `EVAL_MAX_RESPONSE_LEN` | `${EVAL_MAX_RESPONSE_LEN}` | [L142](../rl_miles/run_gspo.sh#L142) |
| `EVAL_RESPONSE_BUDGET` | `${EVAL_RESPONSE_BUDGET}` | [L142](../rl_miles/run_gspo.sh#L142) |
| `JUDGE_BASE_URL` | `${JUDGE_BASE_URL:-http://<JUDGE_HOST>:<JUDGE_PORT>/v1}` | [L145](../rl_miles/run_gspo.sh#L145) |
| `JUDGE_MODEL` | `${JUDGE_MODEL:-DeepSeek-V4-Flash-0731}` | [L146](../rl_miles/run_gspo.sh#L146) |
| `JUDGE_API_KEY` | `${JUDGE_API_KEY:-EMPTY}` | [L147](../rl_miles/run_gspo.sh#L147) |
| `JUDGE_MAX_TOKENS` | `${JUDGE_MAX_TOKENS:-4096}` | [L148](../rl_miles/run_gspo.sh#L148) |
| `JUDGE_CONCURRENCY` | `${JUDGE_CONCURRENCY:-128}` | [L149](../rl_miles/run_gspo.sh#L149) |
| `JUDGE_WEIGHTS` | `${JUDGE_WEIGHTS:-format:1,factuality:2,consistency:2,realism:2,quality:2,efficiency:1}` | [L150](../rl_miles/run_gspo.sh#L150) |
| `JUDGE_ENABLED` | `${JUDGE_ENABLED:-1}` | [L151](../rl_miles/run_gspo.sh#L151) |
| `JUDGE_WEIGHT` | `${JUDGE_WEIGHT:-1.0}` | [L155](../rl_miles/run_gspo.sh#L155) |
| `SIMILARITY_ENABLED` | `${SIMILARITY_ENABLED:-0}` | [L156](../rl_miles/run_gspo.sh#L156) |
| `SIMILARITY_WEIGHT` | `${SIMILARITY_WEIGHT:-0.0}` | [L157](../rl_miles/run_gspo.sh#L157) |
| `SIMILARITY_URL` | `${SIMILARITY_URL:-http://<SIMILARITY_HOST>:<SIMILARITY_PORT>/similarity}` | [L158](../rl_miles/run_gspo.sh#L158) |
| `SIMILARITY_TIMEOUT` | `${SIMILARITY_TIMEOUT:-60}` | [L159](../rl_miles/run_gspo.sh#L159) |
| `SIMILARITY_CONCURRENCY` | `${SIMILARITY_CONCURRENCY:-128}` | [L160](../rl_miles/run_gspo.sh#L160) |
| `THINK_MODE` | `${THINK_MODE:-0}` | [L161](../rl_miles/run_gspo.sh#L161) |
| `JUDGE_PROMPT_DIR` | `${JUDGE_PROMPT_DIR:-${SCRIPT_DIR}/../prompts}` | [L162](../rl_miles/run_gspo.sh#L162) |
| `SGLANG_QWEN3_5_TEXT_ONLY` | `${SGLANG_QWEN3_5_TEXT_ONLY:-1}` | [L163](../rl_miles/run_gspo.sh#L163) |
| `WORLD_SIZE` | `${WORLD_SIZE:-}` | [L165](../rl_miles/run_gspo.sh#L165) |
| `ACTOR_NUM_NODES` | `${ACTOR_NUM_NODES:-${WORLD_SIZE}}` | [L166](../rl_miles/run_gspo.sh#L166) |
| `WORLD_SIZE` | `${WORLD_SIZE}` | [L166](../rl_miles/run_gspo.sh#L166) |
| `ACTOR_NUM_NODES` | `${ACTOR_NUM_NODES:-16}` | [L168](../rl_miles/run_gspo.sh#L168) |
| `ACTOR_NUM_GPUS_PER_NODE` | `${ACTOR_NUM_GPUS_PER_NODE:-${RESOURCE_GPU:-8}}` | [L170](../rl_miles/run_gspo.sh#L170) |
| `RESOURCE_GPU` | `${RESOURCE_GPU:-8}` | [L170](../rl_miles/run_gspo.sh#L170) |
| `ACTOR_NUM_GPUS_PER_NODE` | `${ACTOR_NUM_GPUS_PER_NODE}` | [L171](../rl_miles/run_gspo.sh#L171) |
| `RAY_NUM_GPUS` | `${RAY_NUM_GPUS:-${RESOURCE_GPU:-${ACTOR_NUM_GPUS_PER_NODE}}}` | [L171](../rl_miles/run_gspo.sh#L171) |
| `RESOURCE_GPU` | `${RESOURCE_GPU:-${ACTOR_NUM_GPUS_PER_NODE}}` | [L171](../rl_miles/run_gspo.sh#L171) |
| `HOSTFILE` | `${HOSTFILE:-}` | [L172](../rl_miles/run_gspo.sh#L172) |
| `MASTER_ADDR` | `${MASTER_ADDR:-127.0.0.1}` | [L173](../rl_miles/run_gspo.sh#L173) |
| `MASTER_PORT` | `${MASTER_PORT:-6379}` | [L174](../rl_miles/run_gspo.sh#L174) |
| `SOCKET_IFNAME` | `${SOCKET_IFNAME:-}` | [L175](../rl_miles/run_gspo.sh#L175) |
| `PYTHON_BIN` | `${PYTHON_BIN}` | [L177](../rl_miles/run_gspo.sh#L177) |
| `ACTOR_NUM_NODES` | `${ACTOR_NUM_NODES}` | [L210](../rl_miles/run_gspo.sh#L210) |
| `HOSTFILE` | `${HOSTFILE}` | [L210](../rl_miles/run_gspo.sh#L210) |
| `RANK` | `${RANK:-}` | [L210](../rl_miles/run_gspo.sh#L210) |
| `MASTER_ADDR` | `${MASTER_ADDR}` | [L211](../rl_miles/run_gspo.sh#L211) |
| `LOCAL_IP` | `${LOCAL_IP:-}` | [L221](../rl_miles/run_gspo.sh#L221) |
| `LOCAL_IP` | `${LOCAL_IP}` | [L222](../rl_miles/run_gspo.sh#L222) |
| `SOCKET_IFNAME` | `${SOCKET_IFNAME}` | [L225](../rl_miles/run_gspo.sh#L225) |
| `NO_PROXY` | `${NO_PROXY:-${no_proxy:-}}` | [L231](../rl_miles/run_gspo.sh#L231) |
| `RUNTIME_NO_PROXY` | `${RUNTIME_NO_PROXY}` | [L234](../rl_miles/run_gspo.sh#L234) |
| `RAY_BIN` | `${RAY_BIN}` | [L260](../rl_miles/run_gspo.sh#L260) |
| `RL_DEBUG_LOG` | `${RL_DEBUG_LOG:-1}` | [L272](../rl_miles/run_gspo.sh#L272) |
| `RL_DEBUG_LOG` | `${RL_DEBUG_LOG}` | [L274](../rl_miles/run_gspo.sh#L274) |
| `RANK` | `${RANK:-<unset>}` | [L279](../rl_miles/run_gspo.sh#L279) |
| `WORLD_SIZE` | `${WORLD_SIZE:-<unset>}` | [L279](../rl_miles/run_gspo.sh#L279) |
| `NPROC_PER_NODE` | `${NPROC_PER_NODE:-<unset>}` | [L280](../rl_miles/run_gspo.sh#L280) |
| `RESOURCE_GPU` | `${RESOURCE_GPU:-<unset>}` | [L280](../rl_miles/run_gspo.sh#L280) |
| `MASTER_ADDR` | `${MASTER_ADDR:-<unset>}` | [L281](../rl_miles/run_gspo.sh#L281) |
| `MASTER_PORT` | `${MASTER_PORT:-<unset>}` | [L281](../rl_miles/run_gspo.sh#L281) |
| `DRY_RUN` | `${DRY_RUN}` | [L329](../rl_miles/run_gspo.sh#L329) |
| `MODEL_TAG` | `${MODEL_TAG:?MODEL_TAG must be set by the model-specific wrapper}` | [L395](../rl_miles/run_gspo.sh#L395) |
| `MODEL_HF_DEFAULT` | `${MODEL_HF_DEFAULT:?MODEL_HF_DEFAULT must be set by the model-specific wrapper}` | [L396](../rl_miles/run_gspo.sh#L396) |
| `SAVE_DEFAULT` | `${SAVE_DEFAULT:?SAVE_DEFAULT must be set by the model-specific wrapper}` | [L397](../rl_miles/run_gspo.sh#L397) |
| `TP_DEFAULT` | `${TP_DEFAULT:?TP_DEFAULT must be set by the model-specific wrapper}` | [L398](../rl_miles/run_gspo.sh#L398) |
| `PP_DEFAULT` | `${PP_DEFAULT:?PP_DEFAULT must be set by the model-specific wrapper}` | [L399](../rl_miles/run_gspo.sh#L399) |
| `CP_DEFAULT` | `${CP_DEFAULT:?CP_DEFAULT must be set by the model-specific wrapper}` | [L400](../rl_miles/run_gspo.sh#L400) |
| `EP_DEFAULT` | `${EP_DEFAULT:?EP_DEFAULT must be set by the model-specific wrapper}` | [L401](../rl_miles/run_gspo.sh#L401) |
| `ETP_DEFAULT` | `${ETP_DEFAULT:?ETP_DEFAULT must be set by the model-specific wrapper}` | [L402](../rl_miles/run_gspo.sh#L402) |
| `MAX_TOKENS_PER_GPU_DEFAULT` | `${MAX_TOKENS_PER_GPU_DEFAULT:?MAX_TOKENS_PER_GPU_DEFAULT must be set by the model-specific wrapper}` | [L403](../rl_miles/run_gspo.sh#L403) |
| `LOG_PROBS_MAX_TOKENS_PER_GPU_DEFAULT` | `${LOG_PROBS_MAX_TOKENS_PER_GPU_DEFAULT:-${MAX_TOKENS_PER_GPU_DEFAULT}}` | [L404](../rl_miles/run_gspo.sh#L404) |
| `MAX_TOKENS_PER_GPU_DEFAULT` | `${MAX_TOKENS_PER_GPU_DEFAULT}` | [L404](../rl_miles/run_gspo.sh#L404) |
| `ROLLOUT_GPUS_PER_ENGINE_DEFAULT` | `${ROLLOUT_GPUS_PER_ENGINE_DEFAULT:?ROLLOUT_GPUS_PER_ENGINE_DEFAULT must be set by the model-specific wrapper}` | [L405](../rl_miles/run_gspo.sh#L405) |
| `DECODER_LAST_PIPELINE_NUM_LAYERS_DEFAULT` | `${DECODER_LAST_PIPELINE_NUM_LAYERS_DEFAULT:-}` | [L406](../rl_miles/run_gspo.sh#L406) |
| `MODEL_CONFIG_PATH` | `${MODEL_CONFIG_PATH:-${SCRIPT_DIR}/model_configs/${MODEL_TAG}.sh}` | [L407](../rl_miles/run_gspo.sh#L407) |
| `MODEL_TAG` | `${MODEL_TAG}` | [L407](../rl_miles/run_gspo.sh#L407) |
| `MODEL_CONFIG_PATH` | `${MODEL_CONFIG_PATH}` | [L408](../rl_miles/run_gspo.sh#L408) |
| `TENSOR_PARALLEL_SIZE` | `${TENSOR_PARALLEL_SIZE:-${TP_DEFAULT}}` | [L415](../rl_miles/run_gspo.sh#L415) |
| `TP_DEFAULT` | `${TP_DEFAULT}` | [L415](../rl_miles/run_gspo.sh#L415) |
| `PIPELINE_PARALLEL_SIZE` | `${PIPELINE_PARALLEL_SIZE:-${PP_DEFAULT}}` | [L416](../rl_miles/run_gspo.sh#L416) |
| `PP_DEFAULT` | `${PP_DEFAULT}` | [L416](../rl_miles/run_gspo.sh#L416) |
| `CONTEXT_PARALLEL_SIZE` | `${CONTEXT_PARALLEL_SIZE:-${CP_DEFAULT}}` | [L417](../rl_miles/run_gspo.sh#L417) |
| `CP_DEFAULT` | `${CP_DEFAULT}` | [L417](../rl_miles/run_gspo.sh#L417) |
| `EP_DEFAULT` | `${EP_DEFAULT}` | [L418](../rl_miles/run_gspo.sh#L418) |
| `EXPERT_PARALLEL_SIZE` | `${EXPERT_PARALLEL_SIZE:-${EP_DEFAULT}}` | [L418](../rl_miles/run_gspo.sh#L418) |
| `ETP_DEFAULT` | `${ETP_DEFAULT}` | [L419](../rl_miles/run_gspo.sh#L419) |
| `EXPERT_TENSOR_PARALLEL_SIZE` | `${EXPERT_TENSOR_PARALLEL_SIZE:-${ETP_DEFAULT}}` | [L419](../rl_miles/run_gspo.sh#L419) |
| `DECODER_LAST_PIPELINE_NUM_LAYERS` | `${DECODER_LAST_PIPELINE_NUM_LAYERS+x}` | [L420](../rl_miles/run_gspo.sh#L420) |
| `DECODER_LAST_PIPELINE_NUM_LAYERS_DEFAULT` | `${DECODER_LAST_PIPELINE_NUM_LAYERS_DEFAULT}` | [L421](../rl_miles/run_gspo.sh#L421) |
| `MAX_TOKENS_PER_GPU` | `${MAX_TOKENS_PER_GPU:-${MAX_TOKENS_PER_GPU_DEFAULT}}` | [L423](../rl_miles/run_gspo.sh#L423) |
| `LOG_PROBS_MAX_TOKENS_PER_GPU` | `${LOG_PROBS_MAX_TOKENS_PER_GPU:-${LOG_PROBS_MAX_TOKENS_PER_GPU_DEFAULT}}` | [L424](../rl_miles/run_gspo.sh#L424) |
| `LOG_PROBS_MAX_TOKENS_PER_GPU_DEFAULT` | `${LOG_PROBS_MAX_TOKENS_PER_GPU_DEFAULT}` | [L424](../rl_miles/run_gspo.sh#L424) |
| `LOG_PROBS_CHUNK_SIZE` | `${LOG_PROBS_CHUNK_SIZE:-128}` | [L425](../rl_miles/run_gspo.sh#L425) |
| `ROLLOUT_GPUS_PER_ENGINE` | `${ROLLOUT_GPUS_PER_ENGINE:-${ROLLOUT_GPUS_PER_ENGINE_DEFAULT}}` | [L426](../rl_miles/run_gspo.sh#L426) |
| `ROLLOUT_GPUS_PER_ENGINE_DEFAULT` | `${ROLLOUT_GPUS_PER_ENGINE_DEFAULT}` | [L426](../rl_miles/run_gspo.sh#L426) |
| `TRAIN_DRIVER` | `${TRAIN_DRIVER:-train.py}` | [L427](../rl_miles/run_gspo.sh#L427) |
| `ROLLOUT_NUM_GPUS` | `${ROLLOUT_NUM_GPUS:-}` | [L428](../rl_miles/run_gspo.sh#L428) |
| `ROLLOUT_FUNCTION_PATH` | `${ROLLOUT_FUNCTION_PATH:-world_model_rl.rollout.raw_train_rollout}` | [L429](../rl_miles/run_gspo.sh#L429) |
| `MILES_LOAD` | `${MILES_LOAD:-${MODEL_HF_DEFAULT}}` | [L431](../rl_miles/run_gspo.sh#L431) |
| `MODEL_HF_DEFAULT` | `${MODEL_HF_DEFAULT}` | [L431](../rl_miles/run_gspo.sh#L431) |
| `MILES_HF_CHECKPOINT` | `${MILES_HF_CHECKPOINT:-${MODEL_HF_DEFAULT}}` | [L432](../rl_miles/run_gspo.sh#L432) |
| `MILES_SAVE` | `${MILES_SAVE:-${SAVE_DEFAULT}}` | [L433](../rl_miles/run_gspo.sh#L433) |
| `SAVE_DEFAULT` | `${SAVE_DEFAULT}` | [L433](../rl_miles/run_gspo.sh#L433) |
| `MILES_SAVE` | `${MILES_SAVE}` | [L434](../rl_miles/run_gspo.sh#L434) |
| `ROLLOUT_OUTPUT_DIR` | `${ROLLOUT_OUTPUT_DIR:-${MILES_SAVE}/rollout_jsonl}` | [L434](../rl_miles/run_gspo.sh#L434) |
| `MEGATRON_TO_HF_MODE` | `${MEGATRON_TO_HF_MODE:-bridge}` | [L435](../rl_miles/run_gspo.sh#L435) |
| `MILES_PROMPT_DATA` | `${MILES_PROMPT_DATA:-${PROJECT_ROOT}/data/rl/datasets/world_model_rl/train.jsonl}` | [L436](../rl_miles/run_gspo.sh#L436) |
| `MILES_DATA_SOURCE_PATH` | `${MILES_DATA_SOURCE_PATH:-world_model_rl.data_source.WorldModelRLJsonlDataSource}` | [L437](../rl_miles/run_gspo.sh#L437) |
| `CUSTOM_RM_PATH` | `${CUSTOM_RM_PATH-world_model_rl.reward.custom_rm}` | [L438](../rl_miles/run_gspo.sh#L438) |
| `RM_TYPE` | `${RM_TYPE:-}` | [L439](../rl_miles/run_gspo.sh#L439) |
| `MILES_HF_CHECKPOINT` | `${MILES_HF_CHECKPOINT}` | [L443](../rl_miles/run_gspo.sh#L443) |
| `MILES_PROMPT_DATA` | `${MILES_PROMPT_DATA}` | [L448](../rl_miles/run_gspo.sh#L448) |
| `MILES_LOAD` | `${MILES_LOAD}` | [L456](../rl_miles/run_gspo.sh#L456) |
| `SAVE_INTERVAL` | `${SAVE_INTERVAL}` | [L458](../rl_miles/run_gspo.sh#L458) |
| `MEGATRON_TO_HF_MODE` | `${MEGATRON_TO_HF_MODE}` | [L459](../rl_miles/run_gspo.sh#L459) |
| `REF_LOAD` | `${REF_LOAD}` | [L461](../rl_miles/run_gspo.sh#L461) |
| `MILES_DATA_SOURCE_PATH` | `${MILES_DATA_SOURCE_PATH}` | [L467](../rl_miles/run_gspo.sh#L467) |
| `ROLLOUT_BATCH_SIZE` | `${ROLLOUT_BATCH_SIZE}` | [L473](../rl_miles/run_gspo.sh#L473) |
| `N_SAMPLES_PER_PROMPT` | `${N_SAMPLES_PER_PROMPT}` | [L474](../rl_miles/run_gspo.sh#L474) |
| `GLOBAL_BATCH_SIZE` | `${GLOBAL_BATCH_SIZE}` | [L476](../rl_miles/run_gspo.sh#L476) |
| `MAX_PROMPT_TOKENS` | `${MAX_PROMPT_TOKENS}` | [L477](../rl_miles/run_gspo.sh#L477) |
| `SEQ_LEN` | `${SEQ_LEN}` | [L479](../rl_miles/run_gspo.sh#L479) |
| `ROLLOUT_TEMPERATURE` | `${ROLLOUT_TEMPERATURE}` | [L480](../rl_miles/run_gspo.sh#L480) |
| `THINK_MODE` | `${THINK_MODE}` | [L485](../rl_miles/run_gspo.sh#L485) |
| `ROLLOUT_STOP_TOKEN_IDS` | `${ROLLOUT_STOP_TOKEN_IDS}` | [L495](../rl_miles/run_gspo.sh#L495) |
| `CUSTOM_RM_PATH` | `${CUSTOM_RM_PATH}` | [L498](../rl_miles/run_gspo.sh#L498) |
| `RM_TYPE` | `${RM_TYPE}` | [L501](../rl_miles/run_gspo.sh#L501) |
| `NUM_ROLLOUT` | `${NUM_ROLLOUT}` | [L505](../rl_miles/run_gspo.sh#L505) |
| `NUM_EPOCH` | `${NUM_EPOCH}` | [L507](../rl_miles/run_gspo.sh#L507) |
| `ADVANTAGE_ESTIMATOR` | `${ADVANTAGE_ESTIMATOR:-gspo}` | [L515](../rl_miles/run_gspo.sh#L515) |
| `KL_LOSS_COEF` | `${KL_LOSS_COEF}` | [L516](../rl_miles/run_gspo.sh#L516) |
| `USE_KL_LOSS` | `${USE_KL_LOSS}` | [L522](../rl_miles/run_gspo.sh#L522) |
| `LEARNING_RATE` | `${LEARNING_RATE}` | [L528](../rl_miles/run_gspo.sh#L528) |
| `LR_DECAY_STYLE` | `${LR_DECAY_STYLE}` | [L529](../rl_miles/run_gspo.sh#L529) |
| `MIN_LR` | `${MIN_LR}` | [L530](../rl_miles/run_gspo.sh#L530) |
| `CLIP_GRAD` | `${CLIP_GRAD}` | [L531](../rl_miles/run_gspo.sh#L531) |
| `TENSOR_PARALLEL_SIZE` | `${TENSOR_PARALLEL_SIZE}` | [L541](../rl_miles/run_gspo.sh#L541) |
| `PIPELINE_PARALLEL_SIZE` | `${PIPELINE_PARALLEL_SIZE}` | [L542](../rl_miles/run_gspo.sh#L542) |
| `CONTEXT_PARALLEL_SIZE` | `${CONTEXT_PARALLEL_SIZE}` | [L543](../rl_miles/run_gspo.sh#L543) |
| `EXPERT_PARALLEL_SIZE` | `${EXPERT_PARALLEL_SIZE}` | [L544](../rl_miles/run_gspo.sh#L544) |
| `EXPERT_TENSOR_PARALLEL_SIZE` | `${EXPERT_TENSOR_PARALLEL_SIZE}` | [L545](../rl_miles/run_gspo.sh#L545) |
| `MAX_TOKENS_PER_GPU` | `${MAX_TOKENS_PER_GPU}` | [L550](../rl_miles/run_gspo.sh#L550) |
| `LOG_PROBS_MAX_TOKENS_PER_GPU` | `${LOG_PROBS_MAX_TOKENS_PER_GPU}` | [L551](../rl_miles/run_gspo.sh#L551) |
| `LOG_PROBS_CHUNK_SIZE` | `${LOG_PROBS_CHUNK_SIZE}` | [L552](../rl_miles/run_gspo.sh#L552) |
| `QKV_FORMAT` | `${QKV_FORMAT}` | [L554](../rl_miles/run_gspo.sh#L554) |
| `SEQUENCE_PARALLEL` | `${SEQUENCE_PARALLEL}` | [L557](../rl_miles/run_gspo.sh#L557) |
| `DECODER_LAST_PIPELINE_NUM_LAYERS` | `${DECODER_LAST_PIPELINE_NUM_LAYERS}` | [L561](../rl_miles/run_gspo.sh#L561) |
| `ROLLOUT_GPUS_PER_ENGINE` | `${ROLLOUT_GPUS_PER_ENGINE}` | [L566](../rl_miles/run_gspo.sh#L566) |
| `SGLANG_MEM_FRACTION_STATIC` | `${SGLANG_MEM_FRACTION_STATIC}` | [L567](../rl_miles/run_gspo.sh#L567) |
| `SGLANG_SERVER_CONCURRENCY` | `${SGLANG_SERVER_CONCURRENCY}` | [L568](../rl_miles/run_gspo.sh#L568) |
| `SGLANG_MAX_RUNNING_REQUESTS` | `${SGLANG_MAX_RUNNING_REQUESTS}` | [L569](../rl_miles/run_gspo.sh#L569) |
| `ROLLOUT_NUM_GPUS` | `${ROLLOUT_NUM_GPUS}` | [L571](../rl_miles/run_gspo.sh#L571) |
| `ROLLOUT_FUNCTION_PATH` | `${ROLLOUT_FUNCTION_PATH}` | [L575](../rl_miles/run_gspo.sh#L575) |
| `LOG_INTERVAL` | `${LOG_INTERVAL}` | [L576](../rl_miles/run_gspo.sh#L576) |
| `SAVE_DEBUG_ROLLOUT_DATA` | `${SAVE_DEBUG_ROLLOUT_DATA:-}` | [L593](../rl_miles/run_gspo.sh#L593) |
| `SAVE_DEBUG_ROLLOUT_DATA` | `${SAVE_DEBUG_ROLLOUT_DATA}` | [L594](../rl_miles/run_gspo.sh#L594) |
| `LOAD_DEBUG_ROLLOUT_DATA` | `${LOAD_DEBUG_ROLLOUT_DATA:-}` | [L596](../rl_miles/run_gspo.sh#L596) |
| `LOAD_DEBUG_ROLLOUT_DATA` | `${LOAD_DEBUG_ROLLOUT_DATA}` | [L597](../rl_miles/run_gspo.sh#L597) |
| `LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE` | `${LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE:-}` | [L599](../rl_miles/run_gspo.sh#L599) |
| `LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE` | `${LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE}` | [L600](../rl_miles/run_gspo.sh#L600) |
| `LR_DECAY_ITERS` | `${LR_DECAY_ITERS:-}` | [L602](../rl_miles/run_gspo.sh#L602) |
| `LR_DECAY_ITERS` | `${LR_DECAY_ITERS}` | [L603](../rl_miles/run_gspo.sh#L603) |
| `WANDB_API_KEY` | `${WANDB_API_KEY:-}` | [L607](../rl_miles/run_gspo.sh#L607) |
| `WANDB_PROJECT` | `${WANDB_PROJECT:-world-model-rl}` | [L610](../rl_miles/run_gspo.sh#L610) |
| `WANDB_GROUP` | `${WANDB_GROUP:-qwen35-world-model-gspo}` | [L611](../rl_miles/run_gspo.sh#L611) |
| `EVAL_DATA` | `${EVAL_DATA:-}` | [L616](../rl_miles/run_gspo.sh#L616) |
| `EVAL_DATA` | `${EVAL_DATA}` | [L616](../rl_miles/run_gspo.sh#L616) |
| `EVAL_DATA_LIMIT` | `${EVAL_DATA_LIMIT:-}` | [L623](../rl_miles/run_gspo.sh#L623) |
| `EVAL_DATA_LIMIT` | `${EVAL_DATA_LIMIT}` | [L624](../rl_miles/run_gspo.sh#L624) |
| `EVAL_INTERVAL` | `${EVAL_INTERVAL}` | [L632](../rl_miles/run_gspo.sh#L632) |
| `EVAL_FUNCTION_PATH` | `${EVAL_FUNCTION_PATH:-world_model_rl.eval_rollout.raw_eval_rollout}` | [L633](../rl_miles/run_gspo.sh#L633) |
| `EVAL_DATA_SPEC` | `${EVAL_DATA_SPEC}` | [L634](../rl_miles/run_gspo.sh#L634) |
| `EVAL_INPUT_KEY` | `${EVAL_INPUT_KEY:-prompt}` | [L635](../rl_miles/run_gspo.sh#L635) |
| `EVAL_LABEL_KEY` | `${EVAL_LABEL_KEY:-ground_truth}` | [L636](../rl_miles/run_gspo.sh#L636) |
| `N_SAMPLES_PER_EVAL_PROMPT` | `${N_SAMPLES_PER_EVAL_PROMPT}` | [L637](../rl_miles/run_gspo.sh#L637) |
| `EVAL_MAX_PROMPT_TOKENS` | `${EVAL_MAX_PROMPT_TOKENS}` | [L638](../rl_miles/run_gspo.sh#L638) |
| `EVAL_MAX_CONTEXT_LEN` | `${EVAL_MAX_CONTEXT_LEN}` | [L639](../rl_miles/run_gspo.sh#L639) |
| `EVAL_GLOBAL_BATCH_SIZE` | `${EVAL_GLOBAL_BATCH_SIZE}` | [L644](../rl_miles/run_gspo.sh#L644) |
| `RUNTIME_PYTHONPATH` | `${RUNTIME_PYTHONPATH}` | [L654](../rl_miles/run_gspo.sh#L654) |
| `MILES_PROMPT_DATA_LIMIT` | `${MILES_PROMPT_DATA_LIMIT:-}` | [L655](../rl_miles/run_gspo.sh#L655) |
| `SGLANG_DISABLE_CUDNN_CHECK` | `${SGLANG_DISABLE_CUDNN_CHECK:-1}` | [L657](../rl_miles/run_gspo.sh#L657) |
| `PYTORCH_CUDA_ALLOC_CONF` | `${PYTORCH_CUDA_ALLOC_CONF:-}` | [L658](../rl_miles/run_gspo.sh#L658) |
| `JUDGE_BASE_URL` | `${JUDGE_BASE_URL}` | [L659](../rl_miles/run_gspo.sh#L659) |
| `JUDGE_MODEL` | `${JUDGE_MODEL}` | [L660](../rl_miles/run_gspo.sh#L660) |
| `JUDGE_API_KEY` | `${JUDGE_API_KEY}` | [L661](../rl_miles/run_gspo.sh#L661) |
| `JUDGE_MAX_TOKENS` | `${JUDGE_MAX_TOKENS}` | [L662](../rl_miles/run_gspo.sh#L662) |
| `JUDGE_CONCURRENCY` | `${JUDGE_CONCURRENCY}` | [L663](../rl_miles/run_gspo.sh#L663) |
| `JUDGE_WEIGHTS` | `${JUDGE_WEIGHTS}` | [L664](../rl_miles/run_gspo.sh#L664) |
| `JUDGE_ENABLED` | `${JUDGE_ENABLED}` | [L665](../rl_miles/run_gspo.sh#L665) |
| `JUDGE_WEIGHT` | `${JUDGE_WEIGHT}` | [L666](../rl_miles/run_gspo.sh#L666) |
| `SIMILARITY_ENABLED` | `${SIMILARITY_ENABLED}` | [L667](../rl_miles/run_gspo.sh#L667) |
| `SIMILARITY_WEIGHT` | `${SIMILARITY_WEIGHT}` | [L668](../rl_miles/run_gspo.sh#L668) |
| `SIMILARITY_URL` | `${SIMILARITY_URL}` | [L669](../rl_miles/run_gspo.sh#L669) |
| `SIMILARITY_TIMEOUT` | `${SIMILARITY_TIMEOUT}` | [L670](../rl_miles/run_gspo.sh#L670) |
| `SIMILARITY_CONCURRENCY` | `${SIMILARITY_CONCURRENCY}` | [L671](../rl_miles/run_gspo.sh#L671) |
| `JUDGE_PROMPT_DIR` | `${JUDGE_PROMPT_DIR}` | [L673](../rl_miles/run_gspo.sh#L673) |
| `SGLANG_QWEN3_5_TEXT_ONLY` | `${SGLANG_QWEN3_5_TEXT_ONLY}` | [L674](../rl_miles/run_gspo.sh#L674) |
| `MILES_TORCH_PRELOAD` | `${MILES_TORCH_PRELOAD:-}` | [L675](../rl_miles/run_gspo.sh#L675) |
| `MILES_NCCL_SO` | `${MILES_NCCL_SO:-}` | [L676](../rl_miles/run_gspo.sh#L676) |
| `SGLANG_NCCL_SO_PATH` | `${SGLANG_NCCL_SO_PATH:-${MILES_NCCL_SO:-}}` | [L677](../rl_miles/run_gspo.sh#L677) |
| `VIRTUAL_ENV` | `${VIRTUAL_ENV:-${MILES_VENV}}` | [L679](../rl_miles/run_gspo.sh#L679) |
| `PATH` | `${PATH}` | [L680](../rl_miles/run_gspo.sh#L680) |
| `PYTHONNOUSERSITE` | `${PYTHONNOUSERSITE:-1}` | [L681](../rl_miles/run_gspo.sh#L681) |
| `ROLLOUT_OUTPUT_DIR` | `${ROLLOUT_OUTPUT_DIR}` | [L684](../rl_miles/run_gspo.sh#L684) |
| `MILES_HF_EXPORT_MODE` | `${MILES_HF_EXPORT_MODE:-}` | [L685](../rl_miles/run_gspo.sh#L685) |
| `RUNTIME_PYTHONPATH` | `env("RUNTIME_PYTHONPATH"),` | [L694](../rl_miles/run_gspo.sh#L694) |
| `PATH` | `env("PATH"),` | [L695](../rl_miles/run_gspo.sh#L695) |
| `VIRTUAL_ENV` | `env("VIRTUAL_ENV"),` | [L696](../rl_miles/run_gspo.sh#L696) |
| `PYTHONNOUSERSITE` | `env("PYTHONNOUSERSITE", "1"),` | [L697](../rl_miles/run_gspo.sh#L697) |
| `MILES_TORCH_PRELOAD` | `env("MILES_TORCH_PRELOAD"),` | [L700](../rl_miles/run_gspo.sh#L700) |
| `MILES_NCCL_SO` | `env("MILES_NCCL_SO"),` | [L701](../rl_miles/run_gspo.sh#L701) |
| `MILES_TORCH_PRELOAD` | `env("MILES_TORCH_PRELOAD", ""),` | [L703](../rl_miles/run_gspo.sh#L703) |
| `MILES_NCCL_SO` | `env("MILES_NCCL_SO", ""),` | [L704](../rl_miles/run_gspo.sh#L704) |
| `SGLANG_NCCL_SO_PATH` | `env("SGLANG_NCCL_SO_PATH", ""),` | [L705](../rl_miles/run_gspo.sh#L705) |
| `PYTORCH_CUDA_ALLOC_CONF` | `env("PYTORCH_CUDA_ALLOC_CONF", ""),` | [L706](../rl_miles/run_gspo.sh#L706) |
| `SOCKET_IFNAME` | `env("SOCKET_IFNAME", "eth0"),` | [L707](../rl_miles/run_gspo.sh#L707) |
| `MASTER_ADDR` | `env("MASTER_ADDR"),` | [L710](../rl_miles/run_gspo.sh#L710) |
| `RUNTIME_NO_PROXY` | `env("RUNTIME_NO_PROXY"),` | [L711](../rl_miles/run_gspo.sh#L711) |
| `JUDGE_BASE_URL` | `env("JUDGE_BASE_URL"),` | [L713](../rl_miles/run_gspo.sh#L713) |
| `JUDGE_MODEL` | `env("JUDGE_MODEL"),` | [L714](../rl_miles/run_gspo.sh#L714) |
| `JUDGE_API_KEY` | `env("JUDGE_API_KEY"),` | [L715](../rl_miles/run_gspo.sh#L715) |
| `JUDGE_MAX_TOKENS` | `env("JUDGE_MAX_TOKENS", "4096"),` | [L716](../rl_miles/run_gspo.sh#L716) |
| `JUDGE_CONCURRENCY` | `env("JUDGE_CONCURRENCY", "128"),` | [L717](../rl_miles/run_gspo.sh#L717) |
| `JUDGE_WEIGHTS` | `env("JUDGE_WEIGHTS"),` | [L718](../rl_miles/run_gspo.sh#L718) |
| `JUDGE_ENABLED` | `env("JUDGE_ENABLED", "1"),` | [L719](../rl_miles/run_gspo.sh#L719) |
| `JUDGE_WEIGHT` | `env("JUDGE_WEIGHT", "1.0"),` | [L720](../rl_miles/run_gspo.sh#L720) |
| `SIMILARITY_ENABLED` | `env("SIMILARITY_ENABLED", "0"),` | [L721](../rl_miles/run_gspo.sh#L721) |
| `SIMILARITY_WEIGHT` | `env("SIMILARITY_WEIGHT", "0.0"),` | [L722](../rl_miles/run_gspo.sh#L722) |
| `SIMILARITY_URL` | `env("SIMILARITY_URL"),` | [L723](../rl_miles/run_gspo.sh#L723) |
| `SIMILARITY_TIMEOUT` | `env("SIMILARITY_TIMEOUT", "60"),` | [L724](../rl_miles/run_gspo.sh#L724) |
| `SIMILARITY_CONCURRENCY` | `env("SIMILARITY_CONCURRENCY", "128"),` | [L725](../rl_miles/run_gspo.sh#L725) |
| `THINK_MODE` | `env("THINK_MODE", "0"),` | [L726](../rl_miles/run_gspo.sh#L726) |
| `JUDGE_PROMPT_DIR` | `env("JUDGE_PROMPT_DIR"),` | [L727](../rl_miles/run_gspo.sh#L727) |
| `EVAL_DATA` | `env("EVAL_DATA"),` | [L728](../rl_miles/run_gspo.sh#L728) |
| `WM_EVAL_DATA_LIMIT` | `env("WM_EVAL_DATA_LIMIT"),` | [L729](../rl_miles/run_gspo.sh#L729) |
| `ROLLOUT_OUTPUT_DIR` | `env("ROLLOUT_OUTPUT_DIR"),` | [L730](../rl_miles/run_gspo.sh#L730) |
| `MILES_HF_EXPORT_MODE` | `env("MILES_HF_EXPORT_MODE", ""),` | [L731](../rl_miles/run_gspo.sh#L731) |
| `SGLANG_DISABLE_CUDNN_CHECK` | `env("SGLANG_DISABLE_CUDNN_CHECK", "1"),` | [L732](../rl_miles/run_gspo.sh#L732) |
| `SGLANG_QWEN3_5_TEXT_ONLY` | `env("SGLANG_QWEN3_5_TEXT_ONLY", "1"),` | [L733](../rl_miles/run_gspo.sh#L733) |
| `MILES_PROMPT_DATA_LIMIT` | `env("MILES_PROMPT_DATA_LIMIT"),` | [L734](../rl_miles/run_gspo.sh#L734) |
| `TRAIN_DRIVER` | `${TRAIN_DRIVER}` | [L752](../rl_miles/run_gspo.sh#L752) |
| `MODEL_ARGS` | `${MODEL_ARGS[@]}` | [L753](../rl_miles/run_gspo.sh#L753) |
| `CKPT_ARGS` | `${CKPT_ARGS[@]}` | [L754](../rl_miles/run_gspo.sh#L754) |
| `ROLLOUT_ARGS` | `${ROLLOUT_ARGS[@]}` | [L755](../rl_miles/run_gspo.sh#L755) |
| `OPTIMIZER_ARGS` | `${OPTIMIZER_ARGS[@]}` | [L756](../rl_miles/run_gspo.sh#L756) |
| `GSPO_ARGS` | `${GSPO_ARGS[@]}` | [L757](../rl_miles/run_gspo.sh#L757) |
| `PERF_ARGS` | `${PERF_ARGS[@]}` | [L758](../rl_miles/run_gspo.sh#L758) |
| `SGLANG_ARGS` | `${SGLANG_ARGS[@]}` | [L759](../rl_miles/run_gspo.sh#L759) |
| `MISC_ARGS` | `${MISC_ARGS[@]}` | [L760](../rl_miles/run_gspo.sh#L760) |
| `EVAL_ARGS` | `${EVAL_ARGS[@]}` | [L761](../rl_miles/run_gspo.sh#L761) |
| `WANDB_ARGS` | `${WANDB_ARGS[@]}` | [L762](../rl_miles/run_gspo.sh#L762) |
| `NVLINK_COUNT` | `${NVLINK_COUNT}` | [L802](../rl_miles/run_gspo.sh#L802) |
| `HAS_NVLINK` | `${HAS_NVLINK}` | [L803](../rl_miles/run_gspo.sh#L803) |
| `NCCL_NVLS_ENABLE` | `${NCCL_NVLS_ENABLE:-${HAS_NVLINK}}` | [L803](../rl_miles/run_gspo.sh#L803) |
| `RAY_STOP_BEFORE_START` | `${RAY_STOP_BEFORE_START}` | [L809](../rl_miles/run_gspo.sh#L809) |
| `BOOTSTRAP_MODE` | `${BOOTSTRAP_MODE}` | [L818](../rl_miles/run_gspo.sh#L818) |
| `MASTER_PORT` | `${MASTER_PORT}` | [L819](../rl_miles/run_gspo.sh#L819) |
| `RANK` | `${RANK}` | [L819](../rl_miles/run_gspo.sh#L819) |
| `RAY_CLUSTER_READY_TIMEOUT_SEC` | `${RAY_CLUSTER_READY_TIMEOUT_SEC:-300}` | [L819](../rl_miles/run_gspo.sh#L819) |
| `RAY_HEAD_READY_TIMEOUT_SEC` | `${RAY_HEAD_READY_TIMEOUT_SEC:-600}` | [L819](../rl_miles/run_gspo.sh#L819) |
| `RAY_NUM_GPUS` | `${RAY_NUM_GPUS}` | [L823](../rl_miles/run_gspo.sh#L823) |
| `RAY_CLUSTER_READY_POLL_SEC` | `${RAY_CLUSTER_READY_POLL_SEC:-5}` | [L829](../rl_miles/run_gspo.sh#L829) |
| `RUNTIME_ENV_JSON` | `${RUNTIME_ENV_JSON}` | [L834](../rl_miles/run_gspo.sh#L834) |
| `COLOCATE_ARGS` | `${COLOCATE_ARGS[@]}` | [L839](../rl_miles/run_gspo.sh#L839) |

<a id="entry-44"></a>

## rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh

[源码](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L15](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L15) |
| `PROJECT_ROOT` | `${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}` | [L16](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L16) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L16](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L16) |
| `DRY_RUN` | `${DRY_RUN:-0}` | [L17](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L17) |
| `MILES_PROMPT_DATA` | `${MILES_PROMPT_DATA:-${PROJECT_ROOT}/data/rl/datasets/world_model_rl/train.jsonl}` | [L22](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L22) |
| `PROJECT_ROOT` | `${PROJECT_ROOT}` | [L22](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L22) |
| `JUDGE_PROMPT_DIR` | `${JUDGE_PROMPT_DIR:-${SCRIPT_DIR}/../prompts}` | [L33](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L33) |
| `JUDGE_BASE_URL` | `${JUDGE_BASE_URL:-http://<JUDGE_HOST>:<JUDGE_PORT>/v1}` | [L34](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L34) |
| `SMOKE_JUDGE_BASE_URL` | `${SMOKE_JUDGE_BASE_URL:-${JUDGE_BASE_URL:-http://<JUDGE_HOST>:<JUDGE_PORT>/v1}}` | [L34](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L34) |
| `JUDGE_BASE_URL` | `${JUDGE_BASE_URL}` | [L35](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L35) |
| `JUDGE_MODEL` | `${JUDGE_MODEL:-DeepSeek-V4-Flash-0731}` | [L36](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L36) |
| `JUDGE_MAX_TOKENS` | `${JUDGE_MAX_TOKENS:-4096}` | [L37](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L37) |
| `JUDGE_WEIGHT` | `${JUDGE_WEIGHT:-0.9}` | [L38](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L38) |
| `SIMILARITY_ENABLED` | `${SIMILARITY_ENABLED:-1}` | [L39](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L39) |
| `SIMILARITY_WEIGHT` | `${SIMILARITY_WEIGHT:-0.1}` | [L40](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L40) |
| `SIMILARITY_URL` | `${SIMILARITY_URL:-http://<SIMILARITY_HOST>:<SIMILARITY_PORT>/similarity}` | [L41](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L41) |
| `THINK_MODE` | `${THINK_MODE:-1}` | [L42](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L42) |
| `MODEL_TAG` | `${MODEL_TAG:-qwen3.5-122B-A10B}` | [L44](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L44) |
| `MODEL_HF_DEFAULT` | `${MODEL_HF_DEFAULT:-${PROJECT_ROOT}/models/Qwen3.5-122B-A10B}` | [L45](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L45) |
| `SAVE_DEFAULT` | `${SAVE_DEFAULT:-${PROJECT_ROOT}/checkpoints/qwen35_122b_a10b_gspo_160gpu_async}` | [L46](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L46) |
| `MILES_LOAD` | `${MILES_LOAD:-}` | [L48](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L48) |
| `SAVE_DEFAULT` | `${SAVE_DEFAULT}` | [L49](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L49) |
| `MODEL_HF_DEFAULT` | `${MODEL_HF_DEFAULT}` | [L52](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L52) |
| `MILES_SAVE` | `${MILES_SAVE:-${SAVE_DEFAULT}}` | [L55](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L55) |
| `TP_DEFAULT` | `${TP_DEFAULT:-2}` | [L61](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L61) |
| `PP_DEFAULT` | `${PP_DEFAULT:-12}` | [L62](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L62) |
| `CP_DEFAULT` | `${CP_DEFAULT:-4}` | [L63](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L63) |
| `EP_DEFAULT` | `${EP_DEFAULT:-8}` | [L64](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L64) |
| `ETP_DEFAULT` | `${ETP_DEFAULT:-1}` | [L65](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L65) |
| `LEARNING_RATE` | `${LEARNING_RATE:-1e-6}` | [L66](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L66) |
| `LR_DECAY_STYLE` | `${LR_DECAY_STYLE:-cosine}` | [L67](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L67) |
| `MIN_LR` | `${MIN_LR:-1e-7}` | [L68](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L68) |
| `USE_KL_LOSS` | `${USE_KL_LOSS:-1}` | [L69](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L69) |
| `KL_LOSS_COEF` | `${KL_LOSS_COEF:-0.01}` | [L70](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L70) |
| `MILES_LOAD` | `${MILES_LOAD}` | [L71](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L71) |
| `REF_LOAD` | `${REF_LOAD:-${MILES_LOAD}}` | [L71](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L71) |
| `DECODER_LAST_PIPELINE_NUM_LAYERS_DEFAULT` | `${DECODER_LAST_PIPELINE_NUM_LAYERS_DEFAULT:-}` | [L72](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L72) |
| `MAX_TOKENS_PER_GPU_DEFAULT` | `${MAX_TOKENS_PER_GPU_DEFAULT:-4096}` | [L73](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L73) |
| `LOG_PROBS_MAX_TOKENS_PER_GPU_DEFAULT` | `${LOG_PROBS_MAX_TOKENS_PER_GPU_DEFAULT:-4096}` | [L74](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L74) |
| `LOG_PROBS_CHUNK_SIZE` | `${LOG_PROBS_CHUNK_SIZE:-2048}` | [L75](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L75) |
| `ROLLOUT_GPUS_PER_ENGINE_DEFAULT` | `${ROLLOUT_GPUS_PER_ENGINE_DEFAULT:-8}` | [L78](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L78) |
| `SEQ_LEN` | `${SEQ_LEN:-43008}` | [L80](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L80) |
| `MAX_PROMPT_TOKENS` | `${MAX_PROMPT_TOKENS:-32768}` | [L81](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L81) |
| `EVAL_MAX_PROMPT_TOKENS` | `${EVAL_MAX_PROMPT_TOKENS:-65536}` | [L82](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L82) |
| `EVAL_MAX_CONTEXT_LEN` | `${EVAL_MAX_CONTEXT_LEN:-81920}` | [L83](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L83) |
| `EVAL_MAX_RESPONSE_LEN` | `${EVAL_MAX_RESPONSE_LEN:-16384}` | [L84](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L84) |
| `MAX_RESPONSE_TOKENS` | `${MAX_RESPONSE_TOKENS:-10240}` | [L85](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L85) |
| `ROLLOUT_BATCH_SIZE` | `${ROLLOUT_BATCH_SIZE:-256}` | [L86](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L86) |
| `GLOBAL_BATCH_SIZE` | `${GLOBAL_BATCH_SIZE:-2048}` | [L87](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L87) |
| `N_SAMPLES_PER_PROMPT` | `${N_SAMPLES_PER_PROMPT:-8}` | [L88](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L88) |
| `NUM_ROLLOUT` | `${NUM_ROLLOUT:-}` | [L89](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L89) |
| `NUM_EPOCH` | `${NUM_EPOCH:-2}` | [L90](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L90) |
| `COLOCATE` | `${COLOCATE:-0}` | [L93](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L93) |
| `TRAIN_DRIVER` | `export TRAIN_DRIVER="train_async.py"` | [L94](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L94) |
| `ROLLOUT_NUM_GPUS` | `${ROLLOUT_NUM_GPUS:-64}` | [L95](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L95) |
| `ROLLOUT_FUNCTION_PATH` | `${ROLLOUT_FUNCTION_PATH:-world_model_rl.fully_async_rollout.JsonlFullyAsyncRolloutFn}` | [L96](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L96) |
| `EVAL_FUNCTION_PATH` | `${EVAL_FUNCTION_PATH:-world_model_rl.fully_async_rollout.JsonlFullyAsyncRolloutFn}` | [L97](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L97) |
| `SGLANG_MEM_FRACTION_STATIC` | `${SGLANG_MEM_FRACTION_STATIC:-0.8}` | [L99](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L99) |
| `SGLANG_MAX_RUNNING_REQUESTS` | `${SGLANG_MAX_RUNNING_REQUESTS:-32}` | [L100](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L100) |
| `SGLANG_SERVER_CONCURRENCY` | `${SGLANG_SERVER_CONCURRENCY:-32}` | [L101](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L101) |
| `MAX_WEIGHT_STALENESS` | `${MAX_WEIGHT_STALENESS:-1}` | [L102](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L102) |
| `ASYNC_DATA_BUFFER_CAPACITY_FACTOR` | `${ASYNC_DATA_BUFFER_CAPACITY_FACTOR:-1.0}` | [L103](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L103) |
| `ACTOR_NUM_NODES` | `${ACTOR_NUM_NODES:-12}` | [L106](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L106) |
| `RESOURCE_GPU` | `${RESOURCE_GPU:-8}` | [L107](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L107) |
| `ACTOR_NUM_GPUS_PER_NODE` | `${ACTOR_NUM_GPUS_PER_NODE:-${RESOURCE_GPU}}` | [L108](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L108) |
| `RESOURCE_GPU` | `${RESOURCE_GPU}` | [L108](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L108) |
| `NPROC_PER_NODE` | `${NPROC_PER_NODE:-${RESOURCE_GPU}}` | [L109](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L109) |
| `SAVE_INTERVAL` | `${SAVE_INTERVAL:-20}` | [L110](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L110) |
| `EVAL_INTERVAL` | `${EVAL_INTERVAL:-20}` | [L111](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L111) |
| `MAX_WEIGHT_STALENESS` | `${MAX_WEIGHT_STALENESS}` | [L114](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L114) |
| `ASYNC_DATA_BUFFER_CAPACITY_FACTOR` | `${ASYNC_DATA_BUFFER_CAPACITY_FACTOR}` | [L115](../rl_miles/run_qwen35_122b_a10b_gspo_160gpu_async.sh#L115) |

<a id="entry-45"></a>

## rl_miles/run_qwen35_27b_gspo_128gpu_async.sh

[源码](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L15](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L15) |
| `PROJECT_ROOT` | `${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}` | [L16](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L16) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L16](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L16) |
| `DRY_RUN` | `${DRY_RUN:-0}` | [L17](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L17) |
| `MILES_PROMPT_DATA` | `${MILES_PROMPT_DATA:-${PROJECT_ROOT}/data/rl/datasets/world_model_rl/train.jsonl}` | [L20](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L20) |
| `PROJECT_ROOT` | `${PROJECT_ROOT}` | [L20](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L20) |
| `JUDGE_PROMPT_DIR` | `${JUDGE_PROMPT_DIR:-${SCRIPT_DIR}/../prompts}` | [L24](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L24) |
| `JUDGE_BASE_URL` | `${JUDGE_BASE_URL:-http://<JUDGE_HOST>:<JUDGE_PORT>/v1}` | [L25](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L25) |
| `SMOKE_JUDGE_BASE_URL` | `${SMOKE_JUDGE_BASE_URL:-${JUDGE_BASE_URL:-http://<JUDGE_HOST>:<JUDGE_PORT>/v1}}` | [L25](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L25) |
| `JUDGE_BASE_URL` | `${JUDGE_BASE_URL}` | [L26](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L26) |
| `JUDGE_MODEL` | `${JUDGE_MODEL:-DeepSeek-V4-Flash-0731}` | [L27](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L27) |
| `JUDGE_MAX_TOKENS` | `${JUDGE_MAX_TOKENS:-4096}` | [L28](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L28) |
| `THINK_MODE` | `${THINK_MODE:-0}` | [L30](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L30) |
| `JUDGE_WEIGHT` | `${JUDGE_WEIGHT:-1.0}` | [L34](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L34) |
| `SIMILARITY_WEIGHT` | `${SIMILARITY_WEIGHT:-0.0}` | [L35](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L35) |
| `SIMILARITY_URL` | `${SIMILARITY_URL:-http://<SIMILARITY_HOST>:<SIMILARITY_PORT>/similarity}` | [L36](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L36) |
| `MODEL_TAG` | `${MODEL_TAG:-qwen3.5-27B}` | [L38](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L38) |
| `MODEL_HF_DEFAULT` | `${MODEL_HF_DEFAULT:-${PROJECT_ROOT}/models/Qwen3.5-27B}` | [L39](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L39) |
| `SAVE_DEFAULT` | `${SAVE_DEFAULT:-${PROJECT_ROOT}/checkpoints/qwen35_27b_gspo_128gpu_async}` | [L40](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L40) |
| `TP_DEFAULT` | `${TP_DEFAULT:-4}` | [L41](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L41) |
| `PP_DEFAULT` | `${PP_DEFAULT:-4}` | [L42](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L42) |
| `CP_DEFAULT` | `${CP_DEFAULT:-4}` | [L43](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L43) |
| `EP_DEFAULT` | `${EP_DEFAULT:-1}` | [L44](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L44) |
| `ETP_DEFAULT` | `${ETP_DEFAULT:-1}` | [L45](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L45) |
| `DECODER_LAST_PIPELINE_NUM_LAYERS_DEFAULT` | `${DECODER_LAST_PIPELINE_NUM_LAYERS_DEFAULT:-}` | [L46](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L46) |
| `MAX_TOKENS_PER_GPU_DEFAULT` | `${MAX_TOKENS_PER_GPU_DEFAULT:-8192}` | [L47](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L47) |
| `LOG_PROBS_MAX_TOKENS_PER_GPU_DEFAULT` | `${LOG_PROBS_MAX_TOKENS_PER_GPU_DEFAULT:-8192}` | [L48](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L48) |
| `LOG_PROBS_CHUNK_SIZE` | `${LOG_PROBS_CHUNK_SIZE:-4096}` | [L49](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L49) |
| `ROLLOUT_GPUS_PER_ENGINE_DEFAULT` | `${ROLLOUT_GPUS_PER_ENGINE_DEFAULT:-2}` | [L50](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L50) |
| `SEQ_LEN` | `${SEQ_LEN:-40960}` | [L51](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L51) |
| `MAX_PROMPT_TOKENS` | `${MAX_PROMPT_TOKENS:-32768}` | [L52](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L52) |
| `EVAL_MAX_PROMPT_TOKENS` | `${EVAL_MAX_PROMPT_TOKENS:-65536}` | [L53](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L53) |
| `EVAL_MAX_CONTEXT_LEN` | `${EVAL_MAX_CONTEXT_LEN:-81920}` | [L54](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L54) |
| `EVAL_MAX_RESPONSE_LEN` | `${EVAL_MAX_RESPONSE_LEN:-16384}` | [L55](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L55) |
| `MAX_RESPONSE_TOKENS` | `${MAX_RESPONSE_TOKENS:-8192}` | [L56](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L56) |
| `ROLLOUT_BATCH_SIZE` | `${ROLLOUT_BATCH_SIZE:-256}` | [L57](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L57) |
| `GLOBAL_BATCH_SIZE` | `${GLOBAL_BATCH_SIZE:-2048}` | [L58](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L58) |
| `N_SAMPLES_PER_PROMPT` | `${N_SAMPLES_PER_PROMPT:-8}` | [L59](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L59) |
| `NUM_ROLLOUT` | `${NUM_ROLLOUT:-}` | [L60](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L60) |
| `NUM_EPOCH` | `${NUM_EPOCH:-1}` | [L61](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L61) |
| `COLOCATE` | `${COLOCATE:-0}` | [L64](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L64) |
| `TRAIN_DRIVER` | `export TRAIN_DRIVER="train_async.py"` | [L65](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L65) |
| `ROLLOUT_NUM_GPUS` | `${ROLLOUT_NUM_GPUS:-64}` | [L66](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L66) |
| `ROLLOUT_FUNCTION_PATH` | `${ROLLOUT_FUNCTION_PATH:-world_model_rl.fully_async_rollout.JsonlFullyAsyncRolloutFn}` | [L67](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L67) |
| `EVAL_FUNCTION_PATH` | `${EVAL_FUNCTION_PATH:-world_model_rl.fully_async_rollout.JsonlFullyAsyncRolloutFn}` | [L68](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L68) |
| `SGLANG_MEM_FRACTION_STATIC` | `${SGLANG_MEM_FRACTION_STATIC:-0.8}` | [L70](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L70) |
| `SGLANG_MAX_RUNNING_REQUESTS` | `${SGLANG_MAX_RUNNING_REQUESTS:-32}` | [L71](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L71) |
| `SGLANG_SERVER_CONCURRENCY` | `${SGLANG_SERVER_CONCURRENCY:-32}` | [L72](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L72) |
| `MAX_WEIGHT_STALENESS` | `${MAX_WEIGHT_STALENESS:-1}` | [L75](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L75) |
| `ASYNC_DATA_BUFFER_CAPACITY_FACTOR` | `${ASYNC_DATA_BUFFER_CAPACITY_FACTOR:-1.0}` | [L76](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L76) |
| `ACTOR_NUM_NODES` | `${ACTOR_NUM_NODES:-8}` | [L78](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L78) |
| `RESOURCE_GPU` | `${RESOURCE_GPU:-8}` | [L79](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L79) |
| `ACTOR_NUM_GPUS_PER_NODE` | `${ACTOR_NUM_GPUS_PER_NODE:-${RESOURCE_GPU}}` | [L80](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L80) |
| `RESOURCE_GPU` | `${RESOURCE_GPU}` | [L80](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L80) |
| `NPROC_PER_NODE` | `${NPROC_PER_NODE:-${RESOURCE_GPU}}` | [L81](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L81) |
| `SAVE_INTERVAL` | `${SAVE_INTERVAL:-20}` | [L82](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L82) |
| `EVAL_INTERVAL` | `${EVAL_INTERVAL:-20}` | [L83](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L83) |
| `MAX_WEIGHT_STALENESS` | `${MAX_WEIGHT_STALENESS}` | [L86](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L86) |
| `ASYNC_DATA_BUFFER_CAPACITY_FACTOR` | `${ASYNC_DATA_BUFFER_CAPACITY_FACTOR}` | [L87](../rl_miles/run_qwen35_27b_gspo_128gpu_async.sh#L87) |

<a id="entry-46"></a>

## rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh

[源码](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L15](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L15) |
| `PROJECT_ROOT` | `${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}` | [L16](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L16) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L16](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L16) |
| `DRY_RUN` | `${DRY_RUN:-0}` | [L17](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L17) |
| `MILES_PROMPT_DATA` | `${MILES_PROMPT_DATA:-${PROJECT_ROOT}/data/rl/datasets/world_model_rl/train.jsonl}` | [L20](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L20) |
| `PROJECT_ROOT` | `${PROJECT_ROOT}` | [L20](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L20) |
| `JUDGE_PROMPT_DIR` | `${JUDGE_PROMPT_DIR:-${SCRIPT_DIR}/../prompts}` | [L24](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L24) |
| `JUDGE_BASE_URL` | `${JUDGE_BASE_URL:-http://<JUDGE_HOST>:<JUDGE_PORT>/v1}` | [L25](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L25) |
| `SMOKE_JUDGE_BASE_URL` | `${SMOKE_JUDGE_BASE_URL:-${JUDGE_BASE_URL:-http://<JUDGE_HOST>:<JUDGE_PORT>/v1}}` | [L25](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L25) |
| `JUDGE_BASE_URL` | `${JUDGE_BASE_URL}` | [L26](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L26) |
| `JUDGE_MODEL` | `${JUDGE_MODEL:-DeepSeek-V4-Flash-0731}` | [L27](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L27) |
| `JUDGE_MAX_TOKENS` | `${JUDGE_MAX_TOKENS:-4096}` | [L28](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L28) |
| `THINK_MODE` | `${THINK_MODE:-0}` | [L30](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L30) |
| `JUDGE_WEIGHT` | `${JUDGE_WEIGHT:-1.0}` | [L34](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L34) |
| `SIMILARITY_WEIGHT` | `${SIMILARITY_WEIGHT:-0.0}` | [L35](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L35) |
| `SIMILARITY_URL` | `${SIMILARITY_URL:-http://<SIMILARITY_HOST>:<SIMILARITY_PORT>/similarity}` | [L36](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L36) |
| `MODEL_TAG` | `${MODEL_TAG:-qwen3.5-35B-A3B}` | [L38](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L38) |
| `MODEL_HF_DEFAULT` | `${MODEL_HF_DEFAULT:-${PROJECT_ROOT}/models/Qwen3.5-35B-A3B}` | [L39](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L39) |
| `SAVE_DEFAULT` | `${SAVE_DEFAULT:-${PROJECT_ROOT}/checkpoints/qwen35_35b_a3b_gspo_128gpu_async}` | [L40](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L40) |
| `MILES_LOAD` | `${MILES_LOAD:-}` | [L43](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L43) |
| `SAVE_DEFAULT` | `${SAVE_DEFAULT}` | [L44](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L44) |
| `MODEL_HF_DEFAULT` | `${MODEL_HF_DEFAULT}` | [L47](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L47) |
| `TP_DEFAULT` | `${TP_DEFAULT:-2}` | [L50](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L50) |
| `PP_DEFAULT` | `${PP_DEFAULT:-4}` | [L51](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L51) |
| `CP_DEFAULT` | `${CP_DEFAULT:-4}` | [L52](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L52) |
| `EP_DEFAULT` | `${EP_DEFAULT:-8}` | [L53](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L53) |
| `ETP_DEFAULT` | `${ETP_DEFAULT:-1}` | [L54](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L54) |
| `DECODER_LAST_PIPELINE_NUM_LAYERS_DEFAULT` | `${DECODER_LAST_PIPELINE_NUM_LAYERS_DEFAULT:-}` | [L55](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L55) |
| `MAX_TOKENS_PER_GPU_DEFAULT` | `${MAX_TOKENS_PER_GPU_DEFAULT:-8192}` | [L56](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L56) |
| `LOG_PROBS_MAX_TOKENS_PER_GPU_DEFAULT` | `${LOG_PROBS_MAX_TOKENS_PER_GPU_DEFAULT:-8192}` | [L57](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L57) |
| `LOG_PROBS_CHUNK_SIZE` | `${LOG_PROBS_CHUNK_SIZE:-4096}` | [L58](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L58) |
| `ROLLOUT_GPUS_PER_ENGINE_DEFAULT` | `${ROLLOUT_GPUS_PER_ENGINE_DEFAULT:-2}` | [L59](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L59) |
| `SEQ_LEN` | `${SEQ_LEN:-40960}` | [L60](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L60) |
| `MAX_PROMPT_TOKENS` | `${MAX_PROMPT_TOKENS:-32768}` | [L61](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L61) |
| `EVAL_MAX_PROMPT_TOKENS` | `${EVAL_MAX_PROMPT_TOKENS:-65536}` | [L62](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L62) |
| `EVAL_MAX_CONTEXT_LEN` | `${EVAL_MAX_CONTEXT_LEN:-81920}` | [L63](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L63) |
| `EVAL_MAX_RESPONSE_LEN` | `${EVAL_MAX_RESPONSE_LEN:-16384}` | [L64](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L64) |
| `MAX_RESPONSE_TOKENS` | `${MAX_RESPONSE_TOKENS:-8192}` | [L65](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L65) |
| `ROLLOUT_BATCH_SIZE` | `${ROLLOUT_BATCH_SIZE:-256}` | [L66](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L66) |
| `GLOBAL_BATCH_SIZE` | `${GLOBAL_BATCH_SIZE:-2048}` | [L67](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L67) |
| `N_SAMPLES_PER_PROMPT` | `${N_SAMPLES_PER_PROMPT:-8}` | [L68](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L68) |
| `NUM_ROLLOUT` | `${NUM_ROLLOUT:-}` | [L69](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L69) |
| `NUM_EPOCH` | `${NUM_EPOCH:-1}` | [L70](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L70) |
| `COLOCATE` | `${COLOCATE:-0}` | [L73](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L73) |
| `TRAIN_DRIVER` | `export TRAIN_DRIVER="train_async.py"` | [L74](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L74) |
| `ROLLOUT_NUM_GPUS` | `${ROLLOUT_NUM_GPUS:-64}` | [L75](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L75) |
| `ROLLOUT_FUNCTION_PATH` | `${ROLLOUT_FUNCTION_PATH:-world_model_rl.fully_async_rollout.JsonlFullyAsyncRolloutFn}` | [L76](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L76) |
| `EVAL_FUNCTION_PATH` | `${EVAL_FUNCTION_PATH:-world_model_rl.fully_async_rollout.JsonlFullyAsyncRolloutFn}` | [L77](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L77) |
| `SGLANG_MEM_FRACTION_STATIC` | `${SGLANG_MEM_FRACTION_STATIC:-0.8}` | [L79](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L79) |
| `SGLANG_MAX_RUNNING_REQUESTS` | `${SGLANG_MAX_RUNNING_REQUESTS:-32}` | [L80](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L80) |
| `SGLANG_SERVER_CONCURRENCY` | `${SGLANG_SERVER_CONCURRENCY:-32}` | [L81](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L81) |
| `MAX_WEIGHT_STALENESS` | `${MAX_WEIGHT_STALENESS:-1}` | [L84](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L84) |
| `ASYNC_DATA_BUFFER_CAPACITY_FACTOR` | `${ASYNC_DATA_BUFFER_CAPACITY_FACTOR:-1.0}` | [L85](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L85) |
| `ACTOR_NUM_NODES` | `${ACTOR_NUM_NODES:-8}` | [L87](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L87) |
| `RESOURCE_GPU` | `${RESOURCE_GPU:-8}` | [L88](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L88) |
| `ACTOR_NUM_GPUS_PER_NODE` | `${ACTOR_NUM_GPUS_PER_NODE:-${RESOURCE_GPU}}` | [L89](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L89) |
| `RESOURCE_GPU` | `${RESOURCE_GPU}` | [L89](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L89) |
| `NPROC_PER_NODE` | `${NPROC_PER_NODE:-${RESOURCE_GPU}}` | [L90](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L90) |
| `SAVE_INTERVAL` | `${SAVE_INTERVAL:-20}` | [L91](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L91) |
| `EVAL_INTERVAL` | `${EVAL_INTERVAL:-20}` | [L92](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L92) |
| `MAX_WEIGHT_STALENESS` | `${MAX_WEIGHT_STALENESS}` | [L95](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L95) |
| `ASYNC_DATA_BUFFER_CAPACITY_FACTOR` | `${ASYNC_DATA_BUFFER_CAPACITY_FACTOR}` | [L96](../rl_miles/run_qwen35_35b_a3b_gspo_128gpu_async.sh#L96) |

<a id="entry-47"></a>

## rl_miles/run_qwen35_4b_gspo_128gpu_async.sh

[源码](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L17](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L17) |
| `PROJECT_ROOT` | `${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}` | [L18](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L18) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L18](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L18) |
| `DRY_RUN` | `${DRY_RUN:-0}` | [L19](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L19) |
| `MILES_PROMPT_DATA` | `${MILES_PROMPT_DATA:-${PROJECT_ROOT}/data/rl/datasets/world_model_rl/train.jsonl}` | [L22](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L22) |
| `PROJECT_ROOT` | `${PROJECT_ROOT}` | [L22](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L22) |
| `JUDGE_PROMPT_DIR` | `${JUDGE_PROMPT_DIR:-${SCRIPT_DIR}/../prompts}` | [L26](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L26) |
| `JUDGE_BASE_URL` | `${JUDGE_BASE_URL:-http://<JUDGE_HOST>:<JUDGE_PORT>/v1}` | [L27](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L27) |
| `SMOKE_JUDGE_BASE_URL` | `${SMOKE_JUDGE_BASE_URL:-${JUDGE_BASE_URL:-http://<JUDGE_HOST>:<JUDGE_PORT>/v1}}` | [L27](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L27) |
| `JUDGE_BASE_URL` | `${JUDGE_BASE_URL}` | [L28](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L28) |
| `JUDGE_MODEL` | `${JUDGE_MODEL:-DeepSeek-V4-Flash-0731}` | [L29](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L29) |
| `JUDGE_MAX_TOKENS` | `${JUDGE_MAX_TOKENS:-4096}` | [L30](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L30) |
| `THINK_MODE` | `${THINK_MODE:-0}` | [L32](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L32) |
| `JUDGE_WEIGHT` | `${JUDGE_WEIGHT:-1.0}` | [L36](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L36) |
| `SIMILARITY_WEIGHT` | `${SIMILARITY_WEIGHT:-0.0}` | [L37](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L37) |
| `SIMILARITY_URL` | `${SIMILARITY_URL:-http://<SIMILARITY_HOST>:<SIMILARITY_PORT>/similarity}` | [L38](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L38) |
| `MODEL_TAG` | `${MODEL_TAG:-qwen3.5-4B}` | [L40](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L40) |
| `MODEL_HF_DEFAULT` | `${MODEL_HF_DEFAULT:-${PROJECT_ROOT}/models/Qwen3.5-4B}` | [L41](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L41) |
| `SAVE_DEFAULT` | `${SAVE_DEFAULT:-${PROJECT_ROOT}/checkpoints/qwen35_4b_gspo_128gpu_async}` | [L42](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L42) |
| `TP_DEFAULT` | `${TP_DEFAULT:-2}` | [L43](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L43) |
| `PP_DEFAULT` | `${PP_DEFAULT:-1}` | [L44](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L44) |
| `CP_DEFAULT` | `${CP_DEFAULT:-4}` | [L45](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L45) |
| `EP_DEFAULT` | `${EP_DEFAULT:-1}` | [L46](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L46) |
| `ETP_DEFAULT` | `${ETP_DEFAULT:-1}` | [L47](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L47) |
| `DECODER_LAST_PIPELINE_NUM_LAYERS_DEFAULT` | `${DECODER_LAST_PIPELINE_NUM_LAYERS_DEFAULT:-}` | [L48](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L48) |
| `MAX_TOKENS_PER_GPU_DEFAULT` | `${MAX_TOKENS_PER_GPU_DEFAULT:-8192}` | [L49](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L49) |
| `LOG_PROBS_MAX_TOKENS_PER_GPU_DEFAULT` | `${LOG_PROBS_MAX_TOKENS_PER_GPU_DEFAULT:-8192}` | [L50](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L50) |
| `LOG_PROBS_CHUNK_SIZE` | `${LOG_PROBS_CHUNK_SIZE:-4096}` | [L51](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L51) |
| `ROLLOUT_GPUS_PER_ENGINE_DEFAULT` | `${ROLLOUT_GPUS_PER_ENGINE_DEFAULT:-1}` | [L52](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L52) |
| `SEQ_LEN` | `${SEQ_LEN:-40960}` | [L53](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L53) |
| `MAX_PROMPT_TOKENS` | `${MAX_PROMPT_TOKENS:-32768}` | [L54](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L54) |
| `EVAL_MAX_PROMPT_TOKENS` | `${EVAL_MAX_PROMPT_TOKENS:-65536}` | [L55](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L55) |
| `EVAL_MAX_CONTEXT_LEN` | `${EVAL_MAX_CONTEXT_LEN:-81920}` | [L56](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L56) |
| `EVAL_MAX_RESPONSE_LEN` | `${EVAL_MAX_RESPONSE_LEN:-16384}` | [L57](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L57) |
| `MAX_RESPONSE_TOKENS` | `${MAX_RESPONSE_TOKENS:-8192}` | [L58](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L58) |
| `ROLLOUT_BATCH_SIZE` | `${ROLLOUT_BATCH_SIZE:-128}` | [L59](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L59) |
| `GLOBAL_BATCH_SIZE` | `${GLOBAL_BATCH_SIZE:-1024}` | [L60](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L60) |
| `N_SAMPLES_PER_PROMPT` | `${N_SAMPLES_PER_PROMPT:-8}` | [L61](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L61) |
| `NUM_ROLLOUT` | `${NUM_ROLLOUT:-}` | [L62](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L62) |
| `NUM_EPOCH` | `${NUM_EPOCH:-1}` | [L63](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L63) |
| `COLOCATE` | `${COLOCATE:-0}` | [L66](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L66) |
| `TRAIN_DRIVER` | `export TRAIN_DRIVER="train_async.py"` | [L67](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L67) |
| `ROLLOUT_NUM_GPUS` | `${ROLLOUT_NUM_GPUS:-64}` | [L68](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L68) |
| `ROLLOUT_FUNCTION_PATH` | `${ROLLOUT_FUNCTION_PATH:-world_model_rl.fully_async_rollout.JsonlFullyAsyncRolloutFn}` | [L69](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L69) |
| `EVAL_FUNCTION_PATH` | `${EVAL_FUNCTION_PATH:-world_model_rl.fully_async_rollout.JsonlFullyAsyncRolloutFn}` | [L70](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L70) |
| `SGLANG_MEM_FRACTION_STATIC` | `${SGLANG_MEM_FRACTION_STATIC:-0.8}` | [L74](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L74) |
| `SGLANG_MAX_RUNNING_REQUESTS` | `${SGLANG_MAX_RUNNING_REQUESTS:-32}` | [L75](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L75) |
| `SGLANG_SERVER_CONCURRENCY` | `${SGLANG_SERVER_CONCURRENCY:-32}` | [L76](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L76) |
| `MAX_WEIGHT_STALENESS` | `${MAX_WEIGHT_STALENESS:-1}` | [L79](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L79) |
| `ASYNC_DATA_BUFFER_CAPACITY_FACTOR` | `${ASYNC_DATA_BUFFER_CAPACITY_FACTOR:-1.0}` | [L80](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L80) |
| `ACTOR_NUM_NODES` | `${ACTOR_NUM_NODES:-8}` | [L82](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L82) |
| `RESOURCE_GPU` | `${RESOURCE_GPU:-8}` | [L83](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L83) |
| `ACTOR_NUM_GPUS_PER_NODE` | `${ACTOR_NUM_GPUS_PER_NODE:-${RESOURCE_GPU}}` | [L84](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L84) |
| `RESOURCE_GPU` | `${RESOURCE_GPU}` | [L84](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L84) |
| `NPROC_PER_NODE` | `${NPROC_PER_NODE:-${RESOURCE_GPU}}` | [L85](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L85) |
| `SAVE_INTERVAL` | `${SAVE_INTERVAL:-20}` | [L86](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L86) |
| `EVAL_INTERVAL` | `${EVAL_INTERVAL:-20}` | [L87](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L87) |
| `MAX_WEIGHT_STALENESS` | `${MAX_WEIGHT_STALENESS}` | [L90](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L90) |
| `ASYNC_DATA_BUFFER_CAPACITY_FACTOR` | `${ASYNC_DATA_BUFFER_CAPACITY_FACTOR}` | [L91](../rl_miles/run_qwen35_4b_gspo_128gpu_async.sh#L91) |

<a id="entry-48"></a>

## rl_miles/run_qwen35_9b_gspo_128gpu_async.sh

[源码](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L15](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L15) |
| `PROJECT_ROOT` | `${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}` | [L16](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L16) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L16](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L16) |
| `DRY_RUN` | `${DRY_RUN:-0}` | [L17](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L17) |
| `MILES_PROMPT_DATA` | `${MILES_PROMPT_DATA:-${PROJECT_ROOT}/data/rl/datasets/world_model_rl/train.jsonl}` | [L20](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L20) |
| `PROJECT_ROOT` | `${PROJECT_ROOT}` | [L20](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L20) |
| `JUDGE_PROMPT_DIR` | `${JUDGE_PROMPT_DIR:-${SCRIPT_DIR}/../prompts}` | [L24](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L24) |
| `JUDGE_BASE_URL` | `${JUDGE_BASE_URL:-http://<JUDGE_HOST>:<JUDGE_PORT>/v1}` | [L25](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L25) |
| `SMOKE_JUDGE_BASE_URL` | `${SMOKE_JUDGE_BASE_URL:-${JUDGE_BASE_URL:-http://<JUDGE_HOST>:<JUDGE_PORT>/v1}}` | [L25](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L25) |
| `JUDGE_BASE_URL` | `${JUDGE_BASE_URL}` | [L26](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L26) |
| `JUDGE_MODEL` | `${JUDGE_MODEL:-DeepSeek-V4-Flash-0731}` | [L27](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L27) |
| `JUDGE_MAX_TOKENS` | `${JUDGE_MAX_TOKENS:-4096}` | [L28](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L28) |
| `THINK_MODE` | `${THINK_MODE:-0}` | [L30](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L30) |
| `JUDGE_WEIGHT` | `${JUDGE_WEIGHT:-1.0}` | [L34](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L34) |
| `SIMILARITY_WEIGHT` | `${SIMILARITY_WEIGHT:-0.0}` | [L35](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L35) |
| `SIMILARITY_URL` | `${SIMILARITY_URL:-http://<SIMILARITY_HOST>:<SIMILARITY_PORT>/similarity}` | [L36](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L36) |
| `MODEL_TAG` | `${MODEL_TAG:-qwen3.5-9B}` | [L38](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L38) |
| `MODEL_HF_DEFAULT` | `${MODEL_HF_DEFAULT:-${PROJECT_ROOT}/models/Qwen3.5-9B}` | [L39](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L39) |
| `SAVE_DEFAULT` | `${SAVE_DEFAULT:-${PROJECT_ROOT}/checkpoints/qwen35_9b_gspo_128gpu_async}` | [L40](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L40) |
| `MILES_LOAD` | `${MILES_LOAD:-}` | [L43](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L43) |
| `SAVE_DEFAULT` | `${SAVE_DEFAULT}` | [L44](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L44) |
| `MODEL_HF_DEFAULT` | `${MODEL_HF_DEFAULT}` | [L47](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L47) |
| `TP_DEFAULT` | `${TP_DEFAULT:-2}` | [L50](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L50) |
| `PP_DEFAULT` | `${PP_DEFAULT:-2}` | [L51](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L51) |
| `CP_DEFAULT` | `${CP_DEFAULT:-4}` | [L52](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L52) |
| `EP_DEFAULT` | `${EP_DEFAULT:-1}` | [L53](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L53) |
| `ETP_DEFAULT` | `${ETP_DEFAULT:-1}` | [L54](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L54) |
| `DECODER_LAST_PIPELINE_NUM_LAYERS_DEFAULT` | `${DECODER_LAST_PIPELINE_NUM_LAYERS_DEFAULT:-}` | [L55](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L55) |
| `MAX_TOKENS_PER_GPU_DEFAULT` | `${MAX_TOKENS_PER_GPU_DEFAULT:-8192}` | [L56](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L56) |
| `LOG_PROBS_MAX_TOKENS_PER_GPU_DEFAULT` | `${LOG_PROBS_MAX_TOKENS_PER_GPU_DEFAULT:-8192}` | [L57](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L57) |
| `LOG_PROBS_CHUNK_SIZE` | `${LOG_PROBS_CHUNK_SIZE:-4096}` | [L58](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L58) |
| `ROLLOUT_GPUS_PER_ENGINE_DEFAULT` | `${ROLLOUT_GPUS_PER_ENGINE_DEFAULT:-2}` | [L59](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L59) |
| `SEQ_LEN` | `${SEQ_LEN:-40960}` | [L60](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L60) |
| `MAX_PROMPT_TOKENS` | `${MAX_PROMPT_TOKENS:-32768}` | [L61](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L61) |
| `EVAL_MAX_PROMPT_TOKENS` | `${EVAL_MAX_PROMPT_TOKENS:-65536}` | [L62](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L62) |
| `EVAL_MAX_CONTEXT_LEN` | `${EVAL_MAX_CONTEXT_LEN:-81920}` | [L63](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L63) |
| `EVAL_MAX_RESPONSE_LEN` | `${EVAL_MAX_RESPONSE_LEN:-16384}` | [L64](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L64) |
| `MAX_RESPONSE_TOKENS` | `${MAX_RESPONSE_TOKENS:-8192}` | [L65](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L65) |
| `ROLLOUT_BATCH_SIZE` | `${ROLLOUT_BATCH_SIZE:-256}` | [L66](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L66) |
| `GLOBAL_BATCH_SIZE` | `${GLOBAL_BATCH_SIZE:-2048}` | [L67](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L67) |
| `N_SAMPLES_PER_PROMPT` | `${N_SAMPLES_PER_PROMPT:-8}` | [L68](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L68) |
| `NUM_ROLLOUT` | `${NUM_ROLLOUT:-}` | [L69](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L69) |
| `NUM_EPOCH` | `${NUM_EPOCH:-1}` | [L70](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L70) |
| `COLOCATE` | `${COLOCATE:-0}` | [L73](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L73) |
| `TRAIN_DRIVER` | `export TRAIN_DRIVER="train_async.py"` | [L74](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L74) |
| `ROLLOUT_NUM_GPUS` | `${ROLLOUT_NUM_GPUS:-64}` | [L75](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L75) |
| `ROLLOUT_FUNCTION_PATH` | `${ROLLOUT_FUNCTION_PATH:-world_model_rl.fully_async_rollout.JsonlFullyAsyncRolloutFn}` | [L76](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L76) |
| `EVAL_FUNCTION_PATH` | `${EVAL_FUNCTION_PATH:-world_model_rl.fully_async_rollout.JsonlFullyAsyncRolloutFn}` | [L77](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L77) |
| `SGLANG_MEM_FRACTION_STATIC` | `${SGLANG_MEM_FRACTION_STATIC:-0.8}` | [L79](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L79) |
| `SGLANG_MAX_RUNNING_REQUESTS` | `${SGLANG_MAX_RUNNING_REQUESTS:-32}` | [L80](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L80) |
| `SGLANG_SERVER_CONCURRENCY` | `${SGLANG_SERVER_CONCURRENCY:-32}` | [L81](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L81) |
| `MAX_WEIGHT_STALENESS` | `${MAX_WEIGHT_STALENESS:-1}` | [L84](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L84) |
| `ASYNC_DATA_BUFFER_CAPACITY_FACTOR` | `${ASYNC_DATA_BUFFER_CAPACITY_FACTOR:-1.0}` | [L85](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L85) |
| `ACTOR_NUM_NODES` | `${ACTOR_NUM_NODES:-8}` | [L87](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L87) |
| `RESOURCE_GPU` | `${RESOURCE_GPU:-8}` | [L88](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L88) |
| `ACTOR_NUM_GPUS_PER_NODE` | `${ACTOR_NUM_GPUS_PER_NODE:-${RESOURCE_GPU}}` | [L89](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L89) |
| `RESOURCE_GPU` | `${RESOURCE_GPU}` | [L89](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L89) |
| `NPROC_PER_NODE` | `${NPROC_PER_NODE:-${RESOURCE_GPU}}` | [L90](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L90) |
| `SAVE_INTERVAL` | `${SAVE_INTERVAL:-20}` | [L91](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L91) |
| `EVAL_INTERVAL` | `${EVAL_INTERVAL:-20}` | [L92](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L92) |
| `MAX_WEIGHT_STALENESS` | `${MAX_WEIGHT_STALENESS}` | [L95](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L95) |
| `ASYNC_DATA_BUFFER_CAPACITY_FACTOR` | `${ASYNC_DATA_BUFFER_CAPACITY_FACTOR}` | [L96](../rl_miles/run_qwen35_9b_gspo_128gpu_async.sh#L96) |

<a id="entry-49"></a>

## rl_miles/validate_runtime.py

[源码](../rl_miles/validate_runtime.py)


### CLI

| 参数 | 声明 | 位置 |
| --- | --- | --- |
| `--driver` | `default='train_async.py'` | [L49](../rl_miles/validate_runtime.py#L49) |
| `--data-source` | `required=True` | [L50](../rl_miles/validate_runtime.py#L50) |
| `--reward` | `default='world_model_rl.reward.custom_rm'` | [L51](../rl_miles/validate_runtime.py#L51) |
| `--rollout` | `required=True` | [L52](../rl_miles/validate_runtime.py#L52) |
| `--eval-rollout` | `default='world_model_rl.eval_rollout.raw_eval_rollout'` | [L53](../rl_miles/validate_runtime.py#L53) |
| `--judge-enabled` | `default='1'` | [L54](../rl_miles/validate_runtime.py#L54) |
| `--judge-base-url` | `default=''` | [L55](../rl_miles/validate_runtime.py#L55) |
| `--judge-model` | `default=''` | [L56](../rl_miles/validate_runtime.py#L56) |
| `--similarity-enabled` | `default='0'` | [L57](../rl_miles/validate_runtime.py#L57) |
| `--similarity-url` | `default=''` | [L58](../rl_miles/validate_runtime.py#L58) |

<a id="entry-50"></a>

## rl_miles/wheels/download_official_cu130_wheels.py

[源码](../rl_miles/wheels/download_official_cu130_wheels.py)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `MILES_WHEELS_TAG` | `os.environ.get('MILES_WHEELS_TAG', 'cu130-torch213-x86_64')` | [L3](../rl_miles/wheels/download_official_cu130_wheels.py#L3) |
| `MILES_WHEELS_REPO` | `os.environ.get('MILES_WHEELS_REPO', 'yueming-yuan/miles-wheels')` | [L4](../rl_miles/wheels/download_official_cu130_wheels.py#L4) |
| `MILES_WHEELS_DEST` | `os.environ.get('MILES_WHEELS_DEST', 'wheels')` | [L5](../rl_miles/wheels/download_official_cu130_wheels.py#L5) |
| `MILES_WHEELS_ONLY` | `os.environ.get('MILES_WHEELS_ONLY', '')` | [L6](../rl_miles/wheels/download_official_cu130_wheels.py#L6) |

<a id="entry-51"></a>

## rl_miles/world_model_rl/data_source.py

[源码](../rl_miles/world_model_rl/data_source.py)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `MILES_PROMPT_DATA_LIMIT` | `os.environ.get('MILES_PROMPT_DATA_LIMIT', '')` | [L30](../rl_miles/world_model_rl/data_source.py#L30) |

<a id="entry-52"></a>

## rl_miles/world_model_rl/dataloader.py

[源码](../rl_miles/world_model_rl/dataloader.py)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `WM_TEMPLATE_MODEL` | `os.environ.get('WM_TEMPLATE_MODEL', '')` | [L39](../rl_miles/world_model_rl/dataloader.py#L39) |

<a id="entry-53"></a>

## rl_miles/world_model_rl/eval_rollout.py

[源码](../rl_miles/world_model_rl/eval_rollout.py)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `WM_EVAL_DATA_PATH` | `os.environ.get('WM_EVAL_DATA_PATH', '')` | [L31](../rl_miles/world_model_rl/eval_rollout.py#L31) |

<a id="entry-54"></a>

## rl_miles/world_model_rl/fully_async_rollout.py

[源码](../rl_miles/world_model_rl/fully_async_rollout.py)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `WM_EVAL_DATA_PATH` | `os.environ.get('WM_EVAL_DATA_PATH', '')` | [L69](../rl_miles/world_model_rl/fully_async_rollout.py#L69) |

<a id="entry-55"></a>

## rl_miles/world_model_rl/jsonl_writer.py

[源码](../rl_miles/world_model_rl/jsonl_writer.py)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `WM_ROLLOUT_OUTPUT_DIR` | `os.environ.get('WM_ROLLOUT_OUTPUT_DIR', '')` | [L21](../rl_miles/world_model_rl/jsonl_writer.py#L21) |

<a id="entry-56"></a>

## rl_miles/world_model_rl/reward.py

[源码](../rl_miles/world_model_rl/reward.py)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `NO_PROXY` | `os.environ['NO_PROXY']` | [L55](../rl_miles/world_model_rl/reward.py#L55) |
| `no_proxy` | `os.environ['no_proxy']` | [L56](../rl_miles/world_model_rl/reward.py#L56) |
| `WM_JUDGE_PROMPT_DIR` | `os.environ.get('WM_JUDGE_PROMPT_DIR', str(_DEFAULT_PROMPT_ROOT))` | [L84](../rl_miles/world_model_rl/reward.py#L84) |
| `WM_JUDGE_SKIP` | `os.environ.get('WM_JUDGE_SKIP', '')` | [L428](../rl_miles/world_model_rl/reward.py#L428) |
| `WM_JUDGE_ENABLED` | `os.environ.get('WM_JUDGE_ENABLED', str(int(DEFAULT_JUDGE_ENABLED)))` | [L430](../rl_miles/world_model_rl/reward.py#L430) |
| `WM_SIMILARITY_ENABLED` | `_env_flag('WM_SIMILARITY_ENABLED', DEFAULT_SIMILARITY_ENABLED)` | [L444](../rl_miles/world_model_rl/reward.py#L444) |
| `WM_THINK_MODE` | `_env_flag('WM_THINK_MODE', DEFAULT_THINK_MODE)` | [L448](../rl_miles/world_model_rl/reward.py#L448) |
| `WM_JUDGE_WEIGHTS` | `os.environ.get('WM_JUDGE_WEIGHTS')` | [L600](../rl_miles/world_model_rl/reward.py#L600) |
| `WM_JUDGE_MODEL` | `os.environ.get('WM_JUDGE_MODEL', DEFAULT_MODEL)` | [L609](../rl_miles/world_model_rl/reward.py#L609) |
| `WM_JUDGE_API_KEY` | `os.environ.get('WM_JUDGE_API_KEY', DEFAULT_API_KEY)` | [L610](../rl_miles/world_model_rl/reward.py#L610) |
| `WM_JUDGE_TIMEOUT` | `os.environ.get('WM_JUDGE_TIMEOUT', DEFAULT_TIMEOUT)` | [L611](../rl_miles/world_model_rl/reward.py#L611) |
| `WM_JUDGE_CONCURRENCY` | `os.environ.get('WM_JUDGE_CONCURRENCY', DEFAULT_CONCURRENCY)` | [L612](../rl_miles/world_model_rl/reward.py#L612) |
| `WM_JUDGE_WEIGHT` | `os.environ.get('WM_JUDGE_WEIGHT', DEFAULT_JUDGE_WEIGHT)` | [L637](../rl_miles/world_model_rl/reward.py#L637) |
| `WM_SIMILARITY_WEIGHT` | `os.environ.get('WM_SIMILARITY_WEIGHT', DEFAULT_SIMILARITY_WEIGHT)` | [L638](../rl_miles/world_model_rl/reward.py#L638) |
| `WM_SIMILARITY_TIMEOUT` | `os.environ.get('WM_SIMILARITY_TIMEOUT', DEFAULT_SIMILARITY_TIMEOUT)` | [L651](../rl_miles/world_model_rl/reward.py#L651) |
| `WM_SIMILARITY_CONCURRENCY` | `os.environ.get('WM_SIMILARITY_CONCURRENCY', DEFAULT_CONCURRENCY)` | [L652](../rl_miles/world_model_rl/reward.py#L652) |
| `WM_SIMILARITY_URL` | `os.environ.get('WM_SIMILARITY_URL', DEFAULT_SIMILARITY_URL)` | [L673](../rl_miles/world_model_rl/reward.py#L673) |
| `WM_SIMILARITY_RETRIES` | `os.environ.get('WM_SIMILARITY_RETRIES', DEFAULT_RETRIES)` | [L674](../rl_miles/world_model_rl/reward.py#L674) |
| `WM_JUDGE_BASE_URL` | `os.environ.get('WM_JUDGE_BASE_URL')` | [L773](../rl_miles/world_model_rl/reward.py#L773) |
| `WM_JUDGE_MODEL` | `os.environ.get('WM_JUDGE_MODEL', DEFAULT_MODEL)` | [L774](../rl_miles/world_model_rl/reward.py#L774) |
| `WM_JUDGE_MAX_TOKENS` | `os.environ.get('WM_JUDGE_MAX_TOKENS', DEFAULT_MAX_TOKENS)` | [L775](../rl_miles/world_model_rl/reward.py#L775) |
| `WM_JUDGE_RETRIES` | `os.environ.get('WM_JUDGE_RETRIES', DEFAULT_RETRIES)` | [L776](../rl_miles/world_model_rl/reward.py#L776) |

<a id="entry-57"></a>

## runtime/prepare_sources.py

[源码](../runtime/prepare_sources.py)


### CLI

| 参数 | 声明 | 位置 |
| --- | --- | --- |
| `--component` | `choices=sorted(manifest); action='append'; required=True` | [L55](../runtime/prepare_sources.py#L55) |
| `--destination` | `type=Path; help='override checkout directory (one component only)'` | [L56](../runtime/prepare_sources.py#L56) |
| `--check` | `action='store_true'; help='offline check; never clone, install or modify sources'` | [L57](../runtime/prepare_sources.py#L57) |

<a id="entry-58"></a>

## runtime/validate_swift.py

[源码](../runtime/validate_swift.py)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `SWIFT_TRAIN_USER_LOSS` | `os.environ['SWIFT_TRAIN_USER_LOSS']` | [L27](../runtime/validate_swift.py#L27) |

### CLI

| 参数 | 声明 | 位置 |
| --- | --- | --- |
| `--train-user-loss` | `action='store_true'` | [L11](../runtime/validate_swift.py#L11) |

<a id="entry-59"></a>

## sft_code/setup_runtime.sh

[源码](../sft_code/setup_runtime.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L3](../sft_code/setup_runtime.sh#L3) |
| `ROOT` | `${ROOT}` | [L4](../sft_code/setup_runtime.sh#L4) |

<a id="entry-60"></a>

## sft_code/train_qwen35_27b_sft_64gpu.sh

[源码](../sft_code/train_qwen35_27b_sft_64gpu.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L4](../sft_code/train_qwen35_27b_sft_64gpu.sh#L4) |
| `SFT_MODEL_SIZE` | `export SFT_MODEL_SIZE=27b` | [L5](../sft_code/train_qwen35_27b_sft_64gpu.sh#L5) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L6](../sft_code/train_qwen35_27b_sft_64gpu.sh#L6) |

<a id="entry-61"></a>

## sft_code/train_qwen35_35b_a3b_sft_32gpu.sh

[源码](../sft_code/train_qwen35_35b_a3b_sft_32gpu.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L4](../sft_code/train_qwen35_35b_a3b_sft_32gpu.sh#L4) |
| `SFT_MODEL_SIZE` | `export SFT_MODEL_SIZE=35b_a3b` | [L5](../sft_code/train_qwen35_35b_a3b_sft_32gpu.sh#L5) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L6](../sft_code/train_qwen35_35b_a3b_sft_32gpu.sh#L6) |

<a id="entry-62"></a>

## sft_code/train_qwen35_4b_sft_8gpu.sh

[源码](../sft_code/train_qwen35_4b_sft_8gpu.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L4](../sft_code/train_qwen35_4b_sft_8gpu.sh#L4) |
| `SFT_MODEL_SIZE` | `export SFT_MODEL_SIZE=4b` | [L5](../sft_code/train_qwen35_4b_sft_8gpu.sh#L5) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L6](../sft_code/train_qwen35_4b_sft_8gpu.sh#L6) |

<a id="entry-63"></a>

## sft_code/train_qwen35_9b_sft_8gpu.sh

[源码](../sft_code/train_qwen35_9b_sft_8gpu.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L4](../sft_code/train_qwen35_9b_sft_8gpu.sh#L4) |
| `SFT_MODEL_SIZE` | `export SFT_MODEL_SIZE=9b` | [L5](../sft_code/train_qwen35_9b_sft_8gpu.sh#L5) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L6](../sft_code/train_qwen35_9b_sft_8gpu.sh#L6) |

<a id="entry-64"></a>

## sft_code/train_sft.sh

[源码](../sft_code/train_sft.sh)

| 变量 / 读取项 | 源码表达式 | 位置 |
| --- | --- | --- |
| `LINENO` | `${LINENO}` | [L3](../sft_code/train_sft.sh#L3) |
| `BASH_SOURCE` | `${BASH_SOURCE[0]}` | [L29](../sft_code/train_sft.sh#L29) |
| `PROJECT_ROOT` | `${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}` | [L30](../sft_code/train_sft.sh#L30) |
| `SCRIPT_DIR` | `${SCRIPT_DIR}` | [L30](../sft_code/train_sft.sh#L30) |
| `CODE_ROOT` | `${CODE_ROOT:-${PROJECT_ROOT}/cpt}` | [L31](../sft_code/train_sft.sh#L31) |
| `PROJECT_ROOT` | `${PROJECT_ROOT}` | [L31](../sft_code/train_sft.sh#L31) |
| `CODE_ROOT` | `${CODE_ROOT}` | [L32](../sft_code/train_sft.sh#L32) |
| `SWIFT_ROOT` | `${SWIFT_ROOT:-${CODE_ROOT}/ms-swift}` | [L32](../sft_code/train_sft.sh#L32) |
| `SWIFT_VENV` | `${SWIFT_VENV:-${SCRIPT_DIR}/.venv}` | [L33](../sft_code/train_sft.sh#L33) |
| `MEGATRON_ROOT` | `${MEGATRON_ROOT:-}` | [L34](../sft_code/train_sft.sh#L34) |
| `SFT_MODEL_SIZE` | `${SFT_MODEL_SIZE:-9b}` | [L35](../sft_code/train_sft.sh#L35) |
| `SFT_MODEL_SIZE` | `${SFT_MODEL_SIZE}` | [L38](../sft_code/train_sft.sh#L38) |
| `MODEL_PATH` | `${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/cpt-hf/qwen35_${SFT_MODEL_SIZE}}` | [L64](../sft_code/train_sft.sh#L64) |
| `MCORE_MODEL` | `${MCORE_MODEL:-}` | [L65](../sft_code/train_sft.sh#L65) |
| `DEFAULT_MODEL_TYPE` | `${DEFAULT_MODEL_TYPE}` | [L66](../sft_code/train_sft.sh#L66) |
| `MODEL_TYPE` | `${MODEL_TYPE:-${DEFAULT_MODEL_TYPE}}` | [L66](../sft_code/train_sft.sh#L66) |
| `TEMPLATE` | `${TEMPLATE:-qwen3_5}` | [L67](../sft_code/train_sft.sh#L67) |
| `DATASET_PATH` | `${DATASET_PATH:-${PROJECT_ROOT}/data/sft_data.jsonl}` | [L68](../sft_code/train_sft.sh#L68) |
| `OUTPUT_DIR` | `${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/sft/qwen35_${SFT_MODEL_SIZE}}` | [L69](../sft_code/train_sft.sh#L69) |
| `CACHE_ROOT` | `${CACHE_ROOT:-${PROJECT_ROOT}/cache/sft/qwen35_${SFT_MODEL_SIZE}}` | [L70](../sft_code/train_sft.sh#L70) |
| `LOCAL_RANK` | `${LOCAL_RANK:-}` | [L74](../sft_code/train_sft.sh#L74) |
| `NPROC_PER_NODE` | `${NPROC_PER_NODE:-8}` | [L78](../sft_code/train_sft.sh#L78) |
| `RESOURCE_GPU` | `${RESOURCE_GPU:-${NPROC_PER_NODE:-8}}` | [L78](../sft_code/train_sft.sh#L78) |
| `NPROC_PER_NODE` | `${NPROC_PER_NODE:-${RESOURCE_GPU}}` | [L79](../sft_code/train_sft.sh#L79) |
| `RESOURCE_GPU` | `${RESOURCE_GPU}` | [L79](../sft_code/train_sft.sh#L79) |
| `NNODES` | `${NNODES:-}` | [L80](../sft_code/train_sft.sh#L80) |
| `NNODES` | `${NNODES}` | [L80](../sft_code/train_sft.sh#L80) |
| `WORLD_SIZE` | `${WORLD_SIZE:-}` | [L80](../sft_code/train_sft.sh#L80) |
| `WORLD_SIZE` | `${WORLD_SIZE}` | [L80](../sft_code/train_sft.sh#L80) |
| `NODE_RANK` | `${NODE_RANK:-}` | [L83](../sft_code/train_sft.sh#L83) |
| `NODE_RANK` | `${NODE_RANK}` | [L83](../sft_code/train_sft.sh#L83) |
| `RANK` | `${RANK:-}` | [L83](../sft_code/train_sft.sh#L83) |
| `RANK` | `${RANK}` | [L83](../sft_code/train_sft.sh#L83) |
| `DEFAULT_NNODES` | `${DEFAULT_NNODES}` | [L86](../sft_code/train_sft.sh#L86) |
| `NNODES` | `${NNODES:-${WORLD_SIZE:-${DEFAULT_NNODES}}}` | [L86](../sft_code/train_sft.sh#L86) |
| `WORLD_SIZE` | `${WORLD_SIZE:-${DEFAULT_NNODES}}` | [L86](../sft_code/train_sft.sh#L86) |
| `NODE_RANK` | `${NODE_RANK:-${RANK:-0}}` | [L87](../sft_code/train_sft.sh#L87) |
| `RANK` | `${RANK:-0}` | [L87](../sft_code/train_sft.sh#L87) |
| `MASTER_ADDR` | `${MASTER_ADDR:-${SINGLE_NODE_MASTER_ADDR:-}}` | [L88](../sft_code/train_sft.sh#L88) |
| `SINGLE_NODE_MASTER_ADDR` | `${SINGLE_NODE_MASTER_ADDR:-}` | [L88](../sft_code/train_sft.sh#L88) |
| `MASTER_PORT` | `${MASTER_PORT:-${SINGLE_NODE_MASTER_PORT:-29503}}` | [L89](../sft_code/train_sft.sh#L89) |
| `SINGLE_NODE_MASTER_PORT` | `${SINGLE_NODE_MASTER_PORT:-29503}` | [L89](../sft_code/train_sft.sh#L89) |
| `CUDA_VISIBLE_DEVICES` | `${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}` | [L90](../sft_code/train_sft.sh#L90) |
| `MASTER_ADDR` | `${MASTER_ADDR:-127.0.0.1}` | [L105](../sft_code/train_sft.sh#L105) |
| `MASTER_ADDR` | `${MASTER_ADDR}` | [L106](../sft_code/train_sft.sh#L106) |
| `CUDA_VISIBLE_DEVICES` | `${CUDA_VISIBLE_DEVICES}` | [L109](../sft_code/train_sft.sh#L109) |
| `DEFAULT_TP` | `${DEFAULT_TP}` | [L115](../sft_code/train_sft.sh#L115) |
| `TENSOR_MODEL_PARALLEL_SIZE` | `${TENSOR_MODEL_PARALLEL_SIZE:-${DEFAULT_TP}}` | [L115](../sft_code/train_sft.sh#L115) |
| `DEFAULT_PP` | `${DEFAULT_PP}` | [L116](../sft_code/train_sft.sh#L116) |
| `PIPELINE_MODEL_PARALLEL_SIZE` | `${PIPELINE_MODEL_PARALLEL_SIZE:-${DEFAULT_PP}}` | [L116](../sft_code/train_sft.sh#L116) |
| `CONTEXT_PARALLEL_SIZE` | `${CONTEXT_PARALLEL_SIZE:-4}` | [L117](../sft_code/train_sft.sh#L117) |
| `DEFAULT_EP` | `${DEFAULT_EP}` | [L118](../sft_code/train_sft.sh#L118) |
| `EXPERT_MODEL_PARALLEL_SIZE` | `${EXPERT_MODEL_PARALLEL_SIZE:-${DEFAULT_EP}}` | [L118](../sft_code/train_sft.sh#L118) |
| `EXPERT_TENSOR_PARALLEL_SIZE` | `${EXPERT_TENSOR_PARALLEL_SIZE:-1}` | [L119](../sft_code/train_sft.sh#L119) |
| `CP_COMM_TYPE` | `${CP_COMM_TYPE:-p2p}` | [L120](../sft_code/train_sft.sh#L120) |
| `MICRO_BATCH_SIZE` | `${MICRO_BATCH_SIZE:-${PER_DEVICE_TRAIN_BATCH_SIZE:-1}}` | [L121](../sft_code/train_sft.sh#L121) |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `${PER_DEVICE_TRAIN_BATCH_SIZE:-1}` | [L121](../sft_code/train_sft.sh#L121) |
| `GLOBAL_BATCH_SIZE` | `${GLOBAL_BATCH_SIZE:-128}` | [L122](../sft_code/train_sft.sh#L122) |
| `NUM_TRAIN_EPOCHS` | `${NUM_TRAIN_EPOCHS:-2}` | [L123](../sft_code/train_sft.sh#L123) |
| `TRAIN_ITERS` | `${TRAIN_ITERS:-}` | [L124](../sft_code/train_sft.sh#L124) |
| `MAX_LENGTH` | `${MAX_LENGTH:-65536}` | [L125](../sft_code/train_sft.sh#L125) |
| `TRUNCATION_STRATEGY` | `${TRUNCATION_STRATEGY:-right}` | [L126](../sft_code/train_sft.sh#L126) |
| `LEARNING_RATE` | `${LEARNING_RATE:-1e-5}` | [L127](../sft_code/train_sft.sh#L127) |
| `MIN_LR` | `${MIN_LR:-0}` | [L128](../sft_code/train_sft.sh#L128) |
| `WARMUP_RATIO` | `${WARMUP_RATIO:-0.03}` | [L129](../sft_code/train_sft.sh#L129) |
| `WEIGHT_DECAY` | `${WEIGHT_DECAY:-0.1}` | [L130](../sft_code/train_sft.sh#L130) |
| `MAX_GRAD_NORM` | `${MAX_GRAD_NORM:-1.0}` | [L131](../sft_code/train_sft.sh#L131) |
| `LOSS_SCALE` | `${LOSS_SCALE:-default}` | [L132](../sft_code/train_sft.sh#L132) |
| `PACKING` | `${PACKING:-false}` | [L133](../sft_code/train_sft.sh#L133) |
| `PADDING_FREE` | `${PADDING_FREE:-false}` | [L134](../sft_code/train_sft.sh#L134) |
| `DATASET_NUM_PROC` | `${DATASET_NUM_PROC:-8}` | [L135](../sft_code/train_sft.sh#L135) |
| `DATALOADER_NUM_WORKERS` | `${DATALOADER_NUM_WORKERS:-4}` | [L136](../sft_code/train_sft.sh#L136) |
| `LOAD_FROM_CACHE_FILE` | `${LOAD_FROM_CACHE_FILE:-false}` | [L137](../sft_code/train_sft.sh#L137) |
| `DATASET_SHUFFLE` | `${DATASET_SHUFFLE:-true}` | [L138](../sft_code/train_sft.sh#L138) |
| `SAVE_STEPS` | `${SAVE_STEPS:-10}` | [L139](../sft_code/train_sft.sh#L139) |
| `LOGGING_STEPS` | `${LOGGING_STEPS:-1}` | [L140](../sft_code/train_sft.sh#L140) |
| `SAVE_TOTAL_LIMIT` | `${SAVE_TOTAL_LIMIT:-20}` | [L141](../sft_code/train_sft.sh#L141) |
| `SAVE_SAFETENSORS` | `${SAVE_SAFETENSORS:-true}` | [L142](../sft_code/train_sft.sh#L142) |
| `TRAIN_USER_LOSS` | `${TRAIN_USER_LOSS:-true}` | [L143](../sft_code/train_sft.sh#L143) |
| `FREEZE_VIT` | `${FREEZE_VIT:-true}` | [L144](../sft_code/train_sft.sh#L144) |
| `FREEZE_ALIGNER` | `${FREEZE_ALIGNER:-true}` | [L145](../sft_code/train_sft.sh#L145) |
| `OPTIMIZER_CPU_OFFLOAD` | `${OPTIMIZER_CPU_OFFLOAD:-true}` | [L146](../sft_code/train_sft.sh#L146) |
| `OPTIMIZER_OFFLOAD_FRACTION` | `${OPTIMIZER_OFFLOAD_FRACTION:-1}` | [L147](../sft_code/train_sft.sh#L147) |
| `USE_PRECISION_AWARE_OPTIMIZER` | `${USE_PRECISION_AWARE_OPTIMIZER:-true}` | [L148](../sft_code/train_sft.sh#L148) |
| `DRY_RUN` | `${DRY_RUN:-false}` | [L149](../sft_code/train_sft.sh#L149) |
| `USE_MCORE_GDN` | `${USE_MCORE_GDN:-1}` | [L150](../sft_code/train_sft.sh#L150) |
| `MICRO_BATCH_SIZE` | `${MICRO_BATCH_SIZE}` | [L163](../sft_code/train_sft.sh#L163) |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `${PER_DEVICE_TRAIN_BATCH_SIZE:-}` | [L163](../sft_code/train_sft.sh#L163) |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `${PER_DEVICE_TRAIN_BATCH_SIZE}` | [L163](../sft_code/train_sft.sh#L163) |
| `MAX_GPUS` | `${MAX_GPUS}` | [L168](../sft_code/train_sft.sh#L168) |
| `WORLD_SIZE_TOTAL` | `${WORLD_SIZE_TOTAL}` | [L168](../sft_code/train_sft.sh#L168) |
| `MODEL_PARALLEL_SIZE` | `${MODEL_PARALLEL_SIZE}` | [L172](../sft_code/train_sft.sh#L172) |
| `GDN_HEAD_PARTITIONS` | `${GDN_HEAD_PARTITIONS}` | [L181](../sft_code/train_sft.sh#L181) |
| `USE_MCORE_GDN` | `${USE_MCORE_GDN}` | [L186](../sft_code/train_sft.sh#L186) |
| `CP_COMM_TYPE` | `${CP_COMM_TYPE}` | [L189](../sft_code/train_sft.sh#L189) |
| `DEFAULT_SEQUENCE_PARALLEL` | `${DEFAULT_SEQUENCE_PARALLEL}` | [L193](../sft_code/train_sft.sh#L193) |
| `SEQUENCE_PARALLEL` | `${SEQUENCE_PARALLEL:-${DEFAULT_SEQUENCE_PARALLEL}}` | [L193](../sft_code/train_sft.sh#L193) |
| `SEQUENCE_PARALLEL` | `${SEQUENCE_PARALLEL}` | [L194](../sft_code/train_sft.sh#L194) |
| `EXPERT_PARALLEL_SIZE` | `${EXPERT_PARALLEL_SIZE}` | [L209](../sft_code/train_sft.sh#L209) |
| `STEP_BATCH_SIZE` | `${STEP_BATCH_SIZE}` | [L217](../sft_code/train_sft.sh#L217) |
| `GRADIENT_ACCUMULATION_STEPS` | `${GRADIENT_ACCUMULATION_STEPS:-}` | [L220](../sft_code/train_sft.sh#L220) |
| `GRADIENT_ACCUMULATION_STEPS` | `${GRADIENT_ACCUMULATION_STEPS}` | [L220](../sft_code/train_sft.sh#L220) |
| `GRAD_ACCUM` | `${GRAD_ACCUM}` | [L220](../sft_code/train_sft.sh#L220) |
| `MEGATRON_ROOT` | `${MEGATRON_ROOT:+${MEGATRON_ROOT}:}` | [L225](../sft_code/train_sft.sh#L225) |
| `MEGATRON_ROOT` | `${MEGATRON_ROOT}` | [L225](../sft_code/train_sft.sh#L225) |
| `PYTHONPATH` | `${PYTHONPATH:+:${PYTHONPATH}}` | [L225](../sft_code/train_sft.sh#L225) |
| `PYTHONPATH` | `${PYTHONPATH}` | [L225](../sft_code/train_sft.sh#L225) |
| `SWIFT_ROOT` | `${SWIFT_ROOT}` | [L225](../sft_code/train_sft.sh#L225) |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `${CUDA_DEVICE_MAX_CONNECTIONS:-1}` | [L227](../sft_code/train_sft.sh#L227) |
| `FLA_TILELANG` | `${FLA_TILELANG:-1}` | [L228](../sft_code/train_sft.sh#L228) |
| `HF_HUB_OFFLINE` | `export HF_HUB_OFFLINE=1` | [L232](../sft_code/train_sft.sh#L232) |
| `TRANSFORMERS_OFFLINE` | `export TRANSFORMERS_OFFLINE=1` | [L233](../sft_code/train_sft.sh#L233) |
| `HF_DATASETS_OFFLINE` | `export HF_DATASETS_OFFLINE=1` | [L234](../sft_code/train_sft.sh#L234) |
| `MODELSCOPE_OFFLINE` | `export MODELSCOPE_OFFLINE=1` | [L235](../sft_code/train_sft.sh#L235) |
| `CACHE_ROOT` | `${CACHE_ROOT}` | [L236](../sft_code/train_sft.sh#L236) |
| `TOKENIZERS_PARALLELISM` | `${TOKENIZERS_PARALLELISM:-false}` | [L243](../sft_code/train_sft.sh#L243) |
| `PYTORCH_ALLOC_CONF` | `${PYTORCH_ALLOC_CONF:-expandable_segments:True}` | [L244](../sft_code/train_sft.sh#L244) |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | `${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}` | [L245](../sft_code/train_sft.sh#L245) |
| `TORCH_NCCL_ENABLE_MONITORING` | `${TORCH_NCCL_ENABLE_MONITORING:-0}` | [L246](../sft_code/train_sft.sh#L246) |
| `TORCH_NCCL_BLOCKING_WAIT` | `${TORCH_NCCL_BLOCKING_WAIT:-1}` | [L247](../sft_code/train_sft.sh#L247) |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING` | `${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}` | [L249](../sft_code/train_sft.sh#L249) |
| `NCCL_DEBUG` | `${NCCL_DEBUG:-WARN}` | [L250](../sft_code/train_sft.sh#L250) |
| `IMAGE_MAX_TOKEN_NUM` | `${IMAGE_MAX_TOKEN_NUM:-1024}` | [L251](../sft_code/train_sft.sh#L251) |
| `VIDEO_MAX_TOKEN_NUM` | `${VIDEO_MAX_TOKEN_NUM:-128}` | [L252](../sft_code/train_sft.sh#L252) |
| `FPS_MAX_FRAMES` | `${FPS_MAX_FRAMES:-12}` | [L253](../sft_code/train_sft.sh#L253) |
| `TRAIN_USER_LOSS` | `${TRAIN_USER_LOSS}` | [L254](../sft_code/train_sft.sh#L254) |
| `SWIFT_TRAIN_USER_LOSS` | `export SWIFT_TRAIN_USER_LOSS=1` | [L255](../sft_code/train_sft.sh#L255) |
| `DRY_RUN` | `${DRY_RUN:-0}` | [L259](../sft_code/train_sft.sh#L259) |
| `SWIFT_TRAIN_USER_LOSS` | `${SWIFT_TRAIN_USER_LOSS:-0}` | [L261](../sft_code/train_sft.sh#L261) |
| `SWIFT_VENV` | `${SWIFT_VENV}` | [L262](../sft_code/train_sft.sh#L262) |
| `LOSS_CHECK_ARGS` | `${LOSS_CHECK_ARGS[@]}` | [L263](../sft_code/train_sft.sh#L263) |
| `MODEL_PATH` | `${MODEL_PATH}` | [L268](../sft_code/train_sft.sh#L268) |
| `MODEL_TYPE` | `${MODEL_TYPE}` | [L269](../sft_code/train_sft.sh#L269) |
| `TEMPLATE` | `${TEMPLATE}` | [L270](../sft_code/train_sft.sh#L270) |
| `FREEZE_VIT` | `${FREEZE_VIT}` | [L278](../sft_code/train_sft.sh#L278) |
| `FREEZE_ALIGNER` | `${FREEZE_ALIGNER}` | [L279](../sft_code/train_sft.sh#L279) |
| `TENSOR_MODEL_PARALLEL_SIZE` | `${TENSOR_MODEL_PARALLEL_SIZE}` | [L280](../sft_code/train_sft.sh#L280) |
| `PIPELINE_MODEL_PARALLEL_SIZE` | `${PIPELINE_MODEL_PARALLEL_SIZE}` | [L281](../sft_code/train_sft.sh#L281) |
| `CONTEXT_PARALLEL_SIZE` | `${CONTEXT_PARALLEL_SIZE}` | [L282](../sft_code/train_sft.sh#L282) |
| `GLOBAL_BATCH_SIZE` | `${GLOBAL_BATCH_SIZE}` | [L286](../sft_code/train_sft.sh#L286) |
| `LEARNING_RATE` | `${LEARNING_RATE}` | [L295](../sft_code/train_sft.sh#L295) |
| `MIN_LR` | `${MIN_LR}` | [L296](../sft_code/train_sft.sh#L296) |
| `WARMUP_RATIO` | `${WARMUP_RATIO}` | [L298](../sft_code/train_sft.sh#L298) |
| `WEIGHT_DECAY` | `${WEIGHT_DECAY}` | [L299](../sft_code/train_sft.sh#L299) |
| `MAX_GRAD_NORM` | `${MAX_GRAD_NORM}` | [L300](../sft_code/train_sft.sh#L300) |
| `USE_PRECISION_AWARE_OPTIMIZER` | `${USE_PRECISION_AWARE_OPTIMIZER}` | [L302](../sft_code/train_sft.sh#L302) |
| `OPTIMIZER_CPU_OFFLOAD` | `${OPTIMIZER_CPU_OFFLOAD}` | [L303](../sft_code/train_sft.sh#L303) |
| `OPTIMIZER_OFFLOAD_FRACTION` | `${OPTIMIZER_OFFLOAD_FRACTION}` | [L304](../sft_code/train_sft.sh#L304) |
| `MAX_LENGTH` | `${MAX_LENGTH}` | [L306](../sft_code/train_sft.sh#L306) |
| `TRUNCATION_STRATEGY` | `${TRUNCATION_STRATEGY}` | [L307](../sft_code/train_sft.sh#L307) |
| `LOSS_SCALE` | `${LOSS_SCALE}` | [L308](../sft_code/train_sft.sh#L308) |
| `PACKING` | `${PACKING}` | [L309](../sft_code/train_sft.sh#L309) |
| `PADDING_FREE` | `${PADDING_FREE}` | [L310](../sft_code/train_sft.sh#L310) |
| `DATASET_PATH` | `${DATASET_PATH}` | [L311](../sft_code/train_sft.sh#L311) |
| `LOAD_FROM_CACHE_FILE` | `${LOAD_FROM_CACHE_FILE}` | [L312](../sft_code/train_sft.sh#L312) |
| `DATASET_SHUFFLE` | `${DATASET_SHUFFLE}` | [L313](../sft_code/train_sft.sh#L313) |
| `DATASET_NUM_PROC` | `${DATASET_NUM_PROC}` | [L315](../sft_code/train_sft.sh#L315) |
| `DATALOADER_NUM_WORKERS` | `${DATALOADER_NUM_WORKERS}` | [L316](../sft_code/train_sft.sh#L316) |
| `SAVE_STEPS` | `${SAVE_STEPS}` | [L317](../sft_code/train_sft.sh#L317) |
| `LOGGING_STEPS` | `${LOGGING_STEPS}` | [L318](../sft_code/train_sft.sh#L318) |
| `SAVE_TOTAL_LIMIT` | `${SAVE_TOTAL_LIMIT}` | [L319](../sft_code/train_sft.sh#L319) |
| `SAVE_SAFETENSORS` | `${SAVE_SAFETENSORS}` | [L320](../sft_code/train_sft.sh#L320) |
| `OUTPUT_DIR` | `${OUTPUT_DIR}` | [L323](../sft_code/train_sft.sh#L323) |
| `EXPERT_MODEL_PARALLEL_SIZE` | `${EXPERT_MODEL_PARALLEL_SIZE}` | [L327](../sft_code/train_sft.sh#L327) |
| `EXPERT_TENSOR_PARALLEL_SIZE` | `${EXPERT_TENSOR_PARALLEL_SIZE}` | [L328](../sft_code/train_sft.sh#L328) |
| `MCORE_MODEL` | `${MCORE_MODEL}` | [L336](../sft_code/train_sft.sh#L336) |
| `TRAIN_ITERS` | `${TRAIN_ITERS}` | [L337](../sft_code/train_sft.sh#L337) |
| `NUM_TRAIN_EPOCHS` | `${NUM_TRAIN_EPOCHS}` | [L340](../sft_code/train_sft.sh#L340) |
| `NPROC_PER_NODE` | `${NPROC_PER_NODE}` | [L345](../sft_code/train_sft.sh#L345) |
| `DP_SIZE` | `${DP_SIZE}` | [L347](../sft_code/train_sft.sh#L347) |
| `EXPERT_DP_SIZE` | `${EXPERT_DP_SIZE}` | [L349](../sft_code/train_sft.sh#L349) |
| `TRAIN_ITERS` | `${TRAIN_ITERS:-auto}` | [L352](../sft_code/train_sft.sh#L352) |
| `MCORE_MODEL` | `${MCORE_MODEL:-none}` | [L354](../sft_code/train_sft.sh#L354) |
| `MASTER_PORT` | `${MASTER_PORT}` | [L356](../sft_code/train_sft.sh#L356) |
| `DRY_RUN` | `${DRY_RUN}` | [L358](../sft_code/train_sft.sh#L358) |
| `ARGS` | `${ARGS[@]}` | [L360](../sft_code/train_sft.sh#L360) |
| `LAUNCH` | `${LAUNCH[@]}` | [L360](../sft_code/train_sft.sh#L360) |

<a id="entry-65"></a>

## sft_code/validate_sft_runtime.py

[源码](../sft_code/validate_sft_runtime.py)


### CLI

| 参数 | 声明 | 位置 |
| --- | --- | --- |
| `--model` | `type=Path; required=True` | [L77](../sft_code/validate_sft_runtime.py#L77) |
| `--model-size` | `choices=MODEL_SHAPES; required=True` | [L78](../sft_code/validate_sft_runtime.py#L78) |
| `--model-only` | `action='store_true'` | [L81](../sft_code/validate_sft_runtime.py#L81) |
