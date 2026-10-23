# Embedding 与相似度服务

[English](README.md) | **简体中文** · [项目首页](../../README_zh.md)

为 RL 提供 `POST /similarity`：输入预测 Observation 和 ground truth，返回限制到 [0,1]
的 embedding cosine。122B-A10B 默认需要此服务；其他尺度仅在启用 similarity 时需要。

## 架构

```text
RL reward → similarity API :8100 → embedding replicas :8200..8203
                                  每副本 TP1，各占一张 GPU
```

相似度 HTTP API 不加载大模型，GPU 由 embedding 副本占用。默认是 **4×TP1**，不是 TP4。
`deploy_qwen3_embedding_4gpu_tp4.sh` 是另一个单引擎 TP4 入口，不由默认组合脚本调用。

## 快速开始

从仓库根目录执行，服务 GPU 与训练 GPU 分开分配：

```bash
bash rl_miles/Qwen3-embedding/install_runtime.sh
export MODEL_PATH=/path/to/Qwen3-Embedding
export SERVED_MODEL_NAME=Qwen3-8B-Embedding
export EMBEDDING_MODEL_NAME=Qwen3-8B-Embedding
export GPU_IDS=0,1,2,3
export PORT_BASE=8200
export SIMILARITY_HOST=0.0.0.0 SIMILARITY_PORT=8100
bash rl_miles/Qwen3-embedding/start_similarity_services.sh
curl --noproxy '*' http://127.0.0.1:8100/health
curl --noproxy '*' -H 'Content-Type: application/json' \
  -d '{"text1":"The action succeeds.","text2":"The action succeeds."}' \
  http://127.0.0.1:8100/similarity
```

响应包含 `similarity` 和 `dimension`。`SERVED_MODEL_NAME` 必须与
`EMBEDDING_MODEL_NAME` 一致；使用不同名称时两处一起改。
组合启动器等待 API 健康，但训练前仍应执行一条真实 `/similarity` 请求。

训练节点配置：

```bash
export SIMILARITY_URL=http://similarity.example.invalid:8100/similarity
export SIMILARITY_ENABLED=1
export JUDGE_WEIGHT=0.9 SIMILARITY_WEIGHT=0.1
```

替换示例地址；这里需要完整 `/similarity` 路径，**不是 `/v1`**。

## 参数

| 参数 | 默认值 / 用途 |
| --- | --- |
| `MODEL_PATH` | 必填 embedding HF 权重目录 |
| `VENV_DIR` / `PYTHON_BASE` / `VLLM_VERSION` | 本目录 `.venv` / python3.12 / 0.28.0 |
| `GPU_IDS` / `PORT_BASE` / `WAIT_TIMEOUT` | 多副本 0,1,2,3 / 8200 / 360 秒 |
| `GPU_ID` / `HOST` / `PORT` | 单副本 0 / 127.0.0.1 / 8200；远程 API 访问后端时需调整监听与防火墙 |
| `SERVED_MODEL_NAME` / `EMBEDDING_MODEL_NAME` | 均为 `Qwen3-8B-Embedding` |
| `EMBEDDING_API_BASES` | 逗号分隔后端根地址，**不带 `/v1`**；默认本机 8200..8203 |
| `GPU_MEMORY_UTILIZATION` / `MAX_MODEL_LEN` | 0.90 / 32768 |
| `MAX_NUM_SEQS` / `MAX_NUM_BATCHED_TOKENS` | 8 / 65536 |
| `MAX_NUM_SCHEDULED_TOKENS` | 默认沿用 `MAX_NUM_BATCHED_TOKENS` |
| `KV_CACHE_DTYPE` | fp8；设空可不传此参数，需匹配硬件/运行时 |
| `ENABLE_PREFIX_CACHING` / `ENABLE_CHUNKED_PREFILL` | 1 / 1 |
| `SIMILARITY_HOST` / `SIMILARITY_PORT` | 0.0.0.0 / 8100 |
| `SIMILARITY_MAX_BATCH` / `SIMILARITY_BATCH_TIMEOUT` | 128 / 0.005 秒 |
| `SIMILARITY_QUEUE_LIMIT` / `SIMILARITY_REQUEST_TIMEOUT` | 5000 / 1200 秒；与 RL 客户端 timeout 不同 |
| `TRUNCATE_PROMPT_TOKENS` / `TRUNCATION_SIDE` | 32768 / right |
| `LOG_DIR` | 本目录 logs；保存服务 PID 与日志 |

改动 `GPU_IDS/PORT_BASE` 时，同步修改 `EMBEDDING_API_BASES`；端口列表不会自动重新计算。
使用已有远程 embedding 副本时，直接运行 `similarity_service.py`，不要运行会创建本地副本的组合脚本。
完整参数见 [配置参考](../../docs/CONFIGURATION_zh.md)。

## 停止与访问控制

```bash
bash rl_miles/Qwen3-embedding/stop_similarity_services.sh
```

使用与启动时相同的 `LOG_DIR/GPU_IDS`。这些服务不自带公网认证，只允许可信网络访问；
需要跨安全域时配置自己的认证网关。不要提交服务地址清单、日志或实际请求文本。
