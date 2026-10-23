# Embedding and similarity service

**English** | [简体中文](README_zh.md) · [Home](../../README.md) · [RL](../README.md) · [Configuration](../../docs/CONFIGURATION.md)

Serve `POST /similarity` for predicted observations and ground truth, returning embedding cosine clamped to [0,1].
122B-A10B enables this scorer by default; smaller models require it only when similarity is enabled.

## Architecture

```text
RL reward → similarity API :8100 → embedding replicas :8200..8203
                                  each replica TP1 on one GPU
```

The HTTP layer does not load the embedding model. The default is **four TP1 replicas**, not TP4.
`deploy_qwen3_embedding_4gpu_tp4.sh` is a separate single-engine TP4 entry, not used by the combined launcher.

## Quick start

Run from the project root on GPUs separate from training:

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

Responses contain `similarity` and `dimension`. Keep `SERVED_MODEL_NAME` and `EMBEDDING_MODEL_NAME` identical.
The startup health check is not a substitute for a real `/similarity` request.
On training nodes:

```bash
export SIMILARITY_URL=http://similarity.example.invalid:8100/similarity
export SIMILARITY_ENABLED=1
export JUDGE_WEIGHT=0.9 SIMILARITY_WEIGHT=0.1
```

Replace the example hostname. This URL includes `/similarity`, **not `/v1`**.

## Parameters

| Variable | Default / purpose |
| --- | --- |
| `MODEL_PATH` | Required local embedding HF weights |
| `VENV_DIR` / `PYTHON_BASE` / `VLLM_VERSION` | Local `.venv` / python3.12 / 0.28.0 |
| `GPU_IDS` / `PORT_BASE` / `WAIT_TIMEOUT` | Replica GPUs 0,1,2,3 / 8200 / 360 seconds |
| `GPU_ID` / `HOST` / `PORT` | Single replica 0 / 127.0.0.1 / 8200; adjust bind/firewall for remote backend access |
| `SERVED_MODEL_NAME` / `EMBEDDING_MODEL_NAME` | Both `Qwen3-8B-Embedding` |
| `EMBEDDING_API_BASES` | Comma-separated backend roots **without `/v1`**; localhost 8200..8203 by default |
| `GPU_MEMORY_UTILIZATION` / `MAX_MODEL_LEN` | 0.90 / 32768 |
| `MAX_NUM_SEQS` / `MAX_NUM_BATCHED_TOKENS` | 8 / 65536 |
| `MAX_NUM_SCHEDULED_TOKENS` | Falls back to `MAX_NUM_BATCHED_TOKENS` |
| `KV_CACHE_DTYPE` | fp8; empty omits the flag, validate hardware/runtime support |
| `ENABLE_PREFIX_CACHING` / `ENABLE_CHUNKED_PREFILL` | 1 / 1 |
| `SIMILARITY_HOST` / `SIMILARITY_PORT` | 0.0.0.0 / 8100 |
| `SIMILARITY_MAX_BATCH` / `SIMILARITY_BATCH_TIMEOUT` | 128 / 0.005 seconds |
| `SIMILARITY_QUEUE_LIMIT` / `SIMILARITY_REQUEST_TIMEOUT` | 5000 / 1200 seconds; distinct from the RL client timeout |
| `TRUNCATE_PROMPT_TOKENS` / `TRUNCATION_SIDE` | 32768 / right |
| `LOG_DIR` | Local logs/PIDs directory |

When changing `GPU_IDS/PORT_BASE`, update `EMBEDDING_API_BASES` explicitly; the endpoint list is not recomputed automatically.
For existing remote embedding replicas, run `similarity_service.py` directly instead of the combined local-start script.
See [all settings](../../docs/CONFIGURATION.md).

## Stop and protect access

```bash
bash rl_miles/Qwen3-embedding/stop_similarity_services.sh
```

Use the same `LOG_DIR/GPU_IDS` as startup. Services do not add public API authentication;
restrict access to trusted networks or deploy your own authenticated gateway. Keep logs, requests and endpoint inventories private.
