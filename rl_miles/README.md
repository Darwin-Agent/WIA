# RL: Miles asynchronous GSPO

**English** | [简体中文](README_zh.md) · [Home](../README.md) · [Installation](../ENVIRONMENT.md) · [Configuration](../docs/CONFIGURATION.md)

Separate actor and rollout GPU pools train a language world model: SGLang generates observations, scorer services
provide rewards, Megatron updates the policy, and Miles manages async sampling and weight synchronization.

## Required framework installation

**RL cannot train without Miles, Megatron, Megatron-Bridge, SGLang and their Python/GPU dependencies.**
They are not bundled in the source ZIP. Follow the base-environment requirements in [Installation](../ENVIRONMENT.md),
then run from the repository root:

```bash
bash rl_miles/install_official_miles_env.sh
```

The installer calls `bootstrap_miles.sh` to fetch/check pinned sources and apply runtime patches, then installs the
RL environment. Default sources are `rl_miles/miles`, `rl_miles/Megatron-LM`, `rl_miles/Megatron-Bridge` and
`rl_miles/sglang-miles`; the default venv is `rl_miles/.venv`. No manual clone of arbitrary upstream versions is required.
**Bootstrap alone is not an environment installation.** Complete any additional dependencies listed in the installation
guide and verify imports before training. Training launchers do not automatically install missing frameworks.
Use `MILES_REPO`, `MILES_VENV`, `MILES_MEGATRON_ROOT` and `MILES_SGLANG_ROOT` consistently for custom locations.
An existing environment is reusable only with matching revisions, patches and dependencies.
`bash rl_miles/install_official_miles_env.sh --help` is read-only; `bash rl_miles/bootstrap_miles.sh --check` verifies
source revisions/patches, not GPU readiness. Weights, datasets, all cluster nodes and the required scoring services
must be prepared separately. See [installation and reuse](../ENVIRONMENT.md#installation-and-reuse).

## Models and parallelism

| Model | Launcher | Total nodes / GPUs | Actor nodes / GPUs | TP / PP / CP | EP / ETP | Dense DP | Rollout engines |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| 4B | `run_qwen35_4b_gspo_128gpu_async.sh` | 16 / 128 | 8 / 64 | 2 / 1 / 4 | 1 / 1 | 8 | 64 × 1 GPU |
| 9B | `run_qwen35_9b_gspo_128gpu_async.sh` | 16 / 128 | 8 / 64 | 2 / 2 / 4 | 1 / 1 | 4 | 32 × 2 GPU |
| 27B | `run_qwen35_27b_gspo_128gpu_async.sh` | 16 / 128 | 8 / 64 | 4 / 4 / 4 | 1 / 1 | 1 | 32 × 2 GPU |
| 35B-A3B | `run_qwen35_35b_a3b_gspo_128gpu_async.sh` | 16 / 128 | 8 / 64 | 2 / 4 / 4 | 8 / 1 | 2 | 32 × 2 GPU |
| 122B-A10B | `run_qwen35_122b_a10b_gspo_160gpu_async.sh` | 20 / 160 | 12 / 96 | 2 / 12 / 4 | 8 / 1 | 1 | 8 × 8 GPU |

Each node has 8 GPUs; rollout always uses 64 GPUs. Scoring services need additional resources.
Dense DP = actor GPUs/(TP×PP×CP). Expert DP uses actor GPUs/(ETP×EP×PP): 2 for 35B, 1 for 122B.
Wrappers select `train_async.py`; `COLOCATE=1` is unsupported.

| Model | Rollout batch (prompts) | Samples/prompt | Global batch | Prompt / response tokens | Sequence length |
| --- | ---: | ---: | ---: | --- | ---: |
| 4B | 128 | 8 | 1024 | 32768 / 8192 | 40960 |
| 9B / 27B / 35B-A3B | 256 | 8 | 2048 | 32768 / 8192 | 40960 |
| 122B-A10B | 256 | 8 | 2048 | 32768 / 10240 | 43008 |

Micro batch is set to 1, but `QKV_FORMAT=thd` enables dynamic batching. Token budgets and sequence lengths determine
actual partitions; rollout batch is a prompt count, not a token count, and static accumulation arithmetic is insufficient.

## Services

1. **LLM judge:** enabled for every default profile. Deploy [DeepSeek-V4-Flash](DS-V4-Flash/README.md),
   set `JUDGE_BASE_URL` including `/v1` and `JUDGE_MODEL`; inject `JUDGE_API_KEY` only if authentication is required.
2. **Similarity:** enabled for 122B, optional for smaller models. Deploy [embedding + similarity](Qwen3-embedding/README.md)
   and provide the full `SIMILARITY_URL`, such as `http://similarity.example.invalid:8100/similarity`.
3. **Ray / policy SGLang:** managed by the training launcher on actor+rollout nodes; keep scorer GPUs separate.

Start embedding then similarity when needed, start the judge, validate requests, and only then train.
Loopback URLs cannot address services on other nodes. Use reachable hosts and trusted networks, not public unauthenticated ports.
No private proxy-model service is required.

## Data and checkpoints

Recommended JSONL fields are `prompt` and `ground_truth`, with optional `system/domain/dataset/source`:

```json
{"system":"Predict environment feedback.","prompt":"An action was performed.\n###[Observation]\n","ground_truth":"The action succeeds.","domain":"web","dataset":"ID"}
```

Compatible `messages` and prepared `prompt/label` formats are also accepted. Raw records are rendered according to
runtime CoT settings; prepared strings bypass rendering. To prepare data:

```bash
python rl_miles/prepare_rl_data.py --input /path/to/raw.jsonl --output /path/to/prepared.jsonl \
  --template-model /path/to/hf-model --think-mode off --max-prompt-tokens 32768
```

Use `--think-mode on` for 122B. Known prepared-mode mismatches are rejected; validate legacy records without mode metadata
yourself. Regenerate prepared prompts when switching CoT.

`MILES_HF_CHECKPOINT` must contain a complete HF model; `MILES_LOAD` selects initialization/resume and `MILES_SAVE` outputs.
Wrappers normally resume when the directory selected by `SAVE_DEFAULT` contains `latest_checkpointed_iteration.txt`,
otherwise initialize from HF. Explicitly set `MILES_LOAD/MILES_SAVE` for custom resume paths; changing only
`MILES_SAVE` does not change the automatic resume check. Use a new directory for a new model/experiment.
`REF_LOAD` selects KL reference weights.

## Quick start

Install the [RL environment](../ENVIRONMENT.md) at consistent paths on every node. Example for 4B, from the project root:

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

Replace paths/hosts. Execute on all nodes with `RANK=0..15`; for 122B use the 160-GPU wrapper,
`WORLD_SIZE=20`, ranks 0..19 and a configured `SIMILARITY_URL`. Remove dry-run to train.
**Even dry-run requires readable HF `config.json`.**

Two cluster modes:

- **Per-node launch (recommended):** set `WORLD_SIZE/RANK`, leave `HOSTFILE` empty. Head submits; workers join and block.
  `WORLD_SIZE` is total nodes, while `ACTOR_NUM_NODES` remains 8 or 12, not 16/20.
  The current head readiness wait covers actor node count only; start every rollout node too.
- **Hostfile:** execute on the head only. List one worker address per line, excluding head: 15 or 19 lines.
  The current implementation uses `ssh root@worker` and requires matching executable paths and corresponding access.
  If unavailable, use per-node mode. Keep real host inventories outside the repository.

The launcher starts/stops its Ray runtime; do not share that runtime with other jobs. Restrict dashboard port 8265 to trusted networks.

## Reward configuration

| Model | `THINK_MODE` | `JUDGE_WEIGHT` | `SIMILARITY_WEIGHT` | `SIMILARITY_ENABLED` |
| --- | ---: | ---: | ---: | ---: |
| 4B / 9B / 27B / 35B-A3B | 0 | 1.0 | 0.0 | 0 |
| 122B-A10B | 1 | 0.9 | 0.1 | 1 |

Judge dimensions use 1–5 scores with `format:1,factuality:2,consistency:2,realism:2,quality:2,efficiency:1`.
The weighted mean becomes `(score-1)/4`, clamped to [0,1]. Similarity is embedding cosine clamped to [0,1].

- Both scorers succeed: `JUDGE_WEIGHT × judge_reward + SIMILARITY_WEIGHT × similarity_score`.
- One succeeds: its **unweighted** score. Both fail: local rule reward. Judge-only failure also uses rules.
- `JUDGE_ENABLED=0` selects the local rule path, **not similarity-only mode**.
- Weights are independently clamped to [0,1], not normalized to sum to one.
- More than **10,000 thinking characters** multiplies the fused/fallback reward by 0.9. Exactly 10,000 does not.
  A plain long observation with CoT off is not thinking; explicit think tags remain recognized.
- Thinking is removed before scoring. The judge itself is non-thinking independently of policy CoT.

| Variable | Default / purpose |
| --- | --- |
| `JUDGE_BASE_URL` | Reachable `/v1` URL; comma-separated endpoints supported |
| `JUDGE_MODEL` | `DeepSeek-V4-Flash-0731`; must match the served model |
| `JUDGE_API_KEY` | Runtime environment only; `EMPTY` is an unauthenticated SDK placeholder |
| `JUDGE_MAX_TOKENS` / `JUDGE_CONCURRENCY` | 4096 / 128 |
| `JUDGE_WEIGHTS` | Six-dimension weights, distinct from scorer-mixing weights |
| `JUDGE_PROMPT_DIR` | Top-level `prompts` directory |
| `SIMILARITY_URL` | Full `/similarity` endpoint(s) |
| `SIMILARITY_TIMEOUT` / `SIMILARITY_CONCURRENCY` | 60 seconds / 128 |
| `WM_JUDGE_TIMEOUT` / `WM_JUDGE_RETRIES` | Worker default 300 seconds / 3 attempts; directly forwarded |
| `WM_SIMILARITY_RETRIES` | 3 attempts |
| `WM_JUDGE_SKIP` | Debug bypass of remote judging; changes the reward path |

Launchers map `JUDGE_*`, `SIMILARITY_*` and `THINK_MODE` to worker `WM_*` variables.
Prefer non-WM names when launching training: mapped values may override existing WM values.
Direct reward-module users configure WM names. The retry/timeout/skip controls above are direct pass-through exceptions.

## Training and resource overrides

| Category | Variables and defaults |
| --- | --- |
| Batch/duration | `ROLLOUT_BATCH_SIZE`, `N_SAMPLES_PER_PROMPT`, `GLOBAL_BATCH_SIZE` from table; nonempty `NUM_ROLLOUT` takes precedence |
| Length | `MAX_PROMPT_TOKENS`, `MAX_RESPONSE_TOKENS`, `SEQ_LEN`; response budget is constrained by total context |
| Optimizer | `LEARNING_RATE=1e-6`, `CLIP_GRAD=1`; smaller models `LR_DECAY_STYLE=constant`, `MIN_LR=0`; 122B cosine / `1e-7` |
| KL | Smaller models `USE_KL_LOSS=0`, `KL_LOSS_COEF=0`; 122B 1 / 0.01; optional `REF_LOAD` |
| Algorithm | `ADVANTAGE_ESTIMATOR=gspo`; clip epsilon fixed at `3.5e-4`, entropy coefficient fixed at zero; undeclared environment names do not override them |
| Actor resources | `ACTOR_NUM_NODES`, `ACTOR_NUM_GPUS_PER_NODE`, `RESOURCE_GPU`; Ray per-node allocation `RAY_NUM_GPUS` |
| Final parallel values | `TENSOR_PARALLEL_SIZE`, `PIPELINE_PARALLEL_SIZE`, `CONTEXT_PARALLEL_SIZE`, `EXPERT_PARALLEL_SIZE`, `EXPERT_TENSOR_PARALLEL_SIZE` |
| Parallel fallbacks | Final values take precedence over wrapper `TP_DEFAULT/PP_DEFAULT/CP_DEFAULT/EP_DEFAULT/ETP_DEFAULT` |
| Token batching | `MAX_TOKENS_PER_GPU` / `LOG_PROBS_MAX_TOKENS_PER_GPU`: smaller models 8192, 122B 4096; `LOG_PROBS_CHUNK_SIZE`: smaller models 4096, 122B 2048 |
| Other parallel controls | `SEQUENCE_PARALLEL=1`, `QKV_FORMAT=thd`, optional `DECODER_LAST_PIPELINE_NUM_LAYERS`; validate divisibility |
| Rollout resources | `ROLLOUT_NUM_GPUS=64`, `ROLLOUT_GPUS_PER_ENGINE` from table; disjoint from actor GPUs |
| Rollout performance | `SGLANG_MEM_FRACTION_STATIC=0.8`, `SGLANG_MAX_RUNNING_REQUESTS=32`, `SGLANG_SERVER_CONCURRENCY=32`, `ROLLOUT_TEMPERATURE=1.0` |
| Async | `MAX_WEIGHT_STALENESS=1`, `ASYNC_DATA_BUFFER_CAPACITY_FACTOR=1.0`, forwarded by wrappers |
| Save/eval | `SAVE_INTERVAL=20`, `EVAL_INTERVAL=20`, `LOG_INTERVAL=1`; explicit `EVAL_DATA=` disables eval; missing configured files fail |
| Eval length | `EVAL_MAX_PROMPT_TOKENS=65536`, `EVAL_MAX_RESPONSE_LEN=16384`, `EVAL_MAX_CONTEXT_LEN=81920` |
| Eval batch | `N_SAMPLES_PER_EVAL_PROMPT=1`, optional `EVAL_GLOBAL_BATCH_SIZE`, `EVAL_DATA_LIMIT` |
| Export/output | `MEGATRON_TO_HF_MODE=bridge`, `ROLLOUT_OUTPUT_DIR` defaults under `MILES_SAVE/rollout_jsonl` |
| Source/runtime | `MILES_VENV`, `MILES_REPO`, `MILES_MEGATRON_ROOT`, `MILES_SGLANG_ROOT`, `MILES_EXTRA_PYTHONPATH` |
| Network | `MASTER_ADDR/MASTER_PORT`, `HOSTFILE`, `LOCAL_IP`, `SOCKET_IFNAME`, Ray head/cluster timeout and polling variables |
| Debug | `RL_DEBUG_LOG`, `RAY_STOP_BEFORE_START`, `DEBUG_ROLLOUT_ONLY`, `DEBUG_TRAIN_ONLY`; the last two are mutually exclusive |
| W&B | Setting `WANDB_API_KEY` enables tracking; configure `WANDB_PROJECT/WANDB_GROUP`; no printable key argument is added |

See the [full parameter index](../docs/CONFIGURATION.md), including installation/service knobs and CLI.
Wrappers accept additional Miles arguments, but avoid duplicates of generated options; wrappers can append fixed arguments
after user input. Do not assume universal last-value precedence. `TRAIN_DRIVER` is fixed to async by these wrappers.

## Outputs and extension points

Checkpoints go to `MILES_SAVE`; rollout JSONL goes to `ROLLOUT_OUTPUT_DIR` and may contain sensitive model inputs/responses.
Keep outputs in controlled storage.

- [Custom data/reward hooks](world_model_rl/README.md)
- [Model architecture configurations](model_configs/README.md)
- [Wheels and ABI compatibility](wheels/README.md)
- [Independent evaluation](../eval/README.md), whose metrics do not implicitly include mixed RL rewards or thinking penalties.
