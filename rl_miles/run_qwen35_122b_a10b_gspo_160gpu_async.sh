#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5-122B-A10B MoE, 160-GPU decoupled async GSPO.
#
#   actor   96 GPUs (12 nodes x 8) -> TP2/PP12/CP4/EP8 -> DP1
#   rollout  64 GPUs (8 nodes x 8)  -> 8 SGLang engines, 8 GPUs each
#
# Actor and rollout use separate GPUs; rollout for step N+1 runs while the
# actor trains on step N.
#
# Requires a 20-node Ray cluster. HOSTFILE must contain the 19 worker IPs
# (one per line, excluding the master node).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
if [[ "${DRY_RUN:-0}" != "1" && "${DRY_RUN:-0}" != "true" && -f "${SCRIPT_DIR}/official_env.sh" ]]; then
  source "${SCRIPT_DIR}/official_env.sh"
fi

# Data and checkpoint outputs are controlled by environment variables.
export MILES_PROMPT_DATA="${MILES_PROMPT_DATA:-${PROJECT_ROOT}/data/rl/datasets/world_model_rl/train.jsonl}"
if [[ ! -v EVAL_DATA ]]; then
  export EVAL_DATA="${PROJECT_ROOT}/data/rl/datasets/world_model_rl/test.jsonl"
fi

# Configure judge and similarity endpoints through the environment.
# This wrapper enables thinking and similarity by default. With both scorers
# successful, the default base reward is:
#   0.9 * judge_reward + 0.1 * similarity_score.
# Weights and enable flags remain overridable. Service-failure fallbacks and
# the thinking-length penalty are applied by reward.py as appropriate.
export JUDGE_PROMPT_DIR="${JUDGE_PROMPT_DIR:-${SCRIPT_DIR}/../prompts}"
export JUDGE_BASE_URL="${SMOKE_JUDGE_BASE_URL:-${JUDGE_BASE_URL:-http://<JUDGE_HOST>:<JUDGE_PORT>/v1}}"
export WM_JUDGE_BASE_URL="${JUDGE_BASE_URL}"
export JUDGE_MODEL="${JUDGE_MODEL:-DeepSeek-V4-Flash-0731}"
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"
export JUDGE_WEIGHT="${JUDGE_WEIGHT:-0.9}"
export SIMILARITY_ENABLED="${SIMILARITY_ENABLED:-1}"
export SIMILARITY_WEIGHT="${SIMILARITY_WEIGHT:-0.1}"
export SIMILARITY_URL="${SIMILARITY_URL:-http://<SIMILARITY_HOST>:<SIMILARITY_PORT>/similarity}"
export THINK_MODE="${THINK_MODE:-1}"

export MODEL_TAG="${MODEL_TAG:-qwen3.5-122B-A10B}"
export MODEL_HF_DEFAULT="${MODEL_HF_DEFAULT:-${PROJECT_ROOT}/models/Qwen3.5-122B-A10B}"
export SAVE_DEFAULT="${SAVE_DEFAULT:-${PROJECT_ROOT}/checkpoints/qwen35_122b_a10b_gspo_160gpu_async}"
# Resume only when this run has an actual Megatron checkpoint; otherwise initialize from HF.
if [[ -z "${MILES_LOAD:-}" ]]; then
  if [[ -f "${SAVE_DEFAULT}/latest_checkpointed_iteration.txt" ]]; then
    export MILES_LOAD="${SAVE_DEFAULT}"
  else
    export MILES_LOAD="${MODEL_HF_DEFAULT}"
  fi
fi
export MILES_SAVE="${MILES_SAVE:-${SAVE_DEFAULT}}"

# Core actor grid is TP2 * PP12 * CP4 = 96, so data-parallel size is 1.
# Each PP stage has TP2 * CP4 = 8 ranks; EP8 maps one expert shard per rank.
# Divisibility checks: 48 layers / PP12 = 4 layers/stage;
# 256 experts % EP8 == 0; 96 % (ETP1 * EP8 * PP12) == 0.
export TP_DEFAULT="${TP_DEFAULT:-2}"
export PP_DEFAULT="${PP_DEFAULT:-12}"
export CP_DEFAULT="${CP_DEFAULT:-4}"
export EP_DEFAULT="${EP_DEFAULT:-8}"
export ETP_DEFAULT="${ETP_DEFAULT:-1}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}"
export MIN_LR="${MIN_LR:-1e-7}"
export USE_KL_LOSS="${USE_KL_LOSS:-1}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.01}"
export REF_LOAD="${REF_LOAD:-${MILES_LOAD}}"
export DECODER_LAST_PIPELINE_NUM_LAYERS_DEFAULT="${DECODER_LAST_PIPELINE_NUM_LAYERS_DEFAULT:-}"
export MAX_TOKENS_PER_GPU_DEFAULT="${MAX_TOKENS_PER_GPU_DEFAULT:-4096}"
export LOG_PROBS_MAX_TOKENS_PER_GPU_DEFAULT="${LOG_PROBS_MAX_TOKENS_PER_GPU_DEFAULT:-4096}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-2048}"

# 64 rollout GPUs / 8 GPUs per engine = 8 SGLang engines.
export ROLLOUT_GPUS_PER_ENGINE_DEFAULT="${ROLLOUT_GPUS_PER_ENGINE_DEFAULT:-8}"

export SEQ_LEN="${SEQ_LEN:-43008}"
export MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-32768}"
export EVAL_MAX_PROMPT_TOKENS="${EVAL_MAX_PROMPT_TOKENS:-65536}"
export EVAL_MAX_CONTEXT_LEN="${EVAL_MAX_CONTEXT_LEN:-81920}"
export EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-16384}"
export MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-10240}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2048}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-}"
export NUM_EPOCH="${NUM_EPOCH:-2}"

# Decoupled async mode.
export COLOCATE="${COLOCATE:-0}"
export TRAIN_DRIVER="train_async.py"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-64}"
export ROLLOUT_FUNCTION_PATH="${ROLLOUT_FUNCTION_PATH:-world_model_rl.fully_async_rollout.JsonlFullyAsyncRolloutFn}"
export EVAL_FUNCTION_PATH="${EVAL_FUNCTION_PATH:-world_model_rl.fully_async_rollout.JsonlFullyAsyncRolloutFn}"

export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.8}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-32}"
export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-32}"
export MAX_WEIGHT_STALENESS="${MAX_WEIGHT_STALENESS:-1}"
export ASYNC_DATA_BUFFER_CAPACITY_FACTOR="${ASYNC_DATA_BUFFER_CAPACITY_FACTOR:-1.0}"

# 12 actor nodes + 8 rollout nodes = 20 nodes / 160 GPUs total.
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-12}"
export RESOURCE_GPU="${RESOURCE_GPU:-8}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-${RESOURCE_GPU}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${RESOURCE_GPU}}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-20}"

exec bash "${SCRIPT_DIR}/run_gspo.sh" "$@" \
  --max-weight-staleness "${MAX_WEIGHT_STALENESS}" \
  --async-data-buffer-capacity-factor "${ASYNC_DATA_BUFFER_CAPACITY_FACTOR}" \
  --override-opt-param-scheduler \
  --skip-eval-before-train \
  --empty-unused-memory-level 1 \
  --recompute-loss-function
