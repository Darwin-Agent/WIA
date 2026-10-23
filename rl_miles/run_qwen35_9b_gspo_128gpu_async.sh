#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5-9B, 128-GPU decoupled async GSPO.
#
# Total 128 GPUs split:
#   actor   64 GPUs (8 nodes x 8) -> TP2/PP2/CP4 -> DP4
#   rollout 64 GPUs (32 engines, 2 GPUs each)
#
# With PP2 and 32 decoder layers, leave decoder-last-pipeline-num-layers unset
# so Megatron splits layers evenly across pipeline stages: 16 layers per stage.
#
# Requires a 16-node Ray cluster. HOSTFILE must contain the 15 worker IPs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
if [[ "${DRY_RUN:-0}" != "1" && "${DRY_RUN:-0}" != "true" && -f "${SCRIPT_DIR}/official_env.sh" ]]; then
  source "${SCRIPT_DIR}/official_env.sh"
fi
export MILES_PROMPT_DATA="${MILES_PROMPT_DATA:-${PROJECT_ROOT}/data/rl/datasets/world_model_rl/train.jsonl}"
if [[ ! -v EVAL_DATA ]]; then
  export EVAL_DATA="${PROJECT_ROOT}/data/rl/datasets/world_model_rl/test.jsonl"
fi
export JUDGE_PROMPT_DIR="${JUDGE_PROMPT_DIR:-${SCRIPT_DIR}/../prompts}"
export JUDGE_BASE_URL="${SMOKE_JUDGE_BASE_URL:-${JUDGE_BASE_URL:-http://<JUDGE_HOST>:<JUDGE_PORT>/v1}}"
export WM_JUDGE_BASE_URL="${JUDGE_BASE_URL}"
export JUDGE_MODEL="${JUDGE_MODEL:-DeepSeek-V4-Flash-0731}"
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"
# Non-CoT by default for 4B/9B/27B/35B-A3B; an explicit override is supported.
export THINK_MODE="${THINK_MODE:-0}"
# Default reward is judge-only: weight 1.0, similarity weight 0.0.
# run_gspo.sh defaults SIMILARITY_ENABLED to 0. To use a mixture, explicitly
# enable similarity and configure both weights; environment overrides win.
export JUDGE_WEIGHT="${JUDGE_WEIGHT:-1.0}"
export SIMILARITY_WEIGHT="${SIMILARITY_WEIGHT:-0.0}"
export SIMILARITY_URL="${SIMILARITY_URL:-http://<SIMILARITY_HOST>:<SIMILARITY_PORT>/similarity}"

export MODEL_TAG="${MODEL_TAG:-qwen3.5-9B}"
export MODEL_HF_DEFAULT="${MODEL_HF_DEFAULT:-${PROJECT_ROOT}/models/Qwen3.5-9B}"
export SAVE_DEFAULT="${SAVE_DEFAULT:-${PROJECT_ROOT}/checkpoints/qwen35_9b_gspo_128gpu_async}"
# Resume from the latest saved iteration when available; otherwise start
# from MODEL_HF_DEFAULT. An explicit MILES_LOAD takes precedence.
if [[ -z "${MILES_LOAD:-}" ]]; then
  if [[ -f "${SAVE_DEFAULT}/latest_checkpointed_iteration.txt" ]]; then
    export MILES_LOAD="${SAVE_DEFAULT}"
  else
    export MILES_LOAD="${MODEL_HF_DEFAULT}"
  fi
fi
export TP_DEFAULT="${TP_DEFAULT:-2}"
export PP_DEFAULT="${PP_DEFAULT:-2}"
export CP_DEFAULT="${CP_DEFAULT:-4}"
export EP_DEFAULT="${EP_DEFAULT:-1}"
export ETP_DEFAULT="${ETP_DEFAULT:-1}"
export DECODER_LAST_PIPELINE_NUM_LAYERS_DEFAULT="${DECODER_LAST_PIPELINE_NUM_LAYERS_DEFAULT:-}"
export MAX_TOKENS_PER_GPU_DEFAULT="${MAX_TOKENS_PER_GPU_DEFAULT:-8192}"
export LOG_PROBS_MAX_TOKENS_PER_GPU_DEFAULT="${LOG_PROBS_MAX_TOKENS_PER_GPU_DEFAULT:-8192}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-4096}"
export ROLLOUT_GPUS_PER_ENGINE_DEFAULT="${ROLLOUT_GPUS_PER_ENGINE_DEFAULT:-2}"
export SEQ_LEN="${SEQ_LEN:-40960}"
export MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-32768}"
export EVAL_MAX_PROMPT_TOKENS="${EVAL_MAX_PROMPT_TOKENS:-65536}"
export EVAL_MAX_CONTEXT_LEN="${EVAL_MAX_CONTEXT_LEN:-81920}"
export EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-16384}"
export MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-8192}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2048}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-}"
export NUM_EPOCH="${NUM_EPOCH:-1}"

# Decoupled async mode.
export COLOCATE="${COLOCATE:-0}"
export TRAIN_DRIVER="train_async.py"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-64}"
export ROLLOUT_FUNCTION_PATH="${ROLLOUT_FUNCTION_PATH:-world_model_rl.fully_async_rollout.JsonlFullyAsyncRolloutFn}"
export EVAL_FUNCTION_PATH="${EVAL_FUNCTION_PATH:-world_model_rl.fully_async_rollout.JsonlFullyAsyncRolloutFn}"

export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.8}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-32}"
export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-32}"

# Match the 122B-A10B async staleness and buffer defaults.
export MAX_WEIGHT_STALENESS="${MAX_WEIGHT_STALENESS:-1}"
export ASYNC_DATA_BUFFER_CAPACITY_FACTOR="${ASYNC_DATA_BUFFER_CAPACITY_FACTOR:-1.0}"

export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-8}"
export RESOURCE_GPU="${RESOURCE_GPU:-8}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-${RESOURCE_GPU}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${RESOURCE_GPU}}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-20}"

exec bash "${SCRIPT_DIR}/run_gspo.sh" "$@" \
  --max-weight-staleness "${MAX_WEIGHT_STALENESS}" \
  --async-data-buffer-capacity-factor "${ASYNC_DATA_BUFFER_CAPACITY_FACTOR}" \
  --recompute-loss-function --skip-eval-before-train
