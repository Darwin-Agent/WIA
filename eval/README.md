# Evaluation: generate, judge and report

**English** | [简体中文](README_zh.md) · [Home](../README.md) · [Installation](../ENVIRONMENT.md) · [Configuration](../docs/CONFIGURATION.md)

Predict environment observations from test inputs and compare them against ground truth using domain-aware judges.
Supports local vLLM, user-configured OpenAI-compatible servers, resume, dual judges and checkpoint reports.

## Quick start

```bash
bash eval/install_runtime.sh
export JUDGE_BASE_URL=http://judge.example.invalid:30000/v1
export JUDGE_MODEL=DeepSeek-V4-Flash-0731
bash eval/run.sh --model Qwen3.5-9B --checkpoint /path/to/sft-9b-hf \
  --eval-data /path/to/test.jsonl --think-mode off --run-name 9b-no-cot
```

Replace paths/hosts. Use the [raw RL JSONL schema](../rl_miles/README.md#data-and-checkpoints):
`system/prompt/ground_truth`, optionally `domain/dataset/source`.
`--skip-judge` generates without scoring and therefore does not produce complete judge metrics.

For an existing inference service, install with `API_ONLY=1 bash eval/install_runtime.sh` and set
`--rollout-base-url` plus `--rollout-model-name`. Local checkpoint/tokenizer metadata is still required.
This is a compatible-protocol entry, not a private proxy or hosted-provider-specific adapter.

## CoT controls

| Option | Behavior |
| --- | --- |
| `--think-mode off` | Disable policy thinking |
| `--think-mode on` | Enable policy thinking |
| `--think-mode auto` | AgentWorld and 122B-A10B on; other Qwen sizes off |
| `THINK_MODE` | Environment default; explicit CLI wins; aliases 1/0 and true/false are accepted |

Main evaluator defaults to auto; streaming defaults to off, so explicitly enable on for 122B streaming runs.
The judge remains non-thinking and scores the final observation. Independent evaluation does not apply RL's long-thinking penalty.
Use different `--run-name` values for CoT comparisons; never mix modes in one resumed run.

## Model resources and outputs

All five Qwen3.5 sizes are registered. Default local TP is 8, overridable with `--num-gpus` after checking memory/divisibility.
This is not the RL actor topology or a minimum-memory guarantee. Use a complete HF export.
`convert_moe_for_vllm.py` validates config and creates a linked directory; it **does not convert weights or patch vLLM**.

Outputs under `eval/outputs/<run-name>` by default:

- `predictions.jsonl`: generated responses.
- `judged.jsonl`: scores and errors.
- `metrics.json` and split metrics: sample/domain means, six dimensions, failures/truncation and token counts.
- Per-sample shards for resumable work.

**Without `--resume`, the same run directory is cleared.** Use unique names or resume deliberately.
Judge dimensions match the RL judge, but the main evaluator does not call similarity or compute the mixed RL objective.

## Overrides

Explicit CLI takes precedence over supported environment defaults. Inspect `python eval/run_eval.py --help`
or the [full parameter index](../docs/CONFIGURATION.md).

| Category | Main options |
| --- | --- |
| Model | `--model`, `--checkpoint`, `--template-model`; template follows checkpoint unless overridden |
| Data/output | `--eval-data`, `--eval-limit`, `--output-root`, `--run-name`, `--resume` |
| Local engine | `--num-gpus`, `--base-port` (38000), `--gpu-memory-utilization` (0.80), `--vllm-bin` |
| Precision | `--dtype`, `--kv-cache-dtype`, `--quantization`; must match model and hardware |
| Length | `--max-model-len` (Qwen default 81920), `--max-prompt-tokens` (65536), `--max-response-tokens` (16384), `--prompt-safety-margin` |
| Sampling | `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty`; AgentWorld has specific defaults |
| Concurrency | `--request-concurrency` (128), `--server-timeout` (3600 seconds) |
| Existing service | `--rollout-base-url` / `ROLLOUT_BASE_URL`, `--rollout-model-name` / `ROLLOUT_MODEL_NAME` |
| Judge | `--judge-base-url`, `--judge-model`, `--judge-max-tokens` (4096), `--judge-concurrency` (64) |
| Judge auth | Prefer runtime `JUDGE_API_KEY`; never place actual secrets in examples or committed files |
| Flow | `--skip-judge`, `--inline-judge`, `--no-inline-judge`, `--dry-run`; dry-run still reads model metadata/data |
| Environment | `PYTHON_BIN`, `VLLM_BIN`, `JUDGE_PROMPT_DIR`; installer `VENV_DIR/PYTHON_BASE/VLLM_VERSION` |

## Additional entry points

| File | Purpose |
| --- | --- |
| `run_streaming_eval_worker.py` | Generate against an existing server, then run two configured judges in parallel; resumable |
| `judge_plus.py` | Rejudge existing predictions without regenerating |
| `summarize_rollout_eval.py` | Summarize RL `eval_test_*.jsonl` by checkpoint, ID/OOD and domain into CSV/Markdown |
| `convert_moe_for_vllm.py` | `--source/--dest` linked directory for a complete MoE HF checkpoint |

Streaming needs `--judge-base-url/--judge-model` and `--judge-plus-base-url/--judge-plus-model`.
Inject `JUDGE_API_KEY/JUDGE_PLUS_API_KEY` for authentication. `judge_plus.py` uses `--api-base-url`
or `JUDGE_PLUS_API_BASE_URL`, with optional `--dirs`. No credential-list index or private service is bundled.

```bash
python eval/summarize_rollout_eval.py /path/to/rl-rollouts --split both
```

The summarizer reads existing reward/similarity fields; it does not call models again and is not a generic predictions-file parser.
