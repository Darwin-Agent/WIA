# Evaluation：生成、评分与汇总

[English](README.md) | **简体中文** · [项目首页](../README_zh.md)

读取测试样本，预测 Observation，并用领域 judge 与 ground truth 对照评分。
支持本地 vLLM、自建 OpenAI-compatible 推理服务、断点续跑、双 judge 和 checkpoint 报表。

## 快速开始

```bash
bash eval/install_runtime.sh
export JUDGE_BASE_URL=http://judge.example.invalid:30000/v1
export JUDGE_MODEL=DeepSeek-V4-Flash-0731
bash eval/run.sh --model Qwen3.5-9B --checkpoint /path/to/sft-9b-hf \
  --eval-data /path/to/test.jsonl --think-mode off --run-name 9b-no-cot
```

替换示例地址与路径。测试数据使用 [RL 原始 JSONL 格式](../rl_miles/README_zh.md#数据与权重)：
`system/prompt/ground_truth`，可含 `domain/dataset/source`。
没有 judge 时可用 `--skip-judge` 只生成，不会得到完整 judge 指标。

连接已有推理服务时，安装可使用 `API_ONLY=1 bash eval/install_runtime.sh`，并传入
`--rollout-base-url`、`--rollout-model-name`；仍需本地 checkpoint/tokenizer 配置。
这是兼容协议入口，不包含私有模型代理或托管服务专用适配器。

## CoT

| 参数 | 行为 |
| --- | --- |
| `--think-mode off` | 禁用策略 thinking |
| `--think-mode on` | 启用策略 thinking |
| `--think-mode auto` | AgentWorld 和 122B-A10B 开启；其他 Qwen 尺度关闭 |
| `THINK_MODE` | CLI 未指定时的默认值；接受 on/off/auto、1/0、true/false |

主评测默认 auto，流式入口默认 off；122B 流式评测显式传 on。
judge 始终 non-thinking，只评分最终 Observation；独立评测不应用 RL 的 think 超长惩罚。
CoT on/off 使用不同 `--run-name`，不要把不同模式混入同一续跑目录。

## 模型资源与输出

五个 Qwen3.5 模型均在注册表中；默认本地 TP8，可通过 `--num-gpus` 调整。
这不是 RL actor 拓扑，也不是最低显存保证。使用完整 HF 导出，按硬件验证并行与内存。
`convert_moe_for_vllm.py` 仅校验 MoE config 并创建链接目录，**不转换权重或修补 vLLM**。

输出在 `eval/outputs/<run-name>`（可覆盖）：

- `predictions.jsonl`：生成结果。
- `judged.jsonl`：评分与错误信息。
- `metrics.json`、分组 metrics：样本均分、领域均分、六维分数、失败/截断与 token 统计。
- 按样本保存的 shards：供 `--resume` 跳过已完成项。

**同名目录在不带 `--resume` 时会被清理。** 使用唯一 run name 或显式续跑，避免覆盖结果。
六维权重与 RL judge 一致；主评测不调用 similarity，不应把它的指标直接视为混合 RL reward。

## 参数覆盖

显式 CLI 优先于对应环境默认值。全部 CLI 见 `python eval/run_eval.py --help` 和
[配置参考](../docs/CONFIGURATION_zh.md)。

| 类别 | 主要参数 |
| --- | --- |
| 模型 | `--model`、`--checkpoint`、`--template-model`；模板默认跟随 checkpoint |
| 数据/输出 | `--eval-data`、`--eval-limit`、`--output-root`、`--run-name`、`--resume` |
| 本地引擎 | `--num-gpus`、`--base-port`（38000）、`--gpu-memory-utilization`（0.80）、`--vllm-bin` |
| 精度 | `--dtype`、`--kv-cache-dtype`、`--quantization`；必须与 checkpoint/硬件匹配 |
| 长度 | `--max-model-len`（Qwen 默认 81920）、`--max-prompt-tokens`（65536）、`--max-response-tokens`（16384）、`--prompt-safety-margin` |
| 采样 | `--temperature`、`--top-p`、`--top-k`、`--repetition-penalty`；AgentWorld 有自己的默认采样配置 |
| 并发 | `--request-concurrency`（128）、`--server-timeout`（3600 秒） |
| 已有服务 | `--rollout-base-url` / `ROLLOUT_BASE_URL`、`--rollout-model-name` / `ROLLOUT_MODEL_NAME` |
| judge | `--judge-base-url`、`--judge-model`、`--judge-max-tokens`（4096）、`--judge-concurrency`（64） |
| judge 认证 | 优先由 `JUDGE_API_KEY` 注入；不将真实值写入命令示例/仓库 |
| 流程 | `--skip-judge`、`--inline-judge`、`--no-inline-judge`、`--dry-run`；dry-run 仍读取模型配置和数据 |
| 环境 | `PYTHON_BIN`、`VLLM_BIN`、`JUDGE_PROMPT_DIR`；安装时 `VENV_DIR/PYTHON_BASE/VLLM_VERSION` |

## 其他入口

| 文件 | 用途 |
| --- | --- |
| `run_streaming_eval_worker.py` | 连接已有推理服务，每条生成后并行调用两个用户配置的 judge；支持续跑 |
| `judge_plus.py` | 对现有 `predictions.jsonl` 重新评分，不重新生成 |
| `summarize_rollout_eval.py` | 汇总 RL 导出的 `eval_test_*.jsonl`，按 checkpoint、ID/OOD、领域生成 CSV/Markdown |
| `convert_moe_for_vllm.py` | 使用 `--source/--dest` 为完整 MoE HF 模型准备链接目录 |

双 judge 入口需要 `--judge-base-url/--judge-model` 与
`--judge-plus-base-url/--judge-plus-model`；认证分别从 `JUDGE_API_KEY/JUDGE_PLUS_API_KEY` 注入。
`judge_plus.py` 使用 `--api-base-url` 或 `JUDGE_PLUS_API_BASE_URL`，可用 `--dirs` 指定结果目录。
没有密钥文件索引或预置私有服务。

```bash
python eval/summarize_rollout_eval.py /path/to/rl-rollouts --split both
```

此汇总器读取已有 reward/similarity 字段，不重新调用模型，也不适用于任意格式的预测文件。
