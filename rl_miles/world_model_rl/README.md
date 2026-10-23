# Data loading and reward hooks

**English** | [简体中文](README_zh.md) · [Home](../../README.md)

## Runtime chain

| Module / callable | Responsibility |
| --- | --- |
| `data_source.WorldModelRLJsonlDataSource` | Raw/prepared JSONL, lazy tokenizer, epoch shuffle, grouped samples and state |
| `dataloader` | Record parsing, prompt fitting, observation markers and CoT rendering |
| `fully_async_rollout.JsonlFullyAsyncRolloutFn` | Async training/evaluation adapter and sample output |
| `rollout.raw_train_rollout` / `eval_rollout.raw_eval_rollout` | Non-default compatibility hooks |
| `reward.custom_rm` | Judge parsing, similarity, retries/fallbacks and thinking-length penalty |
| `jsonl_writer` | Sample persistence |
| `reward_random` | Debug alternative; not the default training reward |

Import with `rl_miles` on `PYTHONPATH`; the training launcher propagates it to Ray workers.
Override `MILES_DATA_SOURCE_PATH`, `ROLLOUT_FUNCTION_PATH`, `EVAL_FUNCTION_PATH` or `CUSTOM_RM_PATH`
only when your replacement follows the Miles interface. `CUSTOM_RM_PATH=` explicitly clears that hook;
it does not automatically configure a replacement reward.
Data schemas, CoT defaults and reward settings are in the [RL guide](../README.md).
Do not log or commit raw trajectories, credentials or private endpoint inventories.
