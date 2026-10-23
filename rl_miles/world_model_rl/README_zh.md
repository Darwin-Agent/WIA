# 数据加载与奖励扩展

[English](README.md) | **简体中文** · [项目首页](../../README_zh.md)

## 运行调用链

| 模块 / callable | 职责 |
| --- | --- |
| `data_source.WorldModelRLJsonlDataSource` | 原始/预处理 JSONL、延迟 tokenizer、epoch shuffle、分组采样与状态 |
| `dataloader` | 格式解析、prompt 截断、Observation 标记和 CoT 渲染 |
| `fully_async_rollout.JsonlFullyAsyncRolloutFn` | 异步训练/评测适配和样本输出 |
| `rollout.raw_train_rollout` / `eval_rollout.raw_eval_rollout` | 非默认兼容 hook |
| `reward.custom_rm` | judge 解析、similarity、重试/降级和 think 超长惩罚 |
| `jsonl_writer` | 样本持久化 |
| `reward_random` | 调试替代，不是默认训练奖励 |

导入时 `PYTHONPATH` 需包含 `rl_miles`；训练入口会向 Ray worker 传递。
只在替代实现遵循 Miles 接口时覆盖 `MILES_DATA_SOURCE_PATH`、`ROLLOUT_FUNCTION_PATH`、
`EVAL_FUNCTION_PATH`、`CUSTOM_RM_PATH`。`CUSTOM_RM_PATH=` 可以显式清空 hook，但不会自动配置另一个奖励。
数据格式、CoT 与奖励参数见 [RL 指南](../README_zh.md)。不要公开原始轨迹、凭据或私有服务清单。
