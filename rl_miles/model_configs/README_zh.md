# 模型结构配置

[English](README.md) | **简体中文** · [项目首页](../../README_zh.md)

这些 shell 文件定义 Qwen3.5 4B、9B、27B、35B-A3B、122B-A10B 的 `MODEL_ARGS`。
它们描述层数、头数、维度、专家等**模型结构，不是 GPU 分配**。
actor/rollout 拓扑和 batch 在 [训练入口](../README_zh.md) 中设置。

`MODEL_CONFIG_PATH` 可选择其他结构配置，但必须匹配 HF config 与权重。
不要为了通过并行整除检查而随意改模型层数/头数/专家数；应调整
`TENSOR_PARALLEL_SIZE/PIPELINE_PARALLEL_SIZE/CONTEXT_PARALLEL_SIZE` 和专家并行参数。
MoE 默认 EP8/ETP1，Dense 使用 EP1/ETP1。目录不包含权重。
