# Model architecture configurations

**English** | [简体中文](README_zh.md) · [Home](../../README.md)

These shell files define `MODEL_ARGS` for Qwen3.5 4B, 9B, 27B, 35B-A3B and 122B-A10B.
They describe architecture (layers, heads, dimensions and experts), **not GPU allocation**.
Actor/rollout topology and batches live in the [training wrappers](../README.md).

`MODEL_CONFIG_PATH` can select another configuration; its architecture must match HF config and weights.
Do not change layer/head/expert counts merely to make a topology divisible. Use
`TENSOR_PARALLEL_SIZE/PIPELINE_PARALLEL_SIZE/CONTEXT_PARALLEL_SIZE` and expert parallel controls instead.
MoE defaults use EP8/ETP1. Dense models use EP1/ETP1. No weights are included.
