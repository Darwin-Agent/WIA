# 训练框架补丁

[English](README.md) | **简体中文** · [项目首页](../../README_zh.md)

[运行时指南](../README_zh.md) · [环境安装](../../ENVIRONMENT_zh.md)

| 文件 | 功能 | 适用组件 |
| --- | --- | --- |
| `swift-user-loss.patch` | 允许显式启用的后续 user 正文参与 loss | ms-swift |
| `swift-collator-padding.patch` | collator padding 兼容修正 | ms-swift |
| `swift-loss-mask-alias.patch` | `loss_mask` 转换、首 user/system 掩码校验 | ms-swift |
| `miles-runtime.patch` | CoT 参数、训练/导出/异步运行兼容 | Miles |
| `sglang-miles-runtime.patch` | Miles rollout 使用的 SGLang 兼容 | SGLang-Miles |

通过 `runtime/prepare_sources.py` 与固定 manifest 一起使用，不要直接应用到任意上游版本。
补丁顺序由 manifest 管理；已应用的补丁会检查而不是重复叠加。
修改补丁后需重新验证适用性、导入和实际训练路径。DS 服务补丁不在此目录。
