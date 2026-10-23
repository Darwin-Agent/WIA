# DeepSeek 部署补丁

[English](README.md) | **简体中文** · [项目首页](../../../README_zh.md)

| 补丁 | 用途 |
| --- | --- |
| `0001-env-mp-start-method.patch` | 配置多进程启动方式 |
| `0002-disable-sm100-common-ops.patch` | SM100 common operations 兼容调整 |
| `0003-qwen3-asr-config-register-exist-ok.patch` | Qwen3 ASR config 注册兼容 |

[主安装器](../README_zh.md) 在 SGLang `a5c9636` 上应用这些补丁，四节点安装入口复用同一路径。
不要应用到 Miles rollout checkout。保持 `SKIP_RUNTIME_CHECK=0`，并验证真实导入与推理请求。
