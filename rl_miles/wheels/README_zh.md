# Wheel 与 ABI 兼容

[English](README.md) | **简体中文** · [项目首页](../../README_zh.md)

此目录为 [RL 安装器](../../ENVIRONMENT_zh.md) 提供源码辅助：

- `download_official_cu130_wheels.py` 根据 `MILES_WHEELS_REPO`、`MILES_WHEELS_TAG`
  （默认 `cu130-torch213-x86_64`）、`MILES_WHEELS_DEST` 和可选前缀列表 `MILES_WHEELS_ONLY` 下载公共产物。
- `torch_cow_compat.cpp` 构建供 `official_env.sh` 使用的可选 ABI 兼容层；安装器默认编译。

选择 Python3.12/x86_64/Torch2.13/CUDA13 ABI 匹配的文件。文件大小检查用于发现未完成下载，
不是供应链真实性验证；安装前核验来源。`DOWNLOAD_WHEELS=0` 只跳过下载，不跳过依赖安装，
需事先备齐必要 wheel。源码发行目录不要包含本机 `.so`、wheel 压缩包或未完成下载。
