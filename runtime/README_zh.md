# 框架版本与运行补丁

[English](README.md) | **简体中文** · [项目首页](../README_zh.md)

[项目首页](../README_zh.md) · [环境安装](../ENVIRONMENT_zh.md) · [补丁说明](patches/README_zh.md)

此目录管理框架源码的固定版本、补丁应用和运行前检查。
请保留整个 `runtime/`：训练安装器依赖这里的 manifest 和补丁，而不是任意版本的上游包。

## 固定版本

完整提交与公开源地址见 [revisions.json](revisions.json)。

| 组件 | 提交前缀 | 默认安装位置 |
| --- | --- | --- |
| ms-swift | `225b1a0eb455` | `cpt/ms-swift` |
| mcore-bridge | `1f0b2c4ead2d` | `cpt/mcore-bridge` |
| Miles | `dbbab1566ae4` | `rl_miles/miles` |
| Megatron-LM | `235952df607b` | `rl_miles/Megatron-LM` |
| Megatron-Bridge | `7f0fb3456f8f` | `rl_miles/Megatron-Bridge` |
| SGLang-Miles | `3bbb2812e2ca` | `rl_miles/sglang-miles` |

DS judge 使用另一套 SGLang 提交和补丁，见其[部署指南](../rl_miles/DS-V4-Flash/README_zh.md)。

## 准备与检查源码

通常由阶段安装器调用，无需逐个手工安装。需要单独准备或检查时，从项目根目录执行：

```bash
python3 -S runtime/prepare_sources.py --component swift
python3 -S runtime/prepare_sources.py --component swift --check
bash rl_miles/bootstrap_miles.sh --check
```

| 参数 | 含义 |
| --- | --- |
| `--component` | 选择 manifest 中的组件，可重复传入 |
| `--destination` | 单组件自定义 checkout 位置；使用同一位置进行后续检查 |
| `--check` | 只检查现有源码提交和补丁；不克隆、不安装、不修改 |

工具不会 reset 已有源码或覆盖本地修改。遇到提交不符、补丁冲突时，请准备独立目录，
不要强制覆盖自己的开发 checkout。

## 功能检查

```bash
cpt/ms-swift-venv/bin/python runtime/validate_swift.py --train-user-loss
```

该检查验证当前导入的 Swift 能否保留掩码并训练显式启用的 user 内容。
SFT 启动时还会调用 `sft_code/validate_sft_runtime.py` 检查 GDN CP 能力和模型结构；
RL 启动时调用 `rl_miles/validate_runtime.py` 检查框架/自定义 hook、提示词与服务配置。

这些是最低运行前检查，不会替代完整 GPU、网络或模型效果验证。
更换源码或掩码策略后，重新执行对应检查并重建受影响的数据缓存。
