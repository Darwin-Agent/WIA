# DeepSeek 对话模板

[English](README.md) | **简体中文** · [项目首页](../../../README_zh.md)

`deepseek_v4_flash.jinja` 由 [judge 启动器](../README_zh.md) 的 `CHAT_TEMPLATE` 选择。
模板定义消息和 reasoning 格式，不设置奖励权重；judge 默认 non-thinking。
自定义模板需匹配 tokenizer/模型特殊 token 与 tool-call parser；先用合成请求验证再重启评分服务。
不要把真实会话或凭据写入模板。
