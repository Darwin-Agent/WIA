# DeepSeek 编码参考

[English](README.md) | **简体中文** · [项目首页](../../../README_zh.md)

`encoding_dsv4.py` 提供独立的消息编码与 completion 解析，支持多轮文本、reasoning 和工具调用格式；
它不是独立服务，也不实现 reward。

在本目录运行：

```python
from encoding_dsv4 import encode_messages, parse_message_from_completion_text

messages = [{"role": "user", "content": "What is 2+2?"}]
prompt = encode_messages(messages, thinking_mode="thinking")
completion = "Simple arithmetic.</think>4.<｜end▁of▁sentence｜>"
parsed = parse_message_from_completion_text(completion, thinking_mode="thinking")
```

非思考编码使用 `thinking_mode="chat"`，解析时保持同一模式。
解析器预期格式正确的输出，不自动修复非法工具标记或未闭合块；调用方应处理错误。
在线 [judge 服务](../README_zh.md) 使用自己的模板/parser 参数，只修改此参考模块不会改变已部署服务配置。
