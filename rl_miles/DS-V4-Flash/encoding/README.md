# DeepSeek encoding reference

**English** | [简体中文](README_zh.md) · [Home](../../../README.md)

`encoding_dsv4.py` provides standalone message encoding and completion parsing. It supports multi-turn text,
reasoning and tool-call formatting; it is not a separate server or reward implementation.

From this directory:

```python
from encoding_dsv4 import encode_messages, parse_message_from_completion_text

messages = [{"role": "user", "content": "What is 2+2?"}]
prompt = encode_messages(messages, thinking_mode="thinking")
completion = "Simple arithmetic.</think>4.<｜end▁of▁sentence｜>"
parsed = parse_message_from_completion_text(completion, thinking_mode="thinking")
```

Use `thinking_mode="chat"` for non-thinking encoding. Match the mode when parsing completions.
The parser expects well-formed output and does not repair malformed tool markup or unclosed blocks;
handle errors at the caller. The live [judge service](../README.md) uses its chat template and parser settings,
so changing this reference module alone does not reconfigure the deployed server.
