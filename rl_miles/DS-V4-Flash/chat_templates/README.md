# DeepSeek chat template

**English** | [简体中文](README_zh.md) · [Home](../../../README.md)

`deepseek_v4_flash.jinja` is selected by the [judge launcher](../README.md) through `CHAT_TEMPLATE`.
It defines message and reasoning formatting, not reward weights. The judge profile defaults to non-thinking.
Custom templates must match the tokenizer/model special tokens and tool-call parser; validate a synthetic request
before restarting production scorers. Do not insert real conversations or credentials into the template.
