# Web interaction judge prompt

**English** | [简体中文](README_zh.md) · [Home](../../README.md)

Use `domain="web"` for web interaction samples. [judge_system_prompt.txt](judge_system_prompt.txt)
defines the domain rubric; [shared user template](../judge_user_prompt.txt) supplies prediction/reference content.
Maintain the six-dimension JSON score contract described in [Prompts](../README.md).
Configure the parent prompt root, not this leaf directory, through `JUDGE_PROMPT_DIR`.
Restart scoring workers after customization; use only synthetic/public examples in published templates.
