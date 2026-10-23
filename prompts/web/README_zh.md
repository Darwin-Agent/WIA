# 网页交互评分提示词

[English](README.md) | **简体中文** · [项目首页](../../README_zh.md)

网页交互样本使用 `domain="web"`。[judge_system_prompt.txt](judge_system_prompt.txt)
定义领域评分规则，[公共 user 模板](../judge_user_prompt.txt) 提供预测和参考内容。
修改时保留 [Prompts](../README_zh.md) 说明的六维 JSON 契约。
`JUDGE_PROMPT_DIR` 应指向父级提示词根目录，不是当前叶子目录。
自定义后重启评分 worker；公开模板只使用合成或公开样例。
