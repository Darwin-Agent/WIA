# Judge 提示词

[English](README.md) | **简体中文** · [项目首页](../README_zh.md)

RL reward 与独立 Eval 共用这些提示词。`judge_user_prompt.txt` 组合预测与 ground truth；
各领域目录提供 `judge_system_prompt.txt`。

| 领域 | 任务类型 |
| --- | --- |
| [android](android/README_zh.md) | Android 交互 |
| [mcp](mcp/README_zh.md) | MCP 工具 |
| [os](os/README_zh.md) | 操作系统交互 |
| [search](search/README_zh.md) | 搜索 |
| [swe](swe/README_zh.md) | 软件工程 |
| [terminal](terminal/README_zh.md) | 终端交互 |
| [web](web/README_zh.md) | 网页交互 |

保留 JSON 中的 `scores` 和六个维度：`format`、`factuality`、`consistency`、`realism`、`quality`、`efficiency`。
默认权重为 1、2、2、2、2、1；各维度 1–5 分，归一化为 `(score-1)/4` 并限制到 [0,1]。
领域均分和 judge/similarity 混合权重是不同概念。

启动 RL/Eval 时通过 `JUDGE_PROMPT_DIR` 指定自己的提示词树；RL 映射为 `WM_JUDGE_PROMPT_DIR`，
直接调用 reward 时使用 WM 变量。修改时保留文件名与模板占位符，数据 `domain` 应对应目录名，修改后重启 worker。
不要向公开提示词添加真实用户会话或私有示例。参见 [RL](../rl_miles/README_zh.md) 与 [Eval](../eval/README_zh.md)。
