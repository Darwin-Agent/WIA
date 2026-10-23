# Judge prompts

**English** | [简体中文](README_zh.md) · [Home](../README.md)

Shared by RL reward and independent evaluation. `judge_user_prompt.txt` combines the prediction and ground truth;
each domain directory supplies `judge_system_prompt.txt`.

| Domain | Task family |
| --- | --- |
| [android](android/README.md) | Android interaction |
| [mcp](mcp/README.md) | MCP tools |
| [os](os/README.md) | Operating-system interaction |
| [search](search/README.md) | Search |
| [swe](swe/README.md) | Software engineering |
| [terminal](terminal/README.md) | Terminal interaction |
| [web](web/README.md) | Web interaction |

Keep the JSON score contract with `scores` and six dimensions:
`format`, `factuality`, `consistency`, `realism`, `quality`, `efficiency`.
Default weights are 1,2,2,2,2,1 and scores are 1–5; normalization is `(score-1)/4` clamped to [0,1].
Domain means and scorer-mixing weights are separate concepts.

Set `JUDGE_PROMPT_DIR` when launching RL/Eval to select another prompt tree. RL forwards it as
`WM_JUDGE_PROMPT_DIR`; direct reward-module callers use the WM variable.
Preserve filenames and template placeholders, align data `domain` values, and restart workers after changes.
Never add real user conversations or private examples to published prompts.
See [RL](../rl_miles/README.md) and [Evaluation](../eval/README.md).
