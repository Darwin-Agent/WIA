# DeepSeek-V4-Flash Judge 服务

[English](README.md) | **简体中文** · [项目首页](../../README_zh.md)

为 RL 和独立 Eval 提供 OpenAI-compatible chat completions。此服务只负责评分，
与 Miles 的策略 rollout SGLang 分开部署、分开环境、分开 GPU。

## 安装

需要 DeepSeek-V4-Flash 完整权重、Python 3.12 和兼容的 GPU/CUDA/PyTorch 基础栈。
从仓库根目录执行：

```bash
bash rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh install
```

默认环境为本目录 `.venv`，使用 `--system-site-packages`；请在专用容器中安装。
运行时为 SGLang `a5c9636` + [三份补丁](patches/README_zh.md)，
sglang-kernel 0.4.6.post1、FlashInfer 0.6.13、Transformers 5.12.1、xgrammar 0.2.1。
不要与 RL 的 SGLang-Miles/FlashInfer 环境混用。

## 启动与健康检查

```bash
export MODEL_PATH=/path/to/DeepSeek-V4-Flash
export RESOURCE_GPU=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HOST=0.0.0.0 PORT=30000
export SERVED_MODEL_NAME=DeepSeek-V4-Flash-0731
KEEP_ENTRYPOINT_ALIVE=0 bash rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh serve
curl --noproxy '*' http://127.0.0.1:30000/health
curl --noproxy '*' http://127.0.0.1:30000/v1/models
```

`0.0.0.0` 是监听地址，不能作为客户端目标。训练节点填写可达主机名：

```bash
export JUDGE_BASE_URL=http://judge.example.invalid:30000/v1
export JUDGE_MODEL=DeepSeek-V4-Flash-0731
```

替换示例主机名，并从训练节点验证可达性与 `/v1/models` 中的模型名。
认证由自己的安全网关/服务配置提供，真实 `JUDGE_API_KEY` 只通过运行环境注入。
不要直接暴露未认证的端口到公网。

## 参数

| 参数 | 默认值 / 用途 |
| --- | --- |
| `MODEL_PATH` | 必须填写实际权重；默认 `/preset-models` 仅为容器挂载约定 |
| `VENV_DIR` / `PYTHON_BIN` | 专用 venv / 解释器；`USE_VENV=1` |
| `SGLANG_SRC` / `SGLANG_REPO` / `SGLANG_REF` | 源码目录、公开上游与固定提交；使用独立 checkout |
| `RESOURCE_GPU` | 主 shell 启动器据此设置 TP，默认 8；单独改 `TP_SIZE` 不会覆盖该主入口 |
| `CUDA_VISIBLE_DEVICES` | 可见 GPU 列表；必须与资源设置相符 |
| `HOST` / `PORT` / `SERVED_MODEL_NAME` | 默认 `0.0.0.0` / 30000 / `DeepSeek-V4-Flash-0731` |
| `CONTEXT_LENGTH` / `MEM_FRACTION_STATIC` | 81920 / 0.80 |
| `MAX_RUNNING_REQUESTS` | 128 |
| `CHUNKED_PREFILL_SIZE` / `MAX_PREFILL_TOKENS` | 4096 / 8192 |
| `ATTENTION_BACKEND` / `MOE_RUNNER_BACKEND` | `dsv4` / `marlin` |
| `CUDA_GRAPH_BACKEND_DECODE/PREFILL` | `full` / `disabled` |
| `ENABLE_TWO_BATCH_OVERLAP` / `ENABLE_MIXED_CHUNK` / `ENABLE_DYNAMIC_CHUNKING` | 0 / 0 / 1（主 shell 入口） |
| `CHAT_TEMPLATE` / `REASONING_PARSER` / `TOOL_CALL_PARSER` | 本目录模板 / `deepseek-v4` / `deepseekv4` |
| `SKIP_RUNTIME_CHECK` | 默认 0；不要用 1 掩盖缺失补丁或不兼容依赖 |
| `STARTUP_TIMEOUT` / `HEALTH_INTERVAL` | 1800 / 10 秒 |
| `WARMUP_AFTER_READY` / `WARMUP_TIMEOUT` | 1 / 600 秒 |
| `KEEP_ENTRYPOINT_ALIVE` | 默认 1；0 表示服务准备完成后入口退出，服务仍由其 PID 管理 |
| `LOG_FILE` / `PID_FILE` / `STOP_TIMEOUT` | 日志、PID 路径 / 停止等待默认 30 秒 |

完整参数见 [配置参考](../../docs/CONFIGURATION_zh.md)。shell 和 Python runner 的覆盖规则不同：
直接调用 Python runner 才可用 `ENABLE_DISTRIBUTED`、`DIST_INIT_ADDR`、`NNODES/NODE_RANK`
配置真正的跨节点模型切分；这是高级入口，需自行验证网络与模型资源。

## 多副本与运维

`deploy_sglang_deepseek_v4_flash_4node.sh` / `start_sglang_deepseek_v4_flash_4node.sh`
在每个节点运行**独立模型副本**，不是一个模型跨四台机器 TP。
每个节点分别准备环境、权重和 rank；`node0..node3` 文件只是通用入口的兼容别名，
不会替你填写节点地址或 rank。训练时可用逗号分隔多个 judge URL。

```bash
bash rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh status
bash rl_miles/DS-V4-Flash/deploy_sglang_deepseek_v4_flash.sh stop
```

使用自定义 `PID_FILE/LOG_FILE/PORT` 启动后，运维命令也需保持一致。
单 GPU/CPU-offload 入口为可选实验路径，不是默认 8-GPU 部署的等价替代。
运行日志可能含请求或部署信息，不提交到仓库。

## 目录

- [chat_templates](chat_templates/README_zh.md)：服务使用的对话模板。
- [encoding](encoding/README_zh.md)：编码与完成文本解析参考。
- [patches](patches/README_zh.md)：固定 SGLang 部署补丁。
