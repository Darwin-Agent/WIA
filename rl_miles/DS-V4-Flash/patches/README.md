# DeepSeek deployment patches

**English** | [简体中文](README_zh.md) · [Home](../../../README.md)

| Patch | Purpose |
| --- | --- |
| `0001-env-mp-start-method.patch` | Configurable multiprocessing start method |
| `0002-disable-sm100-common-ops.patch` | Compatibility adjustment for SM100 common operations |
| `0003-qwen3-asr-config-register-exist-ok.patch` | Compatible Qwen3 ASR config registration |

The [main installer](../README.md) applies these against SGLang `a5c9636`; the four-node installer delegates to it.
Do not apply this set to the Miles rollout checkout. Keep `SKIP_RUNTIME_CHECK=0` and validate actual imports and serving.
