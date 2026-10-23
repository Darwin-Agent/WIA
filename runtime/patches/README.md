# Training framework patches

**English** | [简体中文](README_zh.md) · [Home](../../README.md) · [Runtime](../README.md)

| Patch | Purpose | Component |
| --- | --- | --- |
| `swift-user-loss.patch` | Explicitly enable later-user content loss | ms-swift |
| `swift-collator-padding.patch` | Collator padding compatibility | ms-swift |
| `swift-loss-mask-alias.patch` | Normalize masks and validate first-user/system exclusions | ms-swift |
| `miles-runtime.patch` | CoT arguments, training/export and async compatibility | Miles |
| `sglang-miles-runtime.patch` | Miles rollout compatibility | SGLang-Miles |

Apply through `runtime/prepare_sources.py` with the pinned manifest, not against arbitrary upstream versions.
The manifest controls order and already-applied checks. Patch changes require applicability, import and execution validation.
DS service patches live in the deployment directory instead.
