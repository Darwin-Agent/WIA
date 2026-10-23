# Framework versions and runtime patches

**English** | [简体中文](README_zh.md) · [Home](../README.md) · [Installation](../ENVIRONMENT.md) · [Patches](patches/README.md)

This directory manages pinned framework sources, patch application and preflight checks.
Keep `runtime/` in the distribution: stage installers depend on this manifest and patches.

## Pinned sources

Full revisions and public origins are in [revisions.json](revisions.json).

| Component | Revision prefix | Default checkout |
| --- | --- | --- |
| ms-swift | `225b1a0eb455` | `cpt/ms-swift` |
| mcore-bridge | `1f0b2c4ead2d` | `cpt/mcore-bridge` |
| Miles | `dbbab1566ae4` | `rl_miles/miles` |
| Megatron-LM | `235952df607b` | `rl_miles/Megatron-LM` |
| Megatron-Bridge | `7f0fb3456f8f` | `rl_miles/Megatron-Bridge` |
| SGLang-Miles | `3bbb2812e2ca` | `rl_miles/sglang-miles` |

The [DS judge](../rl_miles/DS-V4-Flash/README.md) uses a different SGLang revision and patch set.

## Prepare and verify

Stage installers normally invoke the preparation tool. From the project root:

```bash
python3 -S runtime/prepare_sources.py --component swift
python3 -S runtime/prepare_sources.py --component swift --check
bash rl_miles/bootstrap_miles.sh --check
```

`--component` is repeatable. `--destination` overrides one component's checkout; use the same destination when checking.
`--check` is read-only/offline: it does not clone, install or modify sources.
Revision/patch conflicts fail rather than resetting your checkout. Use a separate directory for incompatible local work.

## Behavioral checks

```bash
cpt/ms-swift-venv/bin/python runtime/validate_swift.py --train-user-loss
```

This checks the imported Swift's mask preservation and explicitly enabled user-content loss.
SFT also validates GDN CP helpers/model structure; RL validates framework/custom hooks, templates and service configuration.
These checks do not replace GPU or live-network validation. Recheck after framework changes and regenerate affected caches.
