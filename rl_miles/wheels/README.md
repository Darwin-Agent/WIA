# Wheels and ABI compatibility

**English** | [简体中文](README_zh.md) · [Home](../../README.md)

Source-only support for the [RL installer](../../ENVIRONMENT.md):

- `download_official_cu130_wheels.py` downloads public release artifacts selected by `MILES_WHEELS_REPO`,
  `MILES_WHEELS_TAG` (default `cu130-torch213-x86_64`), `MILES_WHEELS_DEST` and optional prefix list `MILES_WHEELS_ONLY`.
- `torch_cow_compat.cpp` builds the optional ABI shim used by `official_env.sh`; the installer builds it by default.

Use Python3.12/x86_64/Torch2.13/CUDA13-compatible artifacts. File-size checks protect against incomplete downloads,
not against untrusted supply chains; verify your chosen artifact origin before installation.
`DOWNLOAD_WHEELS=0` skips downloading, not dependency installation. Ensure every required wheel is already present.
Never publish local `.so`, wheel archives or partial downloads as part of the source tree.
