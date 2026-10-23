#!/usr/bin/env bash
set -euo pipefail
# Megatron-SWIFT, CP4; default 8 GPUs, 2 epochs. See train_sft.sh --help.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SFT_MODEL_SIZE=4b
exec bash "${SCRIPT_DIR}/train_sft.sh" "$@"
