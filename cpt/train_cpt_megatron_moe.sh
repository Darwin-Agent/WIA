#!/usr/bin/env bash
set -euo pipefail
# Compatibility name; see the target's --help for its default GPU allocation.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_qwen35_35b_a3b_cpt_128gpu.sh" "$@"
