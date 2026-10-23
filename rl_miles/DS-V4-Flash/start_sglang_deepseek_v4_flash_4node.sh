#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODE="${1:-serve}"
if [[ $# -gt 0 ]]; then
  shift
fi

: "${RESOURCE_GPU:?请设置 RESOURCE_GPU=8}"
: "${WORLD_SIZE:?请设置 WORLD_SIZE（节点总数）}"
: "${RANK:?请设置 RANK=0 到 WORLD_SIZE-1（节点序号）}"
export RESOURCE_GPU
export WORLD_SIZE
export RANK
export ENABLE_DISTRIBUTED=0

echo "启动多节点独立 DeepSeek-V4-Flash：RANK=${RANK}/${WORLD_SIZE}，本节点 GPU=${RESOURCE_GPU}，TP=${RESOURCE_GPU}"
exec "${SCRIPT_DIR}/deploy_sglang_deepseek_v4_flash_4node.sh" "${MODE}" "$@"
