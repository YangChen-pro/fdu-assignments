#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?usage: bash hw3/task2/scripts/train.sh <config.yaml> [max_gpus]}
MAX_GPUS=${2:-8}
TASK2_ROOT=${TASK2_ROOT:-/data/yc/CS60003}
ENV_PY=${PYTHON_BIN:-/data/yc/miniconda/envs/llm-26-gpu/bin/python}
cd "$TASK2_ROOT"
export PYTHONPATH=$TASK2_ROOT/hw3/task2/src:${PYTHONPATH:-}
GPU_IDS=$($ENV_PY hw3/task2/scripts/select_gpus.py --min-free-mib 20000 --max-util 5 --max-gpus "$MAX_GPUS")
if [[ -z "$GPU_IDS" ]]; then
  echo "No sufficiently free GPU found; fallback to CPU."
  $ENV_PY -m hw3_task2.train --config "$CONFIG"
  exit 0
fi
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
NPROC=$($ENV_PY - <<'PY'
import os
print(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
PY
)
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
$ENV_PY -m torch.distributed.run --nproc_per_node "$NPROC" -m hw3_task2.train --config "$CONFIG"
