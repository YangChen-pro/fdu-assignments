#!/usr/bin/env bash
set -euo pipefail
TASK2_ROOT=${TASK2_ROOT:-/data/yc/CS60003}
ENV_PY=${PYTHON_BIN:-/data/yc/miniconda/envs/llm-26-gpu/bin/python}
cd "$TASK2_ROOT"
export PYTHONPATH=$TASK2_ROOT/hw3/task2/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
$ENV_PY -m hw3_task2.train --config hw3/task2/configs/dry_run.yaml --dry-run
