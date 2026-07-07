#!/usr/bin/env bash
set -euo pipefail
TASK2_ROOT=${TASK2_ROOT:-/data/yc/CS60003}
ENV_PY=${PYTHON_BIN:-/data/yc/miniconda/envs/llm-26-gpu/bin/python}
SECRET_ENV=${SECRET_ENV:-$TASK2_ROOT/.helloagents/secrets/hw3.env}
cd "$TASK2_ROOT"
export PYTHONPATH=$TASK2_ROOT/hw3/task2/src:${PYTHONPATH:-}
$ENV_PY -m hw3_task2.export_swanlab --secret-env "$SECRET_ENV" --output-dir hw3/task2/results/swanlab
