#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:?usage: bash hw3/task2/scripts/evaluate.sh <config.yaml> <checkpoint.pt> <name>}
CHECKPOINT=${2:?usage: bash hw3/task2/scripts/evaluate.sh <config.yaml> <checkpoint.pt> <name>}
NAME=${3:?usage: bash hw3/task2/scripts/evaluate.sh <config.yaml> <checkpoint.pt> <name>}
TASK2_ROOT=${TASK2_ROOT:-/data/yc/CS60003}
ENV_PY=${PYTHON_BIN:-/data/yc/miniconda/envs/llm-26-gpu/bin/python}
cd "$TASK2_ROOT"
export PYTHONPATH=$TASK2_ROOT/hw3/task2/src:${PYTHONPATH:-}
$ENV_PY -m hw3_task2.evaluate --config "$CONFIG" --checkpoint "$CHECKPOINT" --name "$NAME" --output "hw3/task2/outputs/eval/${NAME}_splitD"
