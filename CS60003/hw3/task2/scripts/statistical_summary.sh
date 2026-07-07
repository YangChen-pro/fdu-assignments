#!/usr/bin/env bash
set -euo pipefail
TASK2_ROOT=${TASK2_ROOT:-/data/yc/CS60003}
ENV_PY=${PYTHON_BIN:-/data/yc/miniconda/envs/llm-26-gpu/bin/python}
cd "$TASK2_ROOT"
PYTHONPATH="$TASK2_ROOT/hw3/task2/src" "$ENV_PY" -m hw3_task2.statistical_summary --results-dir "hw3/task2/results"
