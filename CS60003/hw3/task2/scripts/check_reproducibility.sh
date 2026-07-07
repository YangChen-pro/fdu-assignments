#!/usr/bin/env bash
set -euo pipefail
TASK2_ROOT=${TASK2_ROOT:-/data/yc/CS60003}
ENV_PY=${PYTHON_BIN:-/data/yc/miniconda/envs/llm-26-gpu/bin/python}
DATA_ROOT=${DATA_ROOT:-/data/yc/CS60003/hw3/task2/data/calvin_lerobot}
cd "$TASK2_ROOT"
PYTHONPATH="$TASK2_ROOT/hw3/task2/src" "$ENV_PY" -m hw3_task2.check_reproducibility \
  --task2-root "hw3/task2" \
  --results-dir "hw3/task2/results" \
  --data-root "$DATA_ROOT" \
  "$@"
