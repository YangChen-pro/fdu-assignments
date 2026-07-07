#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR=${1:?usage: bash hw3/task2/scripts/upload_modelscope.sh <model_dir> <repo_id> <output.json> [path_prefix]}
REPO_ID=${2:?usage: bash hw3/task2/scripts/upload_modelscope.sh <model_dir> <repo_id> <output.json> [path_prefix]}
OUTPUT=${3:?usage: bash hw3/task2/scripts/upload_modelscope.sh <model_dir> <repo_id> <output.json> [path_prefix]}
PATH_PREFIX=${4:-}
TASK2_ROOT=${TASK2_ROOT:-/data/yc/CS60003}
ENV_PY=${PYTHON_BIN:-/data/yc/miniconda/envs/llm-26-gpu/bin/python}
SECRET_ENV=${SECRET_ENV:-$TASK2_ROOT/.helloagents/secrets/hw3.env}
cd "$TASK2_ROOT"
export PYTHONPATH=$TASK2_ROOT/hw3/task2/src:${PYTHONPATH:-}
$ENV_PY -m hw3_task2.upload_modelscope --model-dir "$MODEL_DIR" --repo-id "$REPO_ID" --output "$OUTPUT" --path-prefix "$PATH_PREFIX" --secret-env "$SECRET_ENV"
