#!/usr/bin/env bash
set -euo pipefail
cd "/data/yc/NLP-final-project"
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
CONDA=/data/yc/miniconda/bin/conda
PYTHON_CMD=("$CONDA" run -n minimind python)

"${PYTHON_CMD[@]}" -m task_a_robustness.roberta train \
  --model-name roberta-base \
  --cache-dir checkpoints/hf_cache \
  --output-dir checkpoints/task_a_roberta \
  --epochs 3 \
  --batch-size 16

"${PYTHON_CMD[@]}" -m task_a_robustness.roberta predict \
  --model-dir checkpoints/task_a_roberta \
  --cache-dir checkpoints/hf_cache \
  --split test \
  --output results/task_a/roberta/predictions.csv

"${PYTHON_CMD[@]}" -m task_a_robustness.cli evaluate \
  --split test \
  --predictions results/task_a/roberta/predictions.csv \
  --output results/task_a/roberta/metrics.csv
