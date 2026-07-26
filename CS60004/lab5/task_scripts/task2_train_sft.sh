#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source "/root/miniconda3/etc/profile.d/conda.sh"
conda activate lab3
: "${SWANLAB_API_KEY:?请先设置 SWANLAB_API_KEY 环境变量}"

MODEL=${MODEL:-/root/autodl-tmp/CS60004/models/Qwen2.5-1.5B-Instruct}
DATASET=${DATASET:-lab5/outputs/task2/data/sft_train.jsonl}
VAL_DATASET=${VAL_DATASET:-lab5/outputs/task2/data/sft_val.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-lab5/outputs/task2/megatron_full_sft_all}
TEMPLATE=${TEMPLATE:-qwen2_5}
TUNER_TYPE=${TUNER_TYPE:-full}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
EPOCHS=${EPOCHS:-1}
LR=${LR:-1e-3}
MIN_LR=${MIN_LR:-1e-4}
MAX_LENGTH=${MAX_LENGTH:-6144}
SAVE_STEPS=${SAVE_STEPS:-6000}
EVAL_STEPS=${EVAL_STEPS:-100}
EVAL_ITERS=${EVAL_ITERS:-0}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
DATASET_NUM_PROC=${DATASET_NUM_PROC:-4}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
REPORT_TO=${REPORT_TO:-swanlab}

EXTRA_ARGS=()
if [[ -f "$VAL_DATASET" ]]; then
  EXTRA_ARGS+=(--val_dataset "$VAL_DATASET" --eval_steps "$EVAL_STEPS")
fi
if [[ -n "${TRAIN_ITERS:-}" ]]; then
  EXTRA_ARGS+=(--train_iters "$TRAIN_ITERS")
else
  EXTRA_ARGS+=(--num_train_epochs "$EPOCHS")
fi

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
megatron sft \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --template "$TEMPLATE" \
  --tuner_type "$TUNER_TYPE" \
  --tensor_model_parallel_size 1 \
  --sequence_parallel true \
  --micro_batch_size "$MICRO_BATCH_SIZE" \
  --global_batch_size "$GLOBAL_BATCH_SIZE" \
  --recompute_granularity full \
  --recompute_method uniform \
  --recompute_num_layers 1 \
  --finetune true \
  --cross_entropy_loss_fusion true \
  --lr "$LR" \
  --lr_warmup_fraction 0.05 \
  --min_lr "$MIN_LR" \
  --optimizer_cpu_offload true \
  --output_dir "$OUTPUT_DIR" \
  --report_to "$REPORT_TO" \
  --save_steps "$SAVE_STEPS" \
  --save_safetensors true \
  --max_length "$MAX_LENGTH" \
  --truncation_strategy delete \
  --eval_iters "$EVAL_ITERS" \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --dataset_num_proc "$DATASET_NUM_PROC" \
  --no_save_optim true \
  --no_save_rng true \
  "${EXTRA_ARGS[@]}"
