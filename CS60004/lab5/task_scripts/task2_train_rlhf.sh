#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source "/root/miniconda3/etc/profile.d/conda.sh"
conda activate lab3
: "${SWANLAB_API_KEY:?请先设置 SWANLAB_API_KEY 环境变量}"

SITE_PACKAGES=$(python - <<'PY_SITE'
import site
print(site.getsitepackages()[0])
PY_SITE
)
for lib_dir in "$SITE_PACKAGES/nvidia/cu13/lib" "$SITE_PACKAGES/torch/lib"; do
  if [[ -d "$lib_dir" ]]; then
    export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}"
  fi
done

python - <<'PY_PATCH_SWIFT_GRPO'
from pathlib import Path
import site

trainer_path = Path(site.getsitepackages()[0]) / "swift/megatron/trainers/grpo_trainer.py"
text = trainer_path.read_text(encoding="utf-8")
if "self._metrics[mode].clear()" not in text:
    old = """        if self._metrics[mode]:
            addition_metrics = {
                key: torch.tensor(sum(val) / len(val), device=loss.device)
                for key, val in self._metrics[mode].items()
            }
            avg_metric.update(addition_metrics)

        avg_metric = self._all_reduce_metric(avg_metric)
"""
    new = """        if self._metrics[mode]:
            addition_metrics = {
                key: torch.tensor(sum(val) / len(val), device=loss.device)
                for key, val in self._metrics[mode].items()
            }
            avg_metric.update(addition_metrics)
            # These GRPO metrics are already aggregated by the outer Megatron
            # logging window. Keep only metrics collected since the previous
            # loss call, otherwise SwanLab/logging.jsonl reports historical
            # cumulative averages and the curves become increasingly smoothed.
            self._metrics[mode].clear()

        avg_metric = self._all_reduce_metric(avg_metric)
"""
    if old not in text:
        raise RuntimeError(f"Swift GRPO metric patch target not found: {trainer_path}")
    trainer_path.write_text(text.replace(old, new), encoding="utf-8")
PY_PATCH_SWIFT_GRPO

BASE_MODEL=${BASE_MODEL:-/root/autodl-tmp/CS60004/models/Qwen2.5-1.5B-Instruct}
SFT_OUTPUT_DIR=${SFT_OUTPUT_DIR:-lab5/outputs/task2/megatron_full_sft_all}
DATASET=${DATASET:-lab5/outputs/task2/data/grpo_train.jsonl}
VAL_DATASET=${VAL_DATASET:-lab5/outputs/task2/data/grpo_val.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-lab5/outputs/task2/megatron_grpo_all}
TEMPLATE=${TEMPLATE:-qwen2_5}
TUNER_TYPE=${TUNER_TYPE:-full}
RLHF_TYPE=${RLHF_TYPE:-grpo}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
EPOCHS=${EPOCHS:-1}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-6}
MAX_LENGTH=${MAX_LENGTH:-6144}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-512}
SAVE_STEPS=${SAVE_STEPS:-1000}
EVAL_STEPS=${EVAL_STEPS:-100}
EVAL_ITERS=${EVAL_ITERS:-0}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-8}
NUM_GENERATIONS=${NUM_GENERATIONS:-4}
NUM_GENERATIONS_EVAL=${NUM_GENERATIONS_EVAL:-2}
BETA=${BETA:-0.04}
DATASET_NUM_PROC=${DATASET_NUM_PROC:-4}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
REPORT_TO=${REPORT_TO:-swanlab}
REWARD_FUNCS=${REWARD_FUNCS:-llm_judge}
REWARD_WEIGHTS=${REWARD_WEIGHTS:-1.0}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.20}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-8}
GRPO_JUDGE_BASE_URL=${GRPO_JUDGE_BASE_URL:-http://127.0.0.1:8318/v1}
GRPO_JUDGE_MODEL=${GRPO_JUDGE_MODEL:-Qwen3.5-35B-A3B}
GRPO_JUDGE_CONCURRENCY=${GRPO_JUDGE_CONCURRENCY:-16}
GRPO_JUDGE_MAX_RETRIES=${GRPO_JUDGE_MAX_RETRIES:-2}
GRPO_JUDGE_MAX_TOKENS=${GRPO_JUDGE_MAX_TOKENS:-1024}

find_latest_sft_model() {
  if [[ ! -d "$SFT_OUTPUT_DIR" ]]; then
    return 0
  fi
  {
    find "$SFT_OUTPUT_DIR" -maxdepth 3 -type d -name 'checkpoint-*-merged'
    find "$SFT_OUTPUT_DIR" -maxdepth 3 -type d -name 'checkpoint-*' | grep -E '/checkpoint-[0-9]+$' || true
  } 2>/dev/null | sort -V | tail -n 1 || true
}

if [[ -z "${MODEL:-}" ]]; then
  MODEL=$(find_latest_sft_model)
  if [[ -z "$MODEL" ]]; then
    MODEL="$BASE_MODEL"
  fi
fi
REF_MODEL=${REF_MODEL:-$MODEL}

read -r -a REWARD_FUNC_ARGS <<< "$REWARD_FUNCS"
read -r -a REWARD_WEIGHT_ARGS <<< "$REWARD_WEIGHTS"

EXTRA_ARGS=()
if [[ -f "$VAL_DATASET" ]]; then
  EXTRA_ARGS+=(--val_dataset "$VAL_DATASET" --eval_steps "$EVAL_STEPS")
fi
if [[ -n "${TRAIN_ITERS:-}" ]]; then
  EXTRA_ARGS+=(--train_iters "$TRAIN_ITERS")
else
  EXTRA_ARGS+=(--num_train_epochs "$EPOCHS")
fi

GRPO_JUDGE_BASE_URL="$GRPO_JUDGE_BASE_URL" \
GRPO_JUDGE_MODEL="$GRPO_JUDGE_MODEL" \
GRPO_JUDGE_CONCURRENCY="$GRPO_JUDGE_CONCURRENCY" \
GRPO_JUDGE_MAX_RETRIES="$GRPO_JUDGE_MAX_RETRIES" \
GRPO_JUDGE_MAX_TOKENS="$GRPO_JUDGE_MAX_TOKENS" \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
megatron rlhf \
  --model "$MODEL" \
  --ref_model "$REF_MODEL" \
  --dataset "$DATASET" \
  --template "$TEMPLATE" \
  --tuner_type "$TUNER_TYPE" \
  --rlhf_type "$RLHF_TYPE" \
  --external_plugins lab5/task_scripts/task2_grpo_reward.py \
  --reward_funcs "${REWARD_FUNC_ARGS[@]}" \
  --reward_weights "${REWARD_WEIGHT_ARGS[@]}" \
  --tensor_model_parallel_size 1 \
  --sequence_parallel true \
  --micro_batch_size "$MICRO_BATCH_SIZE" \
  --global_batch_size "$GLOBAL_BATCH_SIZE" \
  --generation_batch_size "$GENERATION_BATCH_SIZE" \
  --num_generations "$NUM_GENERATIONS" \
  --num_generations_eval "$NUM_GENERATIONS_EVAL" \
  --max_completion_length "$MAX_COMPLETION_LENGTH" \
  --beta "$BETA" \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" \
  --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" \
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
  --log_completions true \
  --save_steps "$SAVE_STEPS" \
  --save_safetensors true \
  --max_length "$MAX_LENGTH" \
  --truncation_strategy left \
  --eval_iters "$EVAL_ITERS" \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --dataset_num_proc "$DATASET_NUM_PROC" \
  --no_save_optim true \
  --no_save_rng true \
  "${EXTRA_ARGS[@]}"
