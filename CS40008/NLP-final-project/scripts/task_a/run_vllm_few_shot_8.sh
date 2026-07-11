#!/usr/bin/env bash
set -uo pipefail

cd "/data/yc/NLP-final-project"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/data/yc/NLP-final-project/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/data/yc/NLP-final-project/.cache/huggingface/hub"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"

models=(qwen3_0_6b qwen3_1_7b qwen3_5_0_8b qwen3_5_2b)
# Default to faster non-thinking runs. Override with MODE_LIST="non_thinking thinking" when needed.
read -r -a modes <<< "${MODE_LIST:-non_thinking}"

declare -A display_names=(
  [qwen3_0_6b]="Qwen3-0.6B"
  [qwen3_1_7b]="Qwen3-1.7B"
  [qwen3_5_0_8b]="Qwen3.5-0.8B"
  [qwen3_5_2b]="Qwen3.5-2B"
)

for model in "${models[@]}"; do
  for mode in "${modes[@]}"; do
    name="${display_names[$model]}"
    metrics="results/task_a/vllm/metrics/${name}_${mode}_test_few_shot_8_metrics.csv"
    raw="results/task_a/vllm/raw_outputs/${name}_${mode}_test_few_shot_8_raw_outputs.jsonl"
    batch_size=96
    if [[ "${mode}" == "thinking" ]]; then
      batch_size=24
    fi
    if [[ -f "${metrics}" ]] && [[ -f "${raw}" ]] && [[ "$(wc -l < "${raw}")" -eq 3000 ]]; then
      echo "===== $(date '+%F %T') SKIP few-shot model=${model} mode=${mode} existing=${metrics} ====="
      continue
    fi
    echo "===== $(date '+%F %T') START few-shot model=${model} mode=${mode} batch_size=${batch_size} ====="
    /data/yc/miniconda/bin/conda run -n minimind python -m task_a_robustness.vllm_infer \
      --model "${model}" \
      --mode "${mode}" \
      --split test \
      --templates all \
      --few-shot 8 \
      --few-shot-seed 0 \
      --batch-size "${batch_size}" \
      --max-model-len 18000 \
      --max-num-seqs 32 \
      --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}" \
      --language-model-only \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
      --resume
    status=$?
    if [[ "${status}" -ne 0 ]]; then
      echo "===== $(date '+%F %T') FAIL few-shot model=${model} mode=${mode} status=${status} ====="
      exit "${status}"
    fi
    echo "===== $(date '+%F %T') DONE few-shot model=${model} mode=${mode} ====="
  done
done

echo "===== $(date '+%F %T') ALL DONE few-shot ====="
