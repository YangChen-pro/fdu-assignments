#!/usr/bin/env bash
set -uo pipefail

cd "/data/yc/NLP-final-project"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/data/yc/NLP-final-project/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/data/yc/NLP-final-project/.cache/huggingface/hub"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"

# Defaults target faster non-thinking self-consistency runs. Override before running, e.g.:
# MODEL_LIST="qwen3_0_6b qwen3_1_7b qwen3_5_0_8b qwen3_5_2b" MODE_LIST="non_thinking" REPEAT=5 bash scripts/task_a/run_vllm_self_consistency.sh
read -r -a models <<< "${MODEL_LIST:-qwen3_1_7b qwen3_5_2b}"
read -r -a modes <<< "${MODE_LIST:-non_thinking}"
repeat="${REPEAT:-5}"

declare -A display_names=(
  [qwen3_0_6b]="Qwen3-0.6B"
  [qwen3_1_7b]="Qwen3-1.7B"
  [qwen3_5_0_8b]="Qwen3.5-0.8B"
  [qwen3_5_2b]="Qwen3.5-2B"
)

for model in "${models[@]}"; do
  for mode in "${modes[@]}"; do
    name="${display_names[$model]}"
    suffix="test_zero_shot_self_consistency_k${repeat}"
    metrics="results/task_a/vllm/metrics/${name}_${mode}_${suffix}_metrics.csv"
    raw="results/task_a/vllm/raw_outputs/${name}_${mode}_${suffix}_raw_outputs.jsonl"
    expected_rows=$((1000 * 3 * repeat))
    batch_size=64
    if [[ "${mode}" == "thinking" ]]; then
      batch_size=16
    fi
    if [[ -f "${metrics}" ]] && [[ -f "${raw}" ]] && [[ "$(wc -l < "${raw}")" -eq "${expected_rows}" ]]; then
      echo "===== $(date '+%F %T') SKIP self-consistency model=${model} mode=${mode} existing=${metrics} ====="
      continue
    fi
    echo "===== $(date '+%F %T') START self-consistency model=${model} mode=${mode} repeat=${repeat} batch_size=${batch_size} ====="
    /data/yc/miniconda/bin/conda run -n minimind python -m task_a_robustness.vllm_infer \
      --model "${model}" \
      --mode "${mode}" \
      --split test \
      --templates all \
      --repeat "${repeat}" \
      --batch-size "${batch_size}" \
      --max-model-len 18000 \
      --max-num-seqs 16 \
      --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}" \
      --language-model-only \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
      --resume
    status=$?
    if [[ "${status}" -ne 0 ]]; then
      echo "===== $(date '+%F %T') FAIL self-consistency model=${model} mode=${mode} status=${status} ====="
      exit "${status}"
    fi
    echo "===== $(date '+%F %T') DONE self-consistency model=${model} mode=${mode} ====="
  done
done

echo "===== $(date '+%F %T') ALL DONE self-consistency ====="
