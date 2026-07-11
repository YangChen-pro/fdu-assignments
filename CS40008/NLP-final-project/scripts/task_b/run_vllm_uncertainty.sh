#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python_bin="${PYTHON:-python3}"
model="${MODEL:-qwen3_1_7b}"
thinking_mode="${THINKING_MODE:-thinking}"
repeat="${REPEAT:-5}"
batch_size="${BATCH_SIZE:-32}"
if [[ "${thinking_mode}" == "thinking" ]]; then
  batch_size="${BATCH_SIZE:-16}"
fi

sample_responses="outputs/task_b/vllm/${model}_${thinking_mode}_direct_test_k${repeat}_responses.jsonl"
result_prefix="results/task_b/vllm/${model}_${thinking_mode}"
sample_predictions="${result_prefix}_consistency_samples.csv"
extra_args=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  extra_args+=(--max-samples "${MAX_SAMPLES}")
fi

"${python_bin}" -m task_b_verification_uncertainty.vllm_infer \
  --model "${model}" \
  --thinking-mode "${thinking_mode}" \
  --task-mode direct \
  --split test \
  --repeat "${repeat}" \
  --batch-size "${batch_size}" \
  --max-tokens "${MAX_TOKENS:-128}" \
  --max-model-len "${MAX_MODEL_LEN:-18000}" \
  --max-num-seqs "${MAX_NUM_SEQS:-16}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}" \
  --enforce-eager \
  --language-model-only \
  --resume \
  "${extra_args[@]}" \
  --output "${sample_responses}"
status=$?
if [[ "${status}" -ne 0 ]]; then exit "${status}"; fi

"${python_bin}" -m task_b_verification_uncertainty.cli parse-responses \
  --input "${sample_responses}" \
  --output "${sample_predictions}"
status=$?
if [[ "${status}" -ne 0 ]]; then exit "${status}"; fi

"${python_bin}" -m task_b_verification_uncertainty.cli uncertainty \
  --split test \
  --samples "${sample_predictions}" \
  --method "task_b_consistency_confidence_k${repeat}" \
  --thresholds 0.6 0.8 1.0 \
  --output "${result_prefix}_uncertainty_predictions.csv" \
  --metrics-output "${result_prefix}_uncertainty_metrics.csv"
