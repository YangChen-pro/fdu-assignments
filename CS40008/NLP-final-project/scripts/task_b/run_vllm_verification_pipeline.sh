#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python_bin="${PYTHON:-python3}"
model="${MODEL:-qwen3_1_7b}"
thinking_mode="${THINKING_MODE:-thinking}"
candidate_repeat="${CANDIDATE_REPEAT:-3}"
prompt_tag="${PROMPT_TAG:-aligned}"
batch_size="${BATCH_SIZE:-32}"
sample_suffix="${SAMPLE_SUFFIX:-}"
if [[ -n "${SAMPLE_START:-}" || -n "${SAMPLE_END:-}" ]]; then
  sample_start="${SAMPLE_START:-0}"
  sample_end="${SAMPLE_END:-}"
  sample_suffix="_s${sample_start}"
  if [[ -n "${SAMPLE_END:-}" ]]; then
    sample_suffix="${sample_suffix}_e${sample_end}"
  fi
fi
if [[ "${thinking_mode}" == "thinking" ]]; then
  batch_size="${BATCH_SIZE:-16}"
fi

common_args=(
  --model "${model}"
  --thinking-mode "${thinking_mode}"
  --split test
  --batch-size "${batch_size}"
  --max-tokens "${MAX_TOKENS:-256}"
  --max-model-len "${MAX_MODEL_LEN:-18000}"
  --max-num-seqs "${MAX_NUM_SEQS:-16}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}"
  --enforce-eager
  --language-model-only
  --resume
)
if [[ -n "${SAMPLE_START:-}" ]]; then
  common_args+=(--sample-start "${SAMPLE_START}")
fi
if [[ -n "${SAMPLE_END:-}" ]]; then
  common_args+=(--sample-end "${SAMPLE_END}")
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  common_args+=(--max-samples "${MAX_SAMPLES}")
fi

direct_responses="outputs/task_b/vllm/${model}_${thinking_mode}_${prompt_tag}${sample_suffix:-}_direct_test_responses.jsonl"
constraint_responses="outputs/task_b/vllm/${model}_${thinking_mode}_${prompt_tag}${sample_suffix:-}_constraint_test_responses.jsonl"
candidate_responses="outputs/task_b/vllm/${model}_${thinking_mode}_${prompt_tag}${sample_suffix:-}_candidate_test_k${candidate_repeat}_responses.jsonl"
verifier_responses="outputs/task_b/vllm/${model}_${thinking_mode}_${prompt_tag}${sample_suffix:-}_verifier_test_k${candidate_repeat}_responses.jsonl"
result_prefix="results/task_b/vllm/${model}_${thinking_mode}_${prompt_tag}${sample_suffix:-}"

run_step() {
  "$@"
  local status=$?
  if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
  fi
}

run_step "${python_bin}" -m task_b_verification_uncertainty.vllm_infer \
  "${common_args[@]}" \
  --task-mode direct \
  --output "${direct_responses}"

run_step "${python_bin}" -m task_b_verification_uncertainty.cli parse-responses \
  --input "${direct_responses}" \
  --output "${result_prefix}_direct_predictions.csv"

run_step "${python_bin}" -m task_b_verification_uncertainty.vllm_infer \
  "${common_args[@]}" \
  --task-mode constraint \
  --output "${constraint_responses}"

run_step "${python_bin}" -m task_b_verification_uncertainty.cli parse-responses \
  --input "${constraint_responses}" \
  --output "${result_prefix}_constraint_predictions.csv"

run_step "${python_bin}" -m task_b_verification_uncertainty.vllm_infer \
  "${common_args[@]}" \
  --task-mode candidate \
  --repeat "${candidate_repeat}" \
  --output "${candidate_responses}"

run_step "${python_bin}" -m task_b_verification_uncertainty.vllm_infer \
  "${common_args[@]}" \
  --task-mode verifier \
  --candidates "${candidate_responses}" \
  --output "${verifier_responses}"

run_step "${python_bin}" -m task_b_verification_uncertainty.cli rerank \
  --split test \
  --candidates "${candidate_responses}" \
  --verifier-responses "${verifier_responses}" \
  --output "${result_prefix}_rerank_predictions.csv" \
  --metrics-output "${result_prefix}_rerank_metrics.csv"

run_step "${python_bin}" -m task_b_verification_uncertainty.cli evaluate \
  --split test \
  --predictions "${result_prefix}_direct_predictions.csv" \
  --output "${result_prefix}_direct_metrics.csv"

run_step "${python_bin}" -m task_b_verification_uncertainty.cli evaluate \
  --split test \
  --predictions "${result_prefix}_constraint_predictions.csv" \
  --output "${result_prefix}_constraint_metrics.csv"

run_step "${python_bin}" -m task_b_verification_uncertainty.cli combine-predictions \
  --inputs \
  "${result_prefix}_direct_predictions.csv" \
  "${result_prefix}_constraint_predictions.csv" \
  "${result_prefix}_rerank_predictions.csv" \
  --output "${result_prefix}_predictions.csv"

run_step "${python_bin}" -m task_b_verification_uncertainty.cli evaluate \
  --split test \
  --predictions "${result_prefix}_predictions.csv" \
  --output "${result_prefix}_metrics.csv"
