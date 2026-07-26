#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-models/Qwen3-8B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
ENABLE_REASONING="${ENABLE_REASONING:-false}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

ARGS=(
  serve "${MODEL_PATH}"
  --trust-remote-code
  --host "${HOST}"
  --port "${PORT}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --generation-config vllm
)

if [[ -n "${CHAT_TEMPLATE}" ]]; then
  ARGS+=(--chat-template "${CHAT_TEMPLATE}")
fi

if [[ "${ENABLE_REASONING}" == "true" ]]; then
  ARGS+=(--enable-reasoning --reasoning-parser "${REASONING_PARSER}")
fi

if [[ "${ENFORCE_EAGER}" == "true" ]]; then
  ARGS+=(--enforce-eager)
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  EXTRA_ARRAY=(${EXTRA_ARGS})
  ARGS+=("${EXTRA_ARRAY[@]}")
fi

vllm "${ARGS[@]}"
