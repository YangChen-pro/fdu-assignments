#!/usr/bin/env bash
set -euo pipefail

show_usage() {
    cat <<'USAGE'
Usage:
  lab5/task_scripts/serve_vllm.sh MODEL_PATH [SERVED_MODEL] [-- extra vLLM args]

Environment variables:
  CONDA_ENV                  conda environment name, default: lab3
  HOST                       serve host, default: 0.0.0.0
  PORT                       serve port, default: 8000
  GPU_MEMORY_UTILIZATION     default: 0.85
  MAX_MODEL_LEN              default: 4096

Example:
  lab5/task_scripts/serve_vllm.sh \
    lab5/outputs/task2/megatron_lora_sft_all/v1-20260527-125813/checkpoint-5042-merged \
    task2-sft
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_usage
    exit 0
fi

if [[ $# -lt 1 ]]; then
    show_usage >&2
    exit 2
fi

CONDA_ENV=${CONDA_ENV:-lab3}
MODEL=$1
SERVED_MODEL=${2:-$(basename "$MODEL")}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}

shift
if [[ $# -gt 0 ]]; then
    shift
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

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

vllm serve "$MODEL" \
    --served-model-name "$SERVED_MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    "$@"
