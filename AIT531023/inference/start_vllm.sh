#!/bin/bash

# set -euo pipefail

SESSION_NAME="vllm"
PROJECT_DIR="/raid/data/yc/DeepResearch"

# 你的单卡 omni stage config 路径（请确保文件存在）
OMNI_STAGE_CONFIG="/raid/data/yc/DeepResearch/inference/qwen3_omni_local.yaml"

if ! command -v tmux >/dev/null 2>&1; then
    echo "错误：未安装 tmux，无法使用此脚本。"
    exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "检测到已有 tmux 会话 '$SESSION_NAME'，先关闭旧会话"
    tmux kill-session -t "$SESSION_NAME"
fi

first_window_created=false

start_window() {
    local window_name="$1"
    local gpu="$2"
    local port="$3"
    local model_path="$4"
    local extra_args="${5:-}"

    local cmd="cd \"$PROJECT_DIR\" && \
echo '启动模型: $model_path (GPU $gpu, 端口 $port) extra: $extra_args' && \
CUDA_VISIBLE_DEVICES=\"$gpu\" vllm serve \"$model_path\" \
  --host 0.0.0.0 \
  --port \"$port\" \
  $extra_args \
  --gpu-memory-utilization 0.9 \
  --disable-log-requests"

    if [ "$first_window_created" = false ]; then
        tmux new-session -d -s "$SESSION_NAME" -n "$window_name" "bash -lc \"$cmd\""
        first_window_created=true
    else
        tmux new-window -t "$SESSION_NAME" -n "$window_name" "bash -lc \"$cmd\""
    fi
}

start_window "tongyi" 0 6001 "/raid/data/yc/DeepResearch/models/Tongyi-DeepResearch-30B-A3B"
start_window "summary" 1 10001 "/raid/data/models/Qwen3-30B-A3B-Instruct-2507"

# omni：必须用 stage config 才能真正固定 devices / TP
start_window "omni" 2 10002 "/raid/data/yc/DeepResearch/models/Qwen3-Omni-30B-A3B-Instruct" \
  "--omni --stage-configs-path \"$OMNI_STAGE_CONFIG\""

echo "已经在 tmux session '$SESSION_NAME' 的 'tongyi'、'summary'、'omni' 窗口中启动模型。"
echo "用 \`tmux attach -t $SESSION_NAME\` 观察输出；杀掉会话会终止所有服务。"
