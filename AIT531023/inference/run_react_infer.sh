#!/bin/bash

######################################
# 0. 从 .env 文件加载环境变量
######################################

# 获取当前脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# .env 文件路径（位于上一级目录）
ENV_FILE="$SCRIPT_DIR/../.env"

# 检查 .env 是否存在
if [ ! -f "$ENV_FILE" ]; then
    echo "错误：未找到 .env 文件，路径：$ENV_FILE"
    echo "请将 .env.example 复制为 .env 并按需配置："
    echo "  cp .env.example .env"
    exit 1
fi

echo "正在从 .env 文件加载环境变量..."
set -a  # 自动 export 后续 source 的所有变量
source "$ENV_FILE"
set +a  # 停止自动 export

# 校验关键变量：MODEL_PATH 必须正确配置
if [ "$MODEL_PATH" = "/your/model/path" ] || [ -z "$MODEL_PATH" ]; then
    echo "错误：.env 中未正确配置 MODEL_PATH"
    exit 1
fi

######################################
# 1. 等待现有 vLLM 服务就绪
######################################

echo "请确认 vLLM 服务已手动启动并在端口 6001 上运行，脚本只会等待服务上线。"

timeout=6000
start_time=$(date +%s)

# 主服务端口列表（当前只使用 6001）
# main_ports=(6001 6002 6003 6004 6005 6006 6007 6008)
main_ports=(6001)
echo "模式：所有端口均作为主模型服务（当前仅启用 6001）"

declare -A server_status
for port in "${main_ports[@]}"; do
    server_status[$port]=false
done

echo "等待服务启动..."

while true; do
    all_ready=true

    for port in "${main_ports[@]}"; do
        if [ "${server_status[$port]}" = "false" ]; then
            # 通过访问 /v1/models 判断服务是否就绪
            if curl -s -f http://localhost:$port/v1/models > /dev/null 2>&1; then
                echo "主模型服务（端口 $port）已就绪！"
                server_status[$port]=true
            else
                all_ready=false
            fi
        fi
    done

    # 全部端口就绪则跳出
    if [ "$all_ready" = "true" ]; then
        echo "所有服务已就绪，可以开始推理！"
        break
    fi

    # 超时检查
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -gt $timeout ]; then
        echo -e "\n错误：服务启动超时（超过 ${timeout} 秒）"

        for port in "${main_ports[@]}"; do
            if [ "${server_status[$port]}" = "false" ]; then
                echo "主模型服务（端口 $port）启动失败或未就绪"
            fi
        done

        exit 1
    fi

    printf '等待服务启动中 .....'
    sleep 10
done

# 再次确认是否有失败端口
failed_servers=()
for port in "${main_ports[@]}"; do
    if [ "${server_status[$port]}" = "false" ]; then
        failed_servers+=($port)
    fi
done

if [ ${#failed_servers[@]} -gt 0 ]; then
    echo "错误：以下服务端口启动失败：${failed_servers[*]}"
    exit 1
else
    echo "所有需要的服务均运行正常！"
fi

######################################
# 2. 开始推理（运行 run_multi_react.py）
#    vLLM 服务需要事先手动启动并就绪
######################################

echo "==== 开始推理... ===="

# 切换到 inference 目录（当前脚本所在目录）
cd "$( dirname -- "${BASH_SOURCE[0]}" )"

python -u run_multi_react.py \
  --dataset "$DATASET" \
  --output "$OUTPUT_PATH" \
  --max_workers $MAX_WORKERS \
  --model $MODEL_PATH \
  --temperature $TEMPERATURE \
  --presence_penalty $PRESENCE_PENALTY \
  --total_splits ${WORLD_SIZE:-1} \
  --worker_split $((${RANK:-0} + 1)) \
  --roll_out_count $ROLLOUT_COUNT
