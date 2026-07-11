#!/usr/bin/env bash

# 日志目录
LOG_DIR="/data/yc/llm-26-homework/ollama_log"

# 如果 pid 文件存在，就按 pid 关闭
if [ -f "$LOG_DIR/ollama.pid" ]; then
  kill "$(cat "$LOG_DIR/ollama.pid")" && rm -f "$LOG_DIR/ollama.pid"
  echo "Ollama 已关闭"
else
  echo "未找到 $LOG_DIR/ollama.pid，尝试按进程名关闭..."
  pkill -f "ollama serve" && echo "Ollama 已关闭" || echo "没有找到运行中的 Ollama"
fi