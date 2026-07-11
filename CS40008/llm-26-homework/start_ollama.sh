#!/usr/bin/env bash

LOG_DIR="/data/yc/llm-26-homework/ollama_log"
OLLAMA_BIN="/data/yc/ollama/bin/ollama"

mkdir -p "$LOG_DIR"

export OLLAMA_HOST="http://127.0.0.1:11434"
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"
export OLLAMA_DEBUG="1"

nohup "$OLLAMA_BIN" serve > "$LOG_DIR/ollama.log" 2>&1 &
echo $! > "$LOG_DIR/ollama.pid"

echo "Ollama 已启动"
echo "PID: $(cat "$LOG_DIR/ollama.pid")"
echo "日志文件: $LOG_DIR/ollama.log"
echo "使用的 Ollama: $OLLAMA_BIN"