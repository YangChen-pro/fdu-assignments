#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONTAINER_NAME="neo4j"     # 可改：容器名
IMAGE_TAG="neo4j:4.4"          # 可改：Neo4j 版本（建议先 4.4 LTS）
BIND_HOST="127.0.0.1"          # 可改：绑定主机（默认仅本机可访问，避免无密码暴露到局域网）
HTTP_PORT="7474"               # 可改：Web 端口（浏览器用）
BOLT_PORT="7687"               # 可改：Bolt 端口（驱动用）
AUTH_MODE="none"               # 可改：basic 或 none（none = 不要密码）
NEO4J_USER="neo4j"             # basic 时使用
NEO4J_PASSWORD="your_password" # basic 时使用（与项目 neo_db/config.py 保持一致）

DATA_DIR="${ROOT_DIR}/.neo4j/data" # 可改：数据持久化目录
LOGS_DIR="${ROOT_DIR}/.neo4j/logs" # 可改：日志目录

mkdir -p "${DATA_DIR}" "${LOGS_DIR}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

auth_env=()
if [[ "${AUTH_MODE}" == "none" ]]; then
  auth_env+=(-e "NEO4J_AUTH=none")
else
  auth_env+=(-e "NEO4J_AUTH=${NEO4J_USER}/${NEO4J_PASSWORD}")
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  -p "${BIND_HOST}:${HTTP_PORT}:7474" \
  -p "${BIND_HOST}:${BOLT_PORT}:7687" \
  "${auth_env[@]}" \
  -v "${DATA_DIR}:/data" \
  -v "${LOGS_DIR}:/logs" \
  "${IMAGE_TAG}"

if [[ "${AUTH_MODE}" == "none" ]]; then
  echo "Neo4j is starting: http://localhost:${HTTP_PORT} (no auth)"
else
  echo "Neo4j is starting: http://localhost:${HTTP_PORT} (user: ${NEO4J_USER})"
fi
