#!/usr/bin/env bash
set -euo pipefail

exec bash "$(cd "$(dirname "$0")" && pwd)/kgqa/scripts/start_neo4j.sh"
