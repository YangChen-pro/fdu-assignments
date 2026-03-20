#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONTAINER_NAME="${CONTAINER_NAME:-neo4j}"
NEO4J_STATE_DIR="${NEO4J_STATE_DIR:-${ROOT_DIR}/.neo4j}"

START_AFTER_RESET=0
YES=0

usage() {
  cat <<'EOF'
Usage:
  bash ./reset_docker.sh [--yes] [--start]

What it does:
  - docker rm -f neo4j (or $CONTAINER_NAME)
  - deletes .neo4j (or $NEO4J_STATE_DIR), which resets Neo4j data/logs

Options:
  --yes    Skip confirmation prompt (DANGEROUS: deletes local data directory)
  --start  Start Neo4j again after reset (runs kgqa/scripts/start_neo4j.sh)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START_AFTER_RESET=1; shift ;;
    --yes|-y) YES=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found in PATH." >&2
  exit 1
fi

echo "Stopping/removing container: ${CONTAINER_NAME}"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

if [[ -e "${NEO4J_STATE_DIR}" ]]; then
  if [[ "${YES}" -ne 1 ]]; then
    echo "About to delete Neo4j state dir: ${NEO4J_STATE_DIR}"
    read -r -p "Continue? [y/N] " ans
    case "${ans}" in
      y|Y|yes|YES) ;;
      *) echo "Canceled."; exit 0 ;;
    esac
  fi

  rm -rf "${NEO4J_STATE_DIR}"
  echo "Deleted: ${NEO4J_STATE_DIR}"
else
  echo "State dir not found (nothing to delete): ${NEO4J_STATE_DIR}"
fi

if [[ "${START_AFTER_RESET}" -eq 1 ]]; then
  bash "${ROOT_DIR}/kgqa/scripts/start_neo4j.sh"
fi
