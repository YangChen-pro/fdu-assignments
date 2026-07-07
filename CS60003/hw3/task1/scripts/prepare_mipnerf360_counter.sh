#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
DATA_ROOT=${DATA_ROOT:-$HOME/.cache/mipnerf360}
SCENE=${SCENE:-counter}
RESOLUTION_DIR=${RESOLUTION_DIR:-images}
DATASET_URL=${DATASET_URL:-https://storage.googleapis.com/gresearch/refraw360/360_v2.zip}
ZIP_PATH=${ZIP_PATH:-$DATA_ROOT/360_v2.zip}
TARGET_LINK="$PROJECT_DIR/hw3/assets/background_scene/images"
DOWNLOAD_PROXY=${DOWNLOAD_PROXY:-}

WGET_PROXY_ARGS=()
CURL_PROXY_ARGS=()
if [ -n "$DOWNLOAD_PROXY" ]; then
  WGET_PROXY_ARGS=(-e "use_proxy=yes" -e "http_proxy=$DOWNLOAD_PROXY" -e "https_proxy=$DOWNLOAD_PROXY")
  CURL_PROXY_ARGS=(--proxy "$DOWNLOAD_PROXY")
fi

mkdir -p "$DATA_ROOT" "$PROJECT_DIR/hw3/assets/background_scene"

if [ ! -d "$DATA_ROOT/$SCENE" ]; then
  if [ ! -f "$ZIP_PATH" ]; then
    if command -v wget >/dev/null 2>&1; then
      wget "${WGET_PROXY_ARGS[@]}" -c -O "$ZIP_PATH" "$DATASET_URL"
    else
      curl "${CURL_PROXY_ARGS[@]}" -L -C - -o "$ZIP_PATH" "$DATASET_URL"
    fi
  fi
  unzip -n "$ZIP_PATH" "$SCENE/*" -d "$DATA_ROOT"
fi

SOURCE_DIR="$DATA_ROOT/$SCENE/$RESOLUTION_DIR"
if [ ! -d "$SOURCE_DIR" ]; then
  SOURCE_DIR="$DATA_ROOT/$SCENE/images"
fi
test -d "$SOURCE_DIR" || { echo "missing Mip-NeRF 360 scene images: $SOURCE_DIR" >&2; exit 1; }

if [ -L "$TARGET_LINK" ]; then
  ln -sfn "$SOURCE_DIR" "$TARGET_LINK"
elif [ -e "$TARGET_LINK" ]; then
  echo "background target already exists and is not a symlink: $TARGET_LINK"
  echo "Use it as-is, or move it away before re-linking to $SOURCE_DIR."
else
  ln -s "$SOURCE_DIR" "$TARGET_LINK"
fi

echo "Mip-NeRF 360 $SCENE images ready: $SOURCE_DIR"
echo "Task1 background path: $TARGET_LINK"
