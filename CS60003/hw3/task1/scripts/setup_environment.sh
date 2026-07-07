#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
ENV_NAME=${ENV_NAME:-hw3-task1}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
DOWNLOAD_PROXY=${DOWNLOAD_PROXY:-}
CONDA_BASE=${CONDA_BASE:-}

WGET_PROXY_ARGS=()
CURL_PROXY_ARGS=()
if [ -n "$DOWNLOAD_PROXY" ]; then
  WGET_PROXY_ARGS=(-e "use_proxy=yes" -e "http_proxy=$DOWNLOAD_PROXY" -e "https_proxy=$DOWNLOAD_PROXY")
  CURL_PROXY_ARGS=(--proxy "$DOWNLOAD_PROXY")
fi

if [ -z "$CONDA_BASE" ]; then
  command -v conda >/dev/null 2>&1 || {
    echo "conda is required; set CONDA_BASE if it is not on PATH" >&2
    exit 1
  }
  CONDA_BASE=$(conda info --base)
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi
conda activate "$ENV_NAME"

python -m pip install -i "$PIP_INDEX_URL" --upgrade pip setuptools wheel
python -m pip install -i "$PIP_INDEX_URL" -r "$PROJECT_DIR/hw3/task1/requirements.txt"
python -m pip install -i "$PIP_INDEX_URL" git+https://github.com/openai/CLIP.git
python -m pip install -i "$PIP_INDEX_URL" nerfstudio gsplat

mkdir -p "$PROJECT_DIR/hw3/task1/external"
cd "$PROJECT_DIR/hw3/task1/external"

if [ ! -d threestudio ]; then
  git clone https://github.com/threestudio-project/threestudio.git
fi
cd threestudio
python -m pip install -i "$PIP_INDEX_URL" -r requirements.txt
python -m pip install -i "$PIP_INDEX_URL" -e .

python - <<'PY'
from pathlib import Path

guidance_dir = Path("threestudio/models/guidance")
for filename in ["zero123_guidance.py", "stable_zero123_guidance.py"]:
    path = guidance_dir / filename
    if not path.exists():
        continue
    text = path.read_text(encoding="utf-8")
    patched = text.replace(
        'torch.load(ckpt, map_location="cpu")',
        'torch.load(ckpt, map_location="cpu", weights_only=False)',
    )
    if patched != text:
        path.write_text(patched, encoding="utf-8")
        print(f"Patched {filename} for trusted Zero123 checkpoints on PyTorch >=2.6")
PY

mkdir -p load/zero123
if [ ! -f load/zero123/sd-objaverse-finetune-c_concat-256.yaml ] && [ -f load/zero123/download.sh ]; then
  bash load/zero123/download.sh
fi
if [ ! -f load/zero123/zero123-xl.ckpt ]; then
  if command -v wget >/dev/null 2>&1; then
    wget "${WGET_PROXY_ARGS[@]}" -c -P load/zero123 https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt
  else
    curl "${CURL_PROXY_ARGS[@]}" -L -C - -o load/zero123/zero123-xl.ckpt https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt
  fi
fi

echo "Install COLMAP, FFmpeg and Blender if they are missing:"
echo "  sudo apt-get update"
echo "  sudo apt-get install -y colmap ffmpeg blender"
