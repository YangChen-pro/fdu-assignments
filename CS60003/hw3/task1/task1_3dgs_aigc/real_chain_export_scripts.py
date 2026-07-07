"""Export shell script generation for the HW3 Task1 real tool chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any



def export_script(config: dict[str, Any], run_dir: Path) -> str:
    chain = config["real_chain"]
    threestudio_root = Path(chain["tools"]["threestudio_launch"]).parent
    trusted_runner = Path(chain["tools"]["nerfstudio_swanlab_runner"]).with_name("run_trusted_torch.py")
    export_resolution = int(chain["object_c"]["export_resolution"])
    texture_size = int(chain["quality"]["mesh_texture_size"])
    nerfstudio_block = _nerfstudio_export_block(run_dir, trusted_runner)
    threestudio_block = _threestudio_export_block(
        run_dir,
        trusted_runner,
        export_resolution,
        texture_size,
    )
    return f"""#!/usr/bin/env bash
set -euo pipefail
export WANDB_MODE=offline
{nerfstudio_block}

cd "{threestudio_root}"
{threestudio_block}
"""


def _nerfstudio_export_block(run_dir: Path, trusted_runner: Path) -> str:
    return f"""\
for target in object_a background; do
  config_path=$(find "{run_dir}/nerfstudio/$target" -name config.yml | sort | tail -1)
  test -n "$config_path" || {{ echo "missing nerfstudio config for $target" >&2; exit 1; }}
  mkdir -p "{run_dir}/exports/$target/splat" "{run_dir}/exports/$target/mesh" "{run_dir}/eval/$target"
  if ! python "{trusted_runner}" ns-eval --load-config "$config_path" --output-path "{run_dir}/eval/$target/metrics.json"; then
    TARGET="$target" METRICS_PATH="{run_dir}/eval/$target/metrics.json" python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["METRICS_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(
    json.dumps(
        {{
            "status": "SKIPPED",
            "target": os.environ["TARGET"],
            "reason": "ns-eval failed, usually because this dataset has no eval image split; geometry export continues.",
        }},
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
PY
  fi
  python "{trusted_runner}" ns-export gaussian-splat --load-config "$config_path" --output-dir "{run_dir}/exports/$target/splat"
  if ! python "{trusted_runner}" ns-export tsdf --load-config "$config_path" --output-dir "{run_dir}/exports/$target/mesh"; then
    TARGET="$target" MARKER_PATH="{run_dir}/exports/$target/mesh/SKIPPED_TSDF.json" python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["MARKER_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(
    json.dumps(
        {{
            "status": "SKIPPED",
            "target": os.environ["TARGET"],
            "reason": "ns-export tsdf is not compatible with this splatfacto output; gaussian-splat export remains the authoritative 3DGS artifact.",
        }},
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
PY
  fi
done
"""


def _threestudio_export_block(
    run_dir: Path,
    trusted_runner: Path,
    export_resolution: int,
    texture_size: int,
) -> str:
    return f"""\
for item in object_b:object_b_threestudio object_c:object_c_zero123; do
  target="${{item%%:*}}"
  trial="${{item#*:}}"
  base_dir="{run_dir}/$trial"
  parsed="$(find "$base_dir" -path "*/configs/parsed.yaml" | sort | tail -1)"
  test -n "$parsed" || {{ echo "missing threestudio parsed config for $target under $base_dir" >&2; exit 1; }}
  trial_dir="$(dirname "$(dirname "$parsed")")"
  ckpt="$trial_dir/ckpts/last.ckpt"
  test -f "$ckpt" || {{ echo "missing threestudio checkpoint for $target: $ckpt" >&2; exit 1; }}
  python "{trusted_runner}" launch.py --config "$parsed" --export --gpu 0 \\
    resume="$ckpt" \\
    system.loggers.wandb.enable=false \\
    system.exporter_type=mesh-exporter \\
    system.exporter.fmt=obj-mtl \\
    system.exporter.save_uv=true \\
    system.exporter.save_texture=true \\
    system.exporter.texture_size={texture_size} \\
    system.geometry.isosurface_method=mc-cpu \\
    system.geometry.isosurface_resolution={export_resolution}
  exported="$(find "$trial_dir/save" -path "*/it*-export/*.obj" | sort | tail -1)"
  test -n "$exported" || {{ echo "missing exported OBJ for $target" >&2; exit 1; }}
  export_dir="$(dirname "$exported")"
  material="$(find "$export_dir" -maxdepth 1 -iname '*.mtl' | sort | head -1)"
  texture="$(find "$export_dir" -maxdepth 1 \\( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \\) | sort | head -1)"
  test -n "$material" || {{ echo "missing exported MTL for $target" >&2; exit 1; }}
  test -n "$texture" || {{ echo "missing exported texture for $target" >&2; exit 1; }}
  target_dir="{run_dir}/exports/$target/mesh"
  mkdir -p "$target_dir"
  cp -a "$export_dir"/. "$target_dir/"
  if [ "$(basename "$exported")" != "model.obj" ]; then
    cp "$exported" "$target_dir/model.obj"
  fi
done
"""
