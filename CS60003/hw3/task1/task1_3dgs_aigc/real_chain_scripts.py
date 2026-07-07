"""Shell script generation for the HW3 Task1 real tool chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from task1_3dgs_aigc.real_chain_export_scripts import export_script
from task1_3dgs_aigc.script_utils import resolve_swanlab_config, swanlab_tag_args, to_cli_list, write_script


def write_scripts(config: dict[str, Any], run_dir: Path, scripts_dir: Path) -> list[Path]:
    scripts_dir.mkdir(parents=True, exist_ok=True)
    scripts = [
        _write_script(scripts_dir / "00_check_tools.sh", _check_tools_script(config)),
        _write_script(scripts_dir / "01_object_a_splatfacto.sh", _splatfacto_script(config, run_dir, "object_a")),
        _write_script(scripts_dir / "02_background_splatfacto.sh", _splatfacto_script(config, run_dir, "background")),
        _write_script(scripts_dir / "03_object_b_threestudio.sh", _object_b_script(config, run_dir)),
        _write_script(scripts_dir / "04_object_c_zero123.sh", _object_c_script(config, run_dir)),
        _write_script(scripts_dir / "05_export_geometry.sh", export_script(config, run_dir)),
        _write_script(scripts_dir / "06_render_blender.sh", _render_script(config, run_dir)),
    ]
    return scripts


def _check_tools_script(config: dict[str, Any]) -> str:
    required = " ".join(config["real_chain"]["tools"]["required_cli"])
    swanlab = resolve_swanlab_config(config)
    return f"""#!/usr/bin/env bash
set -euo pipefail
for tool in {required}; do
  command -v "$tool" >/dev/null || {{ echo "missing required tool: $tool" >&2; exit 1; }}
done
test -f "{swanlab['env_file']}" || {{ echo "missing SwanLab env file: {swanlab['env_file']}" >&2; exit 1; }}
set -a
source "{swanlab['env_file']}"
set +a
test -n "${{SWANLAB_API_KEY:-}}" || {{ echo "missing SWANLAB_API_KEY in env file" >&2; exit 1; }}
test -f "{config['real_chain']['tools']['threestudio_launch']}" || {{ echo "missing threestudio launch.py" >&2; exit 1; }}
test -f "{config['real_chain']['tools']['zero123_config']}" || {{ echo "missing Zero123 threestudio config" >&2; exit 1; }}
test -f "{config['real_chain']['tools']['zero123_config_file']}" || {{ echo "missing Zero123 model config" >&2; exit 1; }}
test -f "{config['real_chain']['tools']['zero123_checkpoint']}" || {{ echo "missing Zero123 checkpoint: {config['real_chain']['tools']['zero123_checkpoint']}" >&2; exit 1; }}
python - <<'PY'
from pathlib import Path
path = Path("{config['real_chain']['tools']['zero123_checkpoint']}")
min_size = 15_000_000_000
if path.stat().st_size < min_size:
    raise SystemExit(f"incomplete Zero123 checkpoint: {{path}} size={{path.stat().st_size}} expected>={{min_size}}")
PY
test -f "{config['real_chain']['tools']['object_mask_attacher']}" || {{ echo "missing object mask attacher" >&2; exit 1; }}
test -f "{config['real_chain']['tools']['object_c_preprocessor']}" || {{ echo "missing object C preprocessor" >&2; exit 1; }}
test -f "{config['real_chain']['tools']['nerfstudio_swanlab_runner']}" || {{ echo "missing Nerfstudio SwanLab runner" >&2; exit 1; }}
test -f "{config['real_chain']['tools']['threestudio_swanlab_runner']}" || {{ echo "missing threestudio SwanLab runner" >&2; exit 1; }}
test -f "{Path(config['real_chain']['tools']['nerfstudio_swanlab_runner']).with_name('run_trusted_torch.py')}" || {{ echo "missing trusted torch runner" >&2; exit 1; }}
python - <<'PY'
import cv2
import swanlab
import wandb
PY
"""


def _splatfacto_script(config: dict[str, Any], run_dir: Path, target: str) -> str:
    quality = config["real_chain"]["quality"]
    context = _splatfacto_context(config, run_dir, target)
    processed = run_dir / "processed" / target
    output = run_dir / "nerfstudio" / target
    swanlab = resolve_swanlab_config(config)
    experiment_name = f"{config['experiment']['name']}-{target}-3dgs"
    return f"""#!/usr/bin/env bash
set -euo pipefail
export QT_QPA_PLATFORM="${{QT_QPA_PLATFORM:-offscreen}}"
export MPLBACKEND="${{MPLBACKEND:-Agg}}"
mkdir -p "{processed}" "{output}"
{context['preamble']}\
{context['process_data_cmd']}
{context['postprocess']}\
python "{config['real_chain']['tools']['nerfstudio_swanlab_runner']}" \\
  --env-file "{swanlab['env_file']}" \\
  --project "{swanlab['project']}" \\
  --group "{swanlab['group']}" \\
  --experiment-name "{experiment_name}" \\
  --mode "{swanlab['mode']}" \\
  --logdir "{run_dir}/swanlab/{target}" \\
  {swanlab_tag_args(swanlab, [target, "3dgs", "nerfstudio"])} \\
  -- ns-train splatfacto-big \\
  --vis tensorboard \\
  --data "{processed}" \\
  --output-dir "{output}" \\
  --max-num-iterations {int(quality['splatfacto_iterations'])} \\
  --pipeline.model.cull-alpha-thresh {quality['cull_alpha_thresh']} \\
  --pipeline.model.use-scale-regularization True \\
  --viewer.quit-on-train-completion True
"""


def _splatfacto_context(config: dict[str, Any], run_dir: Path, target: str) -> dict[str, str]:
    source, matching_method = _splatfacto_source(config, target)
    process_kind = "video" if str(source).lower().endswith((".mp4", ".mov", ".m4v")) else "images"
    processed = run_dir / "processed" / target
    process_data_cmd = (
        f'ns-process-data {process_kind} --data "{source}" --output-dir "{processed}" '
        f'--no-gpu --matching-method {matching_method}'
    )
    preamble = ""
    postprocess = ""
    if target == "object_a":
        quality = config["real_chain"]["quality"]
        target_frames = int(quality["object_a_num_frames_target"])
        if process_kind == "video":
            process_data_cmd += f" --num-frames-target {target_frames}"
        postprocess = _object_a_mask_command(
            config,
            processed,
            run_dir / "preprocessed" / "object_a_mask_report.json",
            expected_frames=target_frames,
        )
    if target == "background" and process_kind == "images":
        process_data_cmd = _background_process_data_cmd(source, processed, matching_method)
    return {
        "preamble": preamble,
        "process_data_cmd": process_data_cmd,
        "postprocess": postprocess,
    }


def _splatfacto_source(config: dict[str, Any], target: str) -> tuple[str, str]:
    data = config["real_chain"]["data"]
    if target == "object_a":
        return str(data.get("object_a_video") or data["object_a_images"]), "exhaustive"
    return str(data.get("background_video") or data["background_images"]), "sequential"


def _object_a_preprocess_preamble(config: dict[str, Any], run_dir: Path, raw_source: str) -> tuple[str, str]:
    source = run_dir / "preprocessed" / "object_a_foreground" / "images"
    mask_output = run_dir / "preprocessed" / "object_a_foreground" / "masks"
    colmap_wrapper = run_dir / "scripts" / "colmap_object_a_wrapper.sh"
    preamble = _object_a_preprocess_cmd(config, raw_source, source, mask_output)
    preamble += _object_a_colmap_wrapper_cmd(colmap_wrapper)
    return preamble, f' --colmap-cmd "{colmap_wrapper}"'


def _object_a_preprocess_cmd(config: dict[str, Any], raw_source: str, source: Path, mask_output: Path) -> str:
    return (
        f'python "{config["real_chain"]["tools"]["object_foreground_preprocessor"]}" '
        f'--input "{raw_source}" --output "{source}" --mask-output "{mask_output}"\n'
    )


def _object_a_mask_command(
    config: dict[str, Any],
    processed: Path,
    report: Path,
    *,
    expected_frames: int,
) -> str:
    quality = config["real_chain"]["quality"]
    return (
        f'python "{config["real_chain"]["tools"]["object_mask_attacher"]}" '
        f'--processed-dir "{processed}" '
        f'--model "{quality["object_a_segmentation_model"]}" '
        f'--expected-frames {expected_frames} '
        f'--min-registration-ratio {quality["object_a_min_registration_ratio"]} '
        f'--min-occupancy {quality["object_a_min_mask_occupancy"]} '
        f'--max-occupancy {quality["object_a_max_mask_occupancy"]} '
        f'--min-refined-area-ratio {quality["object_a_min_refined_area_ratio"]} '
        f'--max-refined-area-ratio {quality["object_a_max_refined_area_ratio"]} '
        f'--report "{report}"\n'
    )


def _object_a_colmap_wrapper_cmd(colmap_wrapper: Path) -> str:
    return (
        f'cat > "{colmap_wrapper}" <<\'COLMAP_OBJECT_A_WRAPPER\'\n'
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "subcommand=\"${1:-}\"\n"
        "if [ -z \"$subcommand\" ]; then exec colmap; fi\n"
        "shift\n"
        "if [ \"$subcommand\" = \"mapper\" ]; then\n"
        "  exec colmap mapper \"$@\" \\\n"
        "    --Mapper.min_num_matches 15 \\\n"
        "    --Mapper.init_min_num_inliers 30 \\\n"
        "    --Mapper.abs_pose_min_num_inliers 15 \\\n"
        "    --Mapper.ba_refine_principal_point 0 \\\n"
        "    --Mapper.ba_global_max_num_iterations 20 \\\n"
        "    --Mapper.ba_local_max_num_iterations 20 \\\n"
        "    --Mapper.ba_global_max_refinements 2 \\\n"
        "    --Mapper.ba_local_max_refinements 1\n"
        "fi\n"
        "exec colmap \"$subcommand\" \"$@\"\n"
        "COLMAP_OBJECT_A_WRAPPER\n"
        f'chmod +x "{colmap_wrapper}"\n'
    )


def _background_process_data_cmd(source: str, processed: Path, matching_method: str) -> str:
    return f"""SOURCE_REAL=$(python -c 'from pathlib import Path; print(Path("{source}").resolve())')
SCENE_ROOT=$(dirname "$SOURCE_REAL")
if [ -d "$SCENE_ROOT/sparse/0" ]; then
  python -c 'import shutil; shutil.rmtree("{processed}", ignore_errors=True)'
  mkdir -p "{processed}/colmap/sparse"
  ln -sfn "$SCENE_ROOT/sparse/0" "{processed}/colmap/sparse/0"
  ns-process-data images --data "$SOURCE_REAL" --output-dir "{processed}" --no-gpu --skip-colmap --colmap-model-path colmap/sparse/0
else
  ns-process-data images --data "$SOURCE_REAL" --output-dir "{processed}" --no-gpu --matching-method {matching_method}
fi"""


def _object_b_script(config: dict[str, Any], run_dir: Path) -> str:
    chain = config["real_chain"]
    output = run_dir / "object_b_threestudio"
    prompt = chain["object_b"]["prompt"].replace('"', '\\"')
    sds_model = chain["object_b"]["sds_model"].replace('"', '\\"')
    swanlab = resolve_swanlab_config(config)
    return f"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p "{output}"
cd "{Path(chain['tools']['threestudio_launch']).parent}"
set -a
source "{swanlab['env_file']}"
set +a
SDS_MODEL="${{HW3_OBJECT_B_SDS_MODEL:-{sds_model}}}"
python "{chain['tools']['threestudio_swanlab_runner']}" \\
  --env-file "{swanlab['env_file']}" \\
  --launch "{chain['tools']['threestudio_launch']}" \\
  --mode "{swanlab['mode']}" \\
  --logdir "{run_dir}/swanlab/object_b" \\
  -- --config configs/dreamfusion-sd.yaml --train --gpu 0 \\
  exp_root_dir="{output}" \\
  name="object_b" \\
  tag="sds" \\
  system.prompt_processor.pretrained_model_name_or_path="$SDS_MODEL" \\
  system.guidance.pretrained_model_name_or_path="$SDS_MODEL" \\
  system.prompt_processor.prompt="{prompt}" \\
  trainer.max_steps={int(chain['object_b']['max_steps'])} \\
  trainer.limit_test_batches={int(chain['object_b']['limit_test_batches'])} \\
  system.guidance.grad_clip=[0,0.5,2.0,10000] \\
  system.prompt_processor.use_perp_neg=true \\
  system.loggers.wandb.enable=true \\
  system.loggers.wandb.project="{swanlab['project']}" \\
  system.loggers.wandb.name="{config['experiment']['name']}-object_b-sds"
"""


def _object_c_script(config: dict[str, Any], run_dir: Path) -> str:
    chain = config["real_chain"]
    output = run_dir / "object_c_zero123"
    threestudio_root = Path(chain["tools"]["threestudio_launch"]).parent
    zero123_config = Path(chain["tools"]["zero123_config"])
    zero123_config_arg = zero123_config.relative_to(threestudio_root) if zero123_config.is_relative_to(threestudio_root) else zero123_config
    object_c = chain["object_c"]
    input_height = to_cli_list(object_c["input_height"])
    input_width = to_cli_list(object_c["input_width"])
    batch_size = to_cli_list(object_c["random_camera_batch_size"])
    height = to_cli_list(object_c["random_camera_height"])
    width = to_cli_list(object_c["random_camera_width"])
    milestones = to_cli_list(object_c["resolution_milestones"])
    resume_checkpoint = str(object_c.get("resume_checkpoint", "")).replace('"', '\\"')
    swanlab = resolve_swanlab_config(config)
    preprocessed_dir = run_dir / "preprocessed" / "object_c"
    preprocessed_image = preprocessed_dir / "object_c_rgba.png"
    preprocess_report = preprocessed_dir / "report.json"
    return f"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p "{output}" "{preprocessed_dir}"
python "{chain['tools']['object_c_preprocessor']}" \\
  --input "{chain['data']['object_c_image']}" \\
  --output "{preprocessed_image}" \\
  --report "{preprocess_report}" \\
  --model "{object_c['preprocess_model']}" \\
  --size {int(object_c['preprocess_size'])} \\
  --padding-ratio {float(object_c['preprocess_padding_ratio'])} \\
  --min-occupancy {float(object_c['preprocess_min_occupancy'])} \\
  --max-occupancy {float(object_c['preprocess_max_occupancy'])}
cd "{threestudio_root}"
set -a
source "{swanlab['env_file']}"
set +a
RESUME_CHECKPOINT="${{HW3_OBJECT_C_RESUME_CHECKPOINT:-{resume_checkpoint}}}"
RESUME_ARGS=()
if [ -n "$RESUME_CHECKPOINT" ]; then
  RESUME_ARGS+=(resume="$RESUME_CHECKPOINT")
fi
python "{chain['tools']['threestudio_swanlab_runner']}" \\
  --env-file "{swanlab['env_file']}" \\
  --launch "{chain['tools']['threestudio_launch']}" \\
  --mode "{swanlab['mode']}" \\
  --logdir "{run_dir}/swanlab/object_c" \\
  -- --config "{zero123_config_arg}" --train --gpu 0 \\
  exp_root_dir="{output}" \\
  name="object_c" \\
  tag="zero123" \\
  data.image_path="{preprocessed_image}" \\
  system.guidance.pretrained_model_name_or_path="{chain['tools']['zero123_checkpoint']}" \\
  system.guidance.pretrained_config="{chain['tools']['zero123_config_file']}" \\
  data.height={input_height} \\
  data.width={input_width} \\
  data.random_camera.batch_size={batch_size} \\
  data.random_camera.height={height} \\
  data.random_camera.width={width} \\
  data.random_camera.resolution_milestones={milestones} \\
  data.resolution_milestones={milestones} \\
  data.default_elevation_deg={float(object_c['default_elevation_deg'])} \\
  system.renderer.num_samples_per_ray={int(object_c['num_samples_per_ray'])} \\
  system.loss.lambda_normal_smooth={float(object_c['lambda_normal_smooth'])} \\
  system.loss.lambda_3d_normal_smooth={float(object_c['lambda_3d_normal_smooth'])} \\
  system.loss.lambda_orient={float(object_c['lambda_orient'])} \\
  trainer.max_steps={int(object_c['zero123_max_steps'])} \\
  trainer.limit_val_batches={int(object_c['limit_val_batches'])} \\
  trainer.limit_test_batches={int(object_c['limit_test_batches'])} \\
  "${{RESUME_ARGS[@]}}" \\
  system.loggers.wandb.enable=true \\
  system.loggers.wandb.project="{swanlab['project']}" \\
  system.loggers.wandb.name="{config['experiment']['name']}-object_c-zero123"
"""


def _render_script(config: dict[str, Any], run_dir: Path) -> str:
    del config
    task_root = Path(__file__).resolve().parents[1]
    wrapper = task_root / "scripts" / "run_trusted_torch.py"
    renderer = task_root / "scripts" / "render_fused_splats.py"
    return f"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p "{run_dir}/renders"
mkdir -p "{run_dir}/renders/fused_splats"
PYTHON_BIN="${{PYTHON_BIN:-python}}"
"$PYTHON_BIN" "{wrapper}" \\
  --logdir "{run_dir}/renders/fused_splats" \\
  "{renderer}" \\
  --run-dir "{run_dir}" \\
  --output-dir "{run_dir}/renders/fused_splats" \\
  --background-max 650000 \\
  --object-a-max 450000 \\
  --object-a-opacity-quantile 0.35 \\
  --object-a-opacity-mult 1.05 \\
  --object-b-max 260000 \\
  --object-c-max 120000 \\
  --camera-focus foreground \\
  --camera-focal-multiplier 1.23 \\
  --camera-radial-sweep 0.03 \\
  --camera-height-sweep 0.05 \\
  --foreground-camera-radius 2.20 \\
  --foreground-camera-height 1.02 \\
  --foreground-camera-target-z 0.34 \\
  --foreground-camera-start-degrees 38.0 \\
  --foreground-camera-arc-degrees 34.0 \\
  --foreground-offset-x 0.45 \\
  --foreground-offset-y -0.35 \\
  --foreground-ground-offset 0.15 \\
  --object-separation 0.34 \\
  --object-a-height 0.62 \\
  --object-b-height 0.72 \\
  --object-c-height 0.24 \\
  --object-c-target-extent 0.30 \\
  --object-a-offset-x -0.30 \\
  --object-a-offset-y 0.08 \\
  --object-b-offset-x 0.00 \\
  --object-b-offset-y -0.08 \\
  --object-c-offset-x 0.30 \\
  --object-c-offset-y 0.08 \\
  --background-clear-width 0.84 \\
  --background-clear-depth 0.68 \\
  --background-clear-height 1.25 \\
  --background-clear-below 0.00 \\
  --background-clear-surface-keep 0.04 \\
  --background-clear-shape ellipse \\
  --support-mat-points 0 \\
  --support-mat-width 0.00 \\
  --support-mat-depth 0.00 \\
  --support-mat-thickness 0.020 \\
  --support-mat-opacity 0.00 \\
  --foreground-color-boost 1.20 \\
  --foreground-saturation-boost 1.14 \\
  --foreground-opacity-boost 1.06 \\
  --foreground-rim-strength 0.10 \\
  --foreground-rim-power 2.20 \\
  --object-a-color-boost 0.82 \\
  --object-a-opacity-boost 0.94 \\
  --camera-index-start 45 \\
  --camera-index-stop 150
"""



def _write_script(path: Path, content: str) -> Path:
    return write_script(path, content)


def expected_outputs(run_dir: Path) -> dict[str, str]:
    """Return the expected final artifact globs for a Task1 run directory."""
    return {
        "object_a_splat": f"{run_dir}/exports/object_a/splat/*.ply",
        "object_a_mesh": f"{run_dir}/exports/object_a/mesh/*",
        "background_splat": f"{run_dir}/exports/background/splat/*.ply",
        "background_mesh": f"{run_dir}/exports/background/mesh/*",
        "object_b_mesh": f"{run_dir}/exports/object_b/mesh/*",
        "object_c_mesh": f"{run_dir}/exports/object_c/mesh/*",
        "render_video": f"{run_dir}/renders/fused_splats/fused_scene.mp4",
    }
