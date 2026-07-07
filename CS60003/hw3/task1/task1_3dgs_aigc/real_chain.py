"""External-tool orchestration for HW3 Task 1.

This module does not replace COLMAP, Nerfstudio, threestudio, Zero123, or
Blender. It provides a reproducible orchestration layer so replacing
`hw3/assets` with real captures can drive the real tools directly.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from task1_3dgs_aigc.real_chain_scripts import expected_outputs, write_scripts
from task1_3dgs_aigc.utils import copy_source_config, make_run_dir, save_json

STRICT_REAL_OUTPUT_FILES = [
    "exports/object_a/splat/splat.ply",
    "exports/background/splat/splat.ply",
    "exports/object_b/mesh/model.obj",
    "exports/object_c/mesh/model.obj",
    "renders/fused_splats/fused_scene.mp4",
    "renders/fused_splats/fused_scene_manifest.json",
]


def run_real_chain(config: dict[str, Any]) -> dict[str, Any]:
    """Prepare or run the Task 1 pipeline."""
    run_dir = make_run_dir(config["experiment"]["output_root"], config["experiment"]["name"])
    scripts_dir = run_dir / "scripts"
    logs_dir = run_dir / "logs"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    copy_source_config(config["config_path"], run_dir)
    save_json(run_dir / "config.json", config)

    input_issues = _validate_real_inputs(
        config,
        require_complete=config["real_chain"]["execution"]["mode"] == "run",
    )
    scripts = write_scripts(config, run_dir, scripts_dir)
    outputs = expected_outputs(run_dir)
    summary = {
        "status": _initial_status(config, input_issues),
        "stage": "real_high_quality",
        "run_dir": run_dir.as_posix(),
        "script_count": len(scripts),
        "scripts": [path.as_posix() for path in scripts],
        "input_issues": input_issues,
        "expected_outputs": outputs,
        "input_policy": "Replace hw3/assets real input directories; do not edit code.",
        "background_dataset": config["real_chain"]["data"].get("background_dataset", ""),
    }
    if summary["status"] == "READY" and _final_outputs_present(run_dir):
        summary["status"] = "PASS"
        _collect_outputs(run_dir)
    save_json(run_dir / "summary.json", summary)

    if config["real_chain"]["execution"]["mode"] == "run":
        _run_scripts(scripts, logs_dir)
        summary["status"] = "PASS"
        _collect_outputs(run_dir)
        save_json(run_dir / "summary.json", summary)
    return summary


def _validate_real_inputs(config: dict[str, Any], *, require_complete: bool) -> list[str]:
    data = config["real_chain"]["data"]
    issues: list[str] = []
    object_a_images = Path(data["object_a_images"])
    object_a_video = str(data.get("object_a_video", "")).strip()
    if not object_a_images.is_dir() and not object_a_video:
        issues.append(f"Missing object A source: {object_a_images} or real_chain.data.object_a_video")
    if object_a_images.is_dir() and not _image_files(object_a_images):
        issues.append(f"Object A image directory has no PNG/JPG files: {object_a_images}")

    object_c_image = Path(data["object_c_image"])
    if not object_c_image.is_file():
        issues.append(f"Missing object C image: {object_c_image}")

    background_images = Path(data["background_images"])
    background_video = str(data.get("background_video", "")).strip()
    if not background_images.is_dir() and not background_video:
        issues.append(f"Missing background 3DGS source: {background_images} or real_chain.data.background_video")
    if background_images.is_dir() and not _image_files(background_images):
        issues.append(f"Background image directory has no PNG/JPG files: {background_images}")
    if require_complete and issues:
        raise ValueError("; ".join(issues))
    return issues


def _initial_status(config: dict[str, Any], input_issues: list[str]) -> str:
    if config["real_chain"]["execution"]["mode"] == "run":
        return "RUNNING"
    return "READY" if not input_issues else "NEEDS_INPUTS"


def _final_outputs_present(run_dir: Path) -> bool:
    return all((run_dir / relative_path).exists() for relative_path in STRICT_REAL_OUTPUT_FILES)



def _run_scripts(scripts: list[Path], logs_dir: Path) -> None:
    for script in scripts:
        log_path = logs_dir / f"{script.stem}.log"
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(["bash", str(script)], check=True, stdout=log, stderr=subprocess.STDOUT)


def _collect_outputs(run_dir: Path) -> None:
    manifest = {"outputs": expected_outputs(run_dir)}
    for path in run_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".ply", ".obj", ".glb", ".png", ".mp4", ".gif", ".json"}:
            manifest.setdefault("observed", []).append(path.as_posix())
    save_json(run_dir / "real_output_manifest.json", manifest)



def _image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
