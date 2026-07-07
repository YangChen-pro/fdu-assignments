"""Validate Task 1 pipeline outputs."""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
import numpy as np


REAL_REQUIRED_FILES = [
    "source_config.yaml",
    "config.json",
    "summary.json",
    "scripts/00_check_tools.sh",
    "scripts/01_object_a_splatfacto.sh",
    "scripts/02_background_splatfacto.sh",
    "scripts/03_object_b_threestudio.sh",
    "scripts/04_object_c_zero123.sh",
    "scripts/05_export_geometry.sh",
    "scripts/06_render_blender.sh",
]

STRICT_REQUIRED_FILES = [
    "exports/object_a/splat/splat.ply",
    "exports/background/splat/splat.ply",
    "exports/object_b/mesh/model.obj",
    "exports/object_c/mesh/model.obj",
    "renders/fused_splats/fused_scene.mp4",
    "renders/fused_splats/fused_scene_manifest.json",
]

STRICT_ASSET_NAMES = {"background", "object_a", "object_b", "object_c"}
REQUIRED_RENDERER = "gsplat fused splat renderer"
REQUIRED_PIPELINE_MODE = "fused_splats"
BANNED_COMPOSITION_SOURCES = [
    "object_a_cutouts",
    "object_b_test_renders",
    "object_c_test_renders",
    "composite_sprite",
    "test_renders",
    "panel",
]
EXPORT_SOURCE_ROOT = "exports"


def _is_under_root(child: Path, root: Path) -> bool:
    try:
        return child.resolve().is_relative_to(root.resolve())
    except AttributeError:
        child_resolved = child.resolve()
        root_resolved = root.resolve()
        return child_resolved == root_resolved or root_resolved in child_resolved.parents




def _sha256_hex(path: Path) -> str:
    """Return lowercase sha256 digest for a file path."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_source_path(run_dir: Path, source: str) -> list[Path]:
    """Normalize source candidates in and outside run dir."""
    path = Path(source)
    candidates = [path]
    if not path.is_absolute():
        run_candidate = run_dir / source
        if run_candidate.resolve() != path.resolve():
            candidates.append(run_candidate)
    return candidates


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Task1 real-chain run directory.")
    parser.add_argument("--strict-real-outputs", action="store_true", help="Validate trained artifacts and final video.")
    return parser.parse_args()


def main() -> None:
    """Check required real-chain outputs and summary fields."""
    args = parse_args()
    run_dir = Path(args.run_dir)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing Task1 summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stage = str(summary.get("stage", ""))
    if stage != "real_high_quality":
        raise ValueError(f"Unsupported Task1 stage in maintained evaluator: {stage}")
    missing = [name for name in REAL_REQUIRED_FILES if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Task1 output files: {missing}")
    allowed_status = {"PASS"} if args.strict_real_outputs else {"PASS", "READY", "NEEDS_INPUTS"}
    if summary.get("status") not in allowed_status:
        raise ValueError(f"Unexpected summary status: {summary.get('status')}")
    if int(summary.get("script_count", 0)) != 7:
        raise ValueError("Task1 real high-quality chain must generate 7 orchestration scripts.")
    if args.strict_real_outputs:
        validate_strict_real_outputs(run_dir)
    print(
        json.dumps(
            {"status": "PASS", "run_dir": run_dir.as_posix(), "strict_real_outputs": args.strict_real_outputs},
            ensure_ascii=False,
        ),
        flush=True,
    )


def validate_strict_real_outputs(run_dir: Path) -> None:
    """Validate trained assets and the final render."""
    missing = [name for name in STRICT_REQUIRED_FILES if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing strict Task1 real outputs: {missing}")
    min_sizes = {
        "exports/object_a/splat/splat.ply": 1024 * 1024,
        "exports/background/splat/splat.ply": 1024 * 1024,
        "exports/object_b/mesh/model.obj": 1024 * 1024,
        "exports/object_c/mesh/model.obj": 1024 * 1024,
        "renders/fused_splats/fused_scene.mp4": 1024 * 1024,
        "renders/fused_splats/fused_scene_manifest.json": 128,
    }
    small = [name for name, size in min_sizes.items() if (run_dir / name).stat().st_size <= size]
    if small:
        raise ValueError(f"Strict Task1 outputs are unexpectedly small: {small}")
    if not list((run_dir / "object_b_threestudio").glob("**/ckpts/last.ckpt")):
        raise FileNotFoundError("Missing object B SDS checkpoint: object_b_threestudio/**/ckpts/last.ckpt")
    if not list((run_dir / "object_c_zero123").glob("**/ckpts/last.ckpt")):
        raise FileNotFoundError("Missing object C Zero123 checkpoint: object_c_zero123/**/ckpts/last.ckpt")
    manifest_path = run_dir / "renders/fused_splats/fused_scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_videos = _normalize_source_path(run_dir, manifest.get("video", ""))
    manifest_video_candidates = [path for path in manifest_videos if path.exists()]
    if not manifest_video_candidates:
        raise FileNotFoundError(f"Manifest video path missing: {manifest.get('video')}")
    manifest_video = manifest_video_candidates[0]
    if manifest.get("run_dir") and Path(str(manifest.get("run_dir"))).resolve() != run_dir.resolve():
        raise ValueError(f"Manifest run_dir mismatch: {manifest.get('run_dir')}")
    expected = {"width": 1920, "height": 1080, "frames": 144, "fps": 24}
    if {key: manifest.get(key) for key in expected} != expected:
        raise ValueError(f"Unexpected final video manifest fields: {manifest}")
    if manifest.get("renderer") != REQUIRED_RENDERER:
        raise ValueError(f"Strict final render must use renderer '{REQUIRED_RENDERER}', got: {manifest.get('renderer')}")
    if manifest.get("pipeline_mode") != REQUIRED_PIPELINE_MODE:
        raise ValueError(
            f"Strict final render must declare pipeline mode '{REQUIRED_PIPELINE_MODE}', got: {manifest.get('pipeline_mode')}"
        )
    validate_unified_3d_manifest(run_dir, manifest)
    validate_textured_mesh_assets(run_dir, manifest)
    validate_camera_payload(manifest)


def validate_unified_3d_manifest(run_dir: Path, manifest: dict) -> None:
    """Ensure strict final video comes from one unified 3D renderer."""
    source_mode = str(manifest.get("source_mode", "")).lower()
    if source_mode != "unified_3d_assets":
        raise ValueError(f"Strict manifest source_mode must be 'unified_3d_assets', got: {manifest.get('source_mode')}")
    source_text = json.dumps(manifest.get("assets", []), ensure_ascii=False)
    if any(name in source_text.lower() for name in BANNED_COMPOSITION_SOURCES):
        raise ValueError("Strict final render cannot use cutout/test composite source modes.")
    assets = manifest.get("assets", [])
    names = {asset.get("name") for asset in assets if isinstance(asset, dict)}
    if names != STRICT_ASSET_NAMES or len(assets) != len(STRICT_ASSET_NAMES):
        raise ValueError(f"Strict final render assets mismatch: {names}")
    if len(names) != len(assets):
        raise ValueError(f"Strict final render assets must be unique; got duplicate names in {names}")
    missing_sources = []
    for asset in assets:
        source = asset.get("source", "") if isinstance(asset, dict) else ""
        name = asset.get("name", "")
        if not source:
            missing_sources.append(str(asset))
            continue
        candidates = _normalize_source_path(run_dir, source)
        export_root = (run_dir / EXPORT_SOURCE_ROOT)
        existing = [candidate for candidate in candidates if candidate.exists()]
        if not existing:
            missing_sources.append(f"{source}(missing)")
            continue
        if not all(_is_under_root(candidate.resolve(), run_dir) for candidate in existing):
            missing_sources.append(f"{source}(source outside run_dir)")
        in_export = any(_is_under_root(candidate.resolve(), export_root) for candidate in existing)
        if not in_export:
            missing_sources.append(f"{source}(source not under exports)")
        source_lower = source.lower()
        if any(name in source_lower for name in BANNED_COMPOSITION_SOURCES):
            missing_sources.append(f"{source}(forbidden source path)")
        if any(k in source_lower for k in {"final_3dgs_backplate", "composite", "renders", "backplate"}):
            missing_sources.append(f"{source}(likely preview source)")
        if name == "background" and not source_lower.endswith(".ply"):
            missing_sources.append(f"{source}(background source must be PLY)")
        if name == "object_a" and not source_lower.endswith(".ply"):
            missing_sources.append(f"{source}(object_a source must be PLY)")
        if name in {"object_b", "object_c"} and not source_lower.endswith(".obj"):
            missing_sources.append(f"{source}(object_c source must be OBJ)")
        if source_lower.startswith("object_a") and "object_a" not in str(asset.get("name", "")):
            missing_sources.append(f"{source}(asset-name mismatch)")
        if source_lower.startswith("object_b") and "object_b" not in str(asset.get("name", "")):
            missing_sources.append(f"{source}(asset-name mismatch)")
        if source_lower.startswith("object_c") and "object_c" not in str(asset.get("name", "")):
            missing_sources.append(f"{source}(asset-name mismatch)")
        if source_lower.startswith("background") and asset.get("name") != "background":
            missing_sources.append(f"{source}(asset-name mismatch)")
        if source_lower and asset.get("name") == "object_a" and "object_a" not in source_lower and "splat.ply" not in source_lower:
            missing_sources.append(f"{source}(object_a source should be object_a splat)")
        if asset.get("name") == "background" and "background" not in source_lower and "splat.ply" not in source_lower:
            missing_sources.append(f"{source}(background source should be background splat)")
        if asset.get("name") == "object_b" and "obj" not in source_lower and "mesh" not in source_lower:
            missing_sources.append(f"{source}(object_b source should be mesh obj)")
        if asset.get("name") == "object_c" and "obj" not in source_lower and "mesh" not in source_lower:
            missing_sources.append(f"{source}(object_c source should be mesh obj)")
        hash_map = manifest.get("asset_hashes") or {}
        if hash_map:
            expected_hash = hash_map.get(name)
            if not expected_hash:
                missing_sources.append(f"{name}(asset hash missing)")
            else:
                actual_hash = _sha256_hex(existing[0])
                if expected_hash.lower() != actual_hash:
                    missing_sources.append(f"{source}(asset hash mismatch)")
    if missing_sources:
        raise FileNotFoundError(f"Strict final render source files missing: {missing_sources}")


def validate_textured_mesh_assets(run_dir: Path, manifest: dict) -> None:
    """Require B/C to preserve UV textures through export and strict rendering."""
    assets = {
        asset.get("name"): asset
        for asset in manifest.get("assets", [])
        if isinstance(asset, dict)
    }
    failures: list[str] = []
    texture_hashes = manifest.get("texture_hashes") or {}
    for name in ("object_b", "object_c"):
        asset = assets.get(name)
        if not asset:
            failures.append(f"{name}(missing asset)")
            continue
        if asset.get("color_mode") != "uv_texture":
            failures.append(f"{name}(strict rendering requires uv_texture color mode)")
        texture_source = str(asset.get("texture_source") or "")
        if not texture_source:
            failures.append(f"{name}(missing texture_source)")
            continue
        texture_candidates = [
            candidate
            for candidate in _normalize_source_path(run_dir, texture_source)
            if candidate.exists()
        ]
        if not texture_candidates:
            failures.append(f"{name}(texture file missing)")
            continue
        texture_path = texture_candidates[0]
        if not _is_under_root(texture_path, run_dir / EXPORT_SOURCE_ROOT):
            failures.append(f"{name}(texture outside exports)")
        obj_candidates = [
            candidate
            for candidate in _normalize_source_path(run_dir, str(asset.get("source") or ""))
            if candidate.exists()
        ]
        if not obj_candidates:
            failures.append(f"{name}(OBJ source missing)")
            continue
        material_files = sorted(obj_candidates[0].parent.glob("*.mtl"))
        if not material_files:
            failures.append(f"{name}(MTL missing)")
        elif not any("map_Kd" in path.read_text(encoding="utf-8", errors="ignore") for path in material_files):
            failures.append(f"{name}(MTL has no map_Kd texture)")
        expected_hash = texture_hashes.get(name)
        if expected_hash and expected_hash.lower() != _sha256_hex(texture_path):
            failures.append(f"{name}(texture hash mismatch)")
    if failures:
        raise ValueError(f"Strict textured mesh validation failed: {failures}")


def validate_camera_payload(manifest: dict) -> None:
    """Validate camera metadata for strict outputs."""
    camera = manifest.get("camera")
    if not isinstance(camera, dict):
        raise ValueError("Strict final render must include camera metadata in manifest.")
    mode = str(camera.get("name") or camera.get("mode") or "").lower()
    if mode not in {"orbit", "stabilized_orbit", "background_trajectory", "foreground_orbit"}:
        raise ValueError(f"Strict final render must use a valid camera mode, got: {camera.get('name')}")
    if mode == "background_trajectory":
        centers = camera.get("centers")
        targets = camera.get("targets")
        if not (centers and isinstance(centers, list) and targets and isinstance(targets, list)):
            raise ValueError("Background trajectory mode must record centers and targets arrays.")
        if len(centers) != len(targets):
            raise ValueError("Background trajectory mode must keep centers and targets same length.")
        if len(centers) == 0:
            raise ValueError("Background trajectory mode must include at least one frame.")
    required = {"focal_scale"}
    if mode in {"orbit", "stabilized_orbit", "foreground_orbit"}:
        required.update({"center", "radius", "height", "target_z"})
    missing = [field for field in required if field not in camera]
    if missing:
        raise ValueError(f"Strict final render camera metadata missing required fields: {missing}")
    focal_scale = float(camera.get("focal_scale", 0.0))
    if focal_scale <= 0 or not np.isfinite(focal_scale):
        raise ValueError(f"Strict final render camera focal_scale must be a positive finite number, got: {camera.get('focal_scale')}")


if __name__ == "__main__":
    main()
