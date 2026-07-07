"""Render HW3 Task1 fusion with the real trained 3D artifacts.

This renderer avoids Blender point-cloud proxies for A/background. It renders
the exported Nerfstudio gaussian-splat PLYs directly with gsplat, and converts
the B/C exported colored OBJ meshes into small surface splats so all four
assets can be composited in a single camera path.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional
from typing import Any

try:
    import cv2
except Exception:
    cv2 = None
try:
    from PIL import Image
except Exception:
    Image = None
import numpy as np
import torch
from gsplat.rendering import rasterization

TASK1_ROOT = Path(__file__).resolve().parents[1]
if str(TASK1_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK1_ROOT))

from task1_3dgs_aigc.mesh_splats import (
    apply_object_b_crystal_style,
    normal_aligned_quaternions,
    quaternion_normals,
    sample_textured_obj,
    surface_splat_scales,
)
from task1_3dgs_aigc.camera_geometry import (
    apply_similarity_to_camera_poses,
)
from task1_3dgs_aigc.gaussian_filtering import (
    color_anchor_foreground_mask,
    dampen_high_luminance_colors,
    spatial_core_foreground_mask,
)

SH_C0 = 0.28209479177387814
RENDERER_NAME = "gsplat fused splat renderer"
PIPELINE_MODE = "fused_splats"
DEFAULT_LIGHT_DIR = np.array([0.38, -0.20, 0.90], dtype=np.float32)
DEFAULT_LIGHT_DIR = DEFAULT_LIGHT_DIR / np.linalg.norm(DEFAULT_LIGHT_DIR)
AMBIENT = 0.36
DIFFUSE = 0.74
FOCAL_SCALE_MIN = 0.7
FOCAL_SCALE_MAX = 1.5
KEYFRAME_INDICES = (0, 47, 143)
PALETTE_COLORS = {
    "object_a": np.array([0.92, 0.89, 0.86], dtype=np.float32),
    "object_b": np.array([0.69, 0.53, 0.93], dtype=np.float32),
    "object_c": np.array([0.72, 0.66, 0.56], dtype=np.float32),
    "background": np.array([0.75, 0.74, 0.73], dtype=np.float32),
}


@dataclass
class SplatAsset:
    name: str
    means: np.ndarray
    quats: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray
    colors: np.ndarray
    normals: Optional[np.ndarray]
    source: str
    texture_source: Optional[str] = None
    color_mode: str = "trained"
    normalization_scale: float = 1.0
    normalization_offset: Optional[np.ndarray] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--frames", type=int, default=144)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--background-max", type=int, default=650_000)
    parser.add_argument("--object-a-max", type=int, default=450_000)
    parser.add_argument("--mesh-max", type=int, default=260_000)
    parser.add_argument("--object-a-opacity-quantile", type=float, default=0.20)
    parser.add_argument("--object-a-opacity-mult", type=float, default=1.50)
    parser.add_argument("--object-a-scale-mult", type=float, default=0.80)
    parser.add_argument("--object-b-max", type=int, default=260_000)
    parser.add_argument("--object-c-max", type=int, default=300_000)
    parser.add_argument("--object-b-scale-max", type=float, default=0.0130)
    parser.add_argument("--object-b-base-scale", type=float, default=0.0038)
    parser.add_argument("--object-c-scale-max", type=float, default=0.0110)
    parser.add_argument("--object-c-base-scale", type=float, default=0.0032)
    parser.add_argument("--camera-radius-scale", type=float, default=0.82)
    parser.add_argument("--camera-height-scale", type=float, default=0.86)
    parser.add_argument("--radius-clip", type=float, default=0.085)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--camera-orbit-eccentricity", type=float, default=0.85)
    parser.add_argument("--camera-radial-sweep", type=float, default=0.10)
    parser.add_argument("--camera-height-sweep", type=float, default=0.18)
    parser.add_argument("--camera-focus", choices=("background", "foreground"), default="background")
    parser.add_argument("--camera-focal-multiplier", type=float, default=1.0)
    parser.add_argument("--camera-index-start", type=int, default=0)
    parser.add_argument("--camera-index-stop", type=int, default=0)
    parser.add_argument("--foreground-camera-radius", type=float, default=1.75)
    parser.add_argument("--foreground-camera-height", type=float, default=0.92)
    parser.add_argument("--foreground-camera-target-z", type=float, default=0.34)
    parser.add_argument("--foreground-camera-focal-scale", type=float, default=1.20)
    parser.add_argument("--foreground-camera-start-degrees", type=float, default=18.0)
    parser.add_argument("--foreground-camera-arc-degrees", type=float, default=54.0)
    parser.add_argument("--foreground-offset-x", type=float, default=0.0)
    parser.add_argument("--foreground-offset-y", type=float, default=0.0)
    parser.add_argument("--foreground-ground-offset", type=float, default=0.0)
    parser.add_argument("--object-separation", type=float, default=0.70)
    parser.add_argument("--object-a-height", type=float, default=0.75)
    parser.add_argument("--object-b-height", type=float, default=0.90)
    parser.add_argument("--object-c-height", type=float, default=0.58)
    parser.add_argument("--object-c-target-extent", type=float, default=0.0)
    parser.add_argument("--object-a-offset-x", type=float, default=None)
    parser.add_argument("--object-a-offset-y", type=float, default=None)
    parser.add_argument("--object-b-offset-x", type=float, default=None)
    parser.add_argument("--object-b-offset-y", type=float, default=None)
    parser.add_argument("--object-c-offset-x", type=float, default=None)
    parser.add_argument("--object-c-offset-y", type=float, default=None)
    parser.add_argument("--background-clear-width", type=float, default=0.0)
    parser.add_argument("--background-clear-depth", type=float, default=0.0)
    parser.add_argument("--background-clear-height", type=float, default=0.0)
    parser.add_argument("--background-clear-below", type=float, default=0.06)
    parser.add_argument("--background-clear-surface-keep", type=float, default=0.0)
    parser.add_argument("--background-clear-shape", choices=("rect", "ellipse"), default="rect")
    parser.add_argument("--support-mat-points", type=int, default=0)
    parser.add_argument("--support-mat-width", type=float, default=0.0)
    parser.add_argument("--support-mat-depth", type=float, default=0.0)
    parser.add_argument("--support-mat-thickness", type=float, default=0.045)
    parser.add_argument("--support-mat-opacity", type=float, default=0.98)
    parser.add_argument("--support-mat-shape", choices=("rect", "ellipse"), default="rect")
    parser.add_argument("--support-mat-color", choices=("dark", "wood"), default="dark")
    parser.add_argument("--foreground-color-boost", type=float, default=1.0)
    parser.add_argument("--foreground-saturation-boost", type=float, default=1.0)
    parser.add_argument("--foreground-opacity-boost", type=float, default=1.0)
    parser.add_argument("--foreground-rim-strength", type=float, default=0.0)
    parser.add_argument("--foreground-rim-power", type=float, default=2.0)
    parser.add_argument("--object-a-color-boost", type=float, default=1.0)
    parser.add_argument("--object-a-opacity-boost", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    assets = build_assets(args, rng)
    camera_cfg = build_camera_config(
        args.run_dir,
        assets,
        total_frames=args.frames,
        radius_scale=args.camera_radius_scale,
        height_scale=args.camera_height_scale,
        orbit_eccentricity=args.camera_orbit_eccentricity,
        radial_sweep=args.camera_radial_sweep,
        height_sweep=args.camera_height_sweep,
        focus_mode=args.camera_focus,
        camera_index_start=args.camera_index_start,
        camera_index_stop=args.camera_index_stop,
        foreground_radius=args.foreground_camera_radius,
        foreground_height=args.foreground_camera_height,
        foreground_target_z=args.foreground_camera_target_z,
        foreground_focal_scale=args.foreground_camera_focal_scale,
        foreground_start_degrees=args.foreground_camera_start_degrees,
        foreground_arc_degrees=args.foreground_camera_arc_degrees,
    )
    camera_cfg["focal_scale"] = float(camera_cfg.get("focal_scale", 0.92)) * float(args.camera_focal_multiplier)
    camera_cfg["radius_clip"] = float(args.radius_clip)
    camera_cfg["gamma"] = float(args.gamma)
    camera_cfg["foreground_color_boost"] = float(args.foreground_color_boost)
    camera_cfg["foreground_saturation_boost"] = float(args.foreground_saturation_boost)
    camera_cfg["foreground_opacity_boost"] = float(args.foreground_opacity_boost)
    camera_cfg["foreground_rim_strength"] = float(args.foreground_rim_strength)
    camera_cfg["foreground_rim_power"] = float(args.foreground_rim_power)
    camera_cfg["object_a_color_boost"] = float(args.object_a_color_boost)
    camera_cfg["object_a_opacity_boost"] = float(args.object_a_opacity_boost)
    tensors = concat_assets(assets, args.device)
    render_sequence(args, frames_dir, tensors, camera_cfg)
    video_path = args.output_dir / "fused_scene.mp4"
    encode_video(frames_dir, video_path, args.fps)
    export_strict_keyframes(frames_dir, args.output_dir)
    manifest = build_manifest(args, assets, video_path, time.time() - start, camera_cfg)
    (args.output_dir / "fused_scene_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def build_assets(args: argparse.Namespace, rng: np.random.Generator) -> list[SplatAsset]:
    """Load and normalize the four trained Task1 assets."""
    background_spec = _gaussian_spec(
        name="background",
        relative_path="exports/background/splat/splat.ply",
        max_points=args.background_max,
        opacity_quantile=0.35,
        crop_percentile=(1.5, 98.5),
        target_height=4.6,
        xy=(0.0, 0.0),
        ground_z=0.0,
        robust_percentile=(3, 97),
        scale_multiplier=1.0,
        scale_max=0.065,
        apply_dataparser=False,
        dataparser_target="background",
        seed=int(rng.integers(0, 2**31)),
    )
    background = _load_normalized_asset(args.run_dir, background_spec)
    background_trajectory = load_background_camera_trajectory(
        args.run_dir,
        background,
        total_frames=1,
        target_width=args.width,
    )
    if background_trajectory is not None:
        placement_center = np.asarray(background_trajectory["focus"], dtype=np.float32)
        object_ground = float(placement_center[2] - 0.45)
    else:
        placement_center = np.percentile(background.means, 50, axis=0)
        object_ground = float(np.percentile(background.means, 35, axis=0)[2])
    placement_center = placement_center.copy()
    placement_center[0] += float(args.foreground_offset_x)
    placement_center[1] += float(args.foreground_offset_y)
    object_ground += float(args.foreground_ground_offset)
    separation = float(args.object_separation)
    if args.background_clear_width > 0 and args.background_clear_depth > 0 and args.background_clear_height > 0:
        background = clear_background_support_volume(
            background,
            center_xy=(float(placement_center[0]), float(placement_center[1])),
            ground_z=object_ground,
            width=float(args.background_clear_width),
            depth=float(args.background_clear_depth),
            height=float(args.background_clear_height),
            below=float(args.background_clear_below),
            surface_keep=args.background_clear_surface_keep,
            shape=args.background_clear_shape,
        )
    if args.support_mat_points > 0 and args.support_mat_width > 0 and args.support_mat_depth > 0:
        background = add_support_mat_to_background(
            background,
            center_xy=(float(placement_center[0]), float(placement_center[1])),
            ground_z=object_ground,
            width=float(args.support_mat_width),
            depth=float(args.support_mat_depth),
            thickness=float(args.support_mat_thickness),
            point_count=int(args.support_mat_points),
            opacity=float(args.support_mat_opacity),
            shape=args.support_mat_shape,
            color=args.support_mat_color,
            seed=int(rng.integers(0, 2**31)),
        )
    object_a_xy = (
        placement_center[0] + (args.object_a_offset_x if args.object_a_offset_x is not None else -separation),
        placement_center[1] + (args.object_a_offset_y if args.object_a_offset_y is not None else -0.02),
    )
    object_b_xy = (
        placement_center[0] + (args.object_b_offset_x if args.object_b_offset_x is not None else 0.0),
        placement_center[1] + (args.object_b_offset_y if args.object_b_offset_y is not None else 0.0),
    )
    object_c_xy = (
        placement_center[0] + (args.object_c_offset_x if args.object_c_offset_x is not None else separation),
        placement_center[1] + (args.object_c_offset_y if args.object_c_offset_y is not None else 0.02),
    )

    # Place foreground assets around the recovered training-camera focus on the support surface.
    obj_specs = [
        _gaussian_spec(
            name="object_a",
            relative_path="exports/object_a/splat/splat.ply",
            max_points=args.object_a_max,
            opacity_quantile=args.object_a_opacity_quantile,
            crop_percentile=(2, 98),
            target_height=args.object_a_height,
            xy=object_a_xy,
            ground_z=object_ground,
            robust_percentile=(5, 95),
            scale_multiplier=args.object_a_scale_mult,
            scale_max=0.028,
            opacity_multiplier=args.object_a_opacity_mult,
            apply_dataparser=False,
            dataparser_target="object_a",
            seed=int(rng.integers(0, 2**31)),
        ),
        _mesh_spec(
            name="object_b",
            relative_path="exports/object_b/mesh/model.obj",
            max_points=args.object_b_max,
            base_scale=args.object_b_base_scale,
            scale_cap=0.013,
            scale_max=args.object_b_scale_max,
            target_height=args.object_b_height,
            xy=object_b_xy,
            ground_z=object_ground,
            seed=int(rng.integers(0, 2**31)),
        ),
        _mesh_spec(
            name="object_c",
            relative_path="exports/object_c/mesh/model.obj",
            max_points=args.object_c_max,
            base_scale=args.object_c_base_scale,
            scale_cap=0.011,
            scale_max=args.object_c_scale_max,
            target_height=args.object_c_height,
            target_extent=args.object_c_target_extent,
            xy=object_c_xy,
            ground_z=object_ground + 0.02,
            seed=int(rng.integers(0, 2**31)),
        ),
    ]
    return [background] + [_load_normalized_asset(args.run_dir, spec, reference=background) for spec in obj_specs]


def clear_background_support_volume(
    asset: SplatAsset,
    *,
    center_xy: tuple[float, float],
    ground_z: float,
    width: float,
    depth: float,
    height: float,
    below: float,
    surface_keep: float,
    shape: str,
) -> SplatAsset:
    """Remove background clutter inside the support footprint before fusion."""
    if width <= 0 or depth <= 0 or height <= 0:
        return asset
    x0, y0 = center_xy
    if shape == "ellipse":
        norm_x = (asset.means[:, 0] - x0) / max(width * 0.5, 1e-6)
        norm_y = (asset.means[:, 1] - y0) / max(depth * 0.5, 1e-6)
        in_xy = norm_x * norm_x + norm_y * norm_y <= 1.0
    else:
        in_xy = (
            (np.abs(asset.means[:, 0] - x0) <= width * 0.5)
            & (np.abs(asset.means[:, 1] - y0) <= depth * 0.5)
        )
    lower_z = ground_z - max(0.0, below)
    if surface_keep > 0:
        lower_z = max(lower_z, ground_z + float(surface_keep))
    in_z = (asset.means[:, 2] >= lower_z) & (
        asset.means[:, 2] <= ground_z + height
    )
    remove = in_xy & in_z
    if not np.any(remove):
        return asset
    keep = ~remove
    print(
        json.dumps(
            {
                "background_support_clear": {
                    "removed_points": int(remove.sum()),
                    "remaining_points": int(keep.sum()),
                    "center_xy": [float(x0), float(y0)],
                    "ground_z": float(ground_z),
                    "width": float(width),
                    "depth": float(depth),
                    "height": float(height),
                    "surface_keep": float(surface_keep),
                    "shape": shape,
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return SplatAsset(
        name=asset.name,
        means=asset.means[keep],
        quats=asset.quats[keep],
        scales=asset.scales[keep],
        opacities=asset.opacities[keep],
        colors=asset.colors[keep],
        normals=asset.normals[keep] if asset.normals is not None else None,
        source=asset.source,
        texture_source=asset.texture_source,
        color_mode=asset.color_mode,
        normalization_scale=asset.normalization_scale,
        normalization_offset=asset.normalization_offset,
    )


def add_support_mat_to_background(
    asset: SplatAsset,
    *,
    center_xy: tuple[float, float],
    ground_z: float,
    width: float,
    depth: float,
    thickness: float,
    point_count: int,
    opacity: float,
    shape: str,
    color: str,
    seed: int,
) -> SplatAsset:
    """Append a thin procedural mat to the background as a local support surface."""
    if point_count <= 0 or width <= 0 or depth <= 0:
        return asset
    rng = np.random.default_rng(seed)
    point_count = max(256, int(point_count))
    top_count = int(point_count * 0.86)
    side_count = point_count - top_count
    x0, y0 = center_xy
    top_z = float(ground_z - max(thickness, 0.0) * 0.22)
    shape = shape if shape in {"rect", "ellipse"} else "rect"
    color = color if color in {"dark", "wood"} else "dark"
    if shape == "ellipse":
        top_angles = rng.uniform(0.0, 2.0 * math.pi, size=top_count)
        top_radii = np.sqrt(rng.random(size=top_count))
        local_top_x = np.cos(top_angles) * top_radii * width * 0.5
        local_top_y = np.sin(top_angles) * top_radii * depth * 0.5
    else:
        local_top_x = rng.uniform(-width * 0.5, width * 0.5, size=top_count)
        local_top_y = rng.uniform(-depth * 0.5, depth * 0.5, size=top_count)
    xs = local_top_x + x0
    ys = local_top_y + y0
    zs = rng.normal(loc=top_z, scale=max(thickness, 1e-4) * 0.015, size=top_count)
    top_points = np.column_stack([xs, ys, zs]).astype(np.float32)
    top_normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (top_count, 1))

    side_points: list[np.ndarray] = []
    side_normals: list[np.ndarray] = []
    if shape == "ellipse" and side_count > 0:
        side_angles = rng.uniform(0.0, 2.0 * math.pi, size=side_count)
        local_side_x = np.cos(side_angles) * width * 0.5
        local_side_y = np.sin(side_angles) * depth * 0.5
        x = (x0 + local_side_x).astype(np.float32)
        y = (y0 + local_side_y).astype(np.float32)
        z = rng.uniform(top_z - max(thickness, 0.001), top_z, size=side_count).astype(np.float32)
        normals = np.column_stack(
            [
                local_side_x / max(width * 0.5, 1e-6),
                local_side_y / max(depth * 0.5, 1e-6),
                np.zeros(side_count, dtype=np.float32),
            ]
        ).astype(np.float32)
        normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.where(normal_lengths < 1e-6, 1.0, normal_lengths)
        side_points.append(np.column_stack([x, y, z]).astype(np.float32))
        side_normals.append(normals.astype(np.float32))
    elif side_count > 0:
        edges = [
            ("x-", -width * 0.5, np.array([-1.0, 0.0, 0.0], dtype=np.float32)),
            ("x+", width * 0.5, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            ("y-", -depth * 0.5, np.array([0.0, -1.0, 0.0], dtype=np.float32)),
            ("y+", depth * 0.5, np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        ]
        for edge_index, (edge_name, offset, normal) in enumerate(edges):
            del edge_name
            n = side_count // len(edges) + (1 if edge_index < side_count % len(edges) else 0)
            if n <= 0:
                continue
            if normal[0] != 0:
                x = np.full(n, x0 + offset, dtype=np.float32)
                y = rng.uniform(-depth * 0.5, depth * 0.5, size=n).astype(np.float32) + y0
            else:
                x = rng.uniform(-width * 0.5, width * 0.5, size=n).astype(np.float32) + x0
                y = np.full(n, y0 + offset, dtype=np.float32)
            z = rng.uniform(top_z - max(thickness, 0.001), top_z, size=n).astype(np.float32)
            side_points.append(np.column_stack([x, y, z]).astype(np.float32))
            side_normals.append(np.tile(normal, (n, 1)))
    if side_points:
        mat_points = np.concatenate([top_points] + side_points, axis=0)
        mat_normals = np.concatenate([top_normals] + side_normals, axis=0)
    else:
        mat_points = top_points
        mat_normals = top_normals

    tangent = float(np.clip(math.sqrt((width * depth) / max(top_count, 1)) * 2.3, 0.008, 0.035))
    mat_scales = surface_splat_scales(
        point_count=mat_points.shape[0],
        tangent_scale=np.full(mat_points.shape[0], tangent, dtype=np.float32),
        thickness_ratio=0.12,
    ).astype(np.float32)
    mat_quats = normal_aligned_quaternions(mat_normals).astype(np.float32)
    if color == "wood":
        norm_x = local_top_x / max(width * 0.5, 1e-6)
        norm_y = local_top_y / max(depth * 0.5, 1e-6)
        radial = np.clip(np.sqrt(norm_x * norm_x + norm_y * norm_y), 0.0, 1.0)
        angles = np.arctan2(norm_y, norm_x)
        slats = 0.5 + 0.5 * np.sin(angles * 30.0)
        grain = 0.5 + 0.5 * np.sin(local_top_x * 22.0 + local_top_y * 7.0)
        shade = (0.90 + 0.08 * slats + 0.04 * grain - 0.06 * radial).astype(np.float32)
        base_color = np.array([0.60, 0.51, 0.39], dtype=np.float32)
        edge_tint = np.array([0.38, 0.30, 0.22], dtype=np.float32)
        top_colors = np.clip(
            base_color[None, :] * shade[:, None]
            + rng.normal(0.0, 0.025, size=(top_count, 3)).astype(np.float32),
            0.20,
            0.72,
        )
        side_colors = np.clip(
            edge_tint[None, :]
            + rng.normal(0.0, 0.018, size=(mat_points.shape[0] - top_count, 3)).astype(np.float32),
            0.14,
            0.48,
        )
    else:
        base_color = np.array([0.12, 0.15, 0.145], dtype=np.float32)
        edge_tint = np.array([0.06, 0.075, 0.07], dtype=np.float32)
        top_colors = np.clip(
            base_color[None, :] + rng.normal(0.0, 0.018, size=(top_count, 3)).astype(np.float32),
            0.03,
            0.26,
        )
        side_colors = np.clip(
            edge_tint[None, :]
            + rng.normal(0.0, 0.012, size=(mat_points.shape[0] - top_count, 3)).astype(np.float32),
            0.02,
            0.18,
        )
    mat_colors = np.concatenate([top_colors, side_colors], axis=0).astype(np.float32)
    if color == "wood" and shape == "ellipse":
        top_opacities = np.clip(
            opacity * (0.18 + 0.82 * (1.0 - np.power(radial, 1.7))),
            0.0,
            1.0,
        ).astype(np.float32)
        side_opacities = np.full(
            mat_points.shape[0] - top_count,
            np.clip(opacity * 0.12, 0.0, 1.0),
            dtype=np.float32,
        )
        mat_opacities = np.concatenate([top_opacities, side_opacities], axis=0).astype(np.float32)
    else:
        mat_opacities = np.full(mat_points.shape[0], np.clip(opacity, 0.0, 1.0), dtype=np.float32)
    background_normals = (
        np.zeros_like(asset.means, dtype=np.float32)
        if asset.normals is None
        else asset.normals.astype(np.float32)
    )
    print(
        json.dumps(
            {
                "background_support_mat": {
                    "points": int(mat_points.shape[0]),
                    "center_xy": [float(x0), float(y0)],
                    "top_z": float(top_z),
                    "width": float(width),
                    "depth": float(depth),
                    "thickness": float(thickness),
                    "shape": shape,
                    "color": color,
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return SplatAsset(
        name=asset.name,
        means=np.concatenate([asset.means, mat_points], axis=0).astype(np.float32),
        quats=np.concatenate([asset.quats, mat_quats], axis=0).astype(np.float32),
        scales=np.concatenate([asset.scales, mat_scales], axis=0).astype(np.float32),
        opacities=np.concatenate([asset.opacities, mat_opacities], axis=0).astype(np.float32),
        colors=np.concatenate([asset.colors, mat_colors], axis=0).astype(np.float32),
        normals=np.concatenate([background_normals, mat_normals], axis=0).astype(np.float32),
        source=asset.source,
        texture_source=asset.texture_source,
        color_mode=asset.color_mode,
        normalization_scale=asset.normalization_scale,
        normalization_offset=asset.normalization_offset,
    )



def _gaussian_spec(
    name: str,
    relative_path: str,
    max_points: int,
    opacity_quantile: float,
    crop_percentile: tuple[float, float],
    target_height: float,
    xy: tuple[float, float],
    ground_z: float,
    robust_percentile: tuple[float, float],
    scale_multiplier: float,
    scale_max: float,
    opacity_multiplier: float = 1.0,
    *,
    apply_dataparser: bool = False,
    dataparser_target: str = "",
    seed: int = 0,
) -> dict[str, Any]:
    return locals()


def _mesh_spec(
    name: str,
    relative_path: str,
    max_points: int,
    base_scale: float,
    scale_max: float,
    scale_cap: float,
    target_height: float,
    xy: tuple[float, float],
    ground_z: float,
    seed: int = 0,
    target_extent: float = 0.0,
) -> dict[str, Any]:
    spec = locals()
    spec.update({"robust_percentile": (1, 99), "scale_multiplier": 1.0})
    spec["apply_dataparser"] = False
    spec["dataparser_target"] = ""
    return spec


def _load_normalized_asset(run_dir: Path, spec: dict, reference: SplatAsset | None = None) -> SplatAsset:
    path = run_dir / spec["relative_path"]
    if "base_scale" in spec:
        asset = load_obj_as_splats(
            path,
            spec["name"],
            max_points=spec["max_points"],
            base_scale=spec["base_scale"],
            scale_cap=spec["scale_cap"],
            seed=spec.get("seed", 0),
        )
    else:
        asset = load_gaussian_ply(
            path, spec["name"], max_points=spec["max_points"],
            opacity_quantile=spec["opacity_quantile"], crop_percentile=spec["crop_percentile"],
        )
        if spec["name"] == "object_a":
            asset = filter_object_a_foreground(asset)
            asset = upright_object_a_to_world_z(asset)
            asset = dampen_object_a_highlights(asset)
    opacity_multiplier = float(spec.get("opacity_multiplier", 1.0))
    asset.opacities = np.clip(asset.opacities * opacity_multiplier, 0.0, 1.0)
    if spec.get("apply_dataparser", False):
        asset = apply_dataparser_transform(
            run_dir,
            asset,
            dataparser_target=spec.get("dataparser_target", ""),
        )
    return normalize_asset(
        asset,
        target_height=spec["target_height"],
        target_extent=spec.get("target_extent", 0.0),
        xy=spec["xy"],
        ground_z=spec["ground_z"],
        robust_percentile=spec["robust_percentile"],
        scale_multiplier=spec["scale_multiplier"],
        scale_max=spec["scale_max"],
        reference=reference,
    )


def filter_object_a_foreground(asset: SplatAsset) -> SplatAsset:
    keep, report = color_anchor_foreground_mask(
        asset.means,
        asset.colors,
        asset.opacities,
    )
    print(json.dumps({"object_a_foreground_filter": report}, ensure_ascii=False), flush=True)
    if report.get("applied"):
        return filter_splat_asset(asset, keep)

    keep, report = spatial_core_foreground_mask(asset.means, asset.opacities)
    print(json.dumps({"object_a_spatial_filter": report}, ensure_ascii=False), flush=True)
    if report.get("applied"):
        return filter_splat_asset(asset, keep)
    return asset


def filter_splat_asset(asset: SplatAsset, keep: np.ndarray) -> SplatAsset:
    return SplatAsset(
        name=asset.name,
        means=asset.means[keep],
        quats=asset.quats[keep],
        scales=asset.scales[keep],
        opacities=asset.opacities[keep],
        colors=asset.colors[keep],
        normals=asset.normals[keep] if asset.normals is not None else None,
        source=asset.source,
        texture_source=asset.texture_source,
        color_mode=asset.color_mode,
        normalization_scale=asset.normalization_scale,
        normalization_offset=asset.normalization_offset,
    )


def upright_object_a_to_world_z(asset: SplatAsset, *, min_angle_degrees: float = 3.0) -> SplatAsset:
    """Rotate object A's dominant axis upright before scene placement."""
    if asset.means.shape[0] < 16:
        print(
            json.dumps(
                {"object_a_upright": {"applied": False, "reason": "too_few_points"}},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return asset
    center = np.median(asset.means, axis=0).astype(np.float32)
    centered = asset.means - center[None, :]
    distances = np.linalg.norm(centered, axis=1)
    core = distances <= np.quantile(distances, 0.92)
    core_points = centered[core]
    if core_points.shape[0] < 16:
        core_points = centered
    covariance = np.cov(core_points.T)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        print(
            json.dumps(
                {"object_a_upright": {"applied": False, "reason": "invalid_covariance"}},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return asset
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))].astype(np.float32)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        print(
            json.dumps(
                {"object_a_upright": {"applied": False, "reason": "degenerate_axis"}},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return asset
    axis = axis / axis_norm
    if axis[2] < 0:
        axis = -axis
    target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    cos_angle = float(np.clip(np.dot(axis, target), -1.0, 1.0))
    angle_degrees = math.degrees(math.acos(cos_angle))
    if angle_degrees < min_angle_degrees:
        print(
            json.dumps(
                {
                    "object_a_upright": {
                        "applied": False,
                        "reason": "already_upright",
                        "angle_degrees": angle_degrees,
                        "axis": axis.astype(float).tolist(),
                    }
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return asset

    rotation = rotation_between_vectors(axis, target)
    means = ((asset.means - center[None, :]) @ rotation.T + center[None, :]).astype(np.float32)
    quats = rotate_quaternions(asset.quats, rotation)
    normals = (
        (asset.normals @ rotation.T).astype(np.float32)
        if asset.normals is not None
        else None
    )
    print(
        json.dumps(
            {
                "object_a_upright": {
                    "applied": True,
                    "angle_degrees": angle_degrees,
                    "axis": axis.astype(float).tolist(),
                    "target_axis": target.astype(float).tolist(),
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return SplatAsset(
        name=asset.name,
        means=means,
        quats=quats,
        scales=asset.scales,
        opacities=asset.opacities,
        colors=asset.colors,
        normals=normals,
        source=asset.source,
        texture_source=asset.texture_source,
        color_mode=asset.color_mode,
        normalization_scale=asset.normalization_scale,
        normalization_offset=asset.normalization_offset,
    )


def dampen_object_a_highlights(asset: SplatAsset) -> SplatAsset:
    colors, report = dampen_high_luminance_colors(
        asset.colors,
        luminance_quantile=0.84,
        target_luminance=0.74,
        blend=0.75,
        max_channel=0.84,
    )
    print(json.dumps({"object_a_highlight_damping": report}, ensure_ascii=False), flush=True)
    if not report.get("applied"):
        return asset
    return SplatAsset(
        name=asset.name,
        means=asset.means,
        quats=asset.quats,
        scales=asset.scales,
        opacities=asset.opacities,
        colors=colors,
        normals=asset.normals,
        source=asset.source,
        texture_source=asset.texture_source,
        color_mode=asset.color_mode,
        normalization_scale=asset.normalization_scale,
        normalization_offset=asset.normalization_offset,
    )


def rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    source = source / max(float(np.linalg.norm(source)), 1e-6)
    target = target / max(float(np.linalg.norm(target)), 1e-6)
    cross = np.cross(source, target)
    sin_angle = float(np.linalg.norm(cross))
    cos_angle = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sin_angle < 1e-7:
        if cos_angle > 0:
            return np.eye(3, dtype=np.float32)
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(source, fallback))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        cross = np.cross(source, fallback)
        cross = cross / max(float(np.linalg.norm(cross)), 1e-6)
        return rotation_matrix_from_axis_angle(cross, math.pi)
    axis = cross / sin_angle
    return rotation_matrix_from_axis_angle(axis, math.atan2(sin_angle, cos_angle))


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = (axis / max(float(np.linalg.norm(axis)), 1e-6)).astype(np.float32)
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float32,
    )


def rotate_quaternions(quats: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    rotation_quat = quaternion_from_rotation_matrix(rotation)
    rotated = quaternion_multiply(
        np.repeat(rotation_quat[None, :], quats.shape[0], axis=0),
        quats,
    )
    return normalize_quats(rotated)


def quaternion_from_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    m = np.asarray(rotation, dtype=np.float32)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(max(1.0 + float(m[0, 0] - m[1, 1] - m[2, 2]), 1e-8)) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(max(1.0 + float(m[1, 1] - m[0, 0] - m[2, 2]), 1e-8)) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + float(m[2, 2] - m[0, 0] - m[1, 1]), 1e-8)) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return normalize_quats(np.array([[w, x, y, z]], dtype=np.float32))[0]


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left.T
    w2, x2, y2, z2 = right.T
    return np.column_stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    ).astype(np.float32)


def apply_dataparser_transform(run_dir: Path, asset: SplatAsset, *, dataparser_target: str) -> SplatAsset:
    """Apply nerfstudio dataparser normalization transform and scale to raw asset points."""
    if not dataparser_target:
        return asset
    files = sorted((run_dir / "nerfstudio" / dataparser_target).rglob("dataparser_transforms.json"))
    if not files:
        return asset
    transform_path = files[-1]
    data = json.loads(transform_path.read_text(encoding="utf-8"))
    scale = float(data.get("scale", 1.0))
    matrix = np.array(
        data.get(
            "transform",
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        ),
        dtype=np.float32,
    )
    if matrix.shape != (3, 4):
        return asset
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :4] = matrix
    homogeneous = np.concatenate([asset.means * scale, np.ones((asset.means.shape[0], 1), dtype=np.float32)], axis=1)
    transformed = (homogeneous @ transform.T)[:, :3]
    return SplatAsset(
        name=asset.name,
        means=transformed.astype(np.float32),
        quats=asset.quats,
        scales=asset.scales * scale,
        opacities=asset.opacities,
        colors=asset.colors,
        normals=asset.normals,
        source=asset.source,
        texture_source=asset.texture_source,
        color_mode=asset.color_mode,
        normalization_scale=asset.normalization_scale,
        normalization_offset=asset.normalization_offset,
    )




def _asset_sha256(path: str) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_sequence(
    args: argparse.Namespace,
    frames_dir: Path,
    tensors: dict[str, torch.Tensor],
    camera_cfg: dict[str, Any],
) -> None:
    for frame in range(args.frames):
        image = render_frame(
            frame,
            args.frames,
            args.width,
            args.height,
            args.device,
            camera_cfg=camera_cfg,
            **tensors,
        )
        path = frames_dir / f"frame_{frame + 1:04d}.png"
        write_frame(path, image)


def write_frame(path: Path, image: np.ndarray) -> None:
    if cv2 is not None:
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return
    if Image is None:
        raise RuntimeError("No image writer available: install opencv-python or pillow.")
    Image.fromarray(image).save(path)


def export_strict_keyframes(frames_dir: Path, output_dir: Path) -> None:
    for idx, frame_index in enumerate(KEYFRAME_INDICES, start=1):
        source = frames_dir / f"frame_{frame_index + 1:04d}.png"
        if not source.exists():
            continue
        target = output_dir / f"strict_keyframe_{idx:03d}.png"
        target.write_bytes(source.read_bytes())


def build_manifest(
    args: argparse.Namespace,
    assets: list[SplatAsset],
    video_path: Path,
    elapsed_seconds: float,
    camera_cfg: dict[str, Any],
) -> dict:
    focal_scale = float(camera_cfg.get("focal_scale", 0.9))
    if not math.isfinite(focal_scale) or focal_scale <= 0:
        focal_scale = 0.9
    center = camera_cfg.get("center")
    if center is None and camera_cfg.get("centers"):
        center = np.asarray(camera_cfg["centers"], dtype=np.float32).mean(axis=0).tolist()
    if center is None:
        center = [0.0, 0.0, 0.0]
    asset_hashes = {}
    for asset in assets:
        source_path = args.run_dir / asset.source if not Path(asset.source).is_absolute() else Path(asset.source)
        if not source_path.exists():
            cwd_path = Path.cwd() / asset.source
            if cwd_path.exists():
                source_path = cwd_path
        if not source_path.exists():
            alt_path = Path(asset.source.lstrip("/"))
            if alt_path.exists():
                source_path = alt_path
        asset_hashes[asset.name] = _asset_sha256(source_path.as_posix())
    return {
        "renderer": RENDERER_NAME,
        "run_dir": args.run_dir.as_posix(),
        "pipeline_mode": PIPELINE_MODE,
        "source_mode": "unified_3d_assets",
        "scene_adjustments": {
            "foreground_offset": {
                "x": args.foreground_offset_x,
                "y": args.foreground_offset_y,
                "ground": args.foreground_ground_offset,
                "object_separation": args.object_separation,
                "object_a": [args.object_a_offset_x, args.object_a_offset_y],
                "object_b": [args.object_b_offset_x, args.object_b_offset_y],
                "object_c": [args.object_c_offset_x, args.object_c_offset_y],
                "object_c_target_extent": args.object_c_target_extent,
            },
            "camera": {
                "focus": args.camera_focus,
                "focal_multiplier": args.camera_focal_multiplier,
                "foreground_radius": args.foreground_camera_radius,
                "foreground_height": args.foreground_camera_height,
                "foreground_target_z": args.foreground_camera_target_z,
                "foreground_focal_scale": args.foreground_camera_focal_scale,
                "foreground_start_degrees": args.foreground_camera_start_degrees,
                "foreground_arc_degrees": args.foreground_camera_arc_degrees,
            },
            "background_clear": {
                "width": args.background_clear_width,
                "depth": args.background_clear_depth,
                "height": args.background_clear_height,
                "below": args.background_clear_below,
                "surface_keep": args.background_clear_surface_keep,
                "shape": args.background_clear_shape,
            },
            "support_mat": {
                "points": args.support_mat_points,
                "width": args.support_mat_width,
                "depth": args.support_mat_depth,
                "thickness": args.support_mat_thickness,
                "opacity": args.support_mat_opacity,
                "shape": args.support_mat_shape,
                "color": args.support_mat_color,
            },
            "foreground_highlight": {
                "color_boost": args.foreground_color_boost,
                "saturation_boost": args.foreground_saturation_boost,
                "opacity_boost": args.foreground_opacity_boost,
                "rim_strength": args.foreground_rim_strength,
                "rim_power": args.foreground_rim_power,
                "object_a_color_boost": args.object_a_color_boost,
                "object_a_opacity_boost": args.object_a_opacity_boost,
            },
        },
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "fps": args.fps,
        "seed": args.seed,
        "camera": {
            "name": camera_cfg["mode"],
            "center": center,
            "radius": camera_cfg.get("radius"),
            "height": camera_cfg.get("height"),
            "target_z": camera_cfg.get("target_z"),
            "focal_scale": focal_scale,
            "orbit_eccentricity": camera_cfg.get("orbit_eccentricity"),
            "radial_sweep": camera_cfg.get("radial_sweep"),
            "height_sweep": camera_cfg.get("height_sweep"),
            "camera_index_start": camera_cfg.get("camera_index_start"),
            "camera_index_stop": camera_cfg.get("camera_index_stop"),
            "radius_clip": camera_cfg.get("radius_clip"),
            "gamma": camera_cfg.get("gamma"),
            "centers": camera_cfg.get("centers"),
            "targets": camera_cfg.get("targets"),
        },
        "assets": [
            {
                "name": asset.name,
                "source": asset.source,
                "texture_source": asset.texture_source,
                "color_mode": asset.color_mode,
                "splats": int(asset.means.shape[0]),
            }
            for asset in assets
        ],
        "texture_hashes": {
            asset.name: _asset_sha256(asset.texture_source)
            for asset in assets
            if asset.texture_source
        },
        "asset_hashes": asset_hashes,
        "video": video_path.as_posix(),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def load_gaussian_ply(
    path: Path,
    name: str,
    *,
    max_points: int,
    opacity_quantile: float,
    crop_percentile: tuple[float, float],
) -> SplatAsset:
    count, properties, offset = read_ply_header(path)
    data = np.memmap(path, dtype=np.dtype([(prop, "<f4") for prop in properties]), mode="r", offset=offset, shape=(count,))
    means = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    finite = np.isfinite(means).all(axis=1)
    raw_opacity = np.asarray(data["opacity"], dtype=np.float32)
    finite &= np.isfinite(raw_opacity)
    indices = np.flatnonzero(finite)
    lower, upper = crop_percentile
    bounds_min = np.percentile(means[indices], lower, axis=0)
    bounds_max = np.percentile(means[indices], upper, axis=0)
    inside = np.all((means[indices] >= bounds_min) & (means[indices] <= bounds_max), axis=1)
    indices = indices[inside]
    threshold = np.quantile(raw_opacity[indices], opacity_quantile)
    indices = indices[raw_opacity[indices] >= threshold]
    indices = deterministic_sample(indices, max_points, seed=hash(name) & 0xFFFF)

    means = means[indices]
    colors = np.column_stack([data["f_dc_0"][indices], data["f_dc_1"][indices], data["f_dc_2"][indices]])
    colors = np.clip(0.5 + SH_C0 * colors, 0.0, 1.0).astype(np.float32)
    scales = np.exp(np.column_stack([data["scale_0"][indices], data["scale_1"][indices], data["scale_2"][indices]])).astype(np.float32)
    quats = np.column_stack([data["rot_0"][indices], data["rot_1"][indices], data["rot_2"][indices], data["rot_3"][indices]]).astype(np.float32)
    opacities = sigmoid(raw_opacity[indices]).astype(np.float32)
    quats = normalize_quats(quats)
    return SplatAsset(
        name=name,
        means=means.astype(np.float32),
        quats=quats.astype(np.float32),
        scales=scales.astype(np.float32),
        opacities=opacities.astype(np.float32),
        colors=colors.astype(np.float32),
        normals=None,
        source=path.as_posix(),
    )




def load_background_camera_trajectory(
    run_dir: Path,
    background: SplatAsset,
    total_frames: int,
    target_width: int,
    camera_index_start: int = 0,
    camera_index_stop: int = 0,
) -> dict[str, Any] | None:
    path = run_dir / "processed" / "background" / "transforms.json"
    if not path.exists():
        return None
    try:
        from nerfstudio.cameras.camera_utils import focus_of_attention
        from nerfstudio.data.dataparsers.nerfstudio_dataparser import (
            NerfstudioDataParserConfig,
        )

        parser = NerfstudioDataParserConfig(
            data=path.parent,
            train_split_fraction=1.0,
        ).setup()
        outputs = parser.get_dataparser_outputs(split="train")
        camera_to_worlds = outputs.cameras.camera_to_worlds.detach().cpu().numpy()
        raw_focus = focus_of_attention(
            outputs.cameras.camera_to_worlds,
            outputs.cameras.camera_to_worlds[:, :3, 3].mean(dim=0),
        ).detach().cpu().numpy()
        focal_scale = float(
            torch.median(outputs.cameras.fx / outputs.cameras.width)
            .detach()
            .cpu()
            .item()
        )
    except Exception as exc:
        print(
            json.dumps(
                {"background_camera_trajectory": "unavailable", "reason": str(exc)},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return None
    if not np.all(np.isfinite(camera_to_worlds)):
        return None
    start = max(0, int(camera_index_start))
    stop = int(camera_index_stop)
    if stop <= start or stop > len(camera_to_worlds):
        stop = len(camera_to_worlds)
    camera_to_worlds = camera_to_worlds[start:stop]
    if len(camera_to_worlds) == 0:
        return None
    offset = (
        background.normalization_offset
        if background.normalization_offset is not None
        else np.zeros(3, dtype=np.float32)
    )
    camera_to_worlds = apply_similarity_to_camera_poses(
        camera_to_worlds,
        scale=background.normalization_scale,
        offset=offset,
    )
    focus = raw_focus * background.normalization_scale + offset
    indices = np.linspace(0, len(camera_to_worlds) - 1, num=total_frames).astype(int)
    camera_to_worlds = camera_to_worlds[indices]
    centers = camera_to_worlds[:, :3, 3]
    focus_distance = max(
        float(np.median(np.linalg.norm(centers - focus[None, :], axis=1))),
        1e-3,
    )
    targets = np.repeat(focus[None, :], len(centers), axis=0)
    span = np.percentile(background.means, 98, axis=0) - np.percentile(
        background.means, 2, axis=0
    )
    return {
        "mode": "background_trajectory",
        "centers": centers.tolist(),
        "targets": targets.tolist(),
        "focus": focus.tolist(),
        "center": focus.tolist(),
        "radius": focus_distance,
        "height": float(np.median(centers[:, 2]) - focus[2]),
        "target_z": 0.0,
        "focal_scale": focal_scale,
        "camera_index_start": start,
        "camera_index_stop": stop,
    }


def read_ply_header(path: Path) -> tuple[int, list[str], int]:
    count = 0
    properties: list[str] = []
    in_vertex = False
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"Invalid PLY header: {path}")
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("format ") and "binary_little_endian" not in text:
                raise RuntimeError(f"Unsupported PLY format: {text}")
            if text.startswith("element vertex "):
                count = int(text.split()[-1])
                in_vertex = True
            elif text.startswith("element ") and not text.startswith("element vertex "):
                in_vertex = False
            elif in_vertex and text.startswith("property "):
                properties.append(text.split()[-1])
            elif text == "end_header":
                return count, properties, handle.tell()


def load_obj_as_splats(
    path: Path,
    name: str,
    *,
    max_points: int,
    base_scale: float,
    scale_cap: float,
    seed: int,
) -> SplatAsset:
    samples = sample_textured_obj(path, max_points=max_points, seed=seed)
    isotropic_scales = adaptive_scales(
        samples.points,
        base_scale=base_scale,
        scale_cap=scale_cap,
        target_count=max_points,
        rng_seed=seed,
    )
    scales = surface_splat_scales(
        point_count=samples.points.shape[0],
        tangent_scale=isotropic_scales[:, 0],
        thickness_ratio=0.18,
    )
    quats = normal_aligned_quaternions(samples.normals)
    opacities = np.full(samples.points.shape[0], 0.96, dtype=np.float32)
    if samples.has_texture:
        colors = samples.colors
        if name == "object_b":
            colors = apply_object_b_crystal_style(samples.points, colors)
    else:
        colors = enhance_obj_colors(name, samples.colors, samples.normals)
    return SplatAsset(
        name=name,
        means=samples.points.astype(np.float32),
        quats=quats.astype(np.float32),
        scales=scales.astype(np.float32),
        opacities=opacities.astype(np.float32),
        colors=colors.astype(np.float32),
        normals=samples.normals.astype(np.float32),
        source=path.as_posix(),
        texture_source=samples.texture_path.as_posix() if samples.texture_path else None,
        color_mode="uv_texture" if samples.has_texture else "palette_fallback",
    )


def sample_obj_surface(
    vertices: np.ndarray,
    colors: np.ndarray,
    faces: np.ndarray,
    vertex_normals: np.ndarray,
    *,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    faces_arr = np.asarray(faces, dtype=np.int64)
    if faces_arr.size == 0:
        return vertices, colors, vertex_normals
    rng = np.random.default_rng(seed)
    a = vertices[faces_arr[:, 0]]
    b = vertices[faces_arr[:, 1]]
    c = vertices[faces_arr[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    area = np.where(area > 0, area, 1e-12)
    probs = area / area.sum()
    indices = rng.choice(len(faces_arr), size=max_points, replace=True, p=probs)
    triangles = faces_arr[indices]
    tri_a, tri_b, tri_c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    va, vb, vc = vertices[tri_a], vertices[tri_b], vertices[tri_c]
    ca, cb, cc = colors[tri_a], colors[tri_b], colors[tri_c]
    u = rng.random(size=max_points)
    v = rng.random(size=max_points)
    outside = u + v >= 1.0
    u[outside] = 1.0 - u[outside]
    v[outside] = 1.0 - v[outside]
    w = 1.0 - u - v
    points = w[:, None] * va + u[:, None] * vb + v[:, None] * vc
    sampled_colors = w[:, None] * ca + u[:, None] * cb + v[:, None] * cc
    if vertex_normals.shape[0] >= vertices.shape[0]:
        na, nb, nc = vertex_normals[tri_a], vertex_normals[tri_b], vertex_normals[tri_c]
        sampled_normals = w[:, None] * na + u[:, None] * nb + v[:, None] * nc
    else:
        sampled_normals = np.zeros_like(points, dtype=np.float32)
        sampled_normals[:, 2] = 1.0
    sampled_normals = np.where(
        np.linalg.norm(sampled_normals, axis=1, keepdims=True) < 1e-6,
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        sampled_normals,
    )
    return (
        points.astype(np.float32),
        np.clip(sampled_colors.astype(np.float32), 0.0, 1.0),
        sampled_normals.astype(np.float32),
    )


def enhance_obj_colors(name: str, colors: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Enhance OBJ colors and compensate missing or weak texture.

    Many OBJ exports from Zero123/threestudio have weak color channels; fallback to
    palette + normal shading for stable, plausible appearance.
    """
    if colors.size == 0:
        palette = PALETTE_COLORS.get(name, np.array([0.8, 0.8, 0.8], dtype=np.float32))
        return np.repeat(palette[None, :], repeats=normals.shape[0], axis=0)
    base = colors.astype(np.float32)
    cmin, cmax = float(base.min()), float(base.max())
    if cmax - cmin < 1e-5:
        palette = PALETTE_COLORS.get(name, np.array([0.8, 0.8, 0.8], dtype=np.float32))
        base = np.repeat(palette[None, :], repeats=base.shape[0], axis=0)
    base = np.clip(base, 0.0, 1.0)
    normal_factor = (normals @ DEFAULT_LIGHT_DIR).astype(np.float32)
    normal_factor = np.clip(normal_factor, 0.25, 1.0)
    shading = (AMBIENT + DIFFUSE * normal_factor)[:, None]
    shaded = base * shading
    palette = PALETTE_COLORS.get(name, np.array([0.8, 0.8, 0.8], dtype=np.float32))
    fallback = palette * (
        AMBIENT + DIFFUSE * np.linspace(0.7, 1.0, num=base.shape[0], dtype=np.float32)[:, None]
    )
    blend = 0.5 + 0.5 * normal_factor[:, None]
    return np.clip(blend * shaded + (1.0 - blend) * fallback, 0.0, 1.0)


def adaptive_scales(
    means: np.ndarray,
    *,
    base_scale: float,
    scale_cap: float,
    target_count: int,
    rng_seed: int,
) -> np.ndarray:
    del target_count, rng_seed
    mins = np.percentile(means, 0.2, axis=0)
    maxs = np.percentile(means, 99.8, axis=0)
    diag = np.linalg.norm(maxs - mins) + 1e-6
    target = np.full((means.shape[0], 3), base_scale * (diag / 0.9), dtype=np.float32)
    return np.clip(target, 0.0020, scale_cap)


def prune_outliers(points: np.ndarray, colors: np.ndarray, *, percentile: float) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return points, colors
    z = np.linalg.norm(points, axis=1)
    limit = np.quantile(z, percentile)
    keep = z <= limit
    if not np.any(keep):
        return points, colors
    return points[keep], colors[keep]


def build_camera_config(
    run_dir: Path,
    assets: list[SplatAsset],
    total_frames: int,
    *,
    radius_scale: float = 1.0,
    height_scale: float = 1.0,
    orbit_eccentricity: float = 0.85,
    radial_sweep: float = 0.10,
    height_sweep: float = 0.18,
    focus_mode: str = "background",
    camera_index_start: int = 0,
    camera_index_stop: int = 0,
    foreground_radius: float = 1.75,
    foreground_height: float = 0.92,
    foreground_target_z: float = 0.34,
    foreground_focal_scale: float = 1.20,
    foreground_start_degrees: float = 18.0,
    foreground_arc_degrees: float = 54.0,
) -> dict[str, Any]:
    background = next(asset for asset in assets if asset.name == "background")
    if focus_mode == "foreground":
        return build_foreground_camera_config(
            assets,
            radius=foreground_radius,
            height=foreground_height,
            target_z=foreground_target_z,
            focal_scale=foreground_focal_scale,
            start_degrees=foreground_start_degrees,
            arc_degrees=foreground_arc_degrees,
            orbit_eccentricity=orbit_eccentricity,
            radial_sweep=radial_sweep,
            height_sweep=height_sweep,
        )
    trajectory = load_background_camera_trajectory(
        run_dir,
        background,
        total_frames=total_frames,
        target_width=1920,
        camera_index_start=camera_index_start,
        camera_index_stop=camera_index_stop,
    )
    if trajectory is not None:
        trajectory["orbit_eccentricity"] = float(orbit_eccentricity)
        trajectory["radial_sweep"] = float(radial_sweep)
        trajectory["height_sweep"] = float(height_sweep)
        return trajectory
    mins = np.percentile(background.means, 2, axis=0)
    maxs = np.percentile(background.means, 98, axis=0)
    median = np.percentile(background.means, 50, axis=0)
    span = (maxs - mins).astype(float)
    ground_z = float(mins[2])
    center = [float(median[0]), float(median[1]), ground_z + 0.55]
    xy_radius = max(float(span[0]), float(span[1]), 1e-6) * 0.72
    return {
        "mode": "orbit",
        "center": center,
        "radius": float(np.clip(xy_radius * float(radius_scale), 6.0, 12.0)),
        "height": float(max(2.0, (0.30 * span[2] + 1.0) * float(height_scale))),
        "target_z": 0.0,
        "focal_scale": 1.15,
        "orbit_eccentricity": float(orbit_eccentricity),
        "radial_sweep": float(radial_sweep),
        "height_sweep": float(height_sweep),
    }


def build_foreground_camera_config(
    assets: list[SplatAsset],
    *,
    radius: float,
    height: float,
    target_z: float,
    focal_scale: float,
    start_degrees: float,
    arc_degrees: float,
    orbit_eccentricity: float,
    radial_sweep: float,
    height_sweep: float,
) -> dict[str, Any]:
    foreground_assets = [asset for asset in assets if asset.name != "background"]
    if not foreground_assets:
        raise ValueError("Foreground camera focus requires at least one foreground asset.")
    foreground_points = np.concatenate([asset.means for asset in foreground_assets], axis=0)
    mins = np.percentile(foreground_points, 3, axis=0)
    maxs = np.percentile(foreground_points, 97, axis=0)
    center_xy = (mins[:2] + maxs[:2]) * 0.5
    ground_z = float(mins[2])
    span_z = max(float(maxs[2] - mins[2]), 1e-3)
    if target_z <= 0:
        target_z = min(0.55, max(0.25, span_z * 0.45))
    return {
        "mode": "foreground_orbit",
        "center": [float(center_xy[0]), float(center_xy[1]), ground_z],
        "radius": float(max(radius, 0.25)),
        "height": float(max(height, 0.25)),
        "target_z": float(target_z),
        "focal_scale": float(max(focal_scale, 0.1)),
        "orbit_eccentricity": float(orbit_eccentricity),
        "radial_sweep": float(radial_sweep),
        "height_sweep": float(height_sweep),
        "orbit_start_degrees": float(start_degrees),
        "orbit_arc_degrees": float(arc_degrees),
        "foreground_bounds": {
            "mins": mins.astype(float).tolist(),
            "maxs": maxs.astype(float).tolist(),
        },
    }


def _estimate_focal_scale(points: np.ndarray, target_width: float) -> float:
    mins = np.percentile(points, 4, axis=0)
    maxs = np.percentile(points, 96, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    dominant = float(max(span[0], span[1], 1e-6))
    if not np.isfinite(dominant) or dominant <= 0:
        return 0.9
    return float(np.clip(target_width / (3.4 * dominant), FOCAL_SCALE_MIN, FOCAL_SCALE_MAX))


def deterministic_sample(indices: np.ndarray, max_points: int, *, seed: int) -> np.ndarray:
    if indices.shape[0] <= max_points:
        return np.sort(indices)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_points, replace=False))


def normalize_asset(
    asset: SplatAsset,
    *,
    target_height: float,
    target_extent: float = 0.0,
    xy: tuple[float, float],
    ground_z: float,
    robust_percentile: tuple[float, float],
    scale_multiplier: float,
    scale_max: float,
    reference: SplatAsset | None = None,
) -> SplatAsset:
    lower, upper = robust_percentile
    mins = np.percentile(asset.means, lower, axis=0)
    maxs = np.percentile(asset.means, upper, axis=0)
    center = (mins + maxs) * 0.5
    height = max(maxs[2] - mins[2], 1e-6)
    if target_extent > 0:
        extent = max(float(np.max(maxs - mins)), 1e-6)
        scale = target_extent / extent
    else:
        scale = target_height / height
    means = (asset.means - center) * scale
    robust_ground = (mins[2] - center[2]) * scale
    means[:, 0] += xy[0]
    means[:, 1] += xy[1]
    means[:, 2] += ground_z - robust_ground
    scales = np.clip(asset.scales * scale * scale_multiplier, 0.0025, scale_max)
    del reference
    offset = np.array(
        [
            -center[0] * scale + xy[0],
            -center[1] * scale + xy[1],
            -center[2] * scale + ground_z - robust_ground,
        ],
        dtype=np.float32,
    )
    return SplatAsset(
        name=asset.name,
        means=means.astype(np.float32),
        quats=normalize_quats(asset.quats),
        scales=scales.astype(np.float32),
        opacities=asset.opacities.astype(np.float32),
        colors=asset.colors.astype(np.float32),
        normals=asset.normals,
        source=asset.source,
        texture_source=asset.texture_source,
        color_mode=asset.color_mode,
        normalization_scale=float(scale),
        normalization_offset=offset,
    )


def normalize_quats(quats: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    norm = np.where(norm < 1e-6, 1.0, norm)
    return (quats / norm).astype(np.float32)


def quats_to_normals(quats: np.ndarray) -> np.ndarray:
    """Return the local +Z direction for gsplat WXYZ quaternions."""
    return quaternion_normals(quats)


def concat_assets(assets: list[SplatAsset], device: str) -> dict[str, torch.Tensor]:
    means = np.concatenate([asset.means for asset in assets], axis=0)
    quats = np.concatenate([asset.quats for asset in assets], axis=0)
    scales = np.concatenate([asset.scales for asset in assets], axis=0)
    opacities = np.concatenate([asset.opacities for asset in assets], axis=0)
    colors = np.concatenate([asset.colors for asset in assets], axis=0)
    normals = np.concatenate(
        [np.zeros_like(asset.means) if asset.normals is None else asset.normals for asset in assets],
        axis=0,
    )
    asset_ids = np.concatenate(
        [
            np.full(asset.means.shape[0], index, dtype=np.int64)
            for index, asset in enumerate(assets)
        ],
        axis=0,
    )
    return {
        "means": torch.from_numpy(means).to(device),
        "quats": torch.from_numpy(quats).to(device),
        "scales": torch.from_numpy(scales).to(device),
        "opacities": torch.from_numpy(opacities).to(device),
        "colors": torch.from_numpy(colors).to(device),
        "normals": torch.from_numpy(normals).to(device),
        "asset_ids": torch.from_numpy(asset_ids).to(device),
    }


def render_frame(
    frame: int,
    total_frames: int,
    width: int,
    height: int,
    device: str,
    *,
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    normals: torch.Tensor,
    asset_ids: torch.Tensor,
    camera_cfg: dict[str, Any],
) -> np.ndarray:
    mode = camera_cfg.get("mode")
    if mode == "foreground_orbit":
        t = frame / max(total_frames - 1, 1)
        start = math.radians(float(camera_cfg.get("orbit_start_degrees", 0.0)))
        arc = math.radians(float(camera_cfg.get("orbit_arc_degrees", 360.0)))
        phase = start + arc * t
    else:
        phase = 2.0 * math.pi * frame / max(total_frames, 1)
    orbit_eccentricity = float(camera_cfg.get("orbit_eccentricity", 0.85))
    radial_sweep = float(camera_cfg.get("radial_sweep", 0.10))
    height_sweep = float(camera_cfg.get("height_sweep", 0.18))
    if mode == "background_trajectory":
        camera_to_worlds = camera_cfg.get("camera_to_worlds", [])
        centers = camera_cfg.get("centers", [])
        targets = camera_cfg.get("targets", [])
        if camera_to_worlds:
            c2w = torch.tensor(
                np.asarray(camera_to_worlds[frame % len(camera_to_worlds)], dtype=np.float32),
                device=device,
                dtype=torch.float32,
            )
            if c2w.shape == (3, 4):
                full_c2w = torch.eye(4, device=device, dtype=torch.float32)
                full_c2w[:3, :4] = c2w
                c2w = full_c2w
            eye = c2w[:3, 3]
        elif centers:
            center_np = np.asarray(centers[frame % len(centers)], dtype=np.float32)
            target_np = np.asarray(targets[frame % len(targets)], dtype=np.float32)
            eye = torch.tensor(center_np, device=device, dtype=torch.float32)
            target = torch.tensor(target_np, device=device, dtype=torch.float32)
            c2w = camera_to_world(eye, target)
        else:
            center_np = np.asarray(camera_cfg["center"], dtype=np.float32)
            target_np = np.asarray([camera_cfg["center"][0], camera_cfg["center"][1], camera_cfg["center"][2] + camera_cfg.get("target_z", 0.0)], dtype=np.float32)
            eye = torch.tensor(center_np, device=device, dtype=torch.float32)
            target = torch.tensor(target_np, device=device, dtype=torch.float32)
            c2w = camera_to_world(eye, target)
    else:
        center = torch.tensor(camera_cfg["center"], device=device, dtype=torch.float32)
        radius = float(camera_cfg["radius"]) * (1.0 + float(radial_sweep) * math.sin(2.0 * phase))
        eye = torch.tensor(
            [
                center[0] + math.sin(phase) * radius,
                center[1] + math.cos(phase) * radius,
                center[2] + float(camera_cfg["height"]) + height_sweep * math.cos(1.7 * phase),
            ],
            device=device,
            dtype=torch.float32,
        )
        target = center + torch.tensor([0.0, 0.0, float(camera_cfg["target_z"])], device=device, dtype=torch.float32)
        c2w = camera_to_world(eye, target)
    viewmat = nerfstudio_viewmat(c2w).unsqueeze(0)
    focal = float(camera_cfg.get("focal_scale", 0.92)) * width
    # apply light shading for all points before rasterization
    light_vec = torch.tensor(DEFAULT_LIGHT_DIR, device=device, dtype=torch.float32)
    normal_lengths = torch.linalg.norm(normals, dim=1, keepdim=True)
    surface_normals = torch.nn.functional.normalize(normals, dim=1)
    normal_factor = (surface_normals @ light_vec).clamp(-0.25, 1.0)
    view_direction = torch.nn.functional.normalize(eye[None, :] - means, dim=1)
    view_factor = torch.clamp((surface_normals * view_direction).sum(dim=1), 0.0, 1.0)
    surface_shade = (
        (AMBIENT + DIFFUSE * normal_factor) * 0.78
        + 0.22 * view_factor
    ).clamp(0.0, 1.2).unsqueeze(1)
    shade = torch.where(
        normal_lengths > 0.5,
        surface_shade,
        torch.ones_like(surface_shade),
    )
    foreground_mask = asset_ids > 0
    object_a_mask = asset_ids == 1
    foreground_factor = foreground_mask.to(dtype=torch.float32).unsqueeze(1)
    object_a_factor = object_a_mask.to(dtype=torch.float32).unsqueeze(1)
    color_boost = 1.0 + foreground_factor * (float(camera_cfg.get("foreground_color_boost", 1.0)) - 1.0)
    color_boost = color_boost + object_a_factor * (float(camera_cfg.get("object_a_color_boost", 1.0)) - 1.0)
    saturation_boost = 1.0 + foreground_factor * (
        float(camera_cfg.get("foreground_saturation_boost", 1.0)) - 1.0
    )
    luminance = (
        colors[:, 0:1] * 0.2126
        + colors[:, 1:2] * 0.7152
        + colors[:, 2:3] * 0.0722
    )
    boosted_colors = luminance + (colors - luminance) * saturation_boost
    boosted_colors = torch.clamp(boosted_colors * color_boost, 0.0, 1.0)
    rim_strength = max(0.0, float(camera_cfg.get("foreground_rim_strength", 0.0)))
    rim_power = max(0.25, float(camera_cfg.get("foreground_rim_power", 2.0)))
    rim_mask = foreground_mask.unsqueeze(1) & (normal_lengths > 0.5)
    rim = torch.pow(1.0 - view_factor.unsqueeze(1), rim_power) * rim_strength
    rim_color = torch.tensor([1.0, 0.94, 0.78], device=device, dtype=torch.float32)
    boosted_colors = torch.where(
        rim_mask,
        torch.clamp(boosted_colors + rim * rim_color[None, :], 0.0, 1.0),
        boosted_colors,
    )
    boosted_opacities = opacities * (
        1.0
        + foreground_mask.to(dtype=torch.float32) * (float(camera_cfg.get("foreground_opacity_boost", 1.0)) - 1.0)
        + object_a_mask.to(dtype=torch.float32) * (float(camera_cfg.get("object_a_opacity_boost", 1.0)) - 1.0)
    )
    boosted_opacities = torch.clamp(boosted_opacities, 0.0, 1.0)
    lit_colors = torch.clamp(boosted_colors * shade, 0.0, 1.0)
    intrinsics = torch.tensor(
        [[[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        render, alpha, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=boosted_opacities,
            colors=lit_colors,
            viewmats=viewmat,
            Ks=intrinsics,
            width=width,
            height=height,
            packed=True,
            near_plane=0.01,
            far_plane=30.0,
            render_mode="RGB",
            sh_degree=None,
            rasterize_mode="antialiased",
            radius_clip=float(camera_cfg.get("radius_clip", 0.085)),
        )
    background = torch.tensor([0.72, 0.70, 0.67], device=device, dtype=torch.float32)
    image = render[0, :, :, :3] + (1.0 - alpha[0, :, :, :1]) * background
    # tone mapping for image contrast and gamma control.
    image = torch.clamp(image, 0.0, 1.0)
    gamma = float(camera_cfg.get("gamma", 1.0))
    if gamma > 0 and abs(gamma - 1.0) > 1e-6:
        image = torch.pow(image, 1.0 / gamma)
    return (image.detach().cpu().numpy() * 255).astype(np.uint8)


def camera_to_world(eye: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    forward = torch.nn.functional.normalize(target - eye, dim=0)
    up = torch.tensor([0.0, 0.0, 1.0], device=eye.device)
    right = torch.nn.functional.normalize(torch.cross(forward, up, dim=0), dim=0)
    true_up = torch.nn.functional.normalize(torch.cross(right, forward, dim=0), dim=0)
    c2w = torch.eye(4, device=eye.device, dtype=torch.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = eye
    return c2w


def nerfstudio_viewmat(c2w: torch.Tensor) -> torch.Tensor:
    rotation = c2w[:3, :3] * torch.tensor([[1.0, -1.0, -1.0]], device=c2w.device)
    translation = c2w[:3, 3:4]
    rotation_inverse = rotation.transpose(0, 1)
    translation_inverse = -rotation_inverse @ translation
    viewmat = torch.eye(4, device=c2w.device, dtype=torch.float32)
    viewmat[:3, :3] = rotation_inverse
    viewmat[:3, 3:4] = translation_inverse
    return viewmat


def encode_video(frames_dir: Path, output_path: Path, fps: int) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


if __name__ == "__main__":
    main()
