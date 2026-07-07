"""Configuration helpers for HW3 Task1 experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "task1_real_quality_v2",
        "seed": 42,
        "output_root": "hw3/task1/outputs",
    },
    "task1": {
        "stage": "real_high_quality",
    },
    "logging": {
        "swanlab": {
            "enabled": False,
            "project": "cs60003-hw3-task1",
            "mode": "cloud",
            "group": "real-high-quality",
            "tags": ["hw3", "task1", "real-high-quality"],
            "env_file": ".helloagents/secrets/hw3.env",
        },
    },
    "real_chain": {
        "execution": {"mode": "plan"},
        "data": {
            "object_a_images": "hw3/assets/object_a_multiview",
            "object_a_video": "hw3/milk_task1_1080.m4v",
            "object_c_image": "hw3/objectC.HEIC",
            "background_dataset": "mipnerf360_counter",
            "background_download_url": "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip",
            "background_scene": "counter",
            "background_images": "hw3/assets/background_scene/images",
            "background_video": "",
        },
        "tools": {
            "required_cli": ["ns-process-data", "ns-train", "ns-export", "ns-eval", "colmap", "ffmpeg"],
            "threestudio_launch": "hw3/task1/external/threestudio/launch.py",
            "zero123_config": "hw3/task1/external/threestudio/configs/zero123.yaml",
            "zero123_checkpoint": "hw3/task1/external/threestudio/load/zero123/zero123-xl.ckpt",
            "zero123_config_file": "hw3/task1/external/threestudio/load/zero123/sd-objaverse-finetune-c_concat-256.yaml",
            "object_mask_attacher": "hw3/task1/scripts/attach_object_masks.py",
            "object_c_preprocessor": "hw3/task1/scripts/preprocess_object_c.py",
            "nerfstudio_swanlab_runner": "hw3/task1/scripts/run_nerfstudio_swanlab.py",
            "threestudio_swanlab_runner": "hw3/task1/scripts/run_threestudio_swanlab.py",
        },
        "quality": {
            "splatfacto_iterations": 30000,
            "cull_alpha_thresh": 0.005,
            "object_a_num_frames_target": 80,
            "object_a_segmentation_model": "isnet-general-use",
            "object_a_min_registration_ratio": 0.70,
            "object_a_min_mask_occupancy": 0.05,
            "object_a_max_mask_occupancy": 0.80,
            "object_a_min_refined_area_ratio": 0.70,
            "object_a_max_refined_area_ratio": 1.60,
            "mesh_texture_size": 2048,
        },
        "object_b": {
            "prompt": (
                "a small translucent violet crystal mushroom figurine with a jade green carved stem, "
                "physically plausible tabletop object, detailed faceted crystal cap, subtle internal glow, "
                "high quality PBR material, clean geometry, suitable for insertion into a reconstructed counter scene"
            ),
            "sds_model": "runwayml/stable-diffusion-v1-5",
            "max_steps": 15000,
            "limit_test_batches": 0,
        },
        "object_c": {
            "zero123_max_steps": 1200,
            "export_resolution": 256,
            "preprocess_model": "isnet-general-use",
            "preprocess_size": 512,
            "preprocess_padding_ratio": 0.10,
            "preprocess_min_occupancy": 0.45,
            "preprocess_max_occupancy": 0.85,
            "input_height": [128, 256, 512],
            "input_width": [128, 256, 512],
            "random_camera_batch_size": [10, 2, 1],
            "random_camera_height": [64, 128, 192],
            "random_camera_width": [64, 128, 192],
            "resolution_milestones": [200, 300],
            "default_elevation_deg": 32.0,
            "num_samples_per_ray": 512,
            "limit_val_batches": 0,
            "limit_test_batches": 0,
            "lambda_normal_smooth": 8.0,
            "lambda_3d_normal_smooth": 8.0,
            "lambda_orient": 1.0,
            "resume_checkpoint": "",
        },
    },
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge nested dictionaries without mutating either input."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and fill missing values from defaults."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    config = deep_update(DEFAULT_CONFIG, loaded)
    config["config_path"] = str(config_path)
    _validate_config(config)
    return config


def repo_root_from_task_file() -> Path:
    """Return repository root from this package location."""
    return Path(__file__).resolve().parents[3]


def resolve_repo_path(path_value: str | Path, repo_root: Path | None = None) -> Path:
    """Resolve a path relative to repository root unless absolute."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root or repo_root_from_task_file()) / path


def resolve_paths(config: dict[str, Any], output_root: str | None = None) -> None:
    """Resolve configured paths in-place relative to repository root."""
    repo_root = repo_root_from_task_file()
    if output_root is not None:
        config["experiment"]["output_root"] = output_root
    config["experiment"]["output_root"] = str(resolve_repo_path(config["experiment"]["output_root"]))
    report_dir = config["experiment"].get("report_assets_dir", "")
    if report_dir:
        config["experiment"]["report_assets_dir"] = str(resolve_repo_path(report_dir, repo_root))
    real_data = config["real_chain"]["data"]
    for key in ["object_a_images", "object_c_image", "background_images"]:
        real_data[key] = str(resolve_repo_path(real_data[key], repo_root))
    for key in ["object_a_video", "background_video"]:
        if real_data.get(key):
            real_data[key] = str(resolve_repo_path(real_data[key], repo_root))
    tools = config["real_chain"]["tools"]
    for key in [
        "threestudio_launch",
        "zero123_config",
        "zero123_checkpoint",
        "zero123_config_file",
        "object_mask_attacher",
        "object_c_preprocessor",
        "nerfstudio_swanlab_runner",
        "threestudio_swanlab_runner",
    ]:
        tools[key] = str(resolve_repo_path(tools[key], repo_root))
    swanlab = config.get("logging", {}).get("swanlab", {})
    if swanlab.get("env_file"):
        swanlab["env_file"] = str(resolve_repo_path(swanlab["env_file"], repo_root))


def _validate_config(config: dict[str, Any]) -> None:
    stage = str(config.get("task1", {}).get("stage", ""))
    if stage != "real_high_quality":
        raise ValueError(f"Unsupported Task1 stage for maintained chain: {stage}")
    real_chain = config.get("real_chain", {})
    mode = str(real_chain.get("execution", {}).get("mode", ""))
    if mode not in {"plan", "run"}:
        raise ValueError("real_chain.execution.mode must be either 'plan' or 'run'.")
    data = real_chain.get("data", {})
    for key in ["object_a_images", "object_c_image", "background_images"]:
        if key not in data:
            raise ValueError(f"real_chain.data.{key} is required by the maintained config.")
