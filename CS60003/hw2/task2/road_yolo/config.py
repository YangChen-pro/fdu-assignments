"""Configuration loading for HW2 Task2 experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "task2_yolov8s_baseline",
        "seed": 42,
        "output_root": "hw2/task2/outputs",
    },
    "data": {
        "root": "hw2/RoadVehicleImages/trafic_data",
        "yaml": "data_hw2.yaml",
    },
    "model": {
        "name": "yolov8s.pt",
        "nc": 21,
    },
    "train": {
        "epochs": 50,
        "imgsz": 640,
        "batch": 16,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "warmup_epochs": 3,
        "amp": True,
    },
    "logging": {
        "swanlab": {
            "enabled": False,
            "project": "cs60003-hw2-task2",
            "workspace": None,
            "mode": "cloud",
            "group": None,
            "tags": ["hw2", "task2", "road-vehicle"],
            "description": None,
        },
    },
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge nested dicts without mutating inputs."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config and apply defaults."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    config = deep_update(DEFAULT_CONFIG, loaded)
    config["config_path"] = str(config_path)
    return config


def repo_root_from_task_file() -> Path:
    """Return the repository root from this package location."""
    # config.py lives at hw2/task2/road_yolo/config.py → parents[3] is repo root
    return Path(__file__).resolve().parents[3]


def resolve_repo_path(path_value: str | Path, repo_root: Path | None = None) -> Path:
    """Resolve a path relative to the repository root unless already absolute."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root or repo_root_from_task_file()) / path
