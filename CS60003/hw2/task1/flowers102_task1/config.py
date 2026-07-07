"""Configuration loading for HW2 Task1 experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "flowers102_resnet18",
        "seed": 42,
        "output_root": "hw2/task1/outputs",
    },
    "data": {
        "root": "hw2/Flowers102",
        "image_size": 224,
        "batch_size": 64,
        "num_workers": 4,
        "pin_memory": True,
        "augment": "basic",
        "randaugment_ops": 2,
        "randaugment_magnitude": 9,
        "random_erasing": 0.0,
    },
    "model": {
        "name": "resnet18",
        "pretrained": True,
        "num_classes": 102,
    },
    "train": {
        "epochs": 40,
        "optimizer": "sgd",
        "backbone_lr": 1.0e-4,
        "classifier_lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "momentum": 0.9,
        "scheduler": "cosine",
        "amp": True,
        "label_smoothing": 0.0,
        "grad_clip_norm": 0.0,
        "log_interval": 20,
    },
    "eval": {
        "tta": False,
    },
    "logging": {
        "swanlab": {
            "enabled": False,
            "project": "cs60003-hw2-task1",
            "workspace": None,
            "mode": "cloud",
            "group": None,
            "tags": ["hw2", "task1", "flowers102"],
            "description": None,
        },
    },
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge nested dictionaries without mutating the inputs."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config and apply default values."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    config = deep_update(DEFAULT_CONFIG, loaded)
    config["config_path"] = str(config_path)
    return config


def repo_root_from_task_file() -> Path:
    """Return the repository root from this package location."""
    return Path(__file__).resolve().parents[3]


def resolve_repo_path(path_value: str | Path, repo_root: Path | None = None) -> Path:
    """Resolve a path relative to the repository root unless already absolute."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root or repo_root_from_task_file()) / path
