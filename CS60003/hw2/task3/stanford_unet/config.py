"""Configuration helpers for HW2 Task3 experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

CLASS_NAMES = [
    "sky",
    "tree",
    "road",
    "grass",
    "water",
    "building",
    "mountain",
    "foreground_object",
]
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "task3_unet_ce",
        "seed": 42,
        "output_root": "hw2/task3/outputs",
    },
    "data": {
        "root": "hw2/StanfordBackground/iccv09Data",
        "split_dir": "hw2/task3/splits",
        "train_ratio": 0.8,
        "image_size": [240, 320],
        "batch_size": 8,
        "num_workers": 4,
        "pin_memory": True,
        "ignore_index": IGNORE_INDEX,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "augment": {
            "horizontal_flip": 0.5,
            "color_jitter": 0.15,
        },
    },
    "model": {
        "name": "unet",
        "num_classes": NUM_CLASSES,
        "base_channels": 32,
        "bilinear": False,
        "pretrained": False,
    },
    "train": {
        "loss": "ce",
        "epochs": 80,
        "optimizer": "adamw",
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "scheduler": "cosine",
        "amp": True,
        "grad_clip_norm": 1.0,
        "log_interval": 20,
        "early_stopping_patience": 16,
    },
    "eval": {
        "visualize_samples": 8,
    },
    "logging": {
        "swanlab": {
            "enabled": True,
            "project": "cs60003-hw2-task3",
            "workspace": None,
            "mode": "cloud",
            "group": "loss-comparison",
            "tags": ["hw2", "task3", "stanford-background", "unet"],
            "description": "HW2 Task3 U-Net loss comparison on Stanford Background.",
        }
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
    """Return the repository root from this package location."""
    return Path(__file__).resolve().parents[3]


def resolve_repo_path(path_value: str | Path, repo_root: Path | None = None) -> Path:
    """Resolve a path relative to the repository root unless absolute."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root or repo_root_from_task_file()) / path


def resolve_paths(config: dict[str, Any], output_root: str | None = None) -> None:
    """Resolve config paths in-place relative to the repository root."""
    config["data"]["root"] = str(resolve_repo_path(config["data"]["root"]))
    config["data"]["split_dir"] = str(resolve_repo_path(config["data"]["split_dir"]))
    if output_root is not None:
        config["experiment"]["output_root"] = output_root
    config["experiment"]["output_root"] = str(resolve_repo_path(config["experiment"]["output_root"]))


def _validate_config(config: dict[str, Any]) -> None:
    if bool(config.get("model", {}).get("pretrained", False)):
        raise ValueError("Task3 requires random initialization; pretrained must be false.")
    loss_name = str(config.get("train", {}).get("loss", "")).lower()
    if loss_name not in {"ce", "dice", "ce_dice", "ce_dice_lovasz"}:
        raise ValueError(f"Unsupported loss: {loss_name}")
    image_size = config.get("data", {}).get("image_size")
    if not isinstance(image_size, list) or len(image_size) != 2:
        raise ValueError("data.image_size must be [height, width].")
