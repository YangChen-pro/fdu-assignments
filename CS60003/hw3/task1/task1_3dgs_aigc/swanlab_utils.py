"""Optional SwanLab logging helpers for HW3 Task1.

Logging is off by default. When enabled in YAML, the API key must come from the
`SWANLAB_API_KEY` environment variable; this keeps the HW3 temporary key out of
Git-tracked source files and matches the HW2 cloud-recording habit.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from task1_3dgs_aigc.script_utils import load_env_file, resolve_swanlab_config


class NullSwanLabLogger:
    """No-op logger used when SwanLab is disabled."""

    def log(self, metrics: dict[str, Any]) -> None:
        return None

    def finish(self) -> None:
        return None


class SwanLabLogger:
    """Thin wrapper around the SwanLab module."""

    def __init__(self, swanlab_module: Any) -> None:
        self._swanlab = swanlab_module

    def log(self, metrics: dict[str, Any]) -> None:
        self._swanlab.log(metrics)

    def finish(self) -> None:
        self._swanlab.finish()


def create_swanlab_logger(config: dict[str, Any], run_dir: str | Path) -> NullSwanLabLogger | SwanLabLogger:
    """Create a SwanLab logger if `logging.swanlab.enabled` is true."""
    swanlab_config = resolve_swanlab_config(config)
    if not bool(swanlab_config.get("enabled", False)):
        return NullSwanLabLogger()
    if config.get("real_chain", {}).get("execution", {}).get("mode") == "plan":
        return NullSwanLabLogger()

    load_env_file(swanlab_config.get("env_file", ""))
    api_key = os.environ.get("SWANLAB_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SWANLAB_API_KEY is required when SwanLab logging is enabled.")

    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError("Install swanlab before enabling SwanLab logging.") from exc

    swanlab.login(api_key=api_key)
    swanlab.init(
        project=str(swanlab_config.get("project", "cs60003-hw3-task1")),
        experiment_name=str(config["experiment"]["name"]),
        group=str(swanlab_config.get("group", "real-high-quality")),
        mode=str(swanlab_config.get("mode", "cloud")),
        tags=list(swanlab_config.get("tags", [])),
        config=config,
        logdir=str(Path(run_dir) / "swanlab"),
    )
    return SwanLabLogger(swanlab)
