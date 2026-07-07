"""SwanLab integration helpers for HW2 Task1 experiments."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .config import repo_root_from_task_file


class SwanLabLogger:
    """Small wrapper that keeps SwanLab optional for local code paths."""

    def __init__(self, module: Any | None = None) -> None:
        self._module = module

    @property
    def enabled(self) -> bool:
        """Return whether this logger has an active SwanLab run."""
        return self._module is not None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log scalar metrics to SwanLab if an active run exists."""
        if not self.enabled:
            return
        self._module.log(_scalar_metrics(metrics), step=step)

    def log_image(self, key: str, image_path: Path, caption: str) -> None:
        """Log one image to SwanLab if an active run exists."""
        if not self.enabled:
            return
        image = self._module.Image(str(image_path), caption=caption)
        self._module.log({key: image})

    def finish(self) -> None:
        """Close the active SwanLab run."""
        if self.enabled:
            self._module.finish()


def create_swanlab_logger(
    config: dict[str, Any],
    run_dir: Path,
    *,
    experiment_name: str | None = None,
) -> SwanLabLogger:
    """Create a SwanLab logger from experiment config."""
    settings = config.get("logging", {}).get("swanlab", {})
    if not bool(settings.get("enabled", False)):
        return SwanLabLogger()

    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError("SwanLab logging is enabled but swanlab is not installed.") from exc

    api_key = resolve_swanlab_api_key()
    mode = settings.get("mode", "cloud")
    if mode == "cloud" and not api_key:
        raise RuntimeError("SwanLab cloud logging requires SWANLAB_API_KEY or the project key file.")
    if api_key:
        swanlab.login(api_key=api_key)

    swanlab.init(
        project=settings.get("project") or "cs60003-hw2-task1",
        workspace=settings.get("workspace"),
        experiment_name=experiment_name or config["experiment"]["name"],
        description=settings.get("description"),
        group=settings.get("group"),
        tags=settings.get("tags"),
        config=config,
        logdir=str(run_dir / "swanlab"),
        mode=mode,
        reinit=True,
    )
    return SwanLabLogger(swanlab)


def resolve_swanlab_api_key(repo_root: Path | None = None) -> str:
    """Read the temporary homework SwanLab key from env or project notes."""
    key = os.environ.get("SWANLAB_API_KEY", "").strip()
    if key:
        return key

    notes_path = (repo_root or repo_root_from_task_file()) / ".helloagents/modules/hw2.md"
    if not notes_path.is_file():
        return ""
    text = notes_path.read_text(encoding="utf-8")
    match = re.search(r"SwanLab API key[^`]*`([^`]+)`", text)
    return match.group(1).strip() if match else ""


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float | int]:
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
