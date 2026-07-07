"""SwanLab integration for HW2 Task2 (YOLOv8 training)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .config import repo_root_from_task_file


class SwanLabLogger:
    """Thin optional wrapper around an active swanlab run."""

    def __init__(self, module: Any | None = None) -> None:
        self._module = module

    @property
    def enabled(self) -> bool:
        return self._module is not None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled:
            return
        self._module.log(_scalar_metrics(metrics), step=step)

    def log_image(self, key: str, image_path: Path, caption: str) -> None:
        if not self.enabled:
            return
        image = self._module.Image(str(image_path), caption=caption)
        self._module.log({key: image})

    def finish(self) -> None:
        if self.enabled:
            self._module.finish()


def create_swanlab_logger(
    config: dict[str, Any],
    run_dir: Path,
    *,
    experiment_name: str | None = None,
) -> SwanLabLogger:
    """Initialise a SwanLab run from experiment config."""
    settings = config.get("logging", {}).get("swanlab", {})
    if not bool(settings.get("enabled", False)):
        return SwanLabLogger()

    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError("swanlab is not installed but logging.swanlab.enabled=true.") from exc

    api_key = resolve_swanlab_api_key()
    mode = settings.get("mode", "cloud")
    if mode == "cloud" and not api_key:
        raise RuntimeError("SwanLab cloud logging requires SWANLAB_API_KEY env var.")
    if api_key:
        swanlab.login(api_key=api_key)

    swanlab.init(
        project=settings.get("project") or "cs60003-hw2-task2",
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
    """Read the Task2 SwanLab key from SWANLAB_API_KEY env or .helloagents notes."""
    key = os.environ.get("SWANLAB_API_KEY", "").strip()
    if key:
        return key
    notes_path = (repo_root or repo_root_from_task_file()) / ".helloagents/modules/hw2.md"
    if not notes_path.is_file():
        return ""
    text = notes_path.read_text(encoding="utf-8")
    match = re.search(r"SwanLab API [Kk]ey[^`]*`([^`]+)`", text)
    return match.group(1).strip() if match else ""


def build_ultralytics_callback(logger: SwanLabLogger) -> Any:
    """Return an on_fit_epoch_end callback that logs metrics to SwanLab.

    Register with: model.add_callback('on_fit_epoch_end', callback)
    """
    if not logger.enabled:
        return lambda trainer: None

    def on_fit_epoch_end(trainer: Any) -> None:
        epoch = trainer.epoch + 1  # Ultralytics is 0-indexed
        metrics = {k: float(v) for k, v in trainer.metrics.items() if _is_numeric(v)}
        lr = float(trainer.optimizer.param_groups[0]["lr"]) if trainer.optimizer else 0.0
        payload: dict[str, Any] = {
            "train/box_loss": metrics.get("train/box_loss", 0.0),
            "train/cls_loss": metrics.get("train/cls_loss", 0.0),
            "val/box_loss": metrics.get("val/box_loss", 0.0),
            "val/cls_loss": metrics.get("val/cls_loss", 0.0),
            "metrics/mAP50": metrics.get("metrics/mAP50(B)", 0.0),
            "metrics/mAP50-95": metrics.get("metrics/mAP50-95(B)", 0.0),
            "metrics/precision": metrics.get("metrics/precision(B)", 0.0),
            "metrics/recall": metrics.get("metrics/recall(B)", 0.0),
            "lr": lr,
        }
        logger.log(payload, step=epoch)

    return on_fit_epoch_end


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float | int]:
    return {k: v for k, v in metrics.items() if _is_numeric(v)}
