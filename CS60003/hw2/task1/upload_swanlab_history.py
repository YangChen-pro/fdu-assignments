"""Replay existing Task1 histories to SwanLab for report screenshots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from flowers102_task1.config import deep_update, resolve_repo_path
from flowers102_task1.swanlab_utils import create_swanlab_logger
from flowers102_task1.utils import plot_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", default=[], help="Run directory to upload.")
    parser.add_argument("--all", action="store_true", help="Upload every output run with history.csv.")
    parser.add_argument("--output-root", default="hw2/task1/outputs")
    parser.add_argument("--project", default="cs60003-hw2-task1")
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--group", default="task1-history-replay")
    parser.add_argument("--mode", default="cloud", choices=("cloud", "local", "offline", "disabled"))
    parser.add_argument("--skip-curves-image", action="store_true", help="Only upload scalar histories.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = _resolve_run_dirs(args)
    if not run_dirs:
        raise RuntimeError("No run directories found to upload.")

    for run_dir in run_dirs:
        _upload_run(run_dir, args)


def _resolve_run_dirs(args: argparse.Namespace) -> list[Path]:
    explicit = [resolve_repo_path(path) for path in args.run_dir]
    if not args.all:
        return explicit

    output_root = resolve_repo_path(args.output_root)
    discovered = sorted(path.parent for path in output_root.glob("*/history.csv"))
    return sorted({*explicit, *discovered})


def _upload_run(run_dir: Path, args: argparse.Namespace) -> None:
    history_path = run_dir / "history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(f"Missing history.csv: {run_dir}")

    config = _load_json(run_dir / "config.json")
    config = deep_update(config, _swanlab_config(args))
    logger = create_swanlab_logger(config, run_dir, experiment_name=f"replay_{run_dir.name}")
    try:
        last_epoch = _upload_history(logger, history_path)
        _upload_final_metrics(logger, run_dir / "metrics.json", last_epoch)
        if not args.skip_curves_image:
            _upload_curves_image(logger, run_dir, history_path)
    finally:
        logger.finish()
    print(f"uploaded {run_dir}", flush=True)


def _swanlab_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "logging": {
            "swanlab": {
                "enabled": True,
                "project": args.project,
                "workspace": args.workspace,
                "mode": args.mode,
                "group": args.group,
                "tags": ["hw2", "task1", "flowers102", "history-replay"],
                "description": "Replay existing official Task1 training history for report visualization.",
            }
        }
    }


def _upload_history(logger: Any, history_path: Path) -> int:
    last_epoch = 0
    with history_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            epoch = int(row["epoch"])
            logger.log(
                {
                    "train/loss": float(row["train_loss"]),
                    "train/accuracy": float(row["train_acc"]),
                    "val/loss": float(row["val_loss"]),
                    "val/accuracy": float(row["val_acc"]),
                    "Loss/train": float(row["train_loss"]),
                    "Loss/validation": float(row["val_loss"]),
                    "Accuracy/train": float(row["train_acc"]),
                    "Accuracy/validation": float(row["val_acc"]),
                    "lr/backbone": float(row["lr_backbone"]),
                    "lr/classifier": float(row["lr_classifier"]),
                },
                step=epoch,
            )
            last_epoch = epoch
    return last_epoch


def _upload_final_metrics(logger: Any, metrics_path: Path, step: int) -> None:
    metrics = _load_json(metrics_path)
    logger.log(
        {
            "best/epoch": int(metrics.get("best_epoch", 0)),
            "best/val_accuracy": float(metrics.get("best_val_acc", 0.0)),
            "test/loss": float(metrics.get("test_loss", 0.0)),
            "test/accuracy": float(metrics.get("test_acc", 0.0)),
        },
        step=step,
    )


def _upload_curves_image(logger: Any, run_dir: Path, history_path: Path) -> None:
    image_path = run_dir / "swanlab_report_curves.png"
    plot_history(history_path, image_path)
    logger.log_image(
        "report/curves_with_axis_labels",
        image_path,
        "Task1 training curves: x-axis is Epoch; y-axes are Loss and Accuracy.",
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return payload


if __name__ == "__main__":
    main()
