"""Replay existing Task2 training histories to SwanLab for report screenshots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from road_yolo.config import deep_update, resolve_repo_path
from road_yolo.swanlab_utils import create_swanlab_logger
from road_yolo.utils import plot_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", default=[], help="Run directory to upload.")
    parser.add_argument("--all", action="store_true", help="Upload every output run with history.csv.")
    parser.add_argument("--output-root", default="hw2/task2/outputs")
    parser.add_argument("--project", default="cs60003-hw2-task2")
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--group", default="task2-history-replay")
    parser.add_argument("--mode", default="cloud", choices=("cloud", "local", "offline", "disabled"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = _resolve_run_dirs(args)
    if not run_dirs:
        raise RuntimeError("No run directories found to upload.")
    for run_dir in run_dirs:
        _upload_run(run_dir, args)


def _resolve_run_dirs(args: argparse.Namespace) -> list[Path]:
    explicit = [resolve_repo_path(p) for p in args.run_dir]
    if not args.all:
        return explicit
    output_root = resolve_repo_path(args.output_root)
    discovered = sorted(p.parent for p in output_root.glob("*/history.csv"))
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
        curves_path = run_dir / "swanlab_report_curves.png"
        plot_history(history_path, curves_path)
        logger.log_image(
            "report/curves_with_axis_labels",
            curves_path,
            "Task2 loss and mAP curves; x-axis = Epoch.",
        )
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
                "tags": ["hw2", "task2", "road-vehicle", "history-replay"],
                "description": "Replay Task2 official training history for report.",
            }
        }
    }


def _upload_history(logger: Any, history_path: Path) -> int:
    last_epoch = 0
    with history_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            epoch = int(row["epoch"])
            logger.log(
                {
                    "train/box_loss": float(row["train_box_loss"]),
                    "train/cls_loss": float(row["train_cls_loss"]),
                    "val/box_loss": float(row["val_box_loss"]),
                    "val/cls_loss": float(row["val_cls_loss"]),
                    "metrics/mAP50": float(row["mAP50"]),
                    "metrics/mAP50-95": float(row["mAP50_95"]),
                    "metrics/precision": float(row["precision"]),
                    "metrics/recall": float(row["recall"]),
                    "lr": float(row["lr"]),
                },
                step=epoch,
            )
            last_epoch = epoch
    return last_epoch


def _upload_final_metrics(logger: Any, metrics_path: Path, step: int) -> None:
    if not metrics_path.is_file():
        return
    metrics = _load_json(metrics_path)
    logger.log(
        {
            "best/epoch": int(metrics.get("best_epoch", 0)),
            "best/mAP50": float(metrics.get("best_mAP50", 0.0)),
            "best/mAP50-95": float(metrics.get("best_mAP50_95", 0.0)),
        },
        step=step,
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
