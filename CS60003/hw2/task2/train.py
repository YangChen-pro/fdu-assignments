"""Fine-tune YOLOv8 on Road Vehicle Images Dataset for CS60003 HW2 Task2."""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

from road_yolo.config import load_config, resolve_repo_path
from road_yolo.swanlab_utils import build_ultralytics_callback, create_swanlab_logger
from road_yolo.utils import (
    best_epoch_from_history,
    generate_history_from_results_csv,
    make_run_dir,
    plot_history,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML experiment config.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["experiment"].get("seed", 42)))

    if args.output_root:
        config["experiment"]["output_root"] = args.output_root
    config["experiment"]["output_root"] = str(
        resolve_repo_path(config["experiment"]["output_root"])
    )

    device = _select_device(args.device)
    data_yaml = str(
        resolve_repo_path(config["data"]["root"]) / config["data"]["yaml"]
    )

    run_dir = make_run_dir(config["experiment"]["output_root"], config["experiment"]["name"])
    shutil.copy2(args.config, run_dir / "source_config.yaml")
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "env.json", _env_summary(device))
    print(f"run_dir={run_dir}", flush=True)

    logger = create_swanlab_logger(config, run_dir)
    try:
        from ultralytics import YOLO
        model = YOLO(config["model"]["name"])
        callback = build_ultralytics_callback(logger)
        model.add_callback("on_fit_epoch_end", callback)

        train_cfg = config["train"]
        model.train(
            data=data_yaml,
            epochs=int(train_cfg["epochs"]),
            imgsz=int(train_cfg["imgsz"]),
            batch=int(train_cfg["batch"]),
            optimizer=str(train_cfg.get("optimizer", "AdamW")),
            lr0=float(train_cfg.get("lr0", 0.001)),
            lrf=float(train_cfg.get("lrf", 0.01)),
            warmup_epochs=int(train_cfg.get("warmup_epochs", 3)),
            amp=bool(train_cfg.get("amp", True)),
            device=_device_str_for_ultralytics(device),
            project=str(run_dir),
            name="train",
            exist_ok=False,
            verbose=True,
        )

        # Normalise outputs from Ultralytics subdirectory
        train_out = run_dir / "train"
        best_src = train_out / "weights" / "best.pt"
        best_dst = run_dir / "best.pt"
        if best_src.is_file():
            shutil.copy2(best_src, best_dst)

        results_csv = train_out / "results.csv"
        history_csv = run_dir / "history.csv"
        history_rows = generate_history_from_results_csv(results_csv, history_csv)

        curves_path = run_dir / "curves.png"
        if history_rows:
            plot_history(history_csv, curves_path)
            if logger.enabled:
                logger.log_image(
                    "report/curves_with_axis_labels",
                    curves_path,
                    "Task2 loss and mAP curves",
                )

        best_epoch, best_map50, best_map50_95 = best_epoch_from_history(history_rows)
        metrics = {
            "best_epoch": best_epoch,
            "best_mAP50": best_map50,
            "best_mAP50_95": best_map50_95,
            "best_checkpoint": str(best_dst),
        }
        save_json(run_dir / "metrics.json", metrics)
        if logger.enabled:
            logger.log({"best/epoch": best_epoch, "best/mAP50": best_map50, "best/mAP50-95": best_map50_95})

        print(
            f"best_epoch={best_epoch} best_mAP50={best_map50:.4f} best_mAP50_95={best_map50_95:.4f}",
            flush=True,
        )
    finally:
        logger.finish()


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _device_str_for_ultralytics(device: torch.device) -> str:
    """Convert torch.device to the string Ultralytics expects ('0', '1', 'cpu')."""
    if device.type == "cuda":
        return str(device.index) if device.index is not None else "0"
    return "cpu"


def _env_summary(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
    }


if __name__ == "__main__":
    main()
