"""General utilities for HW2 Task2 training and tracking."""

from __future__ import annotations

import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(output_root: str | Path, experiment_name: str) -> Path:
    """Create a timestamped output directory for one run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{timestamp}_{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def generate_history_from_results_csv(results_csv: Path, history_csv: Path) -> list[dict[str, str]]:
    """Convert Ultralytics results.csv to our standard history.csv format.

    Ultralytics columns (after strip): epoch, train/box_loss, train/cls_loss,
    train/dfl_loss, metrics/precision(B), metrics/recall(B), metrics/mAP50(B),
    metrics/mAP50-95(B), val/box_loss, val/cls_loss, val/dfl_loss, lr/pg0, ...
    """
    rows: list[dict[str, str]] = []
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # Strip whitespace from header keys (Ultralytics pads them)
        for raw_row in reader:
            row = {k.strip(): v.strip() for k, v in raw_row.items()}
            history_row = {
                "epoch": row.get("epoch", ""),
                "train_box_loss": row.get("train/box_loss", ""),
                "train_cls_loss": row.get("train/cls_loss", ""),
                "val_box_loss": row.get("val/box_loss", ""),
                "val_cls_loss": row.get("val/cls_loss", ""),
                "mAP50": row.get("metrics/mAP50(B)", ""),
                "mAP50_95": row.get("metrics/mAP50-95(B)", ""),
                "precision": row.get("metrics/precision(B)", ""),
                "recall": row.get("metrics/recall(B)", ""),
                "lr": row.get("lr/pg0", ""),
            }
            rows.append(history_row)

    if rows:
        with history_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def best_epoch_from_history(rows: list[dict[str, str]]) -> tuple[int, float, float]:
    """Return (best_epoch, best_mAP50, best_mAP50_95) by peak mAP50."""
    best_epoch, best_map50, best_map50_95 = 0, 0.0, 0.0
    for row in rows:
        try:
            map50 = float(row["mAP50"])
        except (ValueError, KeyError):
            continue
        if map50 > best_map50:
            best_map50 = map50
            best_map50_95 = float(row.get("mAP50_95", 0.0))
            best_epoch = int(row["epoch"])
    return best_epoch, best_map50, best_map50_95


def plot_history(history_csv: Path, output_path: Path) -> None:
    """Plot loss and mAP curves from history.csv."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[dict[str, str]] = []
    with history_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    epochs = [int(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, [float(r["train_box_loss"]) for r in rows], label="train box_loss")
    axes[0].plot(epochs, [float(r["val_box_loss"]) for r in rows], label="val box_loss")
    axes[0].set_title("Box Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, [float(r["mAP50"]) for r in rows], label="mAP50")
    axes[1].plot(epochs, [float(r["mAP50_95"]) for r in rows], label="mAP50-95")
    axes[1].set_title("mAP")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
