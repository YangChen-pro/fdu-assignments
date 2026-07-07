"""General utilities for HW2 Task3."""

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
    """Set random seeds for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(output_root: str | Path, experiment_name: str) -> Path:
    """Create a timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{timestamp}_{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def append_history(path: str | Path, row: dict[str, Any]) -> None:
    """Append one row to a CSV history file."""
    csv_path = Path(path)
    exists = csv_path.is_file()
    with csv_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def plot_history(history_csv: str | Path, output_path: str | Path) -> None:
    """Plot loss, mIoU and pixel accuracy curves with explicit axis labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_history_rows(history_csv)
    if not rows:
        return

    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], label="train")
    axes[0].plot(epochs, [float(row["val_loss"]) for row in rows], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, [float(row["train_miou"]) for row in rows], label="train")
    axes[1].plot(epochs, [float(row["val_miou"]) for row in rows], label="val")
    axes[1].set_title("Mean IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].legend()

    axes[2].plot(epochs, [float(row["train_pixel_acc"]) for row in rows], label="train")
    axes[2].plot(epochs, [float(row["val_pixel_acc"]) for row in rows], label="val")
    axes[2].set_title("Pixel Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def read_history(path: str | Path) -> list[dict[str, str]]:
    """Read a history CSV into rows."""
    return _read_history_rows(path)


def _read_history_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))
