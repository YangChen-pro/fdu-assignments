"""Train U-Net models for CS60003 HW2 Task3."""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

from stanford_unet.config import load_config, resolve_paths
from stanford_unet.data import build_loaders, compute_class_weights, create_splits, validate_dataset
from stanford_unet.engine import evaluate, fit
from stanford_unet.losses import build_loss
from stanford_unet.models import build_model
from stanford_unet.swanlab_utils import create_swanlab_logger
from stanford_unet.utils import make_run_dir, save_json, set_seed
from stanford_unet.visualization import save_palette_legend


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML experiment config.")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--output-root", default=None, help="Override output root.")
    return parser.parse_args()


def main() -> None:
    """Train one configured Task3 experiment."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["experiment"].get("seed", 42)))
    resolve_paths(config, args.output_root)

    device = _select_device(args.device)
    create_splits(
        Path(config["data"]["root"]),
        Path(config["data"]["split_dir"]),
        float(config["data"].get("train_ratio", 0.8)),
        seed=int(config["experiment"].get("seed", 42)),
    )
    dataset_stats = validate_dataset(config["data"]["root"], config["data"]["split_dir"])
    loaders = build_loaders(config["data"], device)
    _maybe_add_class_weights(config)
    model = build_model(config["model"]).to(device)
    criterion = build_loss(config["train"], num_classes=int(config["model"].get("num_classes", 8))).to(device)
    optimizer = _build_optimizer(model, config["train"])
    scheduler = _build_scheduler(optimizer, config["train"])

    run_dir = make_run_dir(config["experiment"]["output_root"], config["experiment"]["name"])
    shutil.copy2(args.config, run_dir / "source_config.yaml")
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "dataset_stats.json", dataset_stats)
    save_json(run_dir / "env.json", _environment_summary(device))
    save_palette_legend(run_dir / "palette_legend.png")
    print(f"run_dir={run_dir}", flush=True)
    print(f"device={device}", flush=True)

    logger = create_swanlab_logger(config, run_dir)
    try:
        summary = fit(model, loaders, criterion, optimizer, scheduler, device, config, run_dir, logger)
        checkpoint = torch.load(summary["best_checkpoint"], map_location=device)
        model.load_state_dict(checkpoint["model"])
        val_samples = run_dir / "val_samples.png"
        val_result = evaluate(
            model,
            loaders["val"],
            criterion,
            device,
            num_classes=int(config["model"].get("num_classes", 8)),
            sample_path=val_samples,
            max_samples=int(config.get("eval", {}).get("visualize_samples", 8)),
            mean=config["data"].get("mean"),
            std=config["data"].get("std"),
            tta=bool(config.get("eval", {}).get("tta", False)),
            tta_scales=config.get("eval", {}).get("tta_scales"),
        )
        final_metrics: dict[str, Any] = {
            **summary,
            "val_loss": val_result.loss,
            "val_miou": val_result.miou,
            "val_pixel_acc": val_result.pixel_acc,
            "per_class_iou": val_result.per_class_iou,
            "confusion_matrix": val_result.confusion_matrix,
        }
        save_json(run_dir / "metrics.json", final_metrics)
        logger.log(
            {
                "best/epoch": int(summary["best_epoch"]),
                "best/val_miou": float(summary["best_val_miou"]),
                "final/val_loss": val_result.loss,
                "final/val_miou": val_result.miou,
                "final/val_pixel_accuracy": val_result.pixel_acc,
            }
        )
        logger.log_image("report/validation_samples", val_samples, "Task3 validation predictions")
        logger.log_image("report/palette_legend", run_dir / "palette_legend.png", "Stanford Background color palette")
        print(
            f"best_epoch={summary['best_epoch']} best_val_miou={summary['best_val_miou']:.4f} "
            f"val_miou={val_result.miou:.4f} val_pixel_acc={val_result.pixel_acc:.4f}",
            flush=True,
        )
    finally:
        logger.finish()


def _maybe_add_class_weights(config: dict[str, Any]) -> None:
    method = str(config["train"].get("class_weights", "none")).lower()
    if method in {"", "none", "false"}:
        return
    weights = compute_class_weights(
        config["data"],
        num_classes=int(config["model"].get("num_classes", 8)),
        method=method,
        max_weight=float(config["train"].get("class_weight_max", 2.5)),
    )
    config["train"]["class_weights_values"] = weights
    print(f"class_weights={weights}", flush=True)


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _build_optimizer(model: torch.nn.Module, train_config: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(train_config.get("optimizer", "adamw")).lower()
    lr = float(train_config.get("lr", 3.0e-4))
    weight_decay = float(train_config.get("weight_decay", 1.0e-4))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=float(train_config.get("momentum", 0.9)), weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def _build_scheduler(optimizer: torch.optim.Optimizer, train_config: dict[str, Any]):
    name = str(train_config.get("scheduler", "cosine")).lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_config.get("epochs", 80)))
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.3)
    raise ValueError(f"Unsupported scheduler: {name}")


def _environment_summary(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
    }


if __name__ == "__main__":
    main()
