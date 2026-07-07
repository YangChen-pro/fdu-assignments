"""Train Flowers102 classification models for CS60003 HW2 Task1."""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
from pathlib import Path

import torch
from torch import nn
import torchvision

from flowers102_task1.config import load_config, resolve_repo_path
from flowers102_task1.data import build_loaders, validate_dataset
from flowers102_task1.engine import evaluate, fit
from flowers102_task1.models import build_model, build_parameter_groups
from flowers102_task1.swanlab_utils import create_swanlab_logger
from flowers102_task1.utils import make_run_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML experiment config.")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--output-root", default=None, help="Override output root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["experiment"].get("seed", 42)))
    _resolve_paths(config, args.output_root)

    device = _select_device(args.device)
    dataset_stats = validate_dataset(config["data"]["root"])
    loaders = build_loaders(config["data"], device)
    model = build_model(config["model"]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["train"].get("label_smoothing", 0.0)))
    optimizer = _build_optimizer(model, config["train"])
    scheduler = _build_scheduler(optimizer, config["train"])

    run_dir = make_run_dir(config["experiment"]["output_root"], config["experiment"]["name"])
    shutil.copy2(args.config, run_dir / "source_config.yaml")
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "dataset_stats.json", dataset_stats)
    save_json(run_dir / "env.json", _environment_summary(device))
    print(f"run_dir={run_dir}", flush=True)
    print(f"device={device}", flush=True)

    logger = create_swanlab_logger(config, run_dir)
    try:
        summary = fit(
            model=model,
            loaders=loaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            run_dir=run_dir,
            logger=logger,
        )
        checkpoint = torch.load(summary["best_checkpoint"], map_location=device)
        model.load_state_dict(checkpoint["model"])
        test_metrics = evaluate(
            model=model,
            loader=loaders["test"],
            criterion=criterion,
            device=device,
            num_classes=int(config["model"].get("num_classes", 102)),
            tta=bool(config.get("eval", {}).get("tta", False)),
        )
        final_metrics = {**summary, "test_loss": test_metrics["loss"], "test_acc": test_metrics["acc"]}
        logger.log(
            {
                "best/epoch": int(summary["best_epoch"]),
                "best/val_accuracy": float(summary["best_val_acc"]),
                "test/loss": float(test_metrics["loss"]),
                "test/accuracy": float(test_metrics["acc"]),
            }
        )
        save_json(run_dir / "metrics.json", final_metrics)
        save_json(run_dir / "test_details.json", test_metrics)
        print(f"best_val_acc={summary['best_val_acc']:.4f} test_acc={test_metrics['acc']:.4f}", flush=True)
    finally:
        logger.finish()


def _resolve_paths(config: dict, output_root: str | None) -> None:
    config["data"]["root"] = str(resolve_repo_path(config["data"]["root"]))
    if output_root is not None:
        config["experiment"]["output_root"] = output_root
    config["experiment"]["output_root"] = str(resolve_repo_path(config["experiment"]["output_root"]))


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _build_optimizer(model: torch.nn.Module, train_config: dict) -> torch.optim.Optimizer:
    groups = build_parameter_groups(
        model,
        backbone_lr=float(train_config["backbone_lr"]),
        classifier_lr=float(train_config["classifier_lr"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    name = str(train_config.get("optimizer", "sgd")).lower()
    if name == "sgd":
        return torch.optim.SGD(groups, momentum=float(train_config.get("momentum", 0.9)))
    if name == "adamw":
        return torch.optim.AdamW(groups)
    raise ValueError(f"Unsupported optimizer: {name}")


def _build_scheduler(optimizer: torch.optim.Optimizer, train_config: dict):
    name = str(train_config.get("scheduler", "cosine")).lower()
    if name == "none":
        return None
    if name == "cosine":
        t_max = int(train_config.get("scheduler_t_max", train_config["epochs"]))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    raise ValueError(f"Unsupported scheduler: {name}")


def _environment_summary(device: torch.device) -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
    }


if __name__ == "__main__":
    main()
