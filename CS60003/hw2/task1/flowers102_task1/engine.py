"""Training and evaluation loops for Flowers102."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .metrics import accuracy, per_class_accuracy, update_confusion_matrix
from .utils import append_history, plot_history, save_json


@dataclass
class EpochResult:
    """Aggregated metrics for one epoch."""

    loss: float
    acc: float


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    log_interval: int,
    grad_clip_norm: float,
    epoch: int,
) -> EpochResult:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, targets)

        if scaler is None:
            loss.backward()
            _clip_gradients(model, grad_clip_norm)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                _clip_gradients(model, grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        batch_size = int(targets.size(0))
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_seen += batch_size

        if log_interval > 0 and step % log_interval == 0:
            print(
                f"epoch={epoch} step={step}/{len(loader)} "
                f"loss={total_loss / total_seen:.4f} acc={total_correct / total_seen:.4f}",
                flush=True,
            )

    return EpochResult(loss=total_loss / total_seen, acc=total_correct / total_seen)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    tta: bool = False,
) -> dict[str, Any]:
    """Evaluate a model and return aggregate plus per-class metrics."""
    model.eval()
    total_loss = 0.0
    total_seen = 0
    total_acc_weighted = 0.0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = _forward_eval(model, images, tta)
        loss = criterion(logits, targets)

        batch_size = int(targets.size(0))
        total_loss += float(loss.item()) * batch_size
        total_seen += batch_size
        total_acc_weighted += accuracy(logits, targets) * batch_size
        confusion = update_confusion_matrix(confusion, logits, targets)

    return {
        "loss": total_loss / total_seen,
        "acc": total_acc_weighted / total_seen,
        "per_class_acc": per_class_accuracy(confusion),
        "confusion_matrix": confusion.tolist(),
    }


def fit(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    device: torch.device,
    config: dict[str, Any],
    run_dir: Path,
    logger: Any | None = None,
) -> dict[str, Any]:
    """Train a model and save best checkpoint plus metrics."""
    epochs = int(config["train"]["epochs"])
    num_classes = int(config["model"].get("num_classes", 102))
    log_interval = int(config["train"].get("log_interval", 20))
    grad_clip_norm = float(config["train"].get("grad_clip_norm", 0.0))
    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    history_csv = run_dir / "history.csv"

    best_val_acc = -1.0
    best_epoch = 0
    best_path = run_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        train_result = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            scaler,
            log_interval,
            grad_clip_norm,
            epoch,
        )
        val_result = evaluate(model, loaders["val"], criterion, device, num_classes)
        if scheduler is not None:
            scheduler.step()

        append_history(history_csv, _history_row(epoch, train_result, val_result, optimizer))
        _log_epoch(logger, epoch, train_result, val_result, optimizer)
        _print_epoch_summary(epoch, epochs, train_result, val_result)

        if float(val_result["acc"]) > best_val_acc:
            best_val_acc = float(val_result["acc"])
            best_epoch = epoch
            _save_checkpoint(best_path, epoch, model, config, val_result)

    plot_history(history_csv, run_dir / "curves.png")
    summary = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_checkpoint": str(best_path),
    }
    save_json(run_dir / "metrics.json", summary)
    return summary


def _forward_eval(model: nn.Module, images: torch.Tensor, tta: bool) -> torch.Tensor:
    logits = model(images)
    if not tta:
        return logits
    flipped = torch.flip(images, dims=[3])
    return (logits + model(flipped)) * 0.5


def _clip_gradients(model: nn.Module, grad_clip_norm: float) -> None:
    if grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)


def _history_row(
    epoch: int,
    train_result: EpochResult,
    val_result: dict[str, Any],
    optimizer: Optimizer,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "train_loss": f"{train_result.loss:.6f}",
        "train_acc": f"{train_result.acc:.6f}",
        "val_loss": f"{val_result['loss']:.6f}",
        "val_acc": f"{val_result['acc']:.6f}",
        "lr_backbone": f"{optimizer.param_groups[0]['lr']:.8f}",
        "lr_classifier": f"{optimizer.param_groups[-1]['lr']:.8f}",
    }


def _log_epoch(
    logger: Any | None,
    epoch: int,
    train_result: EpochResult,
    val_result: dict[str, Any],
    optimizer: Optimizer,
) -> None:
    if logger is None:
        return
    logger.log(
        {
            "train/loss": train_result.loss,
            "train/accuracy": train_result.acc,
            "val/loss": float(val_result["loss"]),
            "val/accuracy": float(val_result["acc"]),
            "lr/backbone": float(optimizer.param_groups[0]["lr"]),
            "lr/classifier": float(optimizer.param_groups[-1]["lr"]),
        },
        step=epoch,
    )


def _print_epoch_summary(
    epoch: int,
    epochs: int,
    train_result: EpochResult,
    val_result: dict[str, Any],
) -> None:
    print(
        f"epoch={epoch}/{epochs} train_acc={train_result.acc:.4f} "
        f"val_acc={val_result['acc']:.4f}",
        flush=True,
    )


def _save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    config: dict[str, Any],
    val_result: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "config": config,
            "val": val_result,
        },
        path,
    )
