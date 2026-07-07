"""Training and evaluation loops for HW2 Task3."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .config import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES
from .metrics import summarize_confusion, update_confusion_matrix
from .utils import append_history, plot_history, save_json
from .visualization import save_prediction_grid


@dataclass
class EpochResult:
    """Aggregated segmentation metrics for one epoch."""

    loss: float
    pixel_acc: float
    miou: float
    per_class_iou: dict[str, float | None]
    confusion_matrix: list[list[int]]


class ModelEma:
    """Maintain an exponential moving average copy of model weights."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA weights and floating buffers from the source model."""
        model_state = model.state_dict()
        for name, ema_value in self.module.state_dict().items():
            model_value = model_state[name].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


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
    num_classes: int,
    ema: ModelEma | None = None,
) -> EpochResult:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    total_seen = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for step, (images, masks, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, masks)
        main_logits = _main_logits(logits)

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
        if ema is not None:
            ema.update(model)

        batch_size = int(images.size(0))
        total_loss += float(loss.item()) * batch_size
        total_seen += batch_size
        confusion = update_confusion_matrix(confusion, main_logits, masks, num_classes=num_classes)

        if log_interval > 0 and step % log_interval == 0:
            metrics = summarize_confusion(confusion, CLASS_NAMES[:num_classes])
            print(
                f"epoch={epoch} step={step}/{len(loader)} "
                f"loss={total_loss / total_seen:.4f} miou={metrics['miou']:.4f}",
                flush=True,
            )

    return _epoch_result(total_loss, total_seen, confusion, num_classes)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    sample_path: Path | None = None,
    max_samples: int = 0,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    tta: bool = False,
    tta_scales: list[float] | None = None,
) -> EpochResult:
    """Evaluate a segmentation model and optionally save prediction samples."""
    model.eval()
    total_loss = 0.0
    total_seen = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    saved_samples = False

    for images, masks, ids in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = _forward_eval(model, images, tta, tta_scales)
        loss = criterion(logits, masks)
        preds = _main_logits(logits).argmax(dim=1)

        batch_size = int(images.size(0))
        total_loss += float(loss.item()) * batch_size
        total_seen += batch_size
        confusion = update_confusion_matrix(confusion, preds, masks, num_classes=num_classes)

        if sample_path is not None and max_samples > 0 and not saved_samples:
            save_prediction_grid(
                images.detach().cpu(),
                masks.detach().cpu(),
                preds.detach().cpu(),
                sample_path,
                mean or [0.485, 0.456, 0.406],
                std or [0.229, 0.224, 0.225],
                list(ids),
                max_samples=max_samples,
            )
            saved_samples = True

    return _epoch_result(total_loss, total_seen, confusion, num_classes)


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
    """Train a model, save best checkpoint and write run artifacts."""
    train_config = config["train"]
    data_config = config["data"]
    epochs = int(train_config["epochs"])
    num_classes = int(config["model"].get("num_classes", NUM_CLASSES))
    log_interval = int(train_config.get("log_interval", 20))
    grad_clip_norm = float(train_config.get("grad_clip_norm", 0.0))
    patience = int(train_config.get("early_stopping_patience", 0))
    use_amp = bool(train_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    history_csv = run_dir / "history.csv"
    ema = _build_ema(model, train_config)

    best_miou = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
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
            num_classes,
            ema,
        )
        eval_model = ema.module if ema is not None else model
        val_result = evaluate(
            eval_model,
            loaders["val"],
            criterion,
            device,
            num_classes,
            tta=bool(config.get("eval", {}).get("tta", False)),
            tta_scales=config.get("eval", {}).get("tta_scales"),
        )
        if scheduler is not None:
            scheduler.step()

        append_history(history_csv, _history_row(epoch, train_result, val_result, optimizer))
        _log_epoch(logger, epoch, train_result, val_result, optimizer)
        _print_epoch_summary(epoch, epochs, train_result, val_result)

        if val_result.miou > best_miou:
            best_miou = val_result.miou
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(best_path, epoch, eval_model, config, val_result)
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"early_stopping epoch={epoch} patience={patience}", flush=True)
                break

    curves_path = run_dir / "curves.png"
    plot_history(history_csv, curves_path)
    if logger is not None:
        logger.log_image("report/curves_with_axis_labels", curves_path, "Task3 loss, mIoU and pixel accuracy curves")

    summary = {
        "best_epoch": best_epoch,
        "best_val_miou": best_miou,
        "best_checkpoint": str(best_path),
    }
    if ema is not None:
        summary["ema_decay"] = ema.decay
    save_json(run_dir / "metrics.json", summary)
    return summary


def _build_ema(model: nn.Module, train_config: dict[str, Any]) -> ModelEma | None:
    decay = float(train_config.get("ema_decay", 0.0))
    if decay <= 0.0:
        return None
    if not 0.0 < decay < 1.0:
        raise ValueError("train.ema_decay must be between 0 and 1.")
    return ModelEma(model, decay)


def _forward_eval(
    model: nn.Module,
    images: torch.Tensor,
    tta: bool,
    tta_scales: list[float] | None = None,
) -> torch.Tensor | list[torch.Tensor]:
    scales = tta_scales or [1.0]
    logits_sum: torch.Tensor | None = None
    count = 0
    for scale in scales:
        scaled_images = _scale_images(images, float(scale))
        logits = _resize_logits(_main_logits(model(scaled_images)), images.shape[-2:])
        logits_sum = logits if logits_sum is None else logits_sum + logits
        count += 1
        if tta:
            flipped_images = torch.flip(scaled_images, dims=[3])
            flipped_logits = torch.flip(_main_logits(model(flipped_images)), dims=[3])
            logits_sum = logits_sum + _resize_logits(flipped_logits, images.shape[-2:])
            count += 1
    if logits_sum is None:
        return model(images)
    return logits_sum / count


def _main_logits(logits: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    if isinstance(logits, torch.Tensor):
        return logits
    return logits[-1]


def _scale_images(images: torch.Tensor, scale: float) -> torch.Tensor:
    if abs(scale - 1.0) < 1.0e-6:
        return images
    height, width = images.shape[-2:]
    scaled_h = max(16, int(round(height * scale / 16)) * 16)
    scaled_w = max(16, int(round(width * scale / 16)) * 16)
    return F.interpolate(images, size=(scaled_h, scaled_w), mode="bilinear", align_corners=False)


def _resize_logits(logits: torch.Tensor, spatial_size: torch.Size) -> torch.Tensor:
    if logits.shape[-2:] == spatial_size:
        return logits
    return F.interpolate(logits, size=spatial_size, mode="bilinear", align_corners=False)


def _epoch_result(total_loss: float, total_seen: int, confusion: torch.Tensor, num_classes: int) -> EpochResult:
    metrics = summarize_confusion(confusion, CLASS_NAMES[:num_classes])
    return EpochResult(
        loss=total_loss / max(total_seen, 1),
        pixel_acc=float(metrics["pixel_acc"]),
        miou=float(metrics["miou"]),
        per_class_iou=metrics["per_class_iou"],
        confusion_matrix=metrics["confusion_matrix"],
    )


def _history_row(epoch: int, train_result: EpochResult, val_result: EpochResult, optimizer: Optimizer) -> dict[str, str | int]:
    return {
        "epoch": epoch,
        "train_loss": f"{train_result.loss:.6f}",
        "train_miou": f"{train_result.miou:.6f}",
        "train_pixel_acc": f"{train_result.pixel_acc:.6f}",
        "val_loss": f"{val_result.loss:.6f}",
        "val_miou": f"{val_result.miou:.6f}",
        "val_pixel_acc": f"{val_result.pixel_acc:.6f}",
        "lr": f"{optimizer.param_groups[0]['lr']:.8f}",
    }


def _log_epoch(logger: Any | None, epoch: int, train_result: EpochResult, val_result: EpochResult, optimizer: Optimizer) -> None:
    if logger is None:
        return
    payload = {
        "train/loss": train_result.loss,
        "train/miou": train_result.miou,
        "train/pixel_accuracy": train_result.pixel_acc,
        "val/loss": val_result.loss,
        "val/miou": val_result.miou,
        "val/pixel_accuracy": val_result.pixel_acc,
        "lr": float(optimizer.param_groups[0]["lr"]),
    }
    for class_name, value in val_result.per_class_iou.items():
        if value is not None:
            payload[f"val_iou/{class_name}"] = float(value)
    logger.log(payload, step=epoch)


def _print_epoch_summary(epoch: int, epochs: int, train_result: EpochResult, val_result: EpochResult) -> None:
    print(
        f"epoch={epoch}/{epochs} train_loss={train_result.loss:.4f} "
        f"train_miou={train_result.miou:.4f} val_loss={val_result.loss:.4f} "
        f"val_miou={val_result.miou:.4f} val_pixel_acc={val_result.pixel_acc:.4f}",
        flush=True,
    )


def _save_checkpoint(path: Path, epoch: int, model: nn.Module, config: dict[str, Any], val_result: EpochResult) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "config": config,
            "val": {
                "loss": val_result.loss,
                "miou": val_result.miou,
                "pixel_acc": val_result.pixel_acc,
                "per_class_iou": val_result.per_class_iou,
                "confusion_matrix": val_result.confusion_matrix,
            },
        },
        path,
    )


def _clip_gradients(model: nn.Module, grad_clip_norm: float) -> None:
    if grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
