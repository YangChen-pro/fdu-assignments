"""Segmentation metrics for HW2 Task3."""

from __future__ import annotations

from typing import Any

import torch

from .config import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES


def update_confusion_matrix(
    confusion: torch.Tensor,
    logits_or_preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Accumulate a confusion matrix while ignoring unknown pixels."""
    preds = logits_or_preds.argmax(dim=1) if logits_or_preds.ndim == 4 else logits_or_preds
    preds = preds.detach().to("cpu").long().view(-1)
    targets_cpu = targets.detach().to("cpu").long().view(-1)
    valid = (targets_cpu != ignore_index) & (targets_cpu >= 0) & (targets_cpu < num_classes)
    if not bool(valid.any()):
        return confusion
    indices = targets_cpu[valid] * num_classes + preds[valid].clamp(0, num_classes - 1)
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    return confusion + counts.reshape(num_classes, num_classes).to(confusion.dtype)


def summarize_confusion(confusion: torch.Tensor, class_names: list[str] | None = None) -> dict[str, Any]:
    """Return pixel accuracy, per-class IoU and mean IoU."""
    names = class_names or CLASS_NAMES
    matrix = confusion.to(torch.float64)
    true_positive = torch.diag(matrix)
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    union = support + predicted - true_positive
    iou = torch.where(union > 0, true_positive / union.clamp_min(1.0), torch.full_like(union, float("nan")))
    pixel_acc = true_positive.sum() / matrix.sum().clamp_min(1.0)
    valid_iou = iou[~torch.isnan(iou)]
    mean_iou = valid_iou.mean() if valid_iou.numel() else torch.tensor(0.0, dtype=torch.float64)
    return {
        "pixel_acc": float(pixel_acc.item()),
        "miou": float(mean_iou.item()),
        "per_class_iou": {name: _float_or_none(iou[idx]) for idx, name in enumerate(names)},
        "confusion_matrix": confusion.to(torch.int64).tolist(),
    }


def _float_or_none(value: torch.Tensor) -> float | None:
    if torch.isnan(value):
        return None
    return float(value.item())
