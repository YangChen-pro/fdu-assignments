"""Metrics helpers for classification experiments."""

from __future__ import annotations

import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy for a batch."""
    predictions = logits.argmax(dim=1)
    return float((predictions == targets).float().mean().item())


@torch.no_grad()
def update_confusion_matrix(
    matrix: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Accumulate a confusion matrix on CPU."""
    predictions = logits.argmax(dim=1).detach().cpu()
    labels = targets.detach().cpu()
    for label, prediction in zip(labels, predictions):
        matrix[int(label), int(prediction)] += 1
    return matrix


def per_class_accuracy(matrix: torch.Tensor) -> list[float | None]:
    """Compute per-class accuracy from a confusion matrix."""
    values: list[float | None] = []
    for class_index in range(matrix.shape[0]):
        total = int(matrix[class_index].sum().item())
        if total == 0:
            values.append(None)
        else:
            correct = int(matrix[class_index, class_index].item())
            values.append(correct / total)
    return values
