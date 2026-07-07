"""Loss functions for HW2 Task3 semantic segmentation."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .config import IGNORE_INDEX


class DiceLoss(nn.Module):
    """Softmax Dice loss with support for ignored pixels."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = IGNORE_INDEX,
        smooth: float = 1.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute mean Dice loss over classes after masking ignored pixels."""
        probs = torch.softmax(logits, dim=1)
        valid = targets != self.ignore_index
        safe_targets = targets.masked_fill(~valid, 0).long()
        one_hot = F.one_hot(safe_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid_mask = valid.unsqueeze(1).float()
        probs = probs * valid_mask
        one_hot = one_hot * valid_mask
        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dims)
        denominator = torch.sum(probs + one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        if self.class_weights is None:
            return 1.0 - dice.mean()
        weights = self.class_weights.to(dice.device)
        weights = weights / weights.mean().clamp_min(1.0e-6)
        return 1.0 - (dice * weights).sum() / weights.sum().clamp_min(1.0e-6)


class CombinedLoss(nn.Module):
    """Weighted combination of Cross-Entropy and Dice losses."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = IGNORE_INDEX,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: torch.Tensor | None = None,
        lovasz_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index, class_weights=class_weights)
        self.lovasz = LovaszSoftmaxLoss(ignore_index=ignore_index) if lovasz_weight > 0 else None
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.lovasz_weight = lovasz_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted CE + Dice objective."""
        loss = self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)
        if self.lovasz is not None:
            loss = loss + self.lovasz_weight * self.lovasz(logits, targets)
        return loss


class DeepSupervisionLoss(nn.Module):
    """Apply the same segmentation loss to multiple decoder outputs."""

    def __init__(self, base_loss: nn.Module, weights: list[float] | None = None) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights or [0.2, 0.3, 0.5, 1.0]

    def forward(self, logits: torch.Tensor | list[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """Average losses across auxiliary outputs."""
        if isinstance(logits, torch.Tensor):
            return self.base_loss(logits, targets)
        weights = self.weights[-len(logits) :]
        total = logits[-1].new_tensor(0.0)
        weight_sum = 0.0
        for output, weight in zip(logits, weights, strict=True):
            total = total + float(weight) * self.base_loss(output, targets)
            weight_sum += float(weight)
        return total / max(weight_sum, 1.0e-6)


class LovaszSoftmaxLoss(nn.Module):
    """Lovasz-Softmax loss for directly optimizing mIoU."""

    def __init__(self, ignore_index: int = IGNORE_INDEX) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the multiclass Lovasz-Softmax surrogate."""
        probs = torch.softmax(logits, dim=1)
        losses: list[torch.Tensor] = []
        for prob, target in zip(probs, targets, strict=True):
            flat_probs, flat_target = _flatten_probs(prob, target, self.ignore_index)
            if flat_target.numel() > 0:
                losses.append(_lovasz_softmax_flat(flat_probs, flat_target))
        if not losses:
            return logits.new_tensor(0.0)
        return torch.stack(losses).mean()


def build_loss(train_config: dict, num_classes: int, ignore_index: int = IGNORE_INDEX) -> nn.Module:
    """Create the configured segmentation loss."""
    name = str(train_config.get("loss", "ce")).lower()
    class_weights = _class_weights_tensor(train_config.get("class_weights_values"), num_classes)
    if name == "ce":
        return nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
    if name == "dice":
        return DiceLoss(num_classes=num_classes, ignore_index=ignore_index, class_weights=class_weights)
    if name in {"ce_dice", "ce_dice_lovasz"}:
        base_loss = CombinedLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            class_weights=class_weights,
            lovasz_weight=float(train_config.get("lovasz_weight", 0.5)) if name == "ce_dice_lovasz" else 0.0,
        )
        if bool(train_config.get("deep_supervision", False)):
            weights = train_config.get("deep_supervision_weights")
            return DeepSupervisionLoss(base_loss, weights if isinstance(weights, list) else None)
        return base_loss
    raise ValueError(f"Unsupported loss: {name}")


def _class_weights_tensor(values: object, num_classes: int) -> torch.Tensor | None:
    if values is None:
        return None
    if not isinstance(values, list) or len(values) != num_classes:
        raise ValueError("class_weights_values must be a list with one value per class.")
    return torch.tensor([float(value) for value in values], dtype=torch.float32)


def _flatten_probs(prob: torch.Tensor, target: torch.Tensor, ignore_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    prob = prob.permute(1, 2, 0).reshape(-1, prob.shape[0])
    target = target.reshape(-1)
    valid = target != ignore_index
    return prob[valid], target[valid]


def _lovasz_softmax_flat(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    classes = torch.unique(labels)
    for class_id in classes:
        foreground = (labels == class_id).float()
        if foreground.sum() == 0:
            continue
        errors = (foreground - probs[:, int(class_id)]).abs()
        errors_sorted, permutation = torch.sort(errors, descending=True)
        foreground_sorted = foreground[permutation]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(foreground_sorted)))
    if not losses:
        return probs.new_tensor(0.0)
    return torch.stack(losses).mean()


def _lovasz_grad(foreground_sorted: torch.Tensor) -> torch.Tensor:
    positives = foreground_sorted.sum()
    intersection = positives - foreground_sorted.cumsum(0)
    union = positives + (1.0 - foreground_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1.0e-6)
    if foreground_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard
