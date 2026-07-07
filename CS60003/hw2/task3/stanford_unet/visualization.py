"""Visualization utilities for Stanford Background segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import CLASS_NAMES, IGNORE_INDEX

PALETTE = np.array(
    [
        [112, 178, 255],
        [34, 139, 34],
        [128, 128, 128],
        [124, 252, 0],
        [30, 144, 255],
        [205, 133, 63],
        [160, 82, 45],
        [220, 20, 60],
    ],
    dtype=np.uint8,
)
IGNORE_COLOR = np.array([0, 0, 0], dtype=np.uint8)


def mask_to_rgb(mask: torch.Tensor | np.ndarray, ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    """Convert a class-index mask to a color RGB image."""
    mask_np = mask.detach().to("cpu").numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        rgb[mask_np == class_id] = color
    rgb[mask_np == ignore_index] = IGNORE_COLOR
    return rgb


def denormalize_image(image: torch.Tensor, mean: list[float], std: list[float]) -> np.ndarray:
    """Convert a normalized CHW image tensor back to an HWC RGB array."""
    tensor = image.detach().to("cpu").float().clone()
    mean_tensor = torch.tensor(mean, dtype=torch.float32)[:, None, None]
    std_tensor = torch.tensor(std, dtype=torch.float32)[:, None, None]
    tensor = (tensor * std_tensor + mean_tensor).clamp(0.0, 1.0)
    return (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def save_prediction_grid(
    images: torch.Tensor,
    targets: torch.Tensor,
    preds: torch.Tensor,
    output_path: str | Path,
    mean: list[float],
    std: list[float],
    ids: list[str] | tuple[str, ...],
    max_samples: int = 6,
) -> None:
    """Save a grid with original image, ground truth and prediction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample_count = min(max_samples, int(images.shape[0]))
    if sample_count <= 0:
        return
    figure, axes = plt.subplots(sample_count, 3, figsize=(9, 3 * sample_count))
    if sample_count == 1:
        axes = np.asarray([axes])
    for idx in range(sample_count):
        axes[idx, 0].imshow(denormalize_image(images[idx], mean, std))
        axes[idx, 0].set_title(f"Image: {ids[idx]}")
        axes[idx, 1].imshow(mask_to_rgb(targets[idx]))
        axes[idx, 1].set_title("Ground Truth")
        axes[idx, 2].imshow(mask_to_rgb(preds[idx]))
        axes[idx, 2].set_title("Prediction")
        for col in range(3):
            axes[idx, col].axis("off")
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_palette_legend(output_path: str | Path) -> None:
    """Save a compact legend for the fixed segmentation palette."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    patches = [Patch(color=PALETTE[idx] / 255.0, label=name) for idx, name in enumerate(CLASS_NAMES)]
    patches.append(Patch(color=IGNORE_COLOR / 255.0, label="unknown/ignored"))
    figure = plt.figure(figsize=(6, 2.5))
    figure.legend(handles=patches, loc="center", ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
