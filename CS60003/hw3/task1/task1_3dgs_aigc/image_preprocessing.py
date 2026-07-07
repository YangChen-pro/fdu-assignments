"""Pure image transformations shared by Task1 preprocessing scripts."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def select_refined_mask(
    rembg_mask: np.ndarray,
    refined_mask: np.ndarray,
    *,
    min_area_ratio: float,
    max_area_ratio: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Prefer a color-refined mask, but fall back when color refinement misses."""
    rembg_area = int(np.count_nonzero(rembg_mask))
    refined_area = int(np.count_nonzero(refined_mask))
    ratio = refined_area / max(rembg_area, 1)
    accepted_refined = rembg_area > 0 and min_area_ratio <= ratio <= max_area_ratio
    if accepted_refined:
        selected = np.asarray(refined_mask, dtype=np.uint8)
        accepted = True
        reason = ""
    elif rembg_area > 0 and ratio < min_area_ratio:
        selected = np.asarray(rembg_mask, dtype=np.uint8)
        accepted = True
        reason = "rembg_fallback"
    else:
        selected = None
        accepted = False
        reason = "area_ratio"

    return (
        selected,
        {
            "accepted": accepted,
            "reason": reason,
            "rembg_area": rembg_area,
            "refined_area": refined_area,
            "area_ratio": float(ratio),
        },
    )


def mask_occupancy_report(
    mask: np.ndarray,
    *,
    min_occupancy: float,
    max_occupancy: float,
) -> tuple[bool, dict[str, Any]]:
    """Check whether a foreground mask occupies a plausible image fraction."""
    occupancy = float(np.count_nonzero(mask) / max(np.asarray(mask).size, 1))
    valid = min_occupancy <= occupancy <= max_occupancy
    return (
        valid,
        {
            "reason": "" if valid else "mask_occupancy",
            "occupancy": occupancy,
            "min_occupancy": float(min_occupancy),
            "max_occupancy": float(max_occupancy),
        },
    )


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill zero-valued regions that are enclosed by a binary foreground mask."""
    binary = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        padded = np.pad(binary, 1, mode="constant", constant_values=0)
        flooded = padded.copy()
        flood_mask = np.zeros(
            (padded.shape[0] + 2, padded.shape[1] + 2),
            dtype=np.uint8,
        )
        cv2.floodFill(flooded, flood_mask, (0, 0), 255)
        holes = cv2.bitwise_not(flooded)[1:-1, 1:-1]
        return cv2.bitwise_or(binary, holes)

    height, width = binary.shape
    outside = np.zeros_like(binary, dtype=bool)
    stack = [
        (y, x)
        for y in range(height)
        for x in range(width)
        if (y in {0, height - 1} or x in {0, width - 1}) and binary[y, x] == 0
    ]
    while stack:
        y, x = stack.pop()
        if outside[y, x] or binary[y, x] != 0:
            continue
        outside[y, x] = True
        if y > 0:
            stack.append((y - 1, x))
        if y + 1 < height:
            stack.append((y + 1, x))
        if x > 0:
            stack.append((y, x - 1))
        if x + 1 < width:
            stack.append((y, x + 1))
    holes = (binary == 0) & ~outside
    result = binary.copy()
    result[holes] = 255
    return result


def center_rgba_foreground(
    image: Image.Image,
    *,
    output_size: int,
    padding_ratio: float,
    alpha_threshold: int = 8,
) -> tuple[Image.Image, dict[str, Any]]:
    """Crop an RGBA foreground and center it on a transparent square canvas."""
    rgba = image.convert("RGBA")
    array = np.asarray(rgba)
    alpha = array[:, :, 3]
    ys, xs = np.nonzero(alpha > alpha_threshold)
    if xs.size == 0:
        raise ValueError("Foreground alpha mask is empty.")

    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    crop = rgba.crop((left, top, right, bottom))
    crop_width, crop_height = crop.size
    side = max(crop_width, crop_height)
    padded_side = max(1, int(round(side * (1.0 + 2.0 * padding_ratio))))
    canvas = Image.new("RGBA", (padded_side, padded_side), (0, 0, 0, 0))
    offset = ((padded_side - crop_width) // 2, (padded_side - crop_height) // 2)
    canvas.alpha_composite(crop, dest=offset)
    output = canvas.resize((output_size, output_size), Image.Resampling.LANCZOS)

    output_alpha = np.asarray(output)[:, :, 3]
    occupancy = float(np.count_nonzero(output_alpha > alpha_threshold) / output_alpha.size)
    report = {
        "source_size": [int(rgba.width), int(rgba.height)],
        "alpha_bbox": [left, top, right, bottom],
        "crop_size": [crop_width, crop_height],
        "padded_side": padded_side,
        "output_size": [output_size, output_size],
        "foreground_occupancy": occupancy,
        "padding_ratio": float(padding_ratio),
    }
    return output, report
