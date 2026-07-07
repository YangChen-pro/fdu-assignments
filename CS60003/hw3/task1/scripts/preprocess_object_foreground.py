"""Create foreground-focused images for turntable-style object COLMAP.

The HW3 object A photos are a rotating object on a mostly static tabletop
background. COLMAP expects one static scene, so keeping the background can make
the background dominate matching. This preprocessor removes the low-saturation
background while preserving the real object pixels; COLMAP still estimates poses
from the real multi-view images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input image directory.")
    parser.add_argument("--output", required=True, help="Output foreground image directory.")
    parser.add_argument("--mask-output", required=True, help="Output binary mask directory.")
    parser.add_argument("--background", default="white", choices=["white", "transparent"])
    return parser.parse_args()


def main() -> None:
    """Generate foreground images and masks."""
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    mask_dir = Path(args.mask_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No input images found: {input_dir}")
    for path in images:
        image = _read_rgb(path)
        mask = _foreground_mask(image)
        foreground = _apply_mask(image, mask, transparent=args.background == "transparent")
        _write_image(output_dir / f"{path.stem}.png", foreground)
        cv2.imwrite(str(mask_dir / f"{path.stem}.png"), mask)
    print(f"foreground_images={len(images)} output={output_dir} masks={mask_dir}", flush=True)


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _foreground_mask(image: np.ndarray) -> np.ndarray:
    mask = _grabcut_mask(image)
    if mask.mean() < 8:
        mask = _hsv_fallback_mask(image)
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = _largest_component(mask)
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
    return mask


def _grabcut_mask(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    rect = (int(width * 0.17), int(height * 0.04), int(width * 0.66), int(height * 0.90))
    mask = np.zeros((height, width), np.uint8)
    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros((1, 65), np.float64)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.grabCut(image_bgr, mask, rect, background, foreground, 5, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def _hsv_fallback_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    color_mask = (saturation > 32) & (value > 35)
    dark_object_mask = (value < 115) & _central_prior(image.shape[:2])
    return np.where(color_mask | dark_object_mask, 255, 0).astype(np.uint8)


def _central_prior(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    y, x = np.ogrid[:height, :width]
    cx, cy = width / 2, height / 2
    rx, ry = width * 0.42, height * 0.47
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 < 1.0


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = int(np.argmax(areas)) + 1
    return np.where(labels == label, 255, 0).astype(np.uint8)


def _apply_mask(image: np.ndarray, mask: np.ndarray, *, transparent: bool) -> np.ndarray:
    if transparent:
        alpha = mask[:, :, None]
        return np.concatenate([image, alpha], axis=2)
    background = np.full_like(image, 255)
    return np.where(mask[:, :, None] > 0, image, background)


def _write_image(path: Path, image: np.ndarray) -> None:
    if image.shape[2] == 4:
        encoded = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    else:
        encoded = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), encoded)


if __name__ == "__main__":
    main()
