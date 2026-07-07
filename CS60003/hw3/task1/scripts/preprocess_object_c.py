"""Convert a real object C photo into a centered Zero123 RGBA conditioning image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from rembg import new_session, remove

TASK1_ROOT = Path(__file__).resolve().parents[1]
if str(TASK1_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK1_ROOT))

from task1_3dgs_aigc.image_preprocessing import center_rgba_foreground


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--model", default="isnet-general-use")
    parser.add_argument("--size", default=512, type=int)
    parser.add_argument("--padding-ratio", default=0.10, type=float)
    parser.add_argument("--min-occupancy", default=0.45, type=float)
    parser.add_argument("--max-occupancy", default=0.85, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_heif_opener()
    source = Image.open(args.input).convert("RGB")
    session = new_session(args.model)
    removed = remove(
        source,
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=8,
    ).convert("RGBA")
    cleaned = _clean_alpha(removed)
    output, report = center_rgba_foreground(
        cleaned,
        output_size=args.size,
        padding_ratio=args.padding_ratio,
    )
    occupancy = float(report["foreground_occupancy"])
    report.update(
        {
            "input": args.input.as_posix(),
            "output": args.output.as_posix(),
            "segmentation_model": args.model,
            "thresholds": {
                "min_occupancy": args.min_occupancy,
                "max_occupancy": args.max_occupancy,
            },
            "passed": args.min_occupancy <= occupancy <= args.max_occupancy,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError(
            f"Object C foreground occupancy {occupancy:.3f} is outside "
            f"[{args.min_occupancy}, {args.max_occupancy}]"
        )
    print(json.dumps(report, ensure_ascii=False), flush=True)


def _clean_alpha(image: Image.Image) -> Image.Image:
    rgba = np.asarray(image.convert("RGBA")).copy()
    alpha = np.where(rgba[:, :, 3] > 16, 255, 0).astype(np.uint8)
    alpha = cv2.morphologyEx(
        alpha,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8),
        iterations=2,
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(alpha, connectivity=8)
    if count > 1:
        label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        alpha = np.where(labels == label, 255, 0).astype(np.uint8)
    rgba[:, :, 3] = alpha
    return Image.fromarray(rgba, mode="RGBA")


if __name__ == "__main__":
    main()
