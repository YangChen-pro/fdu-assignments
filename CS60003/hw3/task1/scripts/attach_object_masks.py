"""Generate object masks after COLMAP and attach them to Nerfstudio metadata."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove

TASK1_ROOT = Path(__file__).resolve().parents[1]
if str(TASK1_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK1_ROOT))

from task1_3dgs_aigc.dataset_masks import attach_mask_paths, registration_report
from task1_3dgs_aigc.image_preprocessing import (
    fill_mask_holes,
    mask_occupancy_report,
    select_refined_mask,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--model", default="isnet-general-use")
    parser.add_argument("--expected-frames", required=True, type=int)
    parser.add_argument("--min-registration-ratio", default=0.70, type=float)
    parser.add_argument("--min-occupancy", default=0.02, type=float)
    parser.add_argument("--max-occupancy", default=0.80, type=float)
    parser.add_argument("--min-refined-area-ratio", default=0.70, type=float)
    parser.add_argument("--max-refined-area-ratio", default=1.60, type=float)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transforms_path = args.processed_dir / "transforms.json"
    source_transforms_path = args.processed_dir / "transforms_colmap.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"Missing Nerfstudio transforms: {transforms_path}")
    if not source_transforms_path.exists():
        shutil.copy2(transforms_path, source_transforms_path)
    transforms = json.loads(source_transforms_path.read_text(encoding="utf-8"))
    frames = transforms.get("frames", [])
    if not frames:
        raise ValueError(f"No registered frames in {transforms_path}")

    images_dir = args.processed_dir / "images"
    extracted = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    extracted_count = max(len(extracted), args.expected_frames)
    for directory in args.processed_dir.glob("masks*"):
        if directory.is_dir():
            shutil.rmtree(directory)
    masks_dir = args.processed_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    session = new_session(args.model)

    mapping: dict[str, Path] = {}
    occupancies: list[float] = []
    accepted_frames: list[dict] = []
    rejected_frames: list[dict] = []
    for frame in frames:
        relative_image = Path(str(frame["file_path"]).removeprefix("./"))
        image_path = args.processed_dir / relative_image
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing registered image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        raw_mask = remove(
            image,
            session=session,
            only_mask=True,
            post_process_mask=True,
        )
        rembg_mask = _clean_mask(np.asarray(raw_mask.convert("L")))
        refined_mask = _green_object_hull(np.asarray(image), rembg_mask)
        mask, refinement = select_refined_mask(
            rembg_mask,
            refined_mask,
            min_area_ratio=args.min_refined_area_ratio,
            max_area_ratio=args.max_refined_area_ratio,
        )
        if mask is None:
            rejected_frames.append({"frame": image_path.stem, **refinement})
            continue
        occupancy = float(np.count_nonzero(mask) / mask.size)
        occupancy_valid, occupancy_metrics = mask_occupancy_report(
            mask,
            min_occupancy=args.min_occupancy,
            max_occupancy=args.max_occupancy,
        )
        if not occupancy_valid:
            rejected_frames.append({"frame": image_path.stem, **occupancy_metrics})
            continue
        occupancies.append(occupancy)
        mask_path = masks_dir / f"{image_path.stem}.png"
        Image.fromarray(mask, mode="L").save(mask_path)
        _write_downscaled_masks(args.processed_dir, mask_path, mask)
        mapping[image_path.stem] = mask_path.relative_to(args.processed_dir)
        accepted_frames.append(frame)

    filtered_transforms = dict(transforms)
    filtered_transforms["frames"] = accepted_frames
    updated = attach_mask_paths(filtered_transforms, mapping)
    report = registration_report(
        extracted_count=extracted_count,
        registered_count=len(accepted_frames),
        mask_occupancies=occupancies,
        min_registration_ratio=args.min_registration_ratio,
        min_occupancy=args.min_occupancy,
        max_occupancy=args.max_occupancy,
    )
    report["segmentation_model"] = args.model
    report["colmap_registered_frames"] = len(frames)
    report["rejected_frames"] = rejected_frames
    report["refinement_thresholds"] = {
        "min_area_ratio": args.min_refined_area_ratio,
        "max_area_ratio": args.max_refined_area_ratio,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError(f"Object A preprocessing validation failed: {report['failures']}")
    transforms_path.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    binary = np.where(mask > 127, 255, 0).astype(np.uint8)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        np.ones((11, 11), np.uint8),
        iterations=2,
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        np.ones((5, 5), np.uint8),
        iterations=1,
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count > 1:
        label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        binary = np.where(labels == label, 255, 0).astype(np.uint8)
    binary = fill_mask_holes(binary)
    return cv2.dilate(binary, np.ones((5, 5), np.uint8), iterations=1)


def _green_object_hull(image_rgb: np.ndarray, rembg_mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    green = (
        (hsv[:, :, 0] >= 28)
        & (hsv[:, :, 0] <= 105)
        & (hsv[:, :, 1] >= 45)
        & (hsv[:, :, 2] >= 25)
    )
    min_dimension = min(image_rgb.shape[:2])
    kernel_size = max(15, int(round(min_dimension * 0.028)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    green_mask = np.where(green, 255, 0).astype(np.uint8)
    green_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_CLOSE,
        np.ones((kernel_size, kernel_size), np.uint8),
        iterations=2,
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(green_mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(rembg_mask)
    best_label = max(
        range(1, count),
        key=lambda label: (
            np.count_nonzero((labels == label) & (rembg_mask > 0))
            + 0.05 * stats[label, cv2.CC_STAT_AREA]
        ),
    )
    component = np.where(labels == best_label, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(rembg_mask)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    result = np.zeros_like(rembg_mask)
    cv2.drawContours(result, [hull], -1, 255, thickness=-1)
    return cv2.dilate(result, np.ones((3, 3), np.uint8), iterations=1)


def _write_downscaled_masks(processed_dir: Path, source: Path, mask: np.ndarray) -> None:
    for image_dir in sorted(processed_dir.glob("images_*")):
        suffix = image_dir.name.removeprefix("images_")
        if not suffix.isdigit():
            continue
        matching_images = list(image_dir.glob(f"{source.stem}.*"))
        if not matching_images:
            continue
        with Image.open(matching_images[0]) as image:
            size = image.size
        target_dir = processed_dir / f"masks_{suffix}"
        target_dir.mkdir(parents=True, exist_ok=True)
        resized = Image.fromarray(mask, mode="L").resize(size, Image.Resampling.NEAREST)
        resized.save(target_dir / source.name)


if __name__ == "__main__":
    main()
