"""Helpers for attaching foreground masks to Nerfstudio datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def attach_mask_paths(
    transforms: dict[str, Any],
    masks_by_stem: dict[str, Path],
) -> dict[str, Any]:
    """Attach one relative mask path to every registered frame."""
    frames = transforms.get("frames", [])
    for frame in frames:
        stem = Path(str(frame["file_path"])).stem
        if stem not in masks_by_stem:
            raise KeyError(f"Missing mask for registered frame: {stem}")
        frame["mask_path"] = masks_by_stem[stem].as_posix()
    return transforms


def registration_report(
    *,
    extracted_count: int,
    registered_count: int,
    mask_occupancies: list[float],
    min_registration_ratio: float,
    min_occupancy: float,
    max_occupancy: float,
) -> dict[str, Any]:
    """Build the quality report used to gate object A training."""
    ratio = registered_count / max(extracted_count, 1)
    invalid_masks = [
        index
        for index, occupancy in enumerate(mask_occupancies)
        if occupancy < min_occupancy or occupancy > max_occupancy
    ]
    failures: list[str] = []
    if ratio < min_registration_ratio:
        failures.append("registration_ratio")
    if registered_count == 0 or len(mask_occupancies) != registered_count:
        failures.append("mask_count")
    if invalid_masks:
        failures.append("mask_occupancy")
    return {
        "passed": not failures,
        "extracted_frames": int(extracted_count),
        "registered_frames": int(registered_count),
        "registration_ratio": float(ratio),
        "masked_frames": len(mask_occupancies),
        "mask_occupancy_min": float(min(mask_occupancies, default=0.0)),
        "mask_occupancy_max": float(max(mask_occupancies, default=0.0)),
        "mask_occupancy_mean": float(
            sum(mask_occupancies) / max(len(mask_occupancies), 1)
        ),
        "invalid_mask_indices": invalid_masks,
        "thresholds": {
            "min_registration_ratio": float(min_registration_ratio),
            "min_occupancy": float(min_occupancy),
            "max_occupancy": float(max_occupancy),
        },
        "failures": failures,
    }
