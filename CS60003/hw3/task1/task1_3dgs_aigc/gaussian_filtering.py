"""Robust foreground filtering for object Gaussian splats."""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def color_anchor_foreground_mask(
    means: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    *,
    mad_multiplier: float = 4.0,
    padding_ratio: float = 0.30,
    min_anchor_points: int = 100,
    min_keep_points: int = 1000,
    anchor_link_ratio: float = 0.014,
    foreground_distance_ratio: float = 0.0025,
) -> tuple[np.ndarray, dict[str, object]]:
    """Keep the spatial object cluster anchored by saturated green splats."""
    means = np.asarray(means, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    opacities = np.asarray(opacities, dtype=np.float32)
    fallback = np.ones(means.shape[0], dtype=bool)
    if means.shape[0] == 0:
        return fallback, {"applied": False, "reason": "empty"}

    red, green, blue = colors.T
    saturation = np.ptp(colors, axis=1)
    anchors = (
        (green > 0.18)
        & (green > red * 1.10)
        & (green > blue * 1.04)
        & (saturation > 0.10)
        & (opacities > 0.25)
    )
    anchor_points = means[anchors]
    if anchor_points.shape[0] < min_anchor_points:
        return fallback, {
            "applied": False,
            "reason": "too_few_anchor_points",
            "anchor_points": int(anchor_points.shape[0]),
        }

    scene_span = np.percentile(means, 99.0, axis=0) - np.percentile(means, 1.0, axis=0)
    scene_diagonal = max(float(np.linalg.norm(scene_span)), 1e-6)
    anchor_tree = cKDTree(anchor_points)
    graph = anchor_tree.sparse_distance_matrix(
        anchor_tree,
        scene_diagonal * float(anchor_link_ratio),
        output_type="coo_matrix",
    )
    _, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels)
    core_points = anchor_points[labels == int(np.argmax(counts))]
    if core_points.shape[0] < min_anchor_points:
        median = np.median(anchor_points, axis=0)
        mad = np.median(np.abs(anchor_points - median), axis=0)
        robust_floor = np.maximum(scene_span * 0.002, 1e-4)
        core = np.all(
            np.abs(anchor_points - median)
            <= float(mad_multiplier) * np.maximum(mad, robust_floor),
            axis=1,
        )
        core_points = anchor_points[core]

    lower = np.percentile(core_points, 1.0, axis=0)
    upper = np.percentile(core_points, 99.0, axis=0)
    padding = (upper - lower) * float(padding_ratio) + scene_span * 0.002
    inside = np.all((means >= lower - padding) & (means <= upper + padding), axis=1)
    distance = cKDTree(core_points).query(means, k=1, workers=-1)[0]
    keep = inside & (
        distance <= scene_diagonal * float(foreground_distance_ratio)
    )
    if int(keep.sum()) < min_keep_points:
        return fallback, {
            "applied": False,
            "reason": "too_few_foreground_points",
            "kept_points": int(keep.sum()),
        }
    return keep, {
        "applied": True,
        "anchor_points": int(anchor_points.shape[0]),
        "core_points": int(core_points.shape[0]),
        "kept_points": int(keep.sum()),
        "scene_diagonal": scene_diagonal,
        "foreground_radius": scene_diagonal * float(foreground_distance_ratio),
        "bounds_min": (lower - padding).tolist(),
        "bounds_max": (upper + padding).tolist(),
    }


def spatial_core_foreground_mask(
    means: np.ndarray,
    opacities: np.ndarray,
    *,
    core_quantile: float = 0.82,
    padding_ratio: float = 0.42,
    min_opacity_quantile: float = 0.35,
    min_keep_points: int = 1000,
) -> tuple[np.ndarray, dict[str, object]]:
    """Keep the dense spatial core when color anchors are unreliable."""
    means = np.asarray(means, dtype=np.float32)
    opacities = np.asarray(opacities, dtype=np.float32)
    fallback = np.ones(means.shape[0], dtype=bool)
    if means.shape[0] == 0:
        return fallback, {"applied": False, "reason": "empty"}

    finite = np.isfinite(means).all(axis=1) & np.isfinite(opacities)
    if int(finite.sum()) < min_keep_points:
        return fallback, {
            "applied": False,
            "reason": "too_few_finite_points",
            "finite_points": int(finite.sum()),
        }

    opacity_threshold = float(np.quantile(opacities[finite], min_opacity_quantile))
    anchors = finite & (opacities >= opacity_threshold)
    if int(anchors.sum()) < min_keep_points:
        anchors = finite
    anchor_points = means[anchors]
    center = np.median(anchor_points, axis=0)
    anchor_distances = np.linalg.norm(anchor_points - center[None, :], axis=1)
    quantile = float(np.clip(core_quantile, 0.50, 0.88))
    core_radius = float(np.quantile(anchor_distances, quantile))
    core_points = anchor_points[anchor_distances <= max(core_radius, 1e-6)]
    if int(core_points.shape[0]) < min_keep_points:
        return fallback, {
            "applied": False,
            "reason": "too_few_core_points",
            "core_points": int(core_points.shape[0]),
        }

    lower = np.percentile(core_points, 1.0, axis=0)
    upper = np.percentile(core_points, 99.0, axis=0)
    span = np.maximum(upper - lower, 1e-6)
    padding = span * float(padding_ratio)
    inside = np.all((means >= lower - padding) & (means <= upper + padding), axis=1)
    keep = finite & inside
    if int(keep.sum()) < min_keep_points:
        return fallback, {
            "applied": False,
            "reason": "too_few_foreground_points",
            "kept_points": int(keep.sum()),
        }
    return keep, {
        "applied": True,
        "anchor_points": int(anchor_points.shape[0]),
        "core_points": int(core_points.shape[0]),
        "kept_points": int(keep.sum()),
        "opacity_threshold": opacity_threshold,
        "center": center.astype(float).tolist(),
        "bounds_min": (lower - padding).astype(float).tolist(),
        "bounds_max": (upper + padding).astype(float).tolist(),
    }


def dampen_high_luminance_colors(
    colors: np.ndarray,
    *,
    luminance_quantile: float = 0.86,
    target_luminance: float = 0.74,
    blend: float = 0.75,
    max_channel: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Reduce isolated over-bright splat colors without changing midtones."""
    colors = np.asarray(colors, dtype=np.float32)
    if colors.shape[0] == 0:
        return colors, {"applied": False, "reason": "empty"}
    luminance = (
        colors[:, 0] * 0.2126
        + colors[:, 1] * 0.7152
        + colors[:, 2] * 0.0722
    )
    finite = np.isfinite(luminance) & np.isfinite(colors).all(axis=1)
    if not np.any(finite):
        return colors, {"applied": False, "reason": "no_finite_colors"}
    threshold = float(np.quantile(luminance[finite], np.clip(luminance_quantile, 0.0, 1.0)))
    high = finite & (luminance >= threshold) & (luminance > target_luminance)
    if not np.any(high):
        return colors, {
            "applied": False,
            "reason": "no_high_luminance_colors",
            "threshold": threshold,
        }
    damped = colors.copy()
    scale = np.minimum(1.0, target_luminance / np.maximum(luminance[high], 1e-6))
    clipped = damped[high] * scale[:, None]
    amount = float(np.clip(blend, 0.0, 1.0))
    damped[high] = clipped * amount + damped[high] * (1.0 - amount)
    if max_channel is not None:
        damped[high] = np.minimum(damped[high], float(max_channel))
    damped = np.clip(damped, 0.0, 1.0).astype(np.float32)
    return damped, {
        "applied": True,
        "threshold": threshold,
        "target_luminance": float(target_luminance),
        "blend": amount,
        "max_channel": None if max_channel is None else float(max_channel),
        "damped_points": int(high.sum()),
        "max_before": float(colors[high].max()),
        "max_after": float(damped[high].max()),
    }
