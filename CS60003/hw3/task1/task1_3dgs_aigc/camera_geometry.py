"""Camera geometry helpers for unified 3D asset normalization."""

from __future__ import annotations

import numpy as np


def apply_similarity_to_camera_poses(
    camera_to_worlds: np.ndarray,
    *,
    scale: float,
    offset: np.ndarray,
) -> np.ndarray:
    """Apply a world-space uniform scale and translation to camera centers."""
    poses = np.asarray(camera_to_worlds, dtype=np.float32).copy()
    if poses.ndim != 3 or poses.shape[1:] not in {(3, 4), (4, 4)}:
        raise ValueError(f"Expected Nx3x4 or Nx4x4 camera poses, got {poses.shape}")
    world_offset = np.asarray(offset, dtype=np.float32)
    if world_offset.shape != (3,):
        raise ValueError(f"Expected a 3D offset, got {world_offset.shape}")
    poses[:, :3, 3] = poses[:, :3, 3] * float(scale) + world_offset
    return poses


def camera_forward_targets(
    camera_to_worlds: np.ndarray,
    *,
    distance: float,
) -> np.ndarray:
    """Return points along each OpenGL/Nerfstudio camera's viewing direction."""
    poses = np.asarray(camera_to_worlds, dtype=np.float32)
    centers = poses[:, :3, 3]
    forward = -poses[:, :3, 2]
    return centers + forward * float(distance)
