from __future__ import annotations

import unittest

import numpy as np

from task1_3dgs_aigc.camera_geometry import (
    apply_similarity_to_camera_poses,
    camera_forward_targets,
)


class CameraGeometryTests(unittest.TestCase):
    def test_similarity_transforms_centers_without_rotating_cameras(self) -> None:
        poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
        poses[0, :3, 3] = [1.0, 2.0, 3.0]
        poses[1, :3, 3] = [-1.0, 0.5, 2.0]

        transformed = apply_similarity_to_camera_poses(
            poses,
            scale=2.0,
            offset=np.array([0.5, -1.0, 4.0], dtype=np.float32),
        )

        np.testing.assert_allclose(
            transformed[:, :3, 3],
            [[2.5, 3.0, 10.0], [-1.5, 0.0, 8.0]],
        )
        np.testing.assert_allclose(transformed[:, :3, :3], poses[:, :3, :3])

    def test_forward_targets_follow_negative_camera_z(self) -> None:
        poses = np.eye(4, dtype=np.float32)[None, :, :]
        poses[0, :3, 3] = [1.0, 2.0, 3.0]

        targets = camera_forward_targets(poses, distance=2.5)

        np.testing.assert_allclose(targets, [[1.0, 2.0, 0.5]])


if __name__ == "__main__":
    unittest.main()
