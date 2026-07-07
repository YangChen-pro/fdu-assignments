from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from task1_3dgs_aigc.mesh_splats import (
    apply_object_b_crystal_style,
    normal_aligned_quaternions,
    quaternion_normals,
    sample_textured_obj,
    surface_splat_scales,
)


class TexturedMeshSplatTests(unittest.TestCase):
    def test_object_b_style_keeps_jade_stem_and_makes_violet_cap(self) -> None:
        points = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.8], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        colors = np.repeat([[0.18, 0.72, 0.24]], len(points), axis=0).astype(np.float32)

        styled = apply_object_b_crystal_style(points, colors)

        self.assertGreater(styled[:2, 1].mean(), styled[:2, 2].mean())
        self.assertGreater(styled[-2:, 2].mean(), styled[-2:, 1].mean())
        self.assertGreater(float(np.ptp(styled[:, 2])), 0.15)

    def test_samples_colors_from_obj_uv_texture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            texture = np.array(
                [
                    [[255, 0, 0], [0, 255, 0]],
                    [[0, 0, 255], [255, 255, 255]],
                ],
                dtype=np.uint8,
            )
            Image.fromarray(texture, mode="RGB").save(root / "texture.png")
            (root / "material.mtl").write_text(
                "newmtl painted\nmap_Kd texture.png\n",
                encoding="utf-8",
            )
            (root / "model.obj").write_text(
                "\n".join(
                    [
                        "mtllib material.mtl",
                        "v 0 0 0",
                        "v 1 0 0",
                        "v 0 1 0",
                        "vt 0 0",
                        "vt 1 0",
                        "vt 0 1",
                        "vn 0 0 1",
                        "usemtl painted",
                        "f 1/1/1 2/2/1 3/3/1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            samples = sample_textured_obj(root / "model.obj", max_points=256, seed=7)

        self.assertTrue(samples.has_texture)
        self.assertEqual(samples.colors.shape, (256, 3))
        self.assertGreater(float(np.ptp(samples.colors[:, 0])), 0.5)
        self.assertGreater(float(np.ptp(samples.colors[:, 1])), 0.5)
        self.assertGreater(float(np.ptp(samples.colors[:, 2])), 0.5)
        self.assertEqual(samples.texture_path.name, "texture.png")

    def test_surface_splats_are_thin_and_aligned_to_normals(self) -> None:
        normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )
        quaternions = normal_aligned_quaternions(normals)
        recovered = quaternion_normals(quaternions)
        scales = surface_splat_scales(
            point_count=len(normals),
            tangent_scale=0.01,
            thickness_ratio=0.18,
        )

        np.testing.assert_allclose(recovered, normals, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(scales[:, 0], scales[:, 1], atol=1e-7)
        np.testing.assert_allclose(scales[:, 2], scales[:, 0] * 0.18, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
