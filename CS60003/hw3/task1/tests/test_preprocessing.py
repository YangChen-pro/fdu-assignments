from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from task1_3dgs_aigc.dataset_masks import attach_mask_paths, registration_report
from task1_3dgs_aigc.image_preprocessing import (
    center_rgba_foreground,
    fill_mask_holes,
    mask_occupancy_report,
    select_refined_mask,
)


class ImagePreprocessingTests(unittest.TestCase):
    def test_centers_off_axis_rgba_subject_on_square_canvas(self) -> None:
        rgba = np.zeros((120, 200, 4), dtype=np.uint8)
        rgba[50:110, 130:190, :3] = [120, 80, 40]
        rgba[50:110, 130:190, 3] = 255

        output, report = center_rgba_foreground(
            Image.fromarray(rgba, mode="RGBA"),
            output_size=256,
            padding_ratio=0.10,
        )

        alpha = np.asarray(output)[:, :, 3]
        ys, xs = np.nonzero(alpha)
        self.assertAlmostEqual(float(xs.mean()), 127.5, delta=2.0)
        self.assertAlmostEqual(float(ys.mean()), 127.5, delta=2.0)
        self.assertGreater(report["foreground_occupancy"], 0.60)
        self.assertLess(report["foreground_occupancy"], 0.80)
        self.assertEqual(report["output_size"], [256, 256])

    def test_attaches_a_mask_path_to_every_registered_frame(self) -> None:
        transforms = {
            "frames": [
                {"file_path": "images/frame_00001.jpg"},
                {"file_path": "./images/frame_00002.png"},
            ]
        }
        mapping = {
            "frame_00001": Path("masks/frame_00001.png"),
            "frame_00002": Path("masks/frame_00002.png"),
        }

        updated = attach_mask_paths(deepcopy(transforms), mapping)

        self.assertEqual(updated["frames"][0]["mask_path"], "masks/frame_00001.png")
        self.assertEqual(updated["frames"][1]["mask_path"], "masks/frame_00002.png")

    def test_registration_report_rejects_too_few_registered_frames(self) -> None:
        report = registration_report(
            extracted_count=60,
            registered_count=30,
            mask_occupancies=[0.25] * 30,
            min_registration_ratio=0.70,
            min_occupancy=0.02,
            max_occupancy=0.80,
        )

        self.assertFalse(report["passed"])
        self.assertAlmostEqual(report["registration_ratio"], 0.5)
        self.assertIn("registration_ratio", report["failures"])

    def test_fills_internal_segmentation_holes_without_expanding_background(self) -> None:
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[2:10, 2:10] = 255
        mask[4:8, 4:8] = 0

        filled = fill_mask_holes(mask)

        self.assertTrue(np.all(filled[2:10, 2:10] == 255))
        self.assertTrue(np.all(filled[:2] == 0))
        self.assertTrue(np.all(filled[:, :2] == 0))

    def test_rejects_refined_mask_when_segmenters_disagree_too_much(self) -> None:
        rembg = np.zeros((20, 20), dtype=np.uint8)
        rembg[6:14, 6:14] = 255
        refined = np.zeros((20, 20), dtype=np.uint8)
        refined[2:18, 2:18] = 255

        selected, metrics = select_refined_mask(
            rembg,
            refined,
            min_area_ratio=0.70,
            max_area_ratio=1.60,
        )

        self.assertIsNone(selected)
        self.assertFalse(metrics["accepted"])
        self.assertEqual(metrics["reason"], "area_ratio")

    def test_falls_back_to_rembg_when_color_refinement_misses_object(self) -> None:
        rembg = np.zeros((20, 20), dtype=np.uint8)
        rembg[5:15, 5:15] = 255
        refined = np.zeros((20, 20), dtype=np.uint8)
        refined[9:11, 9:11] = 255

        selected, metrics = select_refined_mask(
            rembg,
            refined,
            min_area_ratio=0.70,
            max_area_ratio=1.60,
        )

        np.testing.assert_array_equal(selected, rembg)
        self.assertTrue(metrics["accepted"])
        self.assertEqual(metrics["reason"], "rembg_fallback")

    def test_marks_tiny_masks_as_invalid_occupancy(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[9:11, 9:11] = 255

        valid, metrics = mask_occupancy_report(
            mask,
            min_occupancy=0.05,
            max_occupancy=0.80,
        )

        self.assertFalse(valid)
        self.assertEqual(metrics["reason"], "mask_occupancy")
        self.assertAlmostEqual(metrics["occupancy"], 0.01)

    def test_prefers_color_refined_mask_when_area_agrees(self) -> None:
        rembg = np.zeros((20, 20), dtype=np.uint8)
        rembg[5:15, 5:15] = 255
        refined = np.zeros((20, 20), dtype=np.uint8)
        refined[4:16, 4:16] = 255

        selected, metrics = select_refined_mask(
            rembg,
            refined,
            min_area_ratio=0.70,
            max_area_ratio=1.60,
        )

        np.testing.assert_array_equal(selected, refined)
        self.assertTrue(metrics["accepted"])


if __name__ == "__main__":
    unittest.main()
