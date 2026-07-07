from __future__ import annotations

import unittest

import numpy as np

from task1_3dgs_aigc.gaussian_filtering import (
    color_anchor_foreground_mask,
    dampen_high_luminance_colors,
    spatial_core_foreground_mask,
)


class GaussianForegroundFilteringTests(unittest.TestCase):
    def test_green_anchor_keeps_multicolor_object_and_removes_far_artifacts(self) -> None:
        rng = np.random.default_rng(7)
        green_object = rng.normal(0.0, 0.08, size=(600, 3)).astype(np.float32)
        logo_object = (
            green_object[:180] + rng.normal(0.0, 0.004, size=(180, 3))
        ).astype(np.float32)
        artifacts = rng.normal(3.0, 0.20, size=(300, 3)).astype(np.float32)
        means = np.concatenate([green_object, logo_object, artifacts], axis=0)
        colors = np.concatenate(
            [
                np.repeat([[0.12, 0.62, 0.20]], len(green_object), axis=0),
                np.repeat([[0.92, 0.20, 0.10]], len(logo_object), axis=0),
                np.repeat([[0.18, 0.54, 0.22]], len(artifacts), axis=0),
            ],
            axis=0,
        ).astype(np.float32)
        opacities = np.full(means.shape[0], 0.9, dtype=np.float32)

        keep, report = color_anchor_foreground_mask(
            means,
            colors,
            opacities,
            min_keep_points=200,
        )

        self.assertTrue(report["applied"])
        self.assertGreater(keep[: len(green_object)].mean(), 0.95)
        self.assertGreater(
            keep[len(green_object) : len(green_object) + len(logo_object)].mean(),
            0.95,
        )
        self.assertLess(keep[-len(artifacts) :].mean(), 0.05)

    def test_spatial_core_fallback_removes_far_non_green_artifacts(self) -> None:
        rng = np.random.default_rng(11)
        milk_box = rng.normal([0.0, 0.0, 0.0], [0.08, 0.05, 0.22], size=(900, 3)).astype(np.float32)
        near_detail = rng.normal([0.02, 0.01, 0.08], [0.03, 0.02, 0.04], size=(120, 3)).astype(np.float32)
        artifacts = rng.normal([1.5, 1.5, 1.5], [0.08, 0.08, 0.08], size=(180, 3)).astype(np.float32)
        means = np.concatenate([milk_box, near_detail, artifacts], axis=0)
        opacities = np.full(means.shape[0], 0.85, dtype=np.float32)

        keep, report = spatial_core_foreground_mask(
            means,
            opacities,
            min_keep_points=500,
        )

        self.assertTrue(report["applied"])
        self.assertGreater(keep[: len(milk_box)].mean(), 0.90)
        self.assertGreater(keep[len(milk_box) : len(milk_box) + len(near_detail)].mean(), 0.80)
        self.assertLess(keep[-len(artifacts) :].mean(), 0.05)

    def test_high_luminance_damping_reduces_bright_splat_speckles(self) -> None:
        colors = np.concatenate(
            [
                np.full((80, 3), [0.42, 0.39, 0.35], dtype=np.float32),
                np.full((20, 3), [0.98, 0.96, 0.93], dtype=np.float32),
            ],
            axis=0,
        )

        damped, report = dampen_high_luminance_colors(
            colors,
            luminance_quantile=0.78,
            target_luminance=0.74,
            blend=0.75,
            max_channel=0.84,
        )

        self.assertTrue(report["applied"])
        self.assertLess(damped[80:].max(), 0.85)
        self.assertLess(damped[80:].mean(), colors[80:].mean() - 0.08)
        self.assertAlmostEqual(float(damped[:80].mean()), float(colors[:80].mean()), delta=0.02)


if __name__ == "__main__":
    unittest.main()
