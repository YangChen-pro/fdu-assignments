from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from task1_3dgs_aigc.config import DEFAULT_CONFIG
from task1_3dgs_aigc.real_chain import run_real_chain
from task1_3dgs_aigc.real_chain_export_scripts import export_script
from task1_3dgs_aigc.real_chain_scripts import (
    _object_c_script,
    _render_script,
    _splatfacto_script,
)


class QualityPipelineScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = deepcopy(DEFAULT_CONFIG)
        self.run_dir = Path("/tmp/task1-quality-run")

    def test_object_a_video_keeps_rgb_for_colmap_then_attaches_masks(self) -> None:
        script = _splatfacto_script(self.config, self.run_dir, "object_a")

        self.assertIn("ns-process-data video", script)
        self.assertIn('--data "hw3/milk_task1_1080.m4v"', script)
        self.assertIn("--num-frames-target 80", script)
        process_index = script.index("ns-process-data video")
        mask_index = script.index("attach_object_masks.py")
        train_index = script.index("ns-train splatfacto-big")
        self.assertLess(process_index, mask_index)
        self.assertLess(mask_index, train_index)
        self.assertIn("--min-registration-ratio 0.7", script)
        self.assertIn("--min-occupancy 0.05", script)
        self.assertIn("--max-refined-area-ratio 1.6", script)
        self.assertNotIn("--colmap-cmd", script)

    def test_object_c_is_normalized_before_high_quality_zero123(self) -> None:
        script = _object_c_script(self.config, self.run_dir)

        self.assertIn("preprocess_object_c.py", script)
        self.assertIn('--input "hw3/objectC.HEIC"', script)
        self.assertIn("object_c_rgba.png", script)
        self.assertIn("data.image_path=", script)
        self.assertIn("data.height=[128,256,512]", script)
        self.assertIn("data.width=[128,256,512]", script)
        self.assertIn("data.random_camera.batch_size=[10,2,1]", script)
        self.assertIn("data.random_camera.height=[64,128,192]", script)
        self.assertIn("data.random_camera.width=[64,128,192]", script)
        self.assertIn("data.random_camera.resolution_milestones=[200,300]", script)
        self.assertIn("data.default_elevation_deg=32.0", script)
        self.assertIn("system.renderer.num_samples_per_ray=512", script)
        self.assertIn("system.loss.lambda_normal_smooth=8.0", script)
        self.assertIn("system.loss.lambda_3d_normal_smooth=8.0", script)
        self.assertIn("system.loss.lambda_orient=1.0", script)

    def test_threestudio_exports_uv_textures(self) -> None:
        script = export_script(self.config, self.run_dir)

        self.assertIn("system.exporter.fmt=obj-mtl", script)
        self.assertIn("system.exporter.save_uv=true", script)
        self.assertIn("system.exporter.save_texture=true", script)
        self.assertIn("system.exporter.texture_size=2048", script)
        self.assertIn("missing exported texture", script)

    def test_final_render_script_uses_cuda_bootstrap_and_per_asset_budgets(self) -> None:
        script = _render_script(self.config, self.run_dir)

        self.assertIn("run_trusted_torch.py", script)
        self.assertIn("render_fused_splats.py", script)
        self.assertIn("--object-a-max 450000", script)
        self.assertIn("--object-a-opacity-quantile 0.35", script)
        self.assertIn("--object-a-opacity-mult 1.05", script)
        self.assertIn("--object-b-max 260000", script)
        self.assertIn("--object-c-max 120000", script)
        self.assertIn("--foreground-offset-x 0.45", script)
        self.assertIn("--foreground-offset-y -0.35", script)
        self.assertIn("--foreground-ground-offset 0.15", script)
        self.assertIn("--object-separation 0.34", script)
        self.assertIn("--object-a-offset-y 0.08", script)
        self.assertIn("--object-a-height 0.62", script)
        self.assertIn("--object-b-height 0.72", script)
        self.assertIn("--object-c-height 0.24", script)
        self.assertIn("--object-c-target-extent 0.30", script)
        self.assertIn("--camera-focus foreground", script)
        self.assertIn("--camera-focal-multiplier 1.23", script)
        self.assertIn("--foreground-camera-radius 2.20", script)
        self.assertIn("--foreground-camera-height 1.02", script)
        self.assertIn("--foreground-camera-target-z 0.34", script)
        self.assertIn("--foreground-camera-start-degrees 38.0", script)
        self.assertIn("--foreground-camera-arc-degrees 34.0", script)
        self.assertIn("--object-c-offset-x 0.30", script)
        self.assertIn("--background-clear-width 0.84", script)
        self.assertIn("--background-clear-depth 0.68", script)
        self.assertIn("--background-clear-height 1.25", script)
        self.assertIn("--background-clear-below 0.00", script)
        self.assertIn("--background-clear-surface-keep 0.04", script)
        self.assertIn("--background-clear-shape ellipse", script)
        self.assertIn("--support-mat-points 0", script)
        self.assertIn("--support-mat-width 0.00", script)
        self.assertIn("--support-mat-depth 0.00", script)
        self.assertIn("--support-mat-opacity 0.00", script)
        self.assertIn("--foreground-color-boost 1.20", script)
        self.assertIn("--foreground-rim-strength 0.10", script)
        self.assertIn("--object-a-color-boost 0.82", script)
        self.assertIn("--object-a-opacity-boost 0.94", script)
        self.assertIn("--camera-index-start 45", script)
        self.assertIn("--camera-index-stop 150", script)
        self.assertNotIn("../scripts/render_fused_splats.py", script)

    def test_exported_gaussians_are_not_dataparser_transformed_twice(self) -> None:
        renderer = (
            Path(__file__).resolve().parents[1] / "scripts" / "render_fused_splats.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("apply_dataparser=True", renderer)

    def test_trained_gaussians_are_not_lit_with_quaternion_pseudo_normals(self) -> None:
        renderer = (
            Path(__file__).resolve().parents[1] / "scripts" / "render_fused_splats.py"
        ).read_text(encoding="utf-8")

        self.assertIn("normals=None", renderer)
        self.assertIn("normal_lengths > 0.5", renderer)
        self.assertIn("torch.ones_like(surface_shade)", renderer)
        self.assertIn('default=1.50', renderer)
        self.assertIn('default=0.80', renderer)
        self.assertIn("scale_multiplier=1.0", renderer)
        self.assertIn("scale_max=0.065", renderer)
        self.assertIn("train_split_fraction=1.0", renderer)
        self.assertIn('background_trajectory["focus"]', renderer)
        self.assertIn("--foreground-offset-x", renderer)
        self.assertIn("--camera-focus", renderer)
        self.assertIn("build_foreground_camera_config", renderer)
        self.assertIn("upright_object_a_to_world_z", renderer)
        self.assertIn("spatial_core_foreground_mask", renderer)
        self.assertIn("dampen_object_a_highlights", renderer)
        self.assertIn("--camera-focal-multiplier", renderer)
        self.assertIn('camera_cfg["focal_scale"]', renderer)
        self.assertIn("args.camera_focal_multiplier", renderer)
        self.assertIn("args.camera_focus", renderer)
        self.assertIn("--camera-index-start", renderer)
        self.assertIn("--object-a-offset-x", renderer)
        self.assertIn("clear_background_support_volume", renderer)
        self.assertIn("--background-clear-surface-keep", renderer)
        self.assertIn("--background-clear-shape", renderer)
        self.assertIn("surface_keep=args.background_clear_surface_keep", renderer)
        self.assertIn("shape=args.background_clear_shape", renderer)
        self.assertIn("add_support_mat_to_background", renderer)
        self.assertIn("--support-mat-shape", renderer)
        self.assertIn("--support-mat-color", renderer)
        self.assertIn('"surface_keep": args.background_clear_surface_keep', renderer)
        self.assertIn('"shape": args.background_clear_shape', renderer)
        self.assertIn('"shape": args.support_mat_shape', renderer)
        self.assertIn('"color": args.support_mat_color', renderer)
        self.assertIn("asset_ids", renderer)
        self.assertIn("foreground_mask = asset_ids > 0", renderer)
        self.assertIn('"foreground_highlight"', renderer)
        self.assertIn('"scene_adjustments"', renderer)
        self.assertIn("np.repeat(focus[None, :], len(centers), axis=0)", renderer)
        self.assertIn("object_ground = float(placement_center[2] - 0.45)", renderer)
        self.assertIn("target_height=args.object_a_height", renderer)
        self.assertIn("target_height=args.object_b_height", renderer)
        self.assertIn("target_height=args.object_c_height", renderer)
        self.assertNotIn("bg_bottom + 0.015", renderer)

    def test_plan_mode_marks_pass_when_all_expected_outputs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = deepcopy(DEFAULT_CONFIG)
            config["config_path"] = __file__
            config["experiment"]["output_root"] = temp_dir
            config["experiment"]["name"] = "existing-outputs"
            config["real_chain"]["execution"]["mode"] = "plan"
            config["real_chain"]["data"]["object_c_image"] = __file__
            config["real_chain"]["data"]["background_video"] = "existing-background.mp4"
            run_dir = Path(temp_dir) / "existing-outputs"
            strict_outputs = [
                "exports/object_a/splat/splat.ply",
                "exports/background/splat/splat.ply",
                "exports/object_b/mesh/model.obj",
                "exports/object_c/mesh/model.obj",
                "renders/fused_splats/fused_scene.mp4",
                "renders/fused_splats/fused_scene_manifest.json",
            ]
            for relative_path in strict_outputs:
                path = run_dir / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("artifact", encoding="utf-8")

            summary = run_real_chain(config)

            self.assertEqual(summary["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
