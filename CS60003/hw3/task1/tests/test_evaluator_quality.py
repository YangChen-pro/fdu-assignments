from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from evaluate import validate_camera_payload, validate_strict_real_outputs, validate_textured_mesh_assets


class StrictTextureValidationTests(unittest.TestCase):
    def test_validation_does_not_require_result_markdown_files(self) -> None:
        source = Path(validate_strict_real_outputs.__code__.co_filename).read_text(encoding="utf-8")

        self.assertNotIn("SWANLAB_RUNS.md", source)

    def test_accepts_foreground_orbit_camera_metadata(self) -> None:
        validate_camera_payload(
            {
                "camera": {
                    "name": "foreground_orbit",
                    "center": [0.0, 0.0, 1.0],
                    "radius": 2.05,
                    "height": 0.96,
                    "target_z": 0.34,
                    "focal_scale": 1.56,
                }
            }
        )

    def test_accepts_uv_textured_object_b_and_c(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            assets = []
            for name in ("object_b", "object_c"):
                mesh_dir = run_dir / "exports" / name / "mesh"
                mesh_dir.mkdir(parents=True)
                obj = mesh_dir / "model.obj"
                mtl = mesh_dir / "model.mtl"
                texture = mesh_dir / "texture.png"
                obj.write_text("mtllib model.mtl\n", encoding="utf-8")
                mtl.write_text("newmtl material\nmap_Kd texture.png\n", encoding="utf-8")
                texture.write_bytes(b"texture")
                assets.append(
                    {
                        "name": name,
                        "source": obj.as_posix(),
                        "texture_source": texture.as_posix(),
                        "color_mode": "uv_texture",
                    }
                )

            validate_textured_mesh_assets(run_dir, {"assets": assets})

    def test_rejects_palette_fallback_for_strict_mesh_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            mesh_dir = run_dir / "exports" / "object_b" / "mesh"
            mesh_dir.mkdir(parents=True)
            obj = mesh_dir / "model.obj"
            obj.write_text("v 0 0 0\n", encoding="utf-8")
            manifest = {
                "assets": [
                    {
                        "name": "object_b",
                        "source": obj.as_posix(),
                        "texture_source": "",
                        "color_mode": "palette_fallback",
                    }
                ]
            }

            with self.assertRaises(ValueError):
                validate_textured_mesh_assets(run_dir, manifest)


if __name__ == "__main__":
    unittest.main()
