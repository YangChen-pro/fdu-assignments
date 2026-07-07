from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from upload_modelscope import _find_weight_files, _prune_remote_subdir


class FakeHubApi:
    def __init__(self, files: list[dict[str, str]]) -> None:
        self.files = files
        self.deleted: list[str] = []

    def list_repo_files(
        self,
        repo_id: str,
        repo_type: str,
        revision: str = "master",
        recursive: bool = True,
    ) -> list[dict[str, str]]:
        del repo_id, repo_type, revision, recursive
        return self.files

    def delete_files(
        self,
        repo_id: str,
        repo_type: str,
        file_paths: list[str],
        revision: str = "master",
    ) -> dict[str, object]:
        del repo_id, repo_type, revision
        self.deleted.extend(file_paths)
        return {"deleted_files": file_paths, "failed_files": []}


class ModelScopeUploadTests(unittest.TestCase):
    def test_selects_exactly_six_current_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            expected = [
                run_dir / "exports/background/splat/splat.ply",
                run_dir / "exports/object_a/splat/splat.ply",
                run_dir
                / "nerfstudio/background/background/splatfacto/new/nerfstudio_models/step-000029999.ckpt",
                run_dir
                / "nerfstudio/object_a/object_a/splatfacto/final/nerfstudio_models/step-000029999.ckpt",
                run_dir / "object_b_threestudio/object_b/new/ckpts/last.ckpt",
                run_dir / "object_c_zero123/object_c/new/ckpts/last.ckpt",
            ]
            old_background = (
                run_dir
                / "nerfstudio/background/background/splatfacto/old/nerfstudio_models/step-000029999.ckpt"
            )
            old_object_b = run_dir / "object_b_threestudio/object_b/old/ckpts/last.ckpt"
            for path in [*expected, old_background, old_object_b, run_dir / "unrelated/model.pt"]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"weight")

            os.utime(old_background, (1, 1))
            os.utime(expected[2], (2, 2))
            os.utime(old_object_b, (1, 1))
            os.utime(expected[4], (2, 2))

            selected = _find_weight_files(run_dir)

            self.assertEqual(selected, sorted(expected))

    def test_replacement_deletes_only_stale_files_below_requested_subdir(self) -> None:
        api = FakeHubApi(
            [
                {"Path": "README.md", "Type": "blob"},
                {"Path": "hw3/task1/real_high_quality", "Type": "tree"},
                {
                    "Path": "hw3/task1/real_high_quality/old/checkpoint.ckpt",
                    "Type": "blob",
                },
                {
                    "Path": "hw3/task1/real_high_quality/exports/object_a/splat/splat.ply",
                    "Type": "blob",
                },
                {"Path": "hw3/task2/act_splitA/final.pt", "Type": "blob"},
            ]
        )
        git_deleted: list[str] = []

        deleted = _prune_remote_subdir(
            api,
            model_id="youngchen/CS60003",
            remote_subdir="real_high_quality",
            keep_paths={
                "hw3/task1/real_high_quality/exports/object_a/splat/splat.ply",
            },
            delete_files=git_deleted.extend,
        )

        self.assertEqual(
            deleted,
            [
                "hw3/task1/real_high_quality/old/checkpoint.ckpt",
            ],
        )
        self.assertEqual(git_deleted, deleted)
        self.assertEqual(api.deleted, [])


if __name__ == "__main__":
    unittest.main()
