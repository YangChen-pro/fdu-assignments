from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import runtime_bootstrap


class RuntimeBootstrapTests(unittest.TestCase):
    def test_prefers_system_cuda_toolkit_over_conda_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            system_cuda = root / "cuda"
            conda_prefix = root / "conda"
            (system_cuda / "bin").mkdir(parents=True)
            (system_cuda / "bin" / "nvcc").touch()
            (conda_prefix / "bin").mkdir(parents=True)

            env = {
                "CONDA_PREFIX": str(conda_prefix),
                "PATH": "/usr/bin:/bin",
            }
            with (
                patch.dict(os.environ, env, clear=True),
                patch.object(runtime_bootstrap, "SYSTEM_CUDA_HOME", system_cuda),
            ):
                runtime_bootstrap.prepare_cuda_extension_env()

                self.assertEqual(os.environ["CUDA_HOME"], str(system_cuda.resolve()))
                self.assertEqual(os.environ["PATH"].split(":")[0], str(system_cuda.resolve() / "bin"))
                self.assertIn(str(Path(sys.executable).resolve().parent), os.environ["PATH"].split(":"))


if __name__ == "__main__":
    unittest.main()
