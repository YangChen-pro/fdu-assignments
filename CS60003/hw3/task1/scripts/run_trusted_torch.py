"""Run a Python script while allowing trusted HW3 checkpoints to load."""

from __future__ import annotations

import argparse
import runpy
import shutil
import sys
from pathlib import Path

from runtime_bootstrap import add_repo_path, patch_trusted_torch_load, prepare_cuda_extension_env


def parse_args() -> argparse.Namespace:
    """Parse the wrapped command and its arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parser.add_argument("--logdir", default="", help="Optional log directory for extension caches.")
    return parser.parse_args()


def main() -> None:
    """Patch torch.load for trusted local HW3 checkpoints, then run command."""
    args = parse_args()
    prepare_cuda_extension_env(args.logdir or None)
    patch_trusted_torch_load()
    command_path = _resolve_command(args.command)
    command_root = str(Path(command_path).resolve().parent)
    add_repo_path(command_root)
    sys.argv = [command_path, *args.args]
    runpy.run_path(command_path, run_name="__main__")


def _resolve_command(command: str) -> str:
    path = Path(command)
    if path.is_file():
        return str(path)
    resolved = shutil.which(command)
    if resolved:
        return resolved
    raise FileNotFoundError(f"Unable to resolve command: {command}")


if __name__ == "__main__":
    main()
