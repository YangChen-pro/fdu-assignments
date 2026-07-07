"""Run threestudio with SwanLab sync for its WandB scalar logger."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from runtime_bootstrap import add_repo_path, ensure_swanlab_api_key, load_env_file, patch_trusted_torch_load, prepare_cuda_extension_env


def parse_args() -> argparse.Namespace:
    """Parse wrapper arguments and preserve the threestudio CLI after `--`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--launch", required=True)
    parser.add_argument("--mode", default="cloud")
    parser.add_argument("--logdir", required=True)
    parser.add_argument("threestudio_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.threestudio_args and args.threestudio_args[0] == "--":
        args.threestudio_args = args.threestudio_args[1:]
    if not args.threestudio_args:
        raise ValueError("Missing threestudio command after --.")
    return args


def main() -> None:
    """Patch WandB logging into SwanLab, then run threestudio launch.py."""
    args = parse_args()
    load_env_file(args.env_file)
    api_key = ensure_swanlab_api_key()
    prepare_cuda_extension_env(args.logdir)
    patch_trusted_torch_load()

    import swanlab

    swanlab.login(api_key=api_key)
    swanlab.sync_wandb(mode=args.mode, wandb_run=False, logdir=args.logdir)
    launch_path = Path(args.launch)
    launch_root = str(launch_path.parent.resolve())
    add_repo_path(launch_root)
    sys.argv = [str(launch_path), *args.threestudio_args]
    runpy.run_path(str(launch_path), run_name="__main__")


if __name__ == "__main__":
    main()
