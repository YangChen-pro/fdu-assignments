"""Run Nerfstudio training with SwanLab TensorBoard scalar sync enabled."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from runtime_bootstrap import ensure_swanlab_api_key, load_env_file, prepare_cuda_extension_env


def parse_args() -> argparse.Namespace:
    """Parse wrapper arguments and preserve the Nerfstudio CLI after `--`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--mode", default="cloud")
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("nerfstudio_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.nerfstudio_args and args.nerfstudio_args[0] == "--":
        args.nerfstudio_args = args.nerfstudio_args[1:]
    if not args.nerfstudio_args:
        raise ValueError("Missing Nerfstudio command after --.")
    return args


def main() -> None:
    """Initialize SwanLab, sync TensorBoard scalars, then run Nerfstudio."""
    args = parse_args()
    load_env_file(args.env_file)
    api_key = ensure_swanlab_api_key()
    prepare_cuda_extension_env(args.logdir)

    import swanlab
    from nerfstudio.scripts.train import entrypoint

    swanlab.login(api_key=api_key)
    swanlab.init(
        project=args.project,
        experiment_name=args.experiment_name,
        group=args.group,
        mode=args.mode,
        tags=args.tag,
        logdir=args.logdir,
        config={"argv": args.nerfstudio_args},
    )
    swanlab.sync_tensorboard_torch(types=["scalar", "scalars"])
    try:
        sys.argv = args.nerfstudio_args
        if Path(sys.argv[0]).name != "ns-train":
            sys.argv = ["ns-train", *sys.argv]
        entrypoint()
    finally:
        swanlab.finish()


if __name__ == "__main__":
    main()
