"""Evaluate a trained Flowers102 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from flowers102_task1.config import load_config, resolve_repo_path
from flowers102_task1.data import build_loaders, validate_dataset
from flowers102_task1.engine import evaluate
from flowers102_task1.models import build_model
from flowers102_task1.utils import save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML experiment config.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--tta", action="store_true", help="Use horizontal-flip test-time augmentation.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["experiment"].get("seed", 42)))
    config["data"]["root"] = str(resolve_repo_path(config["data"]["root"]))
    validate_dataset(config["data"]["root"])

    device = _select_device(args.device)
    loaders = build_loaders(config["data"], device)
    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])

    metrics = evaluate(
        model=model,
        loader=loaders[args.split],
        criterion=nn.CrossEntropyLoss(),
        device=device,
        num_classes=int(config["model"].get("num_classes", 102)),
        tta=args.tta,
    )
    suffix = "eval_tta" if args.tta else "eval"
    output = Path(args.output) if args.output else Path(args.checkpoint).with_name(f"{args.split}_{suffix}.json")
    save_json(output, metrics)
    print(f"split={args.split} loss={metrics['loss']:.4f} acc={metrics['acc']:.4f}", flush=True)


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


if __name__ == "__main__":
    main()
