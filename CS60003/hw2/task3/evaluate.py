"""Evaluate a trained HW2 Task3 U-Net checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from stanford_unet.data import build_loaders, validate_dataset
from stanford_unet.engine import evaluate
from stanford_unet.losses import build_loss
from stanford_unet.models import build_model
from stanford_unet.utils import save_json, set_seed
from stanford_unet.visualization import save_palette_legend


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint.")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--output-dir", default=None, help="Directory for evaluation outputs.")
    parser.add_argument("--tta-scales", nargs="*", type=float, default=None, help="Optional multi-scale TTA factors.")
    return parser.parse_args()


def main() -> None:
    """Evaluate a saved checkpoint on the validation split."""
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config: dict[str, Any] = checkpoint["config"]
    if args.tta_scales:
        config.setdefault("eval", {})["tta"] = True
        config["eval"]["tta_scales"] = args.tta_scales
    set_seed(int(config["experiment"].get("seed", 42)))
    device = _select_device(args.device)

    loaders = build_loaders(config["data"], device)
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    criterion = build_loss(config["train"], num_classes=int(config["model"].get("num_classes", 8))).to(device)

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "dataset_stats.json", validate_dataset(config["data"]["root"], config["data"]["split_dir"]))
    save_palette_legend(output_dir / "palette_legend.png")
    result = evaluate(
        model,
        loaders["val"],
        criterion,
        device,
        num_classes=int(config["model"].get("num_classes", 8)),
        sample_path=output_dir / "val_samples.png",
        max_samples=int(config.get("eval", {}).get("visualize_samples", 8)),
        mean=config["data"].get("mean"),
        std=config["data"].get("std"),
        tta=bool(config.get("eval", {}).get("tta", False)),
        tta_scales=config.get("eval", {}).get("tta_scales"),
    )
    metrics = {
        "checkpoint": str(checkpoint_path),
        "val_loss": result.loss,
        "val_miou": result.miou,
        "val_pixel_acc": result.pixel_acc,
        "per_class_iou": result.per_class_iou,
        "confusion_matrix": result.confusion_matrix,
    }
    save_json(output_dir / "metrics.json", metrics)
    print(f"val_loss={result.loss:.4f} val_miou={result.miou:.4f} val_pixel_acc={result.pixel_acc:.4f}")


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


if __name__ == "__main__":
    main()
