"""Evaluate a trained YOLOv8 checkpoint on the validation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from road_yolo.config import load_config, resolve_repo_path
from road_yolo.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", default="val", choices=("val", "train"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_yaml = str(
        resolve_repo_path(config["data"]["root"]) / config["data"]["yaml"]
    )

    import torch
    from ultralytics import YOLO
    model = YOLO(args.checkpoint)
    if args.device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    results = model.val(
        data=data_yaml,
        split=args.split,
        device=device,
        imgsz=int(config["train"]["imgsz"]),
        verbose=True,
    )

    metrics = {
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "checkpoint": args.checkpoint,
        "split": args.split,
    }
    out_path = Path(args.checkpoint).parent / f"eval_{args.split}.json"
    save_json(out_path, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
