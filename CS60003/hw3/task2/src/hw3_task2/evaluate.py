from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .data import CalvinActDataset, collate_batch
from .model import build_policy
from .utils import append_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = CalvinActDataset(
        cfg.data.root,
        [cfg.data.eval_split],
        cfg.data.image_size,
        cfg.data.chunk_size,
        args.max_episodes or cfg.data.max_eval_episodes,
        cfg.data.use_wrist_image,
        cfg.train.seed,
        True,
    )
    loader = DataLoader(
        data,
        batch_size=cfg.train.eval_batch_size,
        shuffle=False,
        num_workers=max(1, cfg.train.num_workers // 2),
        pin_memory=True,
        collate_fn=collate_batch,
    )
    model = build_policy(cfg.model, cfg.data.image_size, cfg.data.use_wrist_image, cfg.data.chunk_size, cfg.train.amp).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    total = 0.0
    count = 0
    with torch.no_grad():
        iterator = tqdm(loader, desc=f"eval {args.name} on {cfg.data.eval_split}")
        for batch in iterator:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.train.amp and device.type == "cuda"):
                loss, loss_dict = model(batch)
            total += float(loss_dict["l1_loss"])
            count += 1
            iterator.set_postfix(action_l1=f"{total / max(count, 1):.5f}")
    result = {
        "model_name": args.name,
        "train_split": "+".join(cfg.data.train_splits),
        "eval_split": cfg.data.eval_split,
        "action_l1": total / max(count, 1),
        "num_frames": len(data),
        "num_episodes": len(data.episodes),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
    }
    output = Path(args.output)
    write_json(output.with_suffix(".json"), result)
    append_csv(output.with_suffix(".csv"), result)
    print(result)


if __name__ == "__main__":
    main()
