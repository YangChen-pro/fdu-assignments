from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .data import CalvinActDataset, collate_batch
from .model import build_policy


class BreakdownAccumulator:
    def __init__(self) -> None:
        self.task_sums: dict[int, float] = defaultdict(float)
        self.task_counts: dict[int, float] = defaultdict(float)
        self.episode_sums: dict[int, float] = defaultdict(float)
        self.episode_counts: dict[int, float] = defaultdict(float)
        self.action_dim_sums: torch.Tensor | None = None
        self.action_dim_counts: torch.Tensor | None = None

    def update(self, batch: dict[str, torch.Tensor], pred: torch.Tensor) -> None:
        target = batch["actions"]
        valid = batch["valid"].bool()
        abs_err = torch.abs(pred - target).float()
        masked = abs_err * valid.unsqueeze(-1)
        self._update_action_dims(masked, valid, abs_err.shape[-1])
        frame_sums = masked.sum(dim=(1, 2)).detach().cpu()
        frame_counts = (valid.sum(dim=1).float() * abs_err.shape[-1]).detach().cpu()
        self._update_group(self.task_sums, self.task_counts, batch["task_index"], frame_sums, frame_counts)
        self._update_group(self.episode_sums, self.episode_counts, batch["episode_index"], frame_sums, frame_counts)

    def overall(self) -> float:
        return sum(self.task_sums.values()) / max(sum(self.task_counts.values()), 1.0)

    def _update_action_dims(self, masked: torch.Tensor, valid: torch.Tensor, action_dim: int) -> None:
        dim_sum = masked.sum(dim=(0, 1)).detach().cpu()
        dim_count = valid.sum().detach().cpu().float().repeat(action_dim)
        self.action_dim_sums = dim_sum if self.action_dim_sums is None else self.action_dim_sums + dim_sum
        self.action_dim_counts = dim_count if self.action_dim_counts is None else self.action_dim_counts + dim_count

    @staticmethod
    def _update_group(
        sums: dict[int, float],
        counts: dict[int, float],
        indices: torch.Tensor,
        frame_sums: torch.Tensor,
        frame_counts: torch.Tensor,
    ) -> None:
        for idx, group_idx in enumerate(indices.detach().cpu().tolist()):
            sums[int(group_idx)] += float(frame_sums[idx])
            counts[int(group_idx)] += float(frame_counts[idx])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def predict_actions(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    policy_batch = {
        "observation.images.image": batch["image"],
        "observation.state": batch["state"],
        "observation.images": [batch["image"], batch["wrist_image"]] if "wrist_image" in batch else [batch["image"]],
    }
    if "wrist_image" in batch:
        policy_batch["observation.images.wrist_image"] = batch["wrist_image"]
    previous_use_vae = model.policy.config.use_vae
    model.policy.config.use_vae = False
    try:
        actions_hat, _ = model.policy.model(policy_batch)
    finally:
        model.policy.config.use_vae = previous_use_vae
    return actions_hat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", default="hw3/task2/results")
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


def build_loader(cfg, args: argparse.Namespace) -> tuple[CalvinActDataset, DataLoader]:
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
    return data, loader


def load_model(cfg, checkpoint_path: str, device: torch.device):
    model = build_policy(cfg.model, cfg.data.image_size, cfg.data.use_wrist_image, cfg.data.chunk_size, cfg.train.amp).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def evaluate_breakdown(model, loader: DataLoader, cfg, args: argparse.Namespace, device: torch.device) -> BreakdownAccumulator:
    acc = BreakdownAccumulator()
    with torch.no_grad():
        iterator = tqdm(loader, desc=f"breakdown {args.name} on {cfg.data.eval_split}")
        for batch in iterator:
            batch = move_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.train.amp and device.type == "cuda"):
                pred = predict_actions(model, batch)
            acc.update(batch, pred)
            iterator.set_postfix(action_l1=f"{acc.overall():.5f}")
    return acc


def build_rows(acc: BreakdownAccumulator, name: str, eval_split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    task_rows = build_group_rows(name, eval_split, "task_index", acc.task_sums, acc.task_counts)
    episode_rows = build_group_rows(name, eval_split, "episode_index", acc.episode_sums, acc.episode_counts)
    return task_rows, episode_rows, build_dim_rows(acc, name, eval_split)


def build_group_rows(
    name: str,
    eval_split: str,
    index_name: str,
    sums: dict[int, float],
    counts: dict[int, float],
) -> list[dict[str, Any]]:
    return [
        {
            "model_name": name,
            "eval_split": eval_split,
            index_name: idx,
            "action_l1": sums[idx] / max(counts[idx], 1.0),
            "valid_action_elements": counts[idx],
        }
        for idx in sorted(sums)
    ]


def build_dim_rows(acc: BreakdownAccumulator, name: str, eval_split: str) -> list[dict[str, Any]]:
    if acc.action_dim_sums is None or acc.action_dim_counts is None:
        return []
    rows = []
    pairs = zip(acc.action_dim_sums.tolist(), acc.action_dim_counts.tolist(), strict=True)
    for dim, (err_sum, count) in enumerate(pairs):
        rows.append(
            {
                "model_name": name,
                "eval_split": eval_split,
                "action_dim": dim,
                "action_l1": err_sum / max(count, 1.0),
                "valid_action_count": count,
            }
        )
    return rows


def write_breakdown_outputs(output_dir: Path, name: str, payload: dict[str, Any], rows: tuple[list[dict[str, Any]], ...]) -> None:
    task_rows, episode_rows, dim_rows = rows
    write_json(output_dir / f"{name}_breakdown_summary.json", payload)
    write_csv(output_dir / f"{name}_task_breakdown.csv", task_rows)
    write_csv(output_dir / f"{name}_episode_breakdown.csv", episode_rows)
    write_csv(output_dir / f"{name}_action_dim_breakdown.csv", dim_rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, loader = build_loader(cfg, args)
    model = load_model(cfg, args.checkpoint, device)
    acc = evaluate_breakdown(model, loader, cfg, args, device)
    payload = {
        "model_name": args.name,
        "train_split": "+".join(cfg.data.train_splits),
        "eval_split": cfg.data.eval_split,
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "overall_weighted_action_l1": acc.overall(),
        "num_frames": len(data),
        "num_episodes": len(data.episodes),
    }
    write_breakdown_outputs(Path(args.output_dir), args.name, payload, build_rows(acc, args.name, cfg.data.eval_split))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
