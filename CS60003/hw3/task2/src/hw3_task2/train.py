from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from .config import load_config, save_config
from .data import CalvinActDataset, collate_batch, summarize_dataset
from .model import build_policy
from .secrets import has_any_key, load_env_file
from .utils import append_csv, is_main_process, set_seed, write_json

METRIC_FIELDS = ["step", "epoch", "phase", "train_loss", "train_action_l1", "eval_action_l1", "lr"]


def setup_distributed() -> tuple[bool, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, torch.device("cuda", local_rank)
    return False, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_distributed(distributed: bool) -> None:
    if distributed:
        dist.destroy_process_group()


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def init_swanlab(cfg, output_dir: Path):
    if not cfg.track.enable_swanlab or not is_main_process():
        return None
    load_env_file(cfg.track.secret_env)
    if not has_any_key(["SWANLAB_API_KEY", "SWANLAB_KEY", "SWANLAB_TOKEN"]):
        print("SwanLab key not found; continue without cloud logging.")
        return None
    import swanlab

    return swanlab.init(
        project=cfg.track.project,
        workspace=cfg.track.workspace,
        experiment_name=cfg.train.name,
        mode=cfg.track.mode,
        config={
            "data": cfg.data.__dict__,
            "model": cfg.model.__dict__,
            "train": cfg.train.__dict__,
        },
        logdir=str(output_dir / "swanlab"),
    )


def evaluate(model, loader, device: torch.device, amp: bool, desc: str) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        iterator = tqdm(loader, desc=desc, disable=not is_main_process())
        for batch in iterator:
            batch = move_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
                loss, loss_dict = model(batch)
            total += float(loss_dict["l1_loss"])
            count += 1
            iterator.set_postfix(action_l1=f"{total / max(count, 1):.5f}")
    model.train()
    return total / max(count, 1)


def save_checkpoint(path: Path, model, optimizer, scaler, epoch: int, best_metric: float, cfg) -> None:
    raw_model = model.module if hasattr(model, "module") else model
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_metric": best_metric,
            "config": {
                "data": cfg.data.__dict__,
                "model": cfg.model.__dict__,
                "train": cfg.train.__dict__,
            },
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    distributed, device = setup_distributed()
    set_seed(cfg.train.seed + int(os.environ.get("RANK", "0")))

    output_dir = Path(cfg.train.output_dir) / cfg.train.name
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, output_dir / "config.yaml")
        write_json(output_dir / "dataset_summary.json", summarize_dataset(cfg.data.root, sorted(set(cfg.data.train_splits + [cfg.data.eval_split]))))

    train_data = CalvinActDataset(
        cfg.data.root,
        cfg.data.train_splits,
        cfg.data.image_size,
        cfg.data.chunk_size,
        cfg.data.max_train_episodes,
        cfg.data.use_wrist_image,
        cfg.train.seed,
        is_main_process(),
    )
    eval_data = CalvinActDataset(
        cfg.data.root,
        [cfg.data.eval_split],
        cfg.data.image_size,
        cfg.data.chunk_size,
        cfg.data.max_eval_episodes,
        cfg.data.use_wrist_image,
        cfg.train.seed,
        is_main_process(),
    )
    train_sampler = DistributedSampler(train_data, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_data,
        batch_size=cfg.train.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
        persistent_workers=cfg.train.num_workers > 0,
    )
    eval_loader = DataLoader(
        eval_data,
        batch_size=cfg.train.eval_batch_size,
        shuffle=False,
        num_workers=max(1, cfg.train.num_workers // 2),
        pin_memory=True,
        collate_fn=collate_batch,
    )

    model = build_policy(cfg.model, cfg.data.image_size, cfg.data.use_wrist_image, cfg.data.chunk_size, cfg.train.amp).to(device)
    if distributed:
        model = DDP(model, device_ids=[device.index], output_device=device.index)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and device.type == "cuda")
    run = init_swanlab(cfg, output_dir)
    best_eval = float("inf")
    global_step = 0
    epochs = 1 if args.dry_run else cfg.train.epochs
    dry_steps = cfg.train.dry_run_steps if args.dry_run else None

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        iterator = tqdm(train_loader, desc=f"train {cfg.train.name} epoch {epoch}", disable=not is_main_process())
        for step, batch in enumerate(iterator, start=1):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.train.amp and device.type == "cuda"):
                loss, loss_dict = model(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            loss_value = float(loss.item())
            l1_value = float(loss_dict["l1_loss"])
            iterator.set_postfix(loss=f"{loss_value:.5f}", action_l1=f"{l1_value:.5f}")
            if is_main_process() and global_step % cfg.train.log_every == 0:
                row = {
                    "step": global_step,
                    "epoch": epoch,
                    "phase": "train",
                    "train_loss": loss_value,
                    "train_action_l1": l1_value,
                    "lr": cfg.train.lr,
                }
                append_csv(output_dir / "metrics.csv", row, METRIC_FIELDS)
                if run is not None:
                    import swanlab
                    swanlab.log(row, step=global_step)
            if dry_steps and step >= dry_steps:
                break
            if cfg.train.max_steps_per_epoch and step >= cfg.train.max_steps_per_epoch:
                break

        if epoch % cfg.train.eval_every_epochs == 0:
            eval_l1 = evaluate(model, eval_loader, device, cfg.train.amp, f"eval {cfg.data.eval_split}")
            if distributed:
                value = torch.tensor([eval_l1], device=device)
                dist.all_reduce(value, op=dist.ReduceOp.AVG)
                eval_l1 = float(value.item())
            if is_main_process():
                row = {"step": global_step, "epoch": epoch, "phase": "eval", "eval_action_l1": eval_l1}
                append_csv(output_dir / "metrics.csv", row, METRIC_FIELDS)
                if run is not None:
                    import swanlab
                    swanlab.log(row, step=global_step)
                save_checkpoint(output_dir / "checkpoints" / "latest.pt", model, optimizer, scaler, epoch, best_eval, cfg)
                if eval_l1 < best_eval:
                    best_eval = eval_l1
                    save_checkpoint(output_dir / "checkpoints" / "best.pt", model, optimizer, scaler, epoch, best_eval, cfg)
        if is_main_process() and epoch % cfg.train.save_every_epochs == 0:
            save_checkpoint(output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, optimizer, scaler, epoch, best_eval, cfg)

    if is_main_process():
        save_checkpoint(output_dir / "checkpoints" / "final.pt", model, optimizer, scaler, epochs, best_eval, cfg)
        write_json(output_dir / "train_summary.json", {"name": cfg.train.name, "best_eval_action_l1": best_eval, "steps": global_step})
        if run is not None:
            import swanlab
            swanlab.finish()
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
