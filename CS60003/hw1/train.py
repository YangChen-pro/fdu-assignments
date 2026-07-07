"""训练 HW1 的 MLP 分类器。"""

from __future__ import annotations

import argparse

from mlp_hw1.config import build_train_config
from mlp_hw1.trainer import train_model


TRAIN_PRESETS = ["quick", "default", "full", "best", "final_a", "final_k", "final_l", "final_n", "final_p", "final_o"]


def parse_args() -> argparse.Namespace:
    """解析一组精简且实用的命令行参数。"""
    parser = argparse.ArgumentParser(description="训练 EuroSAT 三层 MLP 分类器")
    parser.add_argument("--preset", default="default", choices=TRAIN_PRESETS)
    parser.add_argument("--activation", default=None, choices=["relu", "tanh", "sigmoid"])
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def resolve_run_name(preset: str) -> str | None:
    """保证报告中命名实验可通过命令行稳定复现。"""
    normalized = preset.lower()
    if normalized == "best":
        return "final_p"
    if normalized in {"final_a", "final_k", "final_l", "final_n", "final_p", "final_o"}:
        return normalized
    return None


def main() -> None:
    """训练入口函数。"""
    args = parse_args()
    config = build_train_config(args.preset)
    if args.activation is not None:
        config.activation = args.activation
    if args.hidden_dim is not None:
        config.hidden_dim = args.hidden_dim
    if args.epochs is not None:
        config.epochs = args.epochs
    config.force_rebuild_cache = args.rebuild_cache
    train_model(config=config, run_name=resolve_run_name(args.preset))


if __name__ == "__main__":
    main()
