"""评估已保存的 HW1 模型权重。"""

from __future__ import annotations

import argparse
from pathlib import Path

from mlp_hw1.config import build_train_config
from mlp_hw1.trainer import evaluate_model


TRAIN_PRESETS = ["quick", "default", "full", "best", "final_a", "final_p", "final_o"]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="评估训练好的 EuroSAT MLP 模型")
    parser.add_argument("--preset", default="default", choices=TRAIN_PRESETS)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    """评估入口函数。"""
    args = parse_args()
    config = build_train_config(args.preset)
    config.force_rebuild_cache = args.rebuild_cache
    evaluate_model(config=config, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
