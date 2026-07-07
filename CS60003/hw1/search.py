"""执行 HW1 的超参数搜索。"""

from __future__ import annotations

import argparse

from mlp_hw1.config import build_search_config
from mlp_hw1.trainer import run_search


def parse_args() -> argparse.Namespace:
    """解析搜索参数。"""
    parser = argparse.ArgumentParser(description="为 EuroSAT MLP 执行超参数搜索")
    parser.add_argument("--preset", default="quick", choices=["quick", "default", "full"])
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    """超参数搜索入口函数。"""
    args = parse_args()
    config = build_search_config(args.preset)
    if args.max_trials is not None:
        config.max_trials = args.max_trials
    config.train_config.force_rebuild_cache = args.rebuild_cache
    run_search(config=config)


if __name__ == "__main__":
    main()
