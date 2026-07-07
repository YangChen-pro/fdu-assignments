"""Run the config-driven HW3 Task 1 pipeline."""

from __future__ import annotations

import argparse
import json

from task1_3dgs_aigc.config import load_config, resolve_paths
from task1_3dgs_aigc.real_chain import run_real_chain
from task1_3dgs_aigc.swanlab_utils import create_swanlab_logger


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output-root", default=None, help="Override output root.")
    return parser.parse_args()


def main() -> None:
    """Run one configured Task1 stage."""
    args = parse_args()
    config = load_config(args.config)
    resolve_paths(config, args.output_root)
    stage = str(config["task1"]["stage"])
    if stage == "real_high_quality":
        summary = run_real_chain(config)
    else:
        raise ValueError(f"Unsupported stage: {stage}")
    _log_summary(config, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def _log_summary(config: dict, summary: dict) -> None:
    logger = create_swanlab_logger(config, summary["run_dir"])
    try:
        metrics = {"task1/status_pass": 1 if summary["status"] == "PASS" else 0}
        metrics.update(
            {
                "task1/script_count": int(summary["script_count"]),
                "task1/ready": 1 if summary["status"] in {"READY", "PASS"} else 0,
                "task1/needs_inputs": 1 if summary["status"] == "NEEDS_INPUTS" else 0,
            }
        )
        logger.log(metrics)
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
