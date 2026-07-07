from __future__ import annotations

import argparse
import os
from pathlib import Path

from .secrets import has_any_key, load_env_file

RUN_IDS = {
    "act_splitA": "05kubpls24j5jrp2wbl1t",
    "act_splitABC": "4si6dcrut2krorrbalfkn",
}
METRIC_KEYS = ["train_loss", "train_action_l1", "eval_action_l1", "lr"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="youngchen/CS60003_HW3_Task2")
    parser.add_argument("--secret-env", default=".helloagents/secrets/hw3.env")
    parser.add_argument("--output-dir", default="hw3/task2/results/swanlab")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    load_env_file(args.secret_env)
    if not has_any_key(["SWANLAB_API_KEY", "SWANLAB_KEY", "SWANLAB_TOKEN"]):
        raise RuntimeError("SwanLab key not found in environment")
    key = os.environ.get("SWANLAB_API_KEY") or os.environ.get("SWANLAB_KEY") or os.environ.get("SWANLAB_TOKEN")

    import swanlab

    api = swanlab.Api(api_key=key)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for name, run_id in RUN_IDS.items():
        run = api.run(f"{args.project}/{run_id}")
        metrics = run.metrics(keys=METRIC_KEYS, sample=args.sample, pandas=True)
        metrics = metrics.reset_index()
        output = output_dir / f"{name}_swanlab_metrics.csv"
        metrics.to_csv(output, index=False)
        exported.append({"name": name, "run_id": run_id, "url": run.url, "rows": len(metrics), "output": str(output)})
    print(exported)


if __name__ == "__main__":
    main()
