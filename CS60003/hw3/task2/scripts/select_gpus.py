from __future__ import annotations

import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-mib", type=int, default=12000)
    parser.add_argument("--max-gpus", type=int, default=8)
    parser.add_argument("--max-util", type=int, default=5)
    args = parser.parse_args()
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    ids: list[str] = []
    for line in result.splitlines():
        idx, total, used, util = [part.strip() for part in line.split(",")]
        free = int(total) - int(used)
        if free >= args.min_free_mib and int(util) <= args.max_util:
            ids.append(idx)
    print(",".join(ids[: args.max_gpus]))


if __name__ == "__main__":
    main()
