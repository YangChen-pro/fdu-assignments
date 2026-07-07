from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

REQUIRED_RESULTS = [
    "task2_results_table.csv",
    "task2_results_table.json",
    "best_checkpoint_eval_table.csv",
    "best_checkpoint_eval_table.json",
    "final_only_eval_table.csv",
    "final_only_eval_table.json",
    "statistical_summary.csv",
    "statistical_summary.json",
    "STATISTICAL_SUMMARY.md",
    "calvin_simulator_probe.md",
    "figures/train_loss_curve.png",
    "figures/train_action_l1_curve.png",
    "figures/eval_action_l1_curve.png",
    "figures/splitA_vs_splitABC_eval_l1.png",
]

PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "lerobot": "lerobot",
    "pyarrow": "pyarrow",
    "PIL": "pillow",
    "tqdm": "tqdm",
    "yaml": "pyyaml",
    "swanlab": "swanlab",
    "modelscope": "modelscope",
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
}

EXPECTED_SPLITS = {
    "splitA": {"episodes": 6089, "frames": 366693},
    "splitB": {"episodes": 6115, "frames": 367096},
    "splitC": {"episodes": 5666, "frames": 337954},
    "splitD": {"episodes": 5124, "frames": 308918},
}


def status(ok: bool, message: str, detail: Any = None) -> dict[str, Any]:
    return {"status": "PASS" if ok else "FAIL", "message": message, "detail": detail}


def check_packages() -> list[dict[str, Any]]:
    rows = []
    for module, package_name in PACKAGES.items():
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, "__version__", "unknown")
            rows.append(status(True, package_name, version))
        except Exception as exc:
            rows.append(status(False, package_name, f"{type(exc).__name__}: {exc}"))
    return rows


def check_nvidia_smi() -> dict[str, Any]:
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv,noheader"], text=True)
        return status(True, "nvidia-smi", output.strip().splitlines())
    except Exception as exc:
        return status(False, "nvidia-smi", f"{type(exc).__name__}: {exc}")


def check_files(root: Path, results_dir: Path) -> list[dict[str, Any]]:
    paths = [root / "requirements.txt", root / "README.md"]
    paths.extend(results_dir / item for item in REQUIRED_RESULTS)
    return [status(path.exists(), str(path), path.stat().st_size if path.exists() else None) for path in paths]


def check_dataset(data_root: Path) -> list[dict[str, Any]]:
    rows = []
    for split, expected in EXPECTED_SPLITS.items():
        split_dir = data_root / split
        meta = split_dir / "meta" / "episodes.jsonl"
        ok = split_dir.is_dir() and meta.is_file()
        detail: dict[str, Any] = {"expected": expected, "path": str(split_dir)}
        info = split_dir / "meta" / "info.json"
        if meta.is_file():
            detail["episodes_jsonl_lines"] = sum(1 for _ in meta.open(errors="ignore"))
        if info.is_file():
            info_payload = json.loads(info.read_text())
            detail["total_episodes"] = info_payload.get("total_episodes")
            detail["total_frames"] = info_payload.get("total_frames")
            ok = ok and detail["total_episodes"] == expected["episodes"] and detail["total_frames"] == expected["frames"]
        rows.append(status(ok, split, detail))
    return rows


def check_result_numbers(results_dir: Path) -> dict[str, Any]:
    table = results_dir / "final_only_eval_table.json"
    if not table.exists():
        return status(False, "final-only metrics", "missing final_only_eval_table.json")
    rows = json.loads(table.read_text())
    by_name = {row["model_name"]: row for row in rows}
    a = float(by_name["act_splitA"]["final_checkpoint_action_l1"])
    abc = float(by_name["act_splitABC"]["final_checkpoint_action_l1"])
    return status(abc < a, "final.pt splitABC beats splitA", {"act_splitA": a, "act_splitABC": abc, "relative_improvement_pct": (a - abc) / a * 100})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task2-root", default="hw3/task2")
    parser.add_argument("--data-root", default="/data/yc/CS60003/hw3/task2/data/calvin_lerobot")
    parser.add_argument("--results-dir", default="hw3/task2/results")
    parser.add_argument("--strict-data", action="store_true", help="数据目录或 split 统计不匹配时返回非零退出码")
    parser.add_argument("--strict-env", action="store_true", help="Python 包或 GPU 检查失败时返回非零退出码")
    args = parser.parse_args()

    task2_root = Path(args.task2_root)
    results_dir = Path(args.results_dir)
    data_root = Path(args.data_root)
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": check_packages(),
        "gpu": check_nvidia_smi(),
        "files": check_files(task2_root, results_dir),
        "dataset": check_dataset(data_root),
        "metrics": check_result_numbers(results_dir),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    file_fail = any(item["status"] == "FAIL" for item in report["files"])
    metric_fail = report["metrics"]["status"] == "FAIL"
    data_fail = any(item["status"] == "FAIL" for item in report["dataset"])
    env_fail = any(item["status"] == "FAIL" for item in report["packages"]) or report["gpu"]["status"] == "FAIL"
    if file_fail or metric_fail or (args.strict_data and data_fail) or (args.strict_env and env_fail):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
