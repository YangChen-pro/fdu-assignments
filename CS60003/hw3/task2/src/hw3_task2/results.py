from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

RUNS = ["act_splitA", "act_splitABC"]
TRAIN_SPLITS = {
    "act_splitA": "splitA",
    "act_splitABC": "splitA+splitB+splitC",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def read_raw_metrics(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if "phase" in header:
            return read_phase_metrics(handle)
        return read_legacy_metrics(reader, header, path, train_rows, eval_rows)


def read_phase_metrics(handle) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    handle.seek(0)
    for row in csv.DictReader(handle):
        normalized = normalize_phase_row(row)
        if normalized["phase"] == "eval":
            eval_rows.append(normalized)
        else:
            train_rows.append(normalized)
    return train_rows, eval_rows


def normalize_phase_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "step": int(row["step"]),
        "epoch": int(row["epoch"]),
        "phase": row.get("phase", ""),
        "train_loss": float(row["train_loss"]) if row.get("train_loss") else "",
        "train_action_l1": float(row["train_action_l1"]) if row.get("train_action_l1") else "",
        "lr": float(row["lr"]) if row.get("lr") else "",
        "eval_action_l1": float(row["eval_action_l1"]) if row.get("eval_action_l1") else "",
    }


def read_legacy_metrics(
    reader: csv.reader,
    header: list[str],
    path: Path,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    for raw in reader:
        if not raw:
            continue
        if len(raw) >= 5:
            train_rows.append(build_train_metric(raw))
        elif len(raw) == 3:
            eval_rows.append(build_eval_metric(raw))
        else:
            raise ValueError(f"unexpected metrics row in {path}: {raw!r}; header={header!r}")
    return train_rows, eval_rows


def build_train_metric(raw: list[str]) -> dict[str, Any]:
    return {
        "step": int(raw[0]),
        "epoch": int(raw[1]),
        "phase": "train",
        "train_loss": float(raw[2]),
        "train_action_l1": float(raw[3]),
        "lr": float(raw[4]),
        "eval_action_l1": "",
    }


def build_eval_metric(raw: list[str]) -> dict[str, Any]:
    return {
        "step": int(raw[0]),
        "epoch": int(raw[1]),
        "phase": "eval",
        "train_loss": "",
        "train_action_l1": "",
        "lr": "",
        "eval_action_l1": float(raw[2]),
    }

def read_swanlab_metrics(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            step = int(float(row["step"]))
            if row.get("train_loss"):
                train_rows.append(
                    {
                        "step": step,
                        "epoch": "",
                        "phase": "train",
                        "train_loss": float(row["train_loss"]),
                        "train_action_l1": float(row["train_action_l1"]) if row.get("train_action_l1") else "",
                        "lr": float(row["lr"]) if row.get("lr") else "",
                        "eval_action_l1": "",
                        "source": "swanlab",
                    }
                )
            if row.get("eval_action_l1"):
                eval_rows.append(
                    {
                        "step": step,
                        "epoch": "",
                        "phase": "eval",
                        "train_loss": "",
                        "train_action_l1": "",
                        "lr": "",
                        "eval_action_l1": float(row["eval_action_l1"]),
                        "source": "swanlab",
                    }
                )
    return train_rows, eval_rows


def normalize_metrics(outputs_dir: Path, results_dir: Path) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for run in RUNS:
        swanlab_path = results_dir / "swanlab" / f"{run}_swanlab_metrics.csv"
        if swanlab_path.is_file():
            train_rows, eval_rows = read_swanlab_metrics(swanlab_path)
            source = "swanlab"
        else:
            train_rows, eval_rows = read_raw_metrics(outputs_dir / run / "metrics.csv")
            source = "local_metrics_csv"
        all_rows = sorted(train_rows + eval_rows, key=lambda item: (item["step"], item["phase"]))
        write_csv(results_dir / "curves" / f"{run}_metrics_clean.csv", all_rows)
        summary[run] = {
            "source": source,
            "train_points": len(train_rows),
            "eval_points": len(eval_rows),
            "last_train_action_l1": train_rows[-1]["train_action_l1"] if train_rows else None,
            "best_eval_action_l1": min((row["eval_action_l1"] for row in eval_rows), default=None),
            "final_eval_action_l1_from_training": eval_rows[-1]["eval_action_l1"] if eval_rows else None,
        }
    return summary


def _series(rows: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for row in rows:
        value = row.get(key, "")
        if value == "" or value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        xs.append(int(row["step"]))
        ys.append(float(value))
    return xs, ys


def plot_curves(results_dir: Path) -> None:
    curves_dir = results_dir / "curves"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows_by_run = read_clean_curves(curves_dir)
    specs = [
        ("train_loss", "Train loss", "train_loss_curve.png"),
        ("train_action_l1", "Train Action L1", "train_action_l1_curve.png"),
        ("eval_action_l1", "Eval Action L1 on splitD", "eval_action_l1_curve.png"),
    ]
    for key, ylabel, filename in specs:
        plot_line_figure(rows_by_run, key, ylabel, figures_dir / filename)
    plot_best_bar(results_dir, figures_dir)


def read_clean_curves(curves_dir: Path) -> dict[str, list[dict[str, Any]]]:
    rows_by_run: dict[str, list[dict[str, Any]]] = {}
    for run in RUNS:
        with (curves_dir / f"{run}_metrics_clean.csv").open(newline="") as handle:
            rows_by_run[run] = list(csv.DictReader(handle))
    return rows_by_run


def plot_line_figure(rows_by_run: dict[str, list[dict[str, Any]]], key: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(8, 5), dpi=160)
    for run, rows in rows_by_run.items():
        xs, ys = _series(rows, key)
        if xs:
            plt.plot(xs, ys, label=run, linewidth=2)
    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_best_bar(results_dir: Path, figures_dir: Path) -> None:
    table_path = results_dir / "task2_results_table.csv"
    if not table_path.exists():
        return
    with table_path.open(newline="") as handle:
        best_rows = list(csv.DictReader(handle))
    if not best_rows:
        return
    labels = [row["model_name"] for row in best_rows]
    values = [float(row["best_eval_action_l1"]) for row in best_rows]
    plt.figure(figsize=(7, 4), dpi=160)
    plt.bar(labels, values, color=["#6b7280", "#2563eb"])
    plt.ylabel("Best Action L1 on splitD")
    plt.title("splitA vs splitA+B+C zero-shot performance")
    for idx, value in enumerate(values):
        plt.text(idx, value, f"{value:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(figures_dir / "splitA_vs_splitABC_eval_l1.png")
    plt.close()

def copy_results(outputs_dir: Path, results_dir: Path) -> dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in ["task2_results_table.csv", "task2_results_table.json"]:
        src = outputs_dir / "eval" / name
        if src.is_file():
            dst = results_dir / name
            dst.write_bytes(src.read_bytes())
            copied.append(str(dst.relative_to(results_dir)))
    for run in RUNS:
        for name in ["train_summary.json", "dataset_summary.json", "results_summary.json"]:
            src = outputs_dir / run / name
            if src.is_file():
                dst = results_dir / f"{run}_{name}"
                dst.write_bytes(src.read_bytes())
                copied.append(str(dst.relative_to(results_dir)))
        eval_prefixes = [f"{run}_splitD", f"{run}_final_splitD"]
        for prefix in eval_prefixes:
            for ext in [".json", ".csv"]:
                src = outputs_dir / "eval" / f"{prefix}{ext}"
                if src.is_file():
                    dst = results_dir / f"{prefix}{ext}"
                    dst.write_bytes(src.read_bytes())
                    copied.append(str(dst.relative_to(results_dir)))
    return {"copied_files": copied}



def build_best_checkpoint_table(results_dir: Path) -> list[dict[str, Any]]:
    source = results_dir / "task2_results_table.csv"
    if not source.exists():
        return []
    rows = []
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "model_name": row["model_name"],
                    "train_split": row["train_split"],
                    "eval_split": row["eval_split"],
                    "training_best_eval_action_l1": float(row["best_eval_action_l1"]),
                    "evaluated_best_checkpoint_action_l1": float(row["final_eval_action_l1"]),
                    "num_eval_frames": int(row["num_eval_frames"]),
                    "num_eval_episodes": int(row["num_eval_episodes"]),
                    "checkpoint_path": row["checkpoint_path"],
                }
            )
    write_csv(results_dir / "best_checkpoint_eval_table.csv", rows)
    write_json(results_dir / "best_checkpoint_eval_table.json", rows)
    return rows

def build_final_only_table(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in RUNS:
        best_path = results_dir / f"{run}_splitD.json"
        final_path = results_dir / f"{run}_final_splitD.json"
        if not best_path.exists() or not final_path.exists():
            continue
        best = read_json(best_path)
        final = read_json(final_path)
        best_l1 = float(best["action_l1"])
        final_l1 = float(final["action_l1"])
        rows.append(
            {
                "model_name": run,
                "train_split": TRAIN_SPLITS[run],
                "eval_split": final["eval_split"],
                "best_checkpoint_action_l1": best_l1,
                "final_checkpoint_action_l1": final_l1,
                "absolute_delta_final_minus_best": final_l1 - best_l1,
                "relative_delta_final_minus_best_pct": (final_l1 - best_l1) / best_l1 * 100.0,
                "num_eval_frames": final["num_frames"],
                "num_eval_episodes": final["num_episodes"],
                "final_checkpoint_path": final["checkpoint_path"],
            }
        )
    if rows:
        write_csv(results_dir / "final_only_eval_table.csv", rows)
        write_json(results_dir / "final_only_eval_table.json", rows)
    return rows


def write_readme(results_dir: Path, summary: dict[str, Any], final_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# HW3 Task2 Results",
        "",
        "该目录只保存可提交的小体积结果证据；完整 checkpoint、日志缓存和数据集仍保留在 135 远程机器与 ModelScope。",
        "",
        "## 核心结果",
        "",
        "- 原始 best 结果表：`task2_results_table.csv` / `task2_results_table.json`。",
        "- 字段无歧义的 best checkpoint 表：`best_checkpoint_eval_table.csv` / `best_checkpoint_eval_table.json`。",
        "- final-only 评估表：`final_only_eval_table.csv` / `final_only_eval_table.json`。",
        "- paired 统计汇总：`STATISTICAL_SUMMARY.md`、`statistical_summary.csv`、`statistical_summary.json`。",
        "- 清洗后的曲线源数据：`curves/*_metrics_clean.csv`。",
        "- 曲线图：`figures/*.png`。",
        "",
    ]
    if final_rows:
        lines.extend(["## Final-only D 评估", ""])
        for row in final_rows:
            lines.append(
                f"- `{row['model_name']}` final Action L1 = `{float(row['final_checkpoint_action_l1']):.10f}`, "
                f"best Action L1 = `{float(row['best_checkpoint_action_l1']):.10f}`。"
            )
        lines.append("")
    lines.extend(["## 曲线数据摘要", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""])
    (results_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="hw3/task2/outputs")
    parser.add_argument("--results-dir", default="hw3/task2/results")
    args = parser.parse_args()
    outputs_dir = Path(args.outputs_dir)
    results_dir = Path(args.results_dir)
    copy_summary = copy_results(outputs_dir, results_dir)
    curve_summary = normalize_metrics(outputs_dir, results_dir)
    build_best_checkpoint_table(results_dir)
    final_rows = build_final_only_table(results_dir)
    plot_curves(results_dir)
    summary = {"copy": copy_summary, "curves": curve_summary}
    write_json(results_dir / "results_manifest.json", summary)
    write_readme(results_dir, summary, final_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
