from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

RUN_A = "act_splitA_final"
RUN_ABC = "act_splitABC_final"
BOOTSTRAP_ROUNDS = 2000
RANDOM_SEED = 20260616


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def paired_values(results_dir: Path, suffix: str, key: str) -> list[dict[str, float]]:
    a_rows = read_csv_rows(results_dir / f"{RUN_A}_{suffix}.csv")
    abc_rows = read_csv_rows(results_dir / f"{RUN_ABC}_{suffix}.csv")
    abc_by_key = {row[key]: row for row in abc_rows}
    pairs: list[dict[str, float]] = []
    for row in a_rows:
        other = abc_by_key.get(row[key])
        if other is None:
            continue
        a_l1 = float(row["action_l1"])
        abc_l1 = float(other["action_l1"])
        weight = float(row.get("valid_action_elements") or row.get("valid_action_count") or 1.0)
        pairs.append({"key": float(row[key]), "a_l1": a_l1, "abc_l1": abc_l1, "delta": a_l1 - abc_l1, "weight": weight})
    return pairs


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def weighted_mean(pairs: list[dict[str, float]], field: str) -> float:
    total_weight = sum(item["weight"] for item in pairs)
    return sum(item[field] * item["weight"] for item in pairs) / max(total_weight, 1.0)


def bootstrap_ci(deltas: list[float]) -> tuple[float, float]:
    if not deltas:
        return 0.0, 0.0
    rng = random.Random(RANDOM_SEED)
    n = len(deltas)
    estimates = []
    for _ in range(BOOTSTRAP_ROUNDS):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        estimates.append(mean(sample))
    estimates.sort()
    low_idx = int(0.025 * (BOOTSTRAP_ROUNDS - 1))
    high_idx = int(0.975 * (BOOTSTRAP_ROUNDS - 1))
    return estimates[low_idx], estimates[high_idx]


def summarize(granularity: str, pairs: list[dict[str, float]]) -> dict[str, Any]:
    deltas = [item["delta"] for item in pairs]
    positive = [delta for delta in deltas if delta > 0]
    ci_low, ci_high = bootstrap_ci(deltas)
    a_mean = mean([item["a_l1"] for item in pairs])
    abc_mean = mean([item["abc_l1"] for item in pairs])
    delta_mean = mean(deltas)
    return {
        "granularity": granularity,
        "paired_count": len(pairs),
        "abc_better_count": len(positive),
        "abc_better_pct": len(positive) / max(len(pairs), 1) * 100.0,
        "mean_action_l1_splitA_final": a_mean,
        "mean_action_l1_splitABC_final": abc_mean,
        "mean_delta_splitA_minus_splitABC": delta_mean,
        "relative_improvement_pct": delta_mean / max(a_mean, 1e-12) * 100.0,
        "bootstrap_95ci_delta_low": ci_low,
        "bootstrap_95ci_delta_high": ci_high,
        "weighted_action_l1_splitA_final": weighted_mean(pairs, "a_l1"),
        "weighted_action_l1_splitABC_final": weighted_mean(pairs, "abc_l1"),
        "weighted_delta_splitA_minus_splitABC": weighted_mean(pairs, "delta"),
    }


def load_final_table(results_dir: Path) -> list[dict[str, str]]:
    return read_csv_rows(results_dir / "final_only_eval_table.csv")


def write_markdown(path: Path, final_table: list[dict[str, str]], summaries: list[dict[str, Any]]) -> None:
    final = {row["model_name"]: row for row in final_table}
    a = float(final["act_splitA"]["final_checkpoint_action_l1"])
    abc = float(final["act_splitABC"]["final_checkpoint_action_l1"])
    lines = [
        "# HW3 Task2 Statistical Summary",
        "",
        "## 主结论",
        "",
        f"本节以未使用 D 环境选择 checkpoint 的 `final.pt` 作为主口径：`act_splitABC` 在 splitD 上的 Action L1 为 `{abc:.10f}`，低于 `act_splitA` 的 `{a:.10f}`。",
        f"相对提升为 `{(a - abc) / a * 100:.2f}%`，说明多环境训练在未见过的 D 环境上仍有稳定优势。",
        "",
        "`best.pt` 只作为参考结果；它使用 splitD 离线误差做 checkpoint 选择，因此不作为最干净的 zero-shot 主结论。",
        "",
        "## Paired 统计",
        "",
        "| 粒度 | 配对数 | ABC 更优数量 | ABC 更优比例 | 平均 L1 提升 | 95% bootstrap CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['granularity']} | {item['paired_count']} | {item['abc_better_count']} | "
            f"{item['abc_better_pct']:.2f}% | {item['mean_delta_splitA_minus_splitABC']:.6f} | "
            f"[{item['bootstrap_95ci_delta_low']:.6f}, {item['bootstrap_95ci_delta_high']:.6f}] |"
        )
    lines.extend([
        "",
        "解释：`平均 L1 提升 = act_splitA_final - act_splitABC_final`，大于 0 表示多环境模型更好。",
        "",
        "## 证据文件",
        "",
        "- `statistical_summary.csv` / `statistical_summary.json`：本页对应的机器可读汇总。",
        "- `act_splitA_final_episode_breakdown.csv` 与 `act_splitABC_final_episode_breakdown.csv`：episode 级配对来源。",
        "- `act_splitA_final_task_breakdown.csv` 与 `act_splitABC_final_task_breakdown.csv`：task 级配对来源。",
        "- `act_splitA_final_action_dim_breakdown.csv` 与 `act_splitABC_final_action_dim_breakdown.csv`：动作维度级配对来源。",
        "",
    ])
    path.write_text("\n".join(lines))



def update_results_readme(results_dir: Path, summaries: list[dict[str, Any]]) -> None:
    readme = results_dir / "README.md"
    if not readme.exists():
        return
    text = readme.read_text()
    text = text.replace(
        "- 主结果表：`task2_results_table.csv` / `task2_results_table.json`。",
        "- 原始 best 结果表：`task2_results_table.csv` / `task2_results_table.json`。\n"
        "- 字段无歧义的 best checkpoint 表：`best_checkpoint_eval_table.csv` / `best_checkpoint_eval_table.json`。",
    )
    if "## Paired 统计摘要" in text:
        readme.write_text(text)
        return
    by_name = {item["granularity"]: item for item in summaries}
    section = [
        "",
        "## Paired 统计摘要",
        "",
        format_summary_line("episode", by_name["episode"]),
        format_summary_line("task", by_name["task"]),
        format_summary_line("action-dim", by_name["action_dim"]),
        "",
        "详见 `STATISTICAL_SUMMARY.md`。",
        "",
    ]
    text = text.replace("\n## 曲线数据摘要\n", "\n" + "\n".join(section) + "## 曲线数据摘要\n")
    readme.write_text(text)


def format_summary_line(label: str, item: dict[str, Any]) -> str:
    return (
        f"- {label} 级：`act_splitABC_final` 在 `{item['abc_better_count']}/{item['paired_count']}` 个配对项上优于 "
        f"`act_splitA_final`，占 `{item['abc_better_pct']:.2f}%`；平均 Action L1 提升 "
        f"`{item['mean_delta_splitA_minus_splitABC']:.6f}`，95% bootstrap CI "
        f"`[{item['bootstrap_95ci_delta_low']:.6f}, {item['bootstrap_95ci_delta_high']:.6f}]`。"
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="hw3/task2/results")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    specs = [
        ("episode", "episode_breakdown", "episode_index"),
        ("task", "task_breakdown", "task_index"),
        ("action_dim", "action_dim_breakdown", "action_dim"),
    ]
    summaries = [summarize(name, paired_values(results_dir, suffix, key)) for name, suffix, key in specs]
    final_table = load_final_table(results_dir)
    write_csv(results_dir / "statistical_summary.csv", summaries)
    write_json(results_dir / "statistical_summary.json", summaries)
    write_markdown(results_dir / "STATISTICAL_SUMMARY.md", final_table, summaries)
    update_results_readme(results_dir, summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
