#!/usr/bin/env python3
"""Summarize Task A metrics into report-ready CSV tables."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAN = float("nan")
RESULTS = ROOT / "results" / "task_a"
VLLM_METRICS = RESULTS / "vllm" / "metrics"
VLLM_PREDS = RESULTS / "vllm" / "predictions"
VLLM_RAW = RESULTS / "vllm" / "raw_outputs"
REPORT_DIR = RESULTS / "report_tables"


def read_metric_file(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["source"] = path.name
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_method(method: str) -> dict[str, str]:
    parts = method.split(":")
    parsed = {"model": "", "mode": "", "setting": "", "template": ""}
    if len(parts) >= 4:
        parsed.update(model=parts[0], mode=parts[1], setting=parts[2], template=parts[3])
    elif len(parts) == 3:
        parsed.update(model=parts[0], mode=parts[1], setting=parts[2])
    elif len(parts) == 1:
        parsed.update(model=parts[0])
    return parsed


def as_float(row: dict[str, str]) -> float:
    return float(row["value"])


def infer_experiment(source: str) -> str:
    if source == "metrics.csv":
        return "roberta_original"
    if source == "order_metrics.csv":
        return "roberta_order"
    if "self_consistency" in source:
        return "self_consistency_k5"
    if "few_shot_8" in source:
        return "few_shot_8"
    if "order_metrics" in source:
        return "order_consistency"
    if "swap_metrics" in source:
        return "order_swap_accuracy"
    if source.endswith("test_metrics.csv"):
        return "zero_shot"
    return "other"


def file_mode(source: str) -> str:
    return "non_thinking" if "non_thinking" in source else "thinking"


def file_model(source: str) -> str:
    return source.replace("_non_thinking", "|non_thinking").replace("_thinking", "|thinking").split("|")[0]


def load_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(VLLM_METRICS.glob("*.csv")):
        rows.extend(read_metric_file(path))
    rows.extend(read_metric_file(RESULTS / "roberta" / "metrics.csv"))
    rows.extend(read_metric_file(RESULTS / "roberta" / "order_metrics.csv"))
    for row in rows:
        row["experiment"] = infer_experiment(row["source"])
    return rows


def metric_lookup(rows: list[dict[str, str]], *, experiment: str, metric: str) -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if row["experiment"] != experiment or row["metric"] != metric:
            continue
        parsed = parse_method(row["method"])
        key = (parsed["model"], parsed["mode"], parsed["template"])
        out[key] = as_float(row)
    return out


def prompt_lookup(rows: list[dict[str, str]], experiment: str) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for row in rows:
        if row["experiment"] != experiment or row["metric"] != "prompt_consistency":
            continue
        parsed = parse_method(row["method"])
        model = parsed["model"] or file_model(row["source"])
        mode = parsed["mode"] or file_mode(row["source"])
        out[(model, mode)] = as_float(row)
    return out


def build_accuracy_table(rows: list[dict[str, str]], experiment: str) -> list[dict[str, object]]:
    prompt = prompt_lookup(rows, experiment)
    table: list[dict[str, object]] = []
    for row in rows:
        if row["experiment"] != experiment or row["metric"] != "accuracy":
            continue
        parsed = parse_method(row["method"])
        model, mode, template = parsed["model"], parsed["mode"], parsed["template"]
        table.append(
            {
                "model": model,
                "mode": mode,
                "template": template,
                "accuracy": f"{as_float(row):.3f}",
                "prompt_consistency": f"{prompt.get((model, mode), NAN):.3f}",
                "source": row["source"],
            }
        )
    table.sort(key=lambda x: (float(x["accuracy"]), x["model"], x["mode"], x["template"]), reverse=True)
    return table


def build_order_table(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    original = metric_lookup(rows, experiment="zero_shot", metric="accuracy")
    swapped = metric_lookup(rows, experiment="order_swap_accuracy", metric="accuracy")
    prompt = prompt_lookup(rows, "zero_shot")
    table: list[dict[str, object]] = []
    for row in rows:
        if row["experiment"] != "order_consistency" or row["metric"] != "order_consistency":
            continue
        if row["method"] == "roberta_finetuned":
            table.append(
                {
                    "model": "RoBERTa",
                    "mode": "supervised",
                    "template": "classifier",
                    "original_accuracy": "0.868",
                    "swap_accuracy": "0.867",
                    "order_consistency": f"{as_float(row):.3f}",
                    "prompt_consistency": "",
                    "source": row["source"],
                }
            )
            continue
        parsed = parse_method(row["method"])
        key = (parsed["model"], parsed["mode"], parsed["template"])
        table.append(
            {
                "model": parsed["model"],
                "mode": parsed["mode"],
                "template": parsed["template"],
                "original_accuracy": f"{original.get(key, NAN):.3f}",
                "swap_accuracy": f"{swapped.get(key, NAN):.3f}",
                "order_consistency": f"{as_float(row):.3f}",
                "prompt_consistency": f"{prompt.get((parsed['model'], parsed['mode']), NAN):.3f}",
                "source": row["source"],
            }
        )
    table.sort(key=lambda x: float(x["order_consistency"]), reverse=True)
    return table


def build_self_consistency_table(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    prompt = prompt_lookup(rows, "self_consistency_k5")
    sampling: dict[tuple[str, str, str], float] = {}
    table: list[dict[str, object]] = []
    for row in rows:
        if row["experiment"] == "self_consistency_k5" and row["metric"] == "sampling_consistency":
            parsed = parse_method(row["method"])
            sampling[(parsed["model"], parsed["mode"], parsed["template"])] = as_float(row)
    for row in rows:
        if row["experiment"] != "self_consistency_k5" or row["metric"] != "accuracy":
            continue
        parsed = parse_method(row["method"])
        key = (parsed["model"], parsed["mode"], parsed["template"])
        table.append(
            {
                "model": parsed["model"],
                "mode": parsed["mode"],
                "template": parsed["template"],
                "accuracy": f"{as_float(row):.3f}",
                "prompt_consistency": f"{prompt.get((parsed['model'], parsed['mode']), NAN):.3f}",
                "sampling_consistency": f"{sampling.get(key, NAN):.3f}",
                "source": row["source"],
            }
        )
    table.sort(key=lambda x: float(x["accuracy"]), reverse=True)
    return table


def build_manifest() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for directory, expected in [(VLLM_RAW, 3000), (VLLM_PREDS, 3001), (VLLM_METRICS, None)]:
        for path in sorted(directory.glob("*")):
            if not path.is_file():
                continue
            line_count = sum(1 for _ in path.open("rb"))
            exp = expected
            if path.name.endswith("samples.csv"):
                exp = 15001
            if "self_consistency" in path.name and path.suffix == ".jsonl":
                exp = 15000
            status = "ok" if exp is None or line_count == exp else "check"
            rows.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "lines": line_count,
                    "expected_lines": exp or "",
                    "status": status,
                }
            )
    return rows


def main() -> None:
    rows = load_rows()
    all_rows = [{**row, "value": f"{as_float(row):.6f}"} for row in rows]
    write_csv(REPORT_DIR / "all_metrics_with_source.csv", all_rows, ["source", "experiment", "method", "metric", "value", "split", "notes"])
    write_csv(REPORT_DIR / "artifact_manifest.csv", build_manifest(), ["path", "lines", "expected_lines", "status"])

    zero = build_accuracy_table(rows, "zero_shot")
    few = build_accuracy_table(rows, "few_shot_8")
    order = build_order_table(rows)
    sc = build_self_consistency_table(rows)
    write_csv(REPORT_DIR / "zero_shot_accuracy.csv", zero, ["model", "mode", "template", "accuracy", "prompt_consistency", "source"])
    write_csv(REPORT_DIR / "few_shot_8_accuracy.csv", few, ["model", "mode", "template", "accuracy", "prompt_consistency", "source"])
    write_csv(REPORT_DIR / "order_consistency.csv", order, ["model", "mode", "template", "original_accuracy", "swap_accuracy", "order_consistency", "prompt_consistency", "source"])
    write_csv(REPORT_DIR / "self_consistency_k5.csv", sc, ["model", "mode", "template", "accuracy", "prompt_consistency", "sampling_consistency", "source"])

    best_zero = zero[0]
    best_few = few[0]
    best_sc = sc[0]
    best_zero_order = next(
        row["order_consistency"]
        for row in order
        if row["model"] == best_zero["model"]
        and row["mode"] == best_zero["mode"]
        and row["template"] == best_zero["template"]
    )
    main_rows = [
        {"method": "RoBERTa fine-tuning", "accuracy": "0.868", "consistency": "order 0.939", "note": "supervised baseline"},
        {
            "method": "Best zero-shot LLM",
            "accuracy": best_zero["accuracy"],
            "consistency": "prompt {}; order {}".format(best_zero["prompt_consistency"], best_zero_order),
            "note": "{} {} {}".format(best_zero["model"], best_zero["mode"], best_zero["template"]),
        },
        {
            "method": "Best 8-shot LLM",
            "accuracy": best_few["accuracy"],
            "consistency": "prompt {}".format(best_few["prompt_consistency"]),
            "note": "{} {} {}".format(best_few["model"], best_few["mode"], best_few["template"]),
        },
        {
            "method": "Best non-thinking self-consistency",
            "accuracy": best_sc["accuracy"],
            "consistency": "prompt {}; sampling {}".format(best_sc["prompt_consistency"], best_sc["sampling_consistency"]),
            "note": "{} {} {} k=5".format(best_sc["model"], best_sc["mode"], best_sc["template"]),
        },
    ]
    write_csv(REPORT_DIR / "main_results.csv", main_rows, ["method", "accuracy", "consistency", "note"])
    print("wrote", REPORT_DIR)
    print("zero_best", zero[0])
    print("few_best", few[0])
    print("self_consistency_best", sc[0])
    print("manifest_status", {row["status"] for row in build_manifest()})


if __name__ == "__main__":
    main()
