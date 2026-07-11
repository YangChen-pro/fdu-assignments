"""Metrics for Task A accuracy and robustness analysis."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .data import Prediction


@dataclass(frozen=True)
class MetricRow:
    """One row for a Task A metrics CSV."""

    method: str
    metric: str
    value: float
    split: str
    notes: str = ""


def accuracy(predictions: list[Prediction]) -> float:
    """Return accuracy for prediction rows."""

    if not predictions:
        raise ValueError("accuracy requires at least one prediction")
    return sum(row.correct for row in predictions) / len(predictions)


def metrics_by_method(predictions: list[Prediction], split: str) -> list[MetricRow]:
    """Compute accuracy for each method in a prediction file."""

    grouped: dict[str, list[Prediction]] = defaultdict(list)
    for row in predictions:
        grouped[row.method].append(row)
    return [
        MetricRow(method=method, metric="accuracy", value=accuracy(rows), split=split)
        for method, rows in sorted(grouped.items())
    ]


def order_consistency(
    original: list[Prediction], swapped: list[Prediction], method: str, split: str
) -> MetricRow:
    """Measure whether predictions flip after sent0/sent1 are swapped."""

    original_map = {row.id: row.pred for row in original if row.method == method}
    swapped_map = {
        row.id.removesuffix("__swap"): row.pred
        for row in swapped
        if row.method == method and row.id.endswith("__swap")
    }
    common_ids = sorted(set(original_map) & set(swapped_map))
    if not common_ids:
        raise ValueError(f"no matching swapped predictions for method={method}")
    consistent = sum(swapped_map[item_id] == 1 - original_map[item_id] for item_id in common_ids)
    return MetricRow(
        method=method,
        metric="order_consistency",
        value=consistent / len(common_ids),
        split=split,
        notes=f"matched={len(common_ids)}",
    )


def order_consistency_by_method(
    original: list[Prediction], swapped: list[Prediction], split: str
) -> list[MetricRow]:
    """Compute order consistency for every method shared by both files."""

    original_methods = {row.method for row in original}
    swapped_methods = {row.method for row in swapped}
    rows: list[MetricRow] = []
    for method in sorted(original_methods & swapped_methods):
        try:
            rows.append(order_consistency(original, swapped, method, split))
        except ValueError:
            continue
    if not rows:
        raise ValueError("no matching order-swap methods between the two prediction files")
    return rows


def prompt_family(method: str) -> str | None:
    """Return the comparable prompt family by dropping only the template suffix."""

    parts = method.split(":")
    if len(parts) < 2:
        return None
    return ":".join(parts[:-1])


def prompt_consistency(predictions: list[Prediction], split: str) -> list[MetricRow]:
    """Compute agreement across prompt-template variants within each method family."""

    by_family: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in predictions:
        family = prompt_family(row.method)
        if family is None:
            continue
        by_family[family][row.id].append(row.pred)

    rows: list[MetricRow] = []
    for family, grouped in sorted(by_family.items()):
        comparable = [values for values in grouped.values() if len(values) > 1]
        if not comparable:
            continue
        stable = sum(len(set(values)) == 1 for values in comparable)
        rows.append(
            MetricRow(
                method=family,
                metric="prompt_consistency",
                value=stable / len(comparable),
                split=split,
                notes=f"items={len(comparable)}",
            )
        )
    return rows


def write_metric_rows(rows: list[MetricRow], path: Path) -> None:
    """Write metric rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["method", "metric", "value", "split", "notes"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row.method,
                    "metric": row.metric,
                    "value": f"{row.value:.6f}",
                    "split": row.split,
                    "notes": row.notes,
                }
            )
