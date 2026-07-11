"""Metrics for Task B verification and uncertainty analysis."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from task_a_robustness.data import Prediction
from task_a_robustness.metrics import accuracy


@dataclass(frozen=True)
class MetricRow:
    """One row for results/task_b_metrics.csv."""

    method: str
    metric: str
    value: float
    split: str
    notes: str = ""


def metrics_by_method(predictions: list[Prediction], split: str) -> list[MetricRow]:
    """Compute accuracy for each method."""

    grouped: dict[str, list[Prediction]] = defaultdict(list)
    for row in predictions:
        grouped[row.method].append(row)
    return [
        MetricRow(method=method, metric="accuracy", value=accuracy(rows), split=split)
        for method, rows in sorted(grouped.items())
    ]


def confidence_bucket_metrics(
    rows: list[tuple[Prediction, float]], method: str, split: str
) -> list[MetricRow]:
    """Compute accuracy in coarse confidence buckets."""

    buckets = {
        "conf_[0.50,0.60)": [],
        "conf_[0.60,0.80)": [],
        "conf_[0.80,1.00]": [],
    }
    for prediction, confidence in rows:
        if confidence < 0.6:
            buckets["conf_[0.50,0.60)"].append(prediction)
        elif confidence < 0.8:
            buckets["conf_[0.60,0.80)"].append(prediction)
        else:
            buckets["conf_[0.80,1.00]"].append(prediction)

    metrics: list[MetricRow] = []
    for bucket, predictions in buckets.items():
        if predictions:
            metrics.append(
                MetricRow(
                    method=method,
                    metric=f"{bucket}_accuracy",
                    value=accuracy(predictions),
                    split=split,
                    notes=f"items={len(predictions)}",
                )
            )
    return metrics


def selective_prediction_metrics(
    rows: list[tuple[Prediction, float]], method: str, split: str, thresholds: list[float]
) -> list[MetricRow]:
    """Compute coverage and accuracy after confidence-threshold abstention."""

    total = len(rows)
    if total == 0:
        raise ValueError("selective_prediction_metrics requires at least one row")

    metrics: list[MetricRow] = []
    for threshold in thresholds:
        selected = [prediction for prediction, confidence in rows if confidence >= threshold]
        coverage = len(selected) / total
        selected_accuracy = accuracy(selected) if selected else 0.0
        suffix = f"threshold={threshold:.2f}"
        metrics.append(
            MetricRow(
                method=method,
                metric="selective_coverage",
                value=coverage,
                split=split,
                notes=suffix,
            )
        )
        metrics.append(
            MetricRow(
                method=method,
                metric="selective_accuracy",
                value=selected_accuracy,
                split=split,
                notes=f"{suffix};items={len(selected)}",
            )
        )
    return metrics


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

