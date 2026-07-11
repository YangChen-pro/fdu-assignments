"""Data loading and prediction file helpers for ComVE Task A."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .paths import processed_split_path


@dataclass(frozen=True)
class ComveExample:
    """One ComVE Task A example in the project-wide schema."""

    id: str
    sent0: str
    sent1: str
    gold: int


@dataclass(frozen=True)
class Prediction:
    """One prediction row using the shared results schema."""

    id: str
    method: str
    pred: int
    gold: int
    correct: int
    notes: str = ""


def load_examples(split: str, path: Path | None = None) -> list[ComveExample]:
    """Load a processed ComVE split with fields id,sent0,sent1,gold."""

    csv_path = path or processed_split_path(split)
    examples: list[ComveExample] = []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        expected = {"id", "sent0", "sent1", "gold"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(f"{csv_path} must contain fields {sorted(expected)}")
        for row in reader:
            examples.append(
                ComveExample(
                    id=row["id"],
                    sent0=row["sent0"],
                    sent1=row["sent1"],
                    gold=int(row["gold"]),
                )
            )
    return examples


def make_order_swapped(examples: Iterable[ComveExample]) -> list[ComveExample]:
    """Swap sent0/sent1 and flip the gold label."""

    return [
        ComveExample(
            id=f"{example.id}__swap",
            sent0=example.sent1,
            sent1=example.sent0,
            gold=1 - example.gold,
        )
        for example in examples
    ]


def write_examples_csv(examples: Iterable[ComveExample], path: Path) -> None:
    """Write examples in the project-wide input schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "sent0", "sent1", "gold"])
        writer.writeheader()
        for example in examples:
            writer.writerow(
                {
                    "id": example.id,
                    "sent0": example.sent0,
                    "sent1": example.sent1,
                    "gold": example.gold,
                }
            )


def write_predictions(predictions: Iterable[Prediction], path: Path) -> None:
    """Write prediction rows using id,method,pred,gold,correct,notes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["id", "method", "pred", "gold", "correct", "notes"]
        )
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(prediction.__dict__)


def load_predictions(path: Path) -> list[Prediction]:
    """Load prediction rows from a CSV file."""

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return [
        Prediction(
            id=row["id"],
            method=row["method"],
            pred=int(row["pred"]),
            gold=int(row["gold"]),
            correct=int(row["correct"]),
            notes=row.get("notes", ""),
        )
        for row in rows
    ]


def json_note(**values: object) -> str:
    """Serialize compact metadata for the notes field."""

    return json.dumps(values, ensure_ascii=False, sort_keys=True)
