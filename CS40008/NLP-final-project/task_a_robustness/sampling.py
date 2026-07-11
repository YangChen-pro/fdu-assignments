"""Self-consistency helpers for repeated LLM samples."""

from __future__ import annotations

from collections import Counter, defaultdict

from .data import Prediction, json_note
from .metrics import MetricRow


def choose_majority(values: list[int]) -> tuple[int, float, dict[int, int]]:
    """Choose a deterministic majority label and return label/share/counts."""

    counts = Counter(values)
    pred = min(
        counts,
        key=lambda label: (-counts[label], label not in (0, 1), label),
    )
    return pred, counts[pred] / len(values), dict(sorted(counts.items()))


def majority_vote(predictions: list[Prediction], method: str) -> list[Prediction]:
    """Collapse repeated predictions into one majority-vote prediction per item."""

    grouped: dict[str, list[Prediction]] = defaultdict(list)
    for row in predictions:
        grouped[row.id].append(row)

    voted: list[Prediction] = []
    for item_id, rows in sorted(grouped.items()):
        pred, vote_share, counts = choose_majority([row.pred for row in rows])
        gold = rows[0].gold
        voted.append(
            Prediction(
                id=item_id,
                method=method,
                pred=pred,
                gold=gold,
                correct=int(pred == gold),
                notes=json_note(k=len(rows), vote_share=round(vote_share, 6), counts=counts),
            )
        )
    return voted


def majority_vote_by_method(predictions: list[Prediction]) -> list[Prediction]:
    """Majority-vote repeated predictions while preserving each method name."""

    grouped: dict[tuple[str, str], list[Prediction]] = defaultdict(list)
    for row in predictions:
        grouped[(row.method, row.id)].append(row)

    voted: list[Prediction] = []
    for (method, item_id), rows in sorted(grouped.items()):
        pred, vote_share, counts = choose_majority([row.pred for row in rows])
        gold = rows[0].gold
        voted.append(
            Prediction(
                id=item_id,
                method=method,
                pred=pred,
                gold=gold,
                correct=int(pred == gold),
                notes=json_note(k=len(rows), vote_share=round(vote_share, 6), counts=counts),
            )
        )
    return voted


def sampling_consistency(predictions: list[Prediction], method: str, split: str) -> MetricRow:
    """Average majority-label share across repeated samples."""

    grouped: dict[str, list[int]] = defaultdict(list)
    for row in predictions:
        grouped[row.id].append(row.pred)
    if not grouped:
        raise ValueError("sampling_consistency requires at least one prediction")
    shares = [choose_majority(values)[1] for values in grouped.values()]
    return MetricRow(
        method=method,
        metric="sampling_consistency",
        value=sum(shares) / len(shares),
        split=split,
        notes=f"items={len(shares)}",
    )


def sampling_consistency_by_method(predictions: list[Prediction], split: str) -> list[MetricRow]:
    """Compute sampling consistency for every method in a sample prediction file."""

    grouped: dict[str, list[Prediction]] = defaultdict(list)
    for row in predictions:
        grouped[row.method].append(row)
    return [
        sampling_consistency(rows, method, split)
        for method, rows in sorted(grouped.items())
    ]
