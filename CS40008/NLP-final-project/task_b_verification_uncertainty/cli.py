"""Command-line helpers for Task B verification and uncertainty experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from task_a_robustness.data import Prediction, json_note, load_examples
from task_a_robustness.data import load_predictions, write_predictions
from task_a_robustness.prompts import parse_label

from .metrics import MetricRow, confidence_bucket_metrics, metrics_by_method
from .metrics import selective_prediction_metrics, write_metric_rows
from .paths import OUTPUTS_DIR, RESULTS_DIR, ensure_runtime_dirs
from .prompts import CANDIDATE_TEMPLATE, CONSTRAINT_TEMPLATE, DIRECT_TEMPLATE
from .prompts import extract_response_text, parse_final_label, parse_reason, parse_score
from .prompts import verifier_prompt, write_prompt_batch


TEMPLATES = {
    "direct": (DIRECT_TEMPLATE, "task_b_direct"),
    "constraint": (CONSTRAINT_TEMPLATE, "task_b_constraint_first"),
    "candidate": (CANDIDATE_TEMPLATE, "task_b_candidate"),
}


def make_prompts(args: argparse.Namespace) -> None:
    """Create JSONL prompt batches for Task B LLM runs."""

    ensure_runtime_dirs()
    examples = load_examples(args.split)
    template, method_prefix = TEMPLATES[args.mode]
    write_prompt_batch(
        examples,
        args.output,
        template=template,
        method_prefix=method_prefix,
        repeat=args.repeat,
    )
    print(f"wrote {len(examples) * args.repeat} prompts to {args.output}")


def parse_responses(args: argparse.Namespace) -> None:
    """Parse direct or constraint-first LLM responses into prediction CSV."""

    ensure_runtime_dirs()
    predictions: list[Prediction] = []
    with args.input.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            payload = json.loads(line)
            response = extract_response_text(payload)
            label = parse_label(response)
            if label is None:
                label = args.invalid_label
            gold = int(payload["gold"])
            predictions.append(
                Prediction(
                    id=str(payload["id"]),
                    method=str(payload["method"]),
                    pred=label,
                    gold=gold,
                    correct=int(label == gold),
                    notes=json_note(
                        sample_index=payload.get("sample_index", 0),
                        reason=parse_reason(response),
                    ),
                )
            )
    write_predictions(predictions, args.output)
    print(f"wrote {len(predictions)} parsed predictions to {args.output}")


def make_verifier_prompts(args: argparse.Namespace) -> None:
    """Create verifier prompts from generated candidate responses."""

    ensure_runtime_dirs()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.candidates.open("r", encoding="utf-8") as input_file, args.output.open(
        "w", encoding="utf-8"
    ) as output_file:
        for line_number, line in enumerate(input_file, start=1):
            payload = json.loads(line)
            response = extract_response_text(payload)
            label = parse_label(response)
            if label is None:
                if args.skip_invalid:
                    continue
                label = args.invalid_label
            candidate_id = candidate_identifier(payload, line_number)
            prompt = verifier_prompt(
                sent0=str(payload["sent0"]),
                sent1=str(payload["sent1"]),
                candidate_label=label,
                candidate_reason=parse_reason(response),
            )
            output_file.write(
                json.dumps(
                    {
                        "candidate_id": candidate_id,
                        "id": payload["id"],
                        "method": "task_b_verifier",
                        "gold": payload["gold"],
                        "candidate_label": label,
                        "candidate_method": payload.get("method", "task_b_candidate"),
                        "sample_index": payload.get("sample_index", 0),
                        "sent0": payload["sent0"],
                        "sent1": payload["sent1"],
                        "candidate_reason": parse_reason(response),
                        "prompt": prompt,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    print(f"wrote {count} verifier prompts to {args.output}")


def rerank(args: argparse.Namespace) -> None:
    """Rerank candidate answers using verifier responses."""

    ensure_runtime_dirs()
    candidates = load_candidate_rows(args.candidates)
    verifier_rows = load_verifier_rows(args.verifier_responses)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for candidate_id, candidate in candidates.items():
        verifier = verifier_rows.get(candidate_id, {})
        score = verifier.get("score")
        final_label = verifier.get("final_label")
        rank_score = int(score) if isinstance(score, int) else args.missing_score
        chosen_label = int(final_label) if final_label in (0, 1) else int(candidate["label"])
        candidate = {
            **candidate,
            "candidate_id": candidate_id,
            "rank_score": rank_score,
            "chosen_label": chosen_label,
            "verifier_reason": verifier.get("reason", ""),
        }
        grouped[str(candidate["id"])].append(candidate)

    predictions: list[Prediction] = []
    for item_id, rows in sorted(grouped.items()):
        best = max(rows, key=lambda row: (int(row["rank_score"]), -int(row["sample_index"])))
        gold = int(best["gold"])
        pred = int(best["chosen_label"])
        predictions.append(
            Prediction(
                id=item_id,
                method=args.method,
                pred=pred,
                gold=gold,
                correct=int(pred == gold),
                notes=json_note(
                    candidate_id=best["candidate_id"],
                    candidate_label=best["label"],
                    rank_score=best["rank_score"],
                    candidate_reason=best["reason"],
                    verifier_reason=best["verifier_reason"],
                ),
            )
        )
    write_predictions(predictions, args.output)
    metric_rows = metrics_by_method(predictions, args.split)
    write_metric_rows(metric_rows, args.metrics_output)
    print(f"wrote {len(predictions)} reranked predictions to {args.output}")


def uncertainty(args: argparse.Namespace) -> None:
    """Build majority predictions and uncertainty metrics from repeated samples."""

    ensure_runtime_dirs()
    samples = [row for row in load_predictions(args.samples) if row.pred in (0, 1)]
    grouped: dict[str, list[Prediction]] = defaultdict(list)
    for row in samples:
        grouped[row.id].append(row)

    voted_with_confidence: list[tuple[Prediction, float]] = []
    for item_id, rows in sorted(grouped.items()):
        counts = Counter(row.pred for row in rows)
        pred, majority_count = counts.most_common(1)[0]
        gold = rows[0].gold
        confidence = majority_count / len(rows)
        voted_with_confidence.append(
            (
                Prediction(
                    id=item_id,
                    method=args.method,
                    pred=int(pred),
                    gold=gold,
                    correct=int(pred == gold),
                    notes=json_note(confidence=round(confidence, 6), samples=len(rows)),
                ),
                confidence,
            )
        )

    predictions = [row for row, _confidence in voted_with_confidence]
    write_predictions(predictions, args.output)
    metric_rows = metrics_by_method(predictions, args.split)
    if voted_with_confidence:
        mean_confidence = sum(conf for _row, conf in voted_with_confidence) / len(voted_with_confidence)
        metric_rows.append(
            MetricRow(
                method=args.method,
                metric="mean_confidence",
                value=mean_confidence,
                split=args.split,
                notes=f"items={len(voted_with_confidence)}",
            )
        )
    metric_rows.extend(confidence_bucket_metrics(voted_with_confidence, args.method, args.split))
    metric_rows.extend(
        selective_prediction_metrics(voted_with_confidence, args.method, args.split, args.thresholds)
    )
    write_metric_rows(metric_rows, args.metrics_output)
    print(f"wrote {len(predictions)} uncertainty predictions to {args.output}")


def evaluate(args: argparse.Namespace) -> None:
    """Compute Task B accuracy metrics from a prediction CSV."""

    ensure_runtime_dirs()
    predictions = load_predictions(args.predictions)
    write_metric_rows(metrics_by_method(predictions, args.split), args.output)
    print(f"wrote metrics to {args.output}")


def combine_predictions(args: argparse.Namespace) -> None:
    """Combine several prediction CSV files into one Task B result file."""

    ensure_runtime_dirs()
    predictions: list[Prediction] = []
    for path in args.inputs:
        predictions.extend(load_predictions(path))
    write_predictions(predictions, args.output)
    print(f"wrote {len(predictions)} combined predictions to {args.output}")


def load_candidate_rows(path: Path) -> dict[str, dict[str, object]]:
    """Load candidate responses keyed by deterministic candidate_id."""

    rows: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            payload = json.loads(line)
            response = extract_response_text(payload)
            label = parse_label(response)
            if label is None:
                continue
            rows[candidate_identifier(payload, line_number)] = {
                "id": str(payload["id"]),
                "gold": int(payload["gold"]),
                "label": label,
                "reason": parse_reason(response),
                "sample_index": int(payload.get("sample_index", 0)),
            }
    return rows


def load_verifier_rows(path: Path) -> dict[str, dict[str, object]]:
    """Load verifier responses keyed by candidate_id."""

    rows: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            payload = json.loads(line)
            response = extract_response_text(payload)
            candidate_id = str(payload["candidate_id"])
            rows[candidate_id] = {
                "score": parse_score(response),
                "final_label": parse_final_label(response),
                "reason": parse_reason(response),
            }
    return rows


def candidate_identifier(payload: dict[str, object], line_number: int) -> str:
    """Return a stable candidate identifier shared by candidate and verifier files."""

    if payload.get("candidate_id"):
        return str(payload["candidate_id"])
    return f"{payload['id']}::{payload.get('method', 'candidate')}::{payload.get('sample_index', line_number)}"


def build_parser() -> argparse.ArgumentParser:
    """Build the Task B command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prompt_parser = subparsers.add_parser("make-prompts")
    prompt_parser.add_argument("--split", default="test")
    prompt_parser.add_argument("--mode", choices=sorted(TEMPLATES), default="direct")
    prompt_parser.add_argument("--repeat", type=int, default=1)
    prompt_parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "task_b_prompts.jsonl")
    prompt_parser.set_defaults(func=make_prompts)

    parse_parser = subparsers.add_parser("parse-responses")
    parse_parser.add_argument("--input", type=Path, required=True)
    parse_parser.add_argument("--output", type=Path, default=RESULTS_DIR / "task_b_predictions.csv")
    parse_parser.add_argument("--invalid-label", type=int, default=-1)
    parse_parser.set_defaults(func=parse_responses)

    verifier_parser = subparsers.add_parser("make-verifier-prompts")
    verifier_parser.add_argument("--candidates", type=Path, required=True)
    verifier_parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "task_b_verifier_prompts.jsonl")
    verifier_parser.add_argument("--invalid-label", type=int, default=-1)
    verifier_parser.add_argument("--skip-invalid", action="store_true")
    verifier_parser.set_defaults(func=make_verifier_prompts)

    rerank_parser = subparsers.add_parser("rerank")
    rerank_parser.add_argument("--split", default="test")
    rerank_parser.add_argument("--candidates", type=Path, required=True)
    rerank_parser.add_argument("--verifier-responses", type=Path, required=True)
    rerank_parser.add_argument("--method", default="task_b_generate_verify_rerank")
    rerank_parser.add_argument("--missing-score", type=int, default=0)
    rerank_parser.add_argument("--output", type=Path, default=RESULTS_DIR / "task_b_rerank_predictions.csv")
    rerank_parser.add_argument(
        "--metrics-output", type=Path, default=RESULTS_DIR / "task_b_rerank_metrics.csv"
    )
    rerank_parser.set_defaults(func=rerank)

    uncertainty_parser = subparsers.add_parser("uncertainty")
    uncertainty_parser.add_argument("--split", default="test")
    uncertainty_parser.add_argument("--samples", type=Path, required=True)
    uncertainty_parser.add_argument("--method", default="task_b_consistency_confidence")
    uncertainty_parser.add_argument("--thresholds", type=float, nargs="+", default=[0.6, 0.8, 1.0])
    uncertainty_parser.add_argument("--output", type=Path, default=RESULTS_DIR / "task_b_uncertainty_predictions.csv")
    uncertainty_parser.add_argument(
        "--metrics-output", type=Path, default=RESULTS_DIR / "task_b_uncertainty_metrics.csv"
    )
    uncertainty_parser.set_defaults(func=uncertainty)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--split", default="test")
    eval_parser.add_argument("--predictions", type=Path, default=RESULTS_DIR / "task_b_predictions.csv")
    eval_parser.add_argument("--output", type=Path, default=RESULTS_DIR / "task_b_metrics.csv")
    eval_parser.set_defaults(func=evaluate)

    combine_parser = subparsers.add_parser("combine-predictions")
    combine_parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    combine_parser.add_argument("--output", type=Path, default=RESULTS_DIR / "task_b_predictions.csv")
    combine_parser.set_defaults(func=combine_predictions)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
