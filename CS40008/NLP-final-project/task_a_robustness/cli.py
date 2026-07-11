"""Command-line helpers for Task A robustness experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import Prediction, load_examples, load_predictions, make_order_swapped
from .data import write_examples_csv, write_predictions
from .metrics import (
    metrics_by_method,
    order_consistency,
    order_consistency_by_method,
    prompt_consistency,
    write_metric_rows,
)
from .paths import (
    OUTPUTS_DIR,
    TASK_A_ROBERTA_RESULTS_DIR,
    TASK_A_VLLM_METRICS_DIR,
    TASK_A_VLLM_PREDICTIONS_DIR,
    ensure_runtime_dirs,
)
from .prompts import parse_label, write_prompt_batch
from .sampling import majority_vote, majority_vote_by_method, sampling_consistency, sampling_consistency_by_method


def validate_data(args: argparse.Namespace) -> None:
    """Validate processed data splits."""

    for split in args.splits:
        examples = load_examples(split)
        counts = {0: 0, 1: 0}
        for example in examples:
            counts[example.gold] += 1
        print(f"{split}: rows={len(examples)}, gold0={counts[0]}, gold1={counts[1]}")


def make_swapped(args: argparse.Namespace) -> None:
    """Create an order-swapped CSV split."""

    ensure_runtime_dirs()
    examples = make_order_swapped(load_examples(args.split))
    write_examples_csv(examples, args.output)
    print(f"wrote {len(examples)} rows to {args.output}")


def make_prompts(args: argparse.Namespace) -> None:
    """Create JSONL prompt batches for LLM runs."""

    ensure_runtime_dirs()
    examples = load_examples(args.split)
    demonstrations = load_examples("train")[: args.few_shot] if args.few_shot else None
    prefix = "llm_few_shot" if demonstrations else "llm_zero_shot"
    write_prompt_batch(
        examples,
        args.output,
        demonstrations=demonstrations,
        method_prefix=prefix,
        repeat=args.repeat,
    )
    print(f"wrote prompts for {len(examples)} examples to {args.output}")


def parse_llm_responses(args: argparse.Namespace) -> None:
    """Parse JSONL LLM responses into the shared prediction CSV schema."""

    ensure_runtime_dirs()
    predictions: list[Prediction] = []
    with args.input.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            payload = json.loads(line)
            response = payload.get("response") or payload.get("output_text") or payload.get("text")
            if response is None:
                raise ValueError(f"line {line_number} has no response/output_text/text field")
            label = parse_label(str(response))
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
                    notes=str(payload.get("notes", "")),
                )
            )
    write_predictions(predictions, args.output)
    print(f"wrote {len(predictions)} parsed predictions to {args.output}")


def self_consistency(args: argparse.Namespace) -> None:
    """Majority-vote repeated prediction rows."""

    ensure_runtime_dirs()
    samples = load_predictions(args.samples)
    if args.method:
        voted = majority_vote(samples, args.method)
        metric_rows = metrics_by_method(voted, args.split)
        metric_rows.append(sampling_consistency(samples, args.method, args.split))
    else:
        voted = majority_vote_by_method(samples)
        metric_rows = metrics_by_method(voted, args.split)
        metric_rows.extend(sampling_consistency_by_method(samples, args.split))
    write_predictions(voted, args.output)
    write_metric_rows(metric_rows, args.metrics_output)
    print(f"wrote {len(voted)} majority-vote rows to {args.output}")


def evaluate_predictions(args: argparse.Namespace) -> None:
    """Compute metrics from a prediction CSV."""

    ensure_runtime_dirs()
    predictions = load_predictions(args.predictions)
    metric_rows = metrics_by_method(predictions, args.split)
    metric_rows.extend(prompt_consistency(predictions, args.split))
    write_metric_rows(metric_rows, args.output)
    print(f"wrote {len(metric_rows)} metric rows to {args.output}")


def evaluate_order_swap(args: argparse.Namespace) -> None:
    """Compute order consistency from original and swapped predictions."""

    ensure_runtime_dirs()
    original = load_predictions(args.original)
    swapped = load_predictions(args.swapped)
    if args.method:
        metric_rows = [order_consistency(original, swapped, args.method, args.split)]
    else:
        metric_rows = order_consistency_by_method(original, swapped, args.split)
    write_metric_rows(metric_rows, args.output)
    print(f"wrote {len(metric_rows)} order consistency metric rows to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    """Build the Task A command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser("validate-data")
    data_parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    data_parser.set_defaults(func=validate_data)

    swap_parser = subparsers.add_parser("make-order-swap")
    swap_parser.add_argument("--split", default="test")
    swap_parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "comve_task_a_test_swap.csv")
    swap_parser.set_defaults(func=make_swapped)

    prompt_parser = subparsers.add_parser("make-prompts")
    prompt_parser.add_argument("--split", default="test")
    prompt_parser.add_argument("--few-shot", type=int, default=0)
    prompt_parser.add_argument("--repeat", type=int, default=1)
    prompt_parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "llm_prompts.jsonl")
    prompt_parser.set_defaults(func=make_prompts)

    parse_parser = subparsers.add_parser("parse-llm-responses")
    parse_parser.add_argument("--input", type=Path, required=True)
    parse_parser.add_argument("--output", type=Path, default=TASK_A_ROBERTA_RESULTS_DIR / "predictions.csv")
    parse_parser.add_argument("--invalid-label", type=int, default=-1)
    parse_parser.set_defaults(func=parse_llm_responses)

    sc_parser = subparsers.add_parser("self-consistency")
    sc_parser.add_argument("--split", default="test")
    sc_parser.add_argument("--samples", type=Path, required=True)
    sc_parser.add_argument("--method", default=None)
    sc_parser.add_argument("--output", type=Path, default=TASK_A_VLLM_PREDICTIONS_DIR / "self_consistency_predictions.csv")
    sc_parser.add_argument("--metrics-output", type=Path, default=TASK_A_VLLM_METRICS_DIR / "self_consistency_metrics.csv")
    sc_parser.set_defaults(func=self_consistency)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--split", default="test")
    eval_parser.add_argument("--predictions", type=Path, default=TASK_A_ROBERTA_RESULTS_DIR / "predictions.csv")
    eval_parser.add_argument("--output", type=Path, default=TASK_A_ROBERTA_RESULTS_DIR / "metrics.csv")
    eval_parser.set_defaults(func=evaluate_predictions)

    order_eval_parser = subparsers.add_parser("evaluate-order-swap")
    order_eval_parser.add_argument("--split", default="test")
    order_eval_parser.add_argument("--original", type=Path, required=True)
    order_eval_parser.add_argument("--swapped", type=Path, required=True)
    order_eval_parser.add_argument("--method", default=None)
    order_eval_parser.add_argument("--output", type=Path, default=TASK_A_ROBERTA_RESULTS_DIR / "order_metrics.csv")
    order_eval_parser.set_defaults(func=evaluate_order_swap)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
