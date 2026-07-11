"""Run Task A LLM inference with vLLM's in-process API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import Prediction, load_examples, make_order_swapped, write_predictions
from .paths import TASK_A_VLLM_METRICS_DIR, TASK_A_VLLM_PREDICTIONS_DIR
from .prompts import parse_label
from .vllm_config import DEFAULT_MAX_TOKENS_BY_MODE, DEFAULT_OUTPUT_DIR, MODEL_SPECS
from .vllm_helpers import (
    build_llm_kwargs,
    clean_output,
    experiment_stage,
    finalise_predictions,
    iter_prompt_items,
    load_resume_rows,
    make_sampling_params,
    output_suffix,
    resume_key,
    select_demonstrations,
    select_templates,
    settings_dict,
    write_raw_outputs,
)


def output_paths(args: argparse.Namespace, spec_name: str, mode: str, suffix: str) -> tuple[Path, Path, Path]:
    """Return final prediction, sample prediction, and metric paths."""

    final_prediction_path = (
        args.predictions
        or TASK_A_VLLM_PREDICTIONS_DIR / f"{spec_name}_{mode}_{suffix}_predictions.csv"
    )
    sample_prediction_path = (
        args.sample_predictions
        or TASK_A_VLLM_PREDICTIONS_DIR / f"{spec_name}_{mode}_{suffix}_samples.csv"
    )
    metrics_path = args.metrics or TASK_A_VLLM_METRICS_DIR / f"{spec_name}_{mode}_{suffix}_metrics.csv"
    return final_prediction_path, sample_prediction_path, metrics_path




def run_model(args: argparse.Namespace) -> None:
    """Run one local Qwen model on Task A with vLLM."""

    spec, settings = validate_args(args)
    context = build_run_context(args, spec, settings)
    predictions, raw_rows, completed = load_resume_state(args, spec, settings, context)
    run_batches(args, spec.name, str(spec.path), settings, context, predictions, raw_rows, completed)
    if len(predictions) != len(context["items"]):
        print_incomplete(
            spec.name, args.mode, len(predictions), len(context["items"]), context["metrics_path"]
        )
        return

    sample_rows, final_rows = finalise_predictions(
        predictions,
        args.repeat,
        args.split,
        context["sample_path"],
        context["final_path"],
        context["metrics_path"],
    )
    print_final(
        spec.name,
        args.mode,
        context["stage"],
        context["final_path"],
        context["sample_path"],
        context["metrics_path"],
        context["raw_path"],
        sample_rows,
        final_rows,
        args.repeat,
        settings.source_note,
    )


def validate_args(args: argparse.Namespace):
    """Validate CLI arguments and return the selected spec/settings."""

    spec = MODEL_SPECS[args.model]
    settings = spec.modes[args.mode]
    if not spec.path.exists():
        raise FileNotFoundError(f"model path does not exist: {spec.path}")
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    if args.few_shot < 0:
        raise ValueError("--few-shot must be >= 0")
    return spec, settings


def build_run_context(args: argparse.Namespace, spec, settings) -> dict[str, object]:
    """Initialize vLLM, prompts, sampling parameters, and output paths."""

    from vllm import LLM

    examples = load_examples(args.split)
    if args.order_swap:
        examples = make_order_swapped(examples)
    if args.max_samples:
        examples = examples[: args.max_samples]

    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(**build_llm_kwargs(args, spec))
    tokenizer = llm.get_tokenizer()

    max_tokens = args.max_tokens or DEFAULT_MAX_TOKENS_BY_MODE[args.mode]
    stage = experiment_stage(args.few_shot, args.repeat)
    suffix = output_suffix(args.split, args.order_swap, stage)
    templates = select_templates(args.templates)
    demonstrations = select_demonstrations(args.few_shot, args.few_shot_seed)
    items = iter_prompt_items(
        examples, templates, spec, settings, args.mode, tokenizer, demonstrations, stage, args.repeat
    )
    final_path, sample_path, metrics_path = output_paths(args, spec.name, args.mode, suffix)
    return {
        "llm": llm,
        "items": items,
        "stage": stage,
        "sampling_params": make_sampling_params(settings, max_tokens),
        "used_settings": settings_dict(settings, max_tokens),
        "final_path": final_path,
        "sample_path": sample_path,
        "metrics_path": metrics_path,
        "raw_path": output_dir / f"{spec.name}_{args.mode}_{suffix}_raw_outputs.jsonl",
    }


def load_resume_state(args: argparse.Namespace, spec, settings, context: dict[str, object]):
    """Load resumable predictions/raw rows when requested."""

    if not args.resume:
        return [], [], set()
    predictions, raw_rows, completed = load_resume_rows(
        context["raw_path"],
        spec,
        args.mode,
        settings,
        context["used_settings"],
        args.order_swap,
        args.few_shot,
        args.repeat,
    )
    if completed:
        payload = {
            "event": "resume_loaded",
            "model": spec.name,
            "mode": args.mode,
            "completed": len(completed),
            "total": len(context["items"]),
            "raw_outputs": str(context["raw_path"]),
        }
        print(json.dumps(payload, ensure_ascii=False), flush=True)
    return predictions, raw_rows, completed


def run_batches(
    args: argparse.Namespace,
    model_name: str,
    model_path: str,
    settings,
    context: dict[str, object],
    predictions: list[Prediction],
    raw_rows: list[dict[str, object]],
    completed: set[tuple[str, str, int]],
) -> None:
    """Generate all unfinished prompt items and persist progress per batch."""

    items = context["items"]
    remaining_items = [item for item in items if resume_key(item) not in completed]
    batch_size = args.batch_size or len(remaining_items)
    for start in range(0, len(remaining_items), batch_size):
        end = min(start + batch_size, len(remaining_items))
        batch_items = remaining_items[start:end]
        print_batch_start(model_name, args.mode, start, end, len(predictions), len(items))
        outputs = context["llm"].generate(
            [str(item["prompt"]) for item in batch_items], context["sampling_params"]
        )
        for item, output in zip(batch_items, outputs):
            add_prediction(
                item, output, predictions, raw_rows, args, model_path, settings, context["used_settings"]
            )
        write_predictions(
            predictions, context["sample_path"] if args.repeat > 1 else context["final_path"]
        )
        write_raw_outputs(raw_rows, context["raw_path"])
        print_batch_done(
            model_name,
            args.mode,
            len(predictions),
            len(items),
            context["final_path"],
            context["sample_path"],
            context["raw_path"],
            args.repeat,
        )


def add_prediction(
    item: dict[str, object],
    output,
    predictions: list[Prediction],
    raw_rows: list[dict[str, object]],
    args: argparse.Namespace,
    model_path: str,
    settings,
    used_settings: dict[str, object],
) -> None:
    """Parse one vLLM output and append prediction/raw rows."""

    raw_text = output.outputs[0].text
    cleaned = clean_output(raw_text)
    parsed = parse_label(cleaned)
    pred = parsed if parsed in (0, 1) else args.invalid_label
    gold = int(item["gold"])
    notes = {
        "model_path": model_path,
        "mode": args.mode,
        "prompt_template": item["prompt_template"],
        "order_swap": args.order_swap,
        "few_shot": args.few_shot,
        "repeat": args.repeat,
        "sample_index": item["sample_index"],
        "settings": used_settings,
        "source_note": settings.source_note,
    }
    predictions.append(
        Prediction(
            id=str(item["id"]),
            method=str(item["method"]),
            pred=pred,
            gold=gold,
            correct=int(pred == gold),
            notes=json.dumps(notes, ensure_ascii=False, sort_keys=True),
        )
    )
    raw_rows.append({**item, "raw_output": raw_text, "clean_output": cleaned, "pred": pred})


def print_batch_start(model: str, mode: str, start: int, end: int, completed: int, total: int) -> None:
    payload = {
        "event": "batch_start",
        "model": model,
        "mode": mode,
        "start": start,
        "end": end,
        "completed_before_batch": completed,
        "total": total,
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def print_batch_done(
    model: str,
    mode: str,
    completed: int,
    total: int,
    final_path: Path,
    sample_path: Path,
    raw_path: Path,
    repeat: int,
) -> None:
    payload = {
        "event": "batch_done",
        "model": model,
        "mode": mode,
        "completed": completed,
        "total": total,
        "predictions": str(final_path),
        "sample_predictions": str(sample_path) if repeat > 1 else None,
        "raw_outputs": str(raw_path),
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def print_incomplete(model: str, mode: str, completed: int, total: int, metrics_path: Path) -> None:
    payload = {
        "event": "incomplete_run",
        "model": model,
        "mode": mode,
        "completed": completed,
        "total": total,
        "metrics": str(metrics_path),
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def print_final(
    model: str,
    mode: str,
    stage: str,
    final_path: Path,
    sample_path: Path,
    metrics_path: Path,
    raw_path: Path,
    sample_rows: int,
    final_rows: int,
    repeat: int,
    source_note: str,
) -> None:
    payload = {
        "model": model,
        "mode": mode,
        "stage": stage,
        "predictions": str(final_path),
        "sample_predictions": str(sample_path) if repeat > 1 else None,
        "metrics": str(metrics_path),
        "raw_outputs": str(raw_path),
        "sample_rows": sample_rows,
        "final_rows": final_rows,
        "settings": source_note,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--mode", choices=["non_thinking", "thinking"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--order-swap", action="store_true")
    parser.add_argument("--templates", nargs="+", default=["all"])
    parser.add_argument("--few-shot", type=int, default=0)
    parser.add_argument("--few-shot-seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--invalid-label", type=int, default=-1)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--sample-predictions", type=Path, default=None)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(func=run_model)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
