"""Shared helpers for Task A vLLM inference runs."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable

from .data import ComveExample, Prediction, load_examples, write_predictions
from .metrics import metrics_by_method, prompt_consistency, write_metric_rows
from .prompts import ZERO_SHOT_TEMPLATES, PromptTemplate, parse_label
from .sampling import majority_vote_by_method, sampling_consistency_by_method
from .vllm_config import GenerationSettings, ModelSpec


def experiment_stage(few_shot: int, repeat: int) -> str:
    """Return the method stage name used in predictions and filenames."""

    parts = [f"few_shot_{few_shot}" if few_shot else "zero_shot"]
    if repeat > 1:
        parts.append(f"self_consistency_k{repeat}")
    return "_".join(parts)


def output_suffix(split: str, order_swap: bool, stage: str) -> str:
    """Build a stable output suffix without changing existing zero-shot filenames."""

    parts = ["swap" if order_swap else split]
    if stage != "zero_shot":
        parts.append(stage)
    return "_".join(parts)


def select_demonstrations(k: int, seed: int = 0) -> list[ComveExample] | None:
    """Select a deterministic balanced few-shot set from the training split."""

    if k <= 0:
        return None
    by_label = {0: [], 1: []}
    for example in load_examples("train"):
        by_label[example.gold].append(example)

    rng = random.Random(seed)
    for examples in by_label.values():
        rng.shuffle(examples)

    counts = {0: k // 2, 1: k // 2}
    for label in range(k % 2):
        counts[label] += 1
    if any(len(by_label[label]) < counts[label] for label in counts):
        raise ValueError(f"not enough training examples for {k}-shot demonstrations")

    selected: list[ComveExample] = []
    for index in range(max(counts.values())):
        for label in (0, 1):
            if index < counts[label]:
                selected.append(by_label[label][index])
    return selected[:k]


def build_user_prompt(
    example: ComveExample,
    template: PromptTemplate,
    demonstrations: list[ComveExample] | None,
) -> str:
    """Render the Task A prompt and add a strict output contract."""

    task_text = template.render(example, demonstrations=demonstrations)
    return (
        f"{task_text}\n\n"
        "Output contract: answer with exactly one digit, either 0 or 1. "
        "Do not include explanation or extra text."
    )


def apply_chat_template(tokenizer, user_prompt: str, settings: GenerationSettings) -> str:
    """Apply the model chat template using README-supported thinking switch."""

    messages = [{"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=settings.enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": settings.enable_thinking},
        )


def clean_output(text: str) -> str:
    """Remove optional thinking block and surrounding whitespace."""

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return text.strip()


def make_sampling_params(settings: GenerationSettings, max_tokens: int):
    """Create vLLM SamplingParams from README-backed settings."""

    from vllm import SamplingParams

    kwargs: dict[str, object] = {
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "top_k": settings.top_k,
        "min_p": settings.min_p,
        "max_tokens": max_tokens,
    }
    if settings.presence_penalty is not None:
        kwargs["presence_penalty"] = settings.presence_penalty
    if settings.repetition_penalty is not None:
        kwargs["repetition_penalty"] = settings.repetition_penalty
    return SamplingParams(**kwargs)


def select_templates(names: list[str]) -> list[PromptTemplate]:
    """Select prompt templates by name or all templates."""

    if names == ["all"]:
        return ZERO_SHOT_TEMPLATES
    available = {template.name: template for template in ZERO_SHOT_TEMPLATES}
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"unknown prompt template(s): {missing}; available={sorted(available)}")
    return [available[name] for name in names]


def iter_prompt_items(
    examples: Iterable[ComveExample],
    templates: list[PromptTemplate],
    spec: ModelSpec,
    settings: GenerationSettings,
    mode: str,
    tokenizer,
    demonstrations: list[ComveExample] | None,
    stage: str,
    repeat: int,
) -> list[dict[str, object]]:
    """Build vLLM prompt payloads."""

    items: list[dict[str, object]] = []
    for example in examples:
        for template in templates:
            method = f"{spec.name}:{mode}:{stage}:{template.name}"
            user_prompt = build_user_prompt(example, template, demonstrations)
            prompt = apply_chat_template(tokenizer, user_prompt, settings)
            for sample_index in range(repeat):
                items.append(
                    {
                        "id": example.id,
                        "method": method,
                        "gold": example.gold,
                        "prompt_template": template.name,
                        "mode": mode,
                        "sample_index": sample_index,
                        "prompt": prompt,
                    }
                )
    return items


def write_raw_outputs(rows: list[dict[str, object]], path: Path) -> None:
    """Write raw model outputs for audit and parsing recovery."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def settings_dict(settings: GenerationSettings, max_tokens: int) -> dict[str, object]:
    """Return serializable settings used for this run."""

    data = {
        "enable_thinking": settings.enable_thinking,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "top_k": settings.top_k,
        "min_p": settings.min_p,
        "max_tokens": max_tokens,
    }
    if settings.presence_penalty is not None:
        data["presence_penalty"] = settings.presence_penalty
    if settings.repetition_penalty is not None:
        data["repetition_penalty"] = settings.repetition_penalty
    return data


def prediction_from_raw_row(
    row: dict[str, object],
    spec: ModelSpec,
    mode: str,
    settings: GenerationSettings,
    used_settings: dict[str, object],
    order_swap: bool,
    few_shot: int,
    repeat: int,
) -> Prediction:
    """Rebuild a prediction row from a previously written raw-output row."""

    pred = int(row["pred"])
    gold = int(row["gold"])
    notes = {
        "model_path": str(spec.path),
        "mode": mode,
        "prompt_template": row["prompt_template"],
        "order_swap": order_swap,
        "few_shot": few_shot,
        "repeat": repeat,
        "sample_index": int(row.get("sample_index", 0)),
        "settings": used_settings,
        "source_note": settings.source_note,
    }
    return Prediction(
        id=str(row["id"]),
        method=str(row["method"]),
        pred=pred,
        gold=gold,
        correct=int(pred == gold),
        notes=json.dumps(notes, ensure_ascii=False, sort_keys=True),
    )


def resume_key(row: dict[str, object]) -> tuple[str, str, int]:
    """Return the unique key for one generated sample."""

    return (str(row["id"]), str(row["prompt_template"]), int(row.get("sample_index", 0)))


def load_resume_rows(
    raw_path: Path,
    spec: ModelSpec,
    mode: str,
    settings: GenerationSettings,
    used_settings: dict[str, object],
    order_swap: bool,
    few_shot: int,
    repeat: int,
) -> tuple[list[Prediction], list[dict[str, object]], set[tuple[str, str, int]]]:
    """Load completed raw rows so interrupted vLLM runs can continue."""

    if not raw_path.exists():
        return [], [], set()
    predictions: list[Prediction] = []
    raw_rows: list[dict[str, object]] = []
    completed: set[tuple[str, str, int]] = set()
    expected_prefix = f"{spec.name}:{mode}:"
    with raw_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if not str(row.get("method", "")).startswith(expected_prefix):
                continue
            key = resume_key(row)
            if key in completed:
                continue
            completed.add(key)
            raw_rows.append(row)
            predictions.append(
                prediction_from_raw_row(
                    row, spec, mode, settings, used_settings, order_swap, few_shot, repeat
                )
            )
    return predictions, raw_rows, completed


def build_llm_kwargs(args: argparse.Namespace, spec: ModelSpec) -> dict[str, object]:
    """Build vLLM engine kwargs from CLI arguments."""

    llm_kwargs: dict[str, object] = {"model": str(spec.path), "trust_remote_code": True}
    for arg_name in (
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "max_model_len",
        "max_num_seqs",
        "enforce_eager",
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            llm_kwargs[arg_name] = value
    if args.language_model_only:
        llm_kwargs["language_model_only"] = True
    return llm_kwargs


def finalise_predictions(
    sample_predictions: list[Prediction],
    repeat: int,
    split: str,
    sample_path: Path,
    final_path: Path,
    metrics_path: Path,
) -> tuple[int, int]:
    """Write final prediction and metric files after generation completes."""

    if repeat > 1:
        final_predictions = majority_vote_by_method(sample_predictions)
        metric_rows = metrics_by_method(final_predictions, split)
        metric_rows.extend(prompt_consistency(final_predictions, split))
        metric_rows.extend(sampling_consistency_by_method(sample_predictions, split))
        write_predictions(sample_predictions, sample_path)
        write_predictions(final_predictions, final_path)
    else:
        final_predictions = sample_predictions
        metric_rows = metrics_by_method(final_predictions, split)
        metric_rows.extend(prompt_consistency(final_predictions, split))
        write_predictions(final_predictions, final_path)
    write_metric_rows(metric_rows, metrics_path)
    return len(sample_predictions), len(final_predictions)
