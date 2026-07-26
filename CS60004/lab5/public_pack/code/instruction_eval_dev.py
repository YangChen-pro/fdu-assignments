#!/usr/bin/env python3
"""Evaluate predictions on the public instruction-following dev set."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from instruction_eval import evaluation_lib


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def instruction_accuracy(outputs):
    instruction_total = sum(len(item.follow_instruction_list) for item in outputs)
    instruction_correct = sum(sum(item.follow_instruction_list) for item in outputs)
    return instruction_correct / instruction_total if instruction_total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="data/instruction_following/dev.jsonl", type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    args = parser.parse_args()

    dev_rows = read_jsonl(args.dev)
    pred_rows = read_jsonl(args.predictions)

    key_to_prompt = {str(row["key"]): row["prompt"] for row in dev_rows}
    key_to_prediction = {}
    for row in pred_rows:
        key = str(row.get("key", ""))
        if key in key_to_prediction:
            raise ValueError(f"Duplicate prediction key: {key}")
        if key not in key_to_prompt:
            raise ValueError(f"Prediction key not found in dev set: {key}")
        prediction = row.get("prediction")
        if prediction is None:
            raise ValueError(f"Prediction row misses `prediction`: {row}")
        key_to_prediction[key] = str(prediction)

    missing = sorted(set(key_to_prompt) - set(key_to_prediction))
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} dev examples. First missing key: {missing[0]}")

    response_rows = [
        {"prompt": row["prompt"], "response": key_to_prediction[str(row["key"])]}
        for row in dev_rows
    ]

    output_dir = args.output_dir
    if output_dir is None:
        temp_ctx = tempfile.TemporaryDirectory()
        output_dir = Path(temp_ctx.name)
    else:
        temp_ctx = None
        output_dir.mkdir(parents=True, exist_ok=True)

    response_path = output_dir / "dev_input_response.jsonl"
    write_jsonl(response_path, response_rows)

    inputs = evaluation_lib.read_prompt_list(args.dev)
    prompt_to_response = evaluation_lib.read_prompt_to_response_dict(response_path)

    outputs = [evaluation_lib.test_instruction_following_strict(inp, prompt_to_response) for inp in inputs]
    if args.output_dir is not None:
        evaluation_lib.write_outputs(output_dir / "eval_results_strict.jsonl", outputs)
    print(f"score: {instruction_accuracy(outputs)}")

    if temp_ctx is not None:
        temp_ctx.cleanup()


if __name__ == "__main__":
    main()
