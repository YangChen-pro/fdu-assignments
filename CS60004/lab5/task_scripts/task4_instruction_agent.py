from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from tqdm import tqdm

from utils import call_chat_completion, ordered_records_for_samples, read_jsonl, records_by_field, write_json, write_jsonl, write_jsonl_record


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Follow the user instruction exactly and answer directly."


def build_messages(prompt: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def request_prediction(args: argparse.Namespace, sample: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    prediction = call_chat_completion(
        base_url=args.api_base,
        model=args.served_model,
        messages=build_messages(str(sample["prompt"]), args.system_prompt),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        timeout=args.timeout,
    )
    return {
        "key": sample["key"],
        "prediction": prediction,
        "latency_s": time.time() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Lab5 Task4 instruction-following predictions with a vLLM-served model."
    )
    parser.add_argument("--input", type=Path, default=Path("lab5/public_pack/data/instruction_following/dev.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("lab5/outputs/task4"))
    parser.add_argument("--output-name", default="instruction_dev")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--served-model", default="Qwen2.5-1.5B-Instruct")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = read_jsonl(args.input)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    run_dir = args.output_dir / args.output_name
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = run_dir / "predictions.jsonl"
    detail_path = run_dir / "details.jsonl"
    metrics_path = run_dir / "metrics.json"

    existing = records_by_field(prediction_path, "key", required_field="prediction")
    existing_details = records_by_field(detail_path, "key", required_field="prediction")
    pending = [sample for sample in samples if str(sample["key"]) not in existing]
    write_jsonl(
        prediction_path,
        ordered_records_for_samples(samples, existing, sample_field="key", output_fields=("key", "prediction")),
    )
    print(
        json.dumps(
            {
                "resume_from": str(prediction_path),
                "already_done": len(existing),
                "pending": len(pending),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    latencies = [
        float(row.get("latency_s", 0.0))
        for row in existing_details.values()
        if isinstance(row.get("latency_s"), int | float)
    ]
    started_at = time.time()
    with (
        prediction_path.open("a", encoding="utf-8") as pred_f,
        detail_path.open("a", encoding="utf-8") as detail_f,
        ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor,
    ):
        for start in range(0, len(pending), args.concurrency):
            batch = pending[start : start + args.concurrency]
            futures = [executor.submit(request_prediction, args, sample) for sample in batch]
            desc = f"instruction {start + 1}-{start + len(batch)}/{len(pending)}"
            for sample, future in zip(batch, tqdm(futures, desc=desc, unit="sample")):
                record = future.result()
                latencies.append(record["latency_s"])
                write_jsonl_record(pred_f, {"key": record["key"], "prediction": record["prediction"]})
                write_jsonl_record(
                    detail_f,
                    {
                        "key": record["key"],
                        "prompt": sample["prompt"],
                        "prediction": record["prediction"],
                        "latency_s": record["latency_s"],
                    },
                )

    metrics = {
        "input_file": str(args.input),
        "num_samples": len(samples),
        "concurrency": max(1, args.concurrency),
        "total_latency_s": time.time() - started_at,
        "avg_latency_s": float(np.mean(latencies)) if latencies else math.nan,
        "p50_latency_s": float(np.percentile(latencies, 50)) if latencies else math.nan,
        "p95_latency_s": float(np.percentile(latencies, 95)) if latencies else math.nan,
        "prediction_file": str(prediction_path),
        "detail_file": str(detail_path),
        "served_model": args.served_model,
        "system_prompt": args.system_prompt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_tokens": args.max_tokens,
    }
    write_json(metrics_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
