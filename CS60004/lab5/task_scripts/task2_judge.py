from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from statistics import mean
import time
from typing import Any

from tqdm import tqdm

from utils import (
    build_role_judge_messages,
    call_chat_completion,
    load_persona,
    ordered_records_for_samples,
    parse_json_object,
    read_jsonl,
    records_by_id,
    write_json,
    write_jsonl,
    write_jsonl_record,
)


def evaluate_one(
    sample: dict[str, Any],
    args: argparse.Namespace,
    data_root: Path,
    baseline_predictions: dict[str, dict[str, Any]],
    candidate_predictions: dict[str, dict[str, Any]],
    memory_details: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sample_id = sample["id"]
    answers = {
        args.baseline_name: baseline_predictions[sample_id]["prediction"],
        args.candidate_name: candidate_predictions[sample_id]["prediction"],
    }
    messages = build_role_judge_messages(
        character_id=sample["character_id"],
        question=sample["question"],
        persona=load_persona(data_root, sample["character_id"]),
        answers=answers,
        retrieved_memories=memory_details.get(sample_id, {}).get("retrieved_memories", []),
    )

    raw = ""
    last_error = ""
    parsed: dict[str, Any] | None = None
    started = time.time()
    allowed_winners = set(answers) | {"tie"}
    for attempt in range(args.max_retries + 1):
        try:
            raw = call_chat_completion(
                base_url=args.base_url,
                model=args.model,
                messages=messages,
                api_key=args.api_key,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                extra_body={"enable_thinking": True} if args.enable_thinking else None,
            )
            parsed = parse_json_object(raw)
            winner = str(parsed.get("winner") or "unknown").strip().strip("[]")
            parsed["winner"] = winner if winner in allowed_winners else "unknown"
            break
        except Exception as exc:
            last_error = repr(exc)
            if attempt >= args.max_retries:
                break
            messages = messages + [
                {"role": "assistant", "content": raw or f"ERROR: {last_error}"},
                {
                    "role": "user",
                    "content": (
                        "The previous response was not valid JSON. Return only one valid JSON object. "
                        f"The winner must be exactly {args.baseline_name}, {args.candidate_name}, or tie. "
                        "Do not use markdown."
                    ),
                },
            ]

    status = "ok" if parsed is not None else "error"
    if parsed is None:
        parsed = {"winner": "error", "error": last_error}
    return {
        "id": sample_id,
        "character_id": sample["character_id"],
        "question": sample["question"],
        "judge_raw": raw,
        "judge": parsed,
        "latency_s": time.time() - started,
        "status": status,
        "error": last_error if status != "ok" else "",
    }


def build_metrics(samples: list[dict[str, Any]], records: dict[str, dict[str, Any]], args: argparse.Namespace, result_path: Path) -> dict[str, Any]:
    rows = ordered_records_for_samples(samples, records, ok_only=False)
    winner_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    score_sums: dict[str, dict[str, float]] = {}
    score_counts: dict[str, int] = {}

    for row in rows:
        status = row.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        judge = row.get("judge", {})
        winner = judge.get("winner", "unknown")
        winner_counts[winner] = winner_counts.get(winner, 0) + 1
        for raw_name, scores in judge.get("scores", {}).items():
            if not isinstance(scores, dict):
                continue
            name = str(raw_name).strip().strip("[]")
            if name not in {args.baseline_name, args.candidate_name}:
                continue
            score_sums.setdefault(name, {})
            score_counts[name] = score_counts.get(name, 0) + 1
            for key, value in scores.items():
                if isinstance(value, int | float):
                    score_sums[name][key] = score_sums[name].get(key, 0.0) + float(value)

    avg_scores = {
        name: {key: value / score_counts[name] for key, value in scores.items()}
        for name, scores in score_sums.items()
        if score_counts.get(name)
    }
    ok_count = status_counts.get("ok", 0)
    return {
        "num_samples": len(samples),
        "num_records": len(rows),
        "ok_count": ok_count,
        "pending_count": len(samples) - ok_count,
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "winner_counts": winner_counts,
        "candidate_win_count": winner_counts.get(args.candidate_name, 0),
        "baseline_win_count": winner_counts.get(args.baseline_name, 0),
        "tie_count": winner_counts.get("tie", 0),
        "status_counts": status_counts,
        "avg_scores": avg_scores,
        "avg_latency_s": mean([row["latency_s"] for row in rows if "latency_s" in row]) if rows else 0.0,
        "result_file": str(result_path),
        "model": args.model,
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
        "memory_details": str(args.memory_details) if args.memory_details else "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge Lab5 Task2 trained model against a Task1 baseline.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--data-root", type=Path, default=Path("lab5/public_pack/data"))
    parser.add_argument("--task1-root", type=Path, default=Path("lab5/outputs/task1"))
    parser.add_argument("--baseline-name", default="persona")
    parser.add_argument("--candidate-name", default="sft")
    parser.add_argument("--candidate-predictions", type=Path, default=Path("lab5/outputs/task2/sft_all/predictions.jsonl"))
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--memory-details", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("lab5/outputs/task2_judge/sft_vs_persona"))
    parser.add_argument("--model", default="Qwen3.5-35B-A3B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8318/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = read_jsonl(args.data_root / "role_interview_eval.jsonl")
    if args.ids_file:
        ids = {
            str(json.loads(line).get("id") if line.strip().startswith("{") else line.strip())
            for line in args.ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        samples = [sample for sample in samples if sample["id"] in ids]
    if args.limit > 0:
        samples = samples[: args.limit]

    baseline_path = args.task1_root / args.baseline_name / "predictions.jsonl"
    baseline_predictions = records_by_id(baseline_path)
    candidate_predictions = records_by_id(args.candidate_predictions)
    memory_details = records_by_id(args.memory_details) if args.memory_details else {}
    samples = [sample for sample in samples if sample["id"] in baseline_predictions and sample["id"] in candidate_predictions]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_count = len(samples)
    result_path = args.output_dir / f"judge_results_{args.candidate_name}_vs_{args.baseline_name}_{sample_count}.jsonl"
    metrics_path = args.output_dir / f"judge_metrics_{args.candidate_name}_vs_{args.baseline_name}_{sample_count}.json"

    existing_records = records_by_id(result_path)
    ok_records = ordered_records_for_samples(samples, existing_records, ok_only=True)
    pending_samples = [sample for sample in samples if existing_records.get(sample["id"], {}).get("status") != "ok"]
    write_jsonl(result_path, ok_records)
    print(json.dumps({"resume_from": str(result_path), "already_ok": len(ok_records), "pending": len(pending_samples)}, ensure_ascii=False), flush=True)

    stop_reason = ""
    with result_path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        for start in range(0, len(pending_samples), args.concurrency):
            batch = pending_samples[start : start + args.concurrency]
            futures = [
                executor.submit(evaluate_one, sample, args, args.data_root, baseline_predictions, candidate_predictions, memory_details)
                for sample in batch
            ]
            desc = f"task2 judge {start + 1}-{start + len(batch)}/{len(pending_samples)}"
            for future in tqdm(futures, desc=desc, unit="sample"):
                record = future.result()
                write_jsonl_record(f, record)
                if record["status"] != "ok":
                    error_text = record.get("error", "")
                    if any(token in error_text for token in ("model_cooldown", "RateLimitError", "usage_limit_reached")):
                        stop_reason = error_text
                print(
                    json.dumps(
                        {
                            "id": record["id"],
                            "status": record["status"],
                            "winner": record["judge"].get("winner", "unknown"),
                            "latency_s": record["latency_s"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if stop_reason:
                print(json.dumps({"stopped": "rate_limit", "reason": stop_reason}, ensure_ascii=False), flush=True)
                break

    final_records = records_by_id(result_path)
    write_jsonl(result_path, ordered_records_for_samples(samples, final_records, ok_only=False))
    metrics = build_metrics(samples, records_by_id(result_path), args, result_path)
    write_json(metrics_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
