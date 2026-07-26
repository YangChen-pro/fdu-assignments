from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any

from tqdm import tqdm

sys.path.insert(0, "lab5/task_scripts")
from utils import (  # noqa: E402
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    DASHSCOPE_JUDGE_MODEL,
    build_role_judge_messages,
    call_chat_completion,
    load_persona,
    parse_json_object,
    read_jsonl,
    write_json,
    write_jsonl,
)


def load_existing_records(path: Path) -> dict[str, dict[str, Any]]:
    """Load the latest record for each sample id from a JSONL result file."""
    if not path.exists():
        return {}

    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = record.get("id")
            if sample_id:
                records[sample_id] = record
    return records


def ordered_records(
    samples: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    include_non_ok: bool,
) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        record = records.get(sample["id"])
        if not record:
            continue
        if include_non_ok or record.get("status") == "ok":
            rows.append(record)
    return rows


def evaluate_one(
    sample: dict[str, Any],
    args: argparse.Namespace,
    data_root: Path,
    predictions: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    sample_id = sample["id"]
    answers = {
        "question": predictions["question"][sample_id]["prediction"],
        "persona": predictions["persona"][sample_id]["prediction"],
        "memory": predictions["memory"][sample_id]["prediction"],
    }
    messages = build_role_judge_messages(
        character_id=sample["character_id"],
        question=sample["question"],
        persona=load_persona(data_root, sample["character_id"]),
        answers=answers,
        retrieved_memories=predictions["memory_details"][sample_id].get("retrieved_memories", []),
    )

    started = time.time()
    raw = ""
    last_error = ""
    parsed: dict[str, Any] | None = None
    for attempt in range(args.max_retries + 1):
        try:
            raw = call_chat_completion(
                base_url=args.base_url,
                model=args.model,
                messages=messages,
                api_key=DASHSCOPE_API_KEY,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                extra_body={"enable_thinking": args.enable_thinking},
            )
            parsed = parse_json_object(raw)
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
                        "Escape all double quotes inside string values. The winner must be exactly "
                        "question, persona, memory, or tie. Do not use markdown."
                    ),
                },
            ]

    latency = time.time() - started
    if parsed is None:
        parsed = {"winner": "error", "memory_helped": False, "error": last_error}
        status = "error"
    else:
        winner = str(parsed.get("winner") or "unknown").strip().strip("[]")
        parsed["winner"] = winner if winner in set(answers) or winner == "tie" else "unknown"
        status = "ok"

    return {
        "id": sample_id,
        "character_id": sample["character_id"],
        "question": sample["question"],
        "judge_raw": raw,
        "judge": parsed,
        "latency_s": latency,
        "status": status,
        "error": last_error if status != "ok" else "",
    }


def build_metrics(
    samples: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    result_path: Path,
) -> dict[str, Any]:
    rows = ordered_records(samples, records, include_non_ok=True)
    latencies = [row["latency_s"] for row in rows if "latency_s" in row]
    winners: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    memory_helped = 0

    for row in rows:
        status = row.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        winner = row.get("judge", {}).get("winner", "unknown")
        winners[winner] = winners.get(winner, 0) + 1
        if row.get("judge", {}).get("memory_helped") is True:
            memory_helped += 1

    ok_count = status_counts.get("ok", 0)
    return {
        "num_samples": len(samples),
        "num_records": len(rows),
        "ok_count": ok_count,
        "pending_count": len(samples) - ok_count,
        "concurrency": args.concurrency,
        "max_retries": args.max_retries,
        "avg_latency_s": mean(latencies) if latencies else 0.0,
        "winner_counts": winners,
        "memory_helped_count": memory_helped,
        "status_counts": status_counts,
        "result_file": str(result_path),
        "model": args.model,
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("lab5/outputs/task1_judge"))
    parser.add_argument("--model", default=DASHSCOPE_JUDGE_MODEL)
    parser.add_argument("--base-url", default=DASHSCOPE_BASE_URL)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--enable-thinking", action="store_true", default=True)
    args = parser.parse_args()

    data_root = Path("lab5/public_pack/data")
    task1_root = Path("lab5/outputs/task1")
    samples = read_jsonl(data_root / "role_interview_eval.jsonl")
    if args.limit > 0:
        samples = samples[: args.limit]
    predictions = {
        "question": {row["id"]: row for row in read_jsonl(task1_root / "question" / "predictions.jsonl")},
        "persona": {row["id"]: row for row in read_jsonl(task1_root / "persona" / "predictions.jsonl")},
        "memory": {row["id"]: row for row in read_jsonl(task1_root / "memory" / "predictions.jsonl")},
        "memory_details": {row["id"]: row for row in read_jsonl(task1_root / "memory" / "details.jsonl")},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_count = len(samples)
    result_path = args.output_dir / f"judge_results_{sample_count}.jsonl"
    metrics_path = args.output_dir / f"judge_metrics_{sample_count}.json"

    existing_records = load_existing_records(result_path)
    ok_records = ordered_records(samples, existing_records, include_non_ok=False)
    pending_samples = [
        sample
        for sample in samples
        if existing_records.get(sample["id"], {}).get("status") != "ok"
    ]
    write_jsonl(result_path, ok_records)
    print(
        json.dumps(
            {
                "resume_from": str(result_path),
                "num_samples": sample_count,
                "already_ok": len(ok_records),
                "pending": len(pending_samples),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    with result_path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        for start in range(0, len(pending_samples), args.concurrency):
            batch = pending_samples[start : start + args.concurrency]
            futures = [executor.submit(evaluate_one, sample, args, data_root, predictions) for sample in batch]
            desc = f"judge pending {start + 1}-{start + len(batch)}/{len(pending_samples)}"
            for future in tqdm(futures, desc=desc, unit="sample"):
                record = future.result()
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
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

    final_records = load_existing_records(result_path)
    write_jsonl(result_path, ordered_records(samples, final_records, include_non_ok=True))
    final_records = load_existing_records(result_path)
    metrics = build_metrics(samples, final_records, args, result_path)
    write_json(metrics_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
