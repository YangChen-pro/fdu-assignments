import argparse
import json
import random
from collections import Counter
from pathlib import Path


PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. You can use\n"
    "basic arithmetic operations (+, -, *, /) and each number can only be used\n"
    "once. Show your work in <think> </think> tags. Return the final answer in\n"
    "<answer> </answer> tags."
)


def parse_args():
    parser = argparse.ArgumentParser(description="从 pass@k 候选结果构造 DPO 数据")
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--valid-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--max-pairs-per-sample", type=int, default=4)
    parser.add_argument(
        "--strategy",
        choices=["balanced", "one-negative-per-positive"],
        default="balanced",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def answer_position(text):
    position = (text or "").lower().find("<answer>")
    return position if position >= 0 else 10**9


def candidate_length(item):
    return len(item.get("prediction") or "")


def positive_key(item):
    return answer_position(item.get("prediction") or ""), candidate_length(item)


def negative_key(item):
    reason_priority = {
        "missing_answer": 0,
        "illegal_character": 1,
        "number_usage_error": 2,
        "ValueError": 3,
        "wrong_result": 4,
    }
    reason = item.get("eval", {}).get("reason", "")
    return reason_priority.get(reason, 9), -candidate_length(item), item.get("index", 0)


def make_pair(sample, chosen, rejected):
    return {
        "sample_id": sample["id"],
        "prompt": sample.get("prompt") or PROMPT_TEMPLATE.format(numbers=sample["numbers"], target=sample["target"]),
        "chosen": chosen["prediction"],
        "rejected": rejected["prediction"],
        "numbers": sample["numbers"],
        "target": sample["target"],
        "source_plan": "r8_ckpt382_pass32_dpo",
        "chosen_index": chosen["index"],
        "rejected_index": rejected["index"],
        "rejected_reason": rejected.get("eval", {}).get("reason", "unknown"),
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    rows = []
    skipped = []
    reason_counts = Counter()
    per_sample_pair_counts = Counter()

    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            candidates = item.get("candidates", [])
            positives = [candidate for candidate in candidates if candidate.get("eval", {}).get("correct")]
            negatives = [candidate for candidate in candidates if not candidate.get("eval", {}).get("correct")]
            if not positives or not negatives:
                skipped.append(
                    {
                        "id": item.get("id"),
                        "has_positive": bool(positives),
                        "has_negative": bool(negatives),
                    }
                )
                continue
            positives.sort(key=positive_key)
            negatives.sort(key=negative_key)
            if args.strategy == "one-negative-per-positive":
                pair_count = min(args.max_pairs_per_sample, len(positives))
            else:
                pair_count = min(args.max_pairs_per_sample, len(positives), len(negatives))
            for offset in range(pair_count):
                chosen = positives[offset % len(positives)]
                rejected = negatives[offset % len(negatives)]
                row = make_pair(item, chosen, rejected)
                rows.append(row)
                reason_counts[row["rejected_reason"]] += 1
                per_sample_pair_counts[item["id"]] += 1

    rng.shuffle(rows)
    valid_size = int(len(rows) * args.valid_ratio)
    valid_rows = rows[:valid_size]
    train_rows = rows[valid_size:]
    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.valid_output), valid_rows)
    summary = {
        "input": args.input,
        "train_output": args.train_output,
        "valid_output": args.valid_output,
        "num_pairs": len(rows),
        "num_train": len(train_rows),
        "num_valid": len(valid_rows),
        "num_skipped_samples": len(skipped),
        "skipped_samples": skipped[:50],
        "rejected_reasons": dict(reason_counts),
        "max_pairs_per_sample": args.max_pairs_per_sample,
        "strategy": args.strategy,
        "pair_count_distribution": dict(Counter(per_sample_pair_counts.values())),
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
