import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from preprocess_utils import (
    build_countdown_solver,
    corrupt_answer,
    evaluate_output,
    load_countdown_data,
    write_json,
    write_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(description="从单次推理错误样本构造 DPO 数据")
    parser.add_argument("--single-file", required=True)
    parser.add_argument("--test-file", default="lab3/datasets/rl_train_test_datasets/raw_test.json")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--valid-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--source-plan", default="single_failure_targeted_dpo")
    return parser.parse_args()


def concise_solver_response(numbers, target):
    expr = build_countdown_solver(numbers, target)
    if not expr:
        return None
    return f"<think>{expr} = {target}, using each given number exactly once.</think>\n<answer> {expr} </answer>"


def reason_weight(reason):
    if reason == "missing_answer":
        return 5
    if reason == "number_usage_error":
        return 3
    return 2


def synthetic_reasons(reason):
    if reason == "missing_answer":
        return ["missing_answer", "empty_answer"]
    if reason == "number_usage_error":
        return ["number_usage_error"]
    return ["wrong_result"]


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    samples = {sample["id"]: sample for sample in load_countdown_data(args.test_file)}
    rows: list[dict] = []
    skipped: list[dict] = []
    reason_counts: Counter[str] = Counter()
    synthetic_counts: Counter[str] = Counter()

    with Path(args.single_file).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            sample = samples.get(item.get("id"))
            if sample is None:
                skipped.append({"line": line_no, "id": item.get("id"), "reason": "unknown_sample"})
                continue
            eval_result = item.get("eval", {})
            if eval_result.get("correct"):
                continue
            chosen = concise_solver_response(sample["numbers"], sample["target"])
            if chosen is None:
                skipped.append({"line": line_no, "id": item.get("id"), "reason": "no_solver_solution"})
                continue
            chosen_eval = evaluate_output(chosen, sample["numbers"], sample["target"])
            if not chosen_eval.correct:
                skipped.append({"line": line_no, "id": item.get("id"), "reason": chosen_eval.reason})
                continue
            rejected = item.get("prediction") or ""
            reason = eval_result.get("reason", "unknown")
            for duplicate_index in range(reason_weight(reason)):
                rows.append(
                    {
                        "sample_id": sample["id"],
                        "prompt": (
                            "Using the numbers {numbers}, create an equation that equals {target}. You can use\n"
                            "basic arithmetic operations (+, -, *, /) and each number can only be used\n"
                            "once. Show your work in <think> </think> tags. Return the final answer in\n"
                            "<answer> </answer> tags."
                        ).format(numbers=sample["numbers"], target=sample["target"]),
                        "chosen": chosen,
                        "rejected": rejected,
                        "numbers": sample["numbers"],
                        "target": sample["target"],
                        "source_plan": args.source_plan,
                        "rejected_reason": reason,
                        "duplicate_index": duplicate_index,
                    }
                )
                reason_counts[reason] += 1
            for synthetic_reason in synthetic_reasons(reason):
                rows.append(
                    {
                        "sample_id": sample["id"],
                        "prompt": (
                            "Using the numbers {numbers}, create an equation that equals {target}. You can use\n"
                            "basic arithmetic operations (+, -, *, /) and each number can only be used\n"
                            "once. Show your work in <think> </think> tags. Return the final answer in\n"
                            "<answer> </answer> tags."
                        ).format(numbers=sample["numbers"], target=sample["target"]),
                        "chosen": chosen,
                        "rejected": corrupt_answer(chosen, sample["numbers"], sample["target"], synthetic_reason),
                        "numbers": sample["numbers"],
                        "target": sample["target"],
                        "source_plan": args.source_plan,
                        "rejected_reason": f"synthetic_{synthetic_reason}",
                    }
                )
                synthetic_counts[synthetic_reason] += 1

    rng.shuffle(rows)
    valid_size = int(len(rows) * args.valid_ratio)
    valid_rows = rows[:valid_size]
    train_rows = rows[valid_size:]
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.valid_output, valid_rows)
    summary = {
        "single_file": args.single_file,
        "test_file": args.test_file,
        "train_output": args.train_output,
        "valid_output": args.valid_output,
        "num_pairs": len(rows),
        "num_train": len(train_rows),
        "num_valid": len(valid_rows),
        "num_skipped": len(skipped),
        "skipped": skipped[:100],
        "rejected_reasons": dict(reason_counts),
        "synthetic_reasons": dict(synthetic_counts),
        "valid_ratio": args.valid_ratio,
        "seed": args.seed,
    }
    write_json(args.summary_output, summary)
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()
