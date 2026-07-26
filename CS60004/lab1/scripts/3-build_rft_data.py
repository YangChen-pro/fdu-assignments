import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from preprocess_utils import (
    TeacherEmptyContentError,
    TeacherModel,
    build_input,
    build_instruction,
    build_plan_b_teacher_prompt,
    check_plan_b_clean_hard_rules,
    parse_plan_b_output,
    read_jsonl,
    validate_output,
    write_jsonl,
)


def normalize_output(raw_text):
    output_obj = parse_plan_b_output(raw_text.strip())
    return json.dumps(output_obj, ensure_ascii=False)


def choose_best_candidate(candidates):
    return min(candidates, key=len)


def build_rft_sample(sample, output_str):
    return {
        "instruction": build_instruction(plan="B"),
        "input": build_input(sample, plan="B"),
        "output": output_str,
    }


def process_rft_sample(i, sample, teacher, num_candidates=4, max_tokens=4096):
    accepted_candidates = []
    reject_reasons = Counter()

    prompt = build_plan_b_teacher_prompt(sample)
    messages = [
        {"role": "system", "content": "你是一个严谨的医学考试题解析助手。"},
        {"role": "user", "content": prompt},
    ]

    for _ in range(num_candidates):
        try:
            raw_text = teacher.chat(messages, max_tokens=max_tokens)
            output_str = normalize_output(raw_text)

            ok, msg = validate_output(output_str, sample["answer"], plan="B")
            if not ok:
                reject_reasons[f"validate:{msg}"] += 1
                continue

            hard_pass, hard_reasons = check_plan_b_clean_hard_rules(output_str)
            if not hard_pass:
                for reason in hard_reasons:
                    reject_reasons[f"hard:{reason}"] += 1
                continue

            accepted_candidates.append(output_str)
        except TeacherEmptyContentError as e:
            reject_reasons[f"empty:{e.finish_reason}"] += 1
        except Exception as e:
            reject_reasons[f"error:{type(e).__name__}"] += 1

    audit = {
        "index": i,
        "accepted": len(accepted_candidates),
        "rejected": dict(reject_reasons),
    }

    if not accepted_candidates:
        return i, None, audit, ValueError("no accepted candidate")

    best_candidate = choose_best_candidate(accepted_candidates)
    return i, build_rft_sample(sample, best_candidate), audit, None


def collect_rft_samples(
    samples,
    teacher,
    num_candidates=4,
    max_tokens=4096,
    num_workers=1,
):
    kept_samples = []
    audits = []

    if num_workers <= 1:
        for i, sample in enumerate(tqdm(samples, desc="Build RFT", ncols=80)):
            _, new_sample, audit, err = process_rft_sample(
                i=i,
                sample=sample,
                teacher=teacher,
                num_candidates=num_candidates,
                max_tokens=max_tokens,
            )
            audits.append(audit)
            if err is not None:
                print(f"[Drop] sample {i}: {err}")
                continue
            kept_samples.append((i, new_sample))
    else:
        indexed_results = []
        pbar = tqdm(total=len(samples), desc="Build RFT", ncols=80)

        for start in range(0, len(samples), num_workers):
            end = min(start + num_workers, len(samples))
            batch_samples = samples[start:end]

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for offset, sample in enumerate(batch_samples):
                    future = executor.submit(
                        process_rft_sample,
                        start + offset,
                        sample,
                        teacher,
                        num_candidates,
                        max_tokens,
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    indexed_results.append(future.result())
                    pbar.update(1)

        pbar.close()
        indexed_results.sort(key=lambda x: x[0])

        for i, new_sample, audit, err in indexed_results:
            audits.append(audit)
            if err is not None:
                print(f"[Drop] sample {i}: {err}")
                continue
            kept_samples.append((i, new_sample))

    kept_samples.sort(key=lambda x: x[0])
    return [sample for _, sample in kept_samples], audits


def build_report(samples, kept_samples, audits, num_candidates):
    reason_counter = Counter()
    accepted_total = 0

    for audit in audits:
        accepted_total += audit["accepted"]
        reason_counter.update(audit["rejected"])

    lines = [
        f"Input samples    : {len(samples)}",
        f"Output samples   : {len(kept_samples)}",
        f"Dropped samples  : {len(samples) - len(kept_samples)}",
        f"Candidates/sample: {num_candidates}",
        f"Accepted total   : {accepted_total}",
    ]

    if audits:
        avg_accepted = accepted_total / len(audits)
        lines.append(f"Avg accepted     : {avg_accepted:.2f}")

    if reason_counter:
        lines.append("Top reject reasons:")
        for key, value in reason_counter.most_common(10):
            lines.append(f"- {key}: {value}")

    return "\n".join(lines) + "\n"


def save_report(report_path, report_text):
    save_dir = os.path.dirname(report_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(report_text)


def parse_args():
    parser = argparse.ArgumentParser(description="Task 3 RFT 拒绝采样数据构造")
    parser.add_argument(
        "--input_path",
        type=str,
        default="datasets/provide_to_students/train.jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="datasets/task3/rft_train.jsonl",
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default="datasets/task3/rft_report.txt",
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="qwen-3.5-9b",
    )
    parser.add_argument(
        "--teacher_base_url",
        type=str,
        default="http://localhost:12345/v1",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    teacher = TeacherModel(
        model_name=args.teacher_model,
        base_url=args.teacher_base_url,
    )

    samples = read_jsonl(args.input_path)
    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    kept_samples, audits = collect_rft_samples(
        samples=samples,
        teacher=teacher,
        num_candidates=args.num_candidates,
        max_tokens=args.max_tokens,
        num_workers=args.num_workers,
    )
    write_jsonl(args.output_path, kept_samples)

    report_text = build_report(
        samples=samples,
        kept_samples=kept_samples,
        audits=audits,
        num_candidates=args.num_candidates,
    )
    save_report(args.report_path, report_text)

    print(report_text.strip())
    print(f"Output path      : {args.output_path}")
    print(f"Report path      : {args.report_path}")


if __name__ == "__main__":
    main()


# python3 scripts/3-build_rft_data.py --num_workers 8 --num_candidates 4
