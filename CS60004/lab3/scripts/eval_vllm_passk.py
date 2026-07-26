import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from preprocess_utils import build_prompt, evaluate_output, load_countdown_data, summarize_results, write_json
from train_utils import vllm_chat_completions

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_: dict):
        return iterable


def parse_args():
    parser = argparse.ArgumentParser(description="评测 vLLM checkpoint 的 single 和 pass@k")
    parser.add_argument("--test-file", default="lab3/datasets/rl_train_test_datasets/raw_test.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--request-concurrency", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    return parser.parse_args()


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_sft_row(prompt, response):
    return {
        "instruction": "Solve the arithmetic puzzle. Return the final expression in <answer> </answer> tags.",
        "input": prompt,
        "output": response,
    }


def main():
    args = parse_args()
    samples = load_countdown_data(args.test_file)
    output_prefix = Path(args.output_prefix)
    generation = {
        "api_base": args.api_base,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "enable_thinking": True,
        "request_concurrency": args.request_concurrency,
        "timeout": 600,
    }
    start = time.time()
    single_rows: list[dict] = []
    passk_rows: list[dict] = []
    correct_message_rows: list[dict] = []
    single_results = []
    passk_results = []

    groups = [
        samples[index : index + args.batch_size]
        for index in range(0, len(samples), args.batch_size)
    ]
    progress = tqdm(groups, total=len(groups), desc="eval", unit="batch")
    for batch in progress:
        prompts = [build_prompt(sample) for sample in batch]
        expanded_prompts = [
            prompt
            for prompt in prompts
            for _ in range(args.candidate_count)
        ]
        outputs = vllm_chat_completions(expanded_prompts, generation, str(Path(args.checkpoint).resolve()))
        cursor = 0
        for sample, prompt in zip(batch, prompts):
            candidates = outputs[cursor : cursor + args.candidate_count]
            cursor += args.candidate_count
            candidate_evals = [
                evaluate_output(candidate, sample["numbers"], sample["target"])
                for candidate in candidates
            ]
            first_eval = candidate_evals[0]
            single_results.append(first_eval)
            single_rows.append(
                {
                    "id": sample["id"],
                    "prediction": candidates[0],
                    "eval": asdict(first_eval),
                }
            )

            correct_indices = [index for index, result in enumerate(candidate_evals) if result.correct]
            pass_eval = candidate_evals[correct_indices[0]] if correct_indices else first_eval
            pass_eval.correct = bool(correct_indices)
            passk_results.append(pass_eval)
            passk_rows.append(
                {
                    "id": sample["id"],
                    "numbers": sample["numbers"],
                    "target": sample["target"],
                    "pass": bool(correct_indices),
                    "correct_indices": correct_indices,
                    "candidates": [
                        {
                            "index": index,
                            "prediction": candidate,
                            "eval": asdict(result),
                        }
                        for index, (candidate, result) in enumerate(zip(candidates, candidate_evals))
                    ],
                }
            )
            for index in correct_indices:
                correct_message_rows.append(make_sft_row(prompt, candidates[index]))
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(done=len(single_rows), correct_messages=len(correct_message_rows))

    single_summary = summarize_results(single_results)
    passk_summary = summarize_results(passk_results)
    meta = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "test_file": args.test_file,
        "num_samples": len(samples),
        "candidate_count": args.candidate_count,
        "runtime_sec": time.time() - start,
        "generation": generation,
        "single_shot": single_summary,
        f"pass_at_{args.candidate_count}": passk_summary,
        "files": {
            "single_predictions": str(output_prefix.with_suffix(".single.jsonl")),
            "passk_candidates": str(output_prefix.with_suffix(f".pass{args.candidate_count}.jsonl")),
            "correct_messages": str(output_prefix.with_suffix(f".pass{args.candidate_count}.correct_messages.jsonl")),
            "summary": str(output_prefix.with_suffix(".summary.json")),
        },
    }

    write_jsonl(Path(meta["files"]["single_predictions"]), single_rows)
    write_jsonl(Path(meta["files"]["passk_candidates"]), passk_rows)
    write_jsonl(Path(meta["files"]["correct_messages"]), correct_message_rows)
    write_json(meta["files"]["summary"], meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
