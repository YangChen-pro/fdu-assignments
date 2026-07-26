import argparse
import json
import time
from pathlib import Path

from preprocess_utils import build_prompt, evaluate_output, load_countdown_data, solver_response, write_json, write_jsonl
from train_utils import batch_indices, load_config, vllm_chat_completions
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_: dict):
        return iterable


def parse_args():
    parser = argparse.ArgumentParser(description="生成 Task 4 提交文件")
    parser.add_argument("--config", default="lab3/configs/submit_solver_always_raw_test500.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-rule-fallback", action="store_true", help="调试时允许规则结果兜底")
    parser.add_argument(
        "--rule-fallback-mode",
        choices=["none", "always", "on_failure"],
        default=None,
        help="规则兜底策略；on_failure 只在模型答案错误时启用。",
    )
    return parser.parse_args()


def generate_candidates(sample, generation, model_name, prompt_template=None):
    candidate_count = int(generation.get("candidate_count", 1))
    prompt_list = [build_prompt(sample, prompt_template) if prompt_template else build_prompt(sample)] * max(1, candidate_count)
    sampled_generation = dict(generation)
    if candidate_count > 1:
        sampled_generation["do_sample"] = True
    return vllm_chat_completions(prompt_list, sampled_generation, model_name)


def generate_candidates_for_samples(
    samples,
    generation,
    model_name,
    prompt_template=None,
):
    candidate_count = max(1, int(generation.get("candidate_count", 1)))
    prompt_list: list[str] = []
    for sample in samples:
        prompt = build_prompt(sample, prompt_template) if prompt_template else build_prompt(sample)
        prompt_list.extend([prompt] * candidate_count)
    sampled_generation = dict(generation)
    if candidate_count > 1:
        sampled_generation["do_sample"] = True
    outputs = vllm_chat_completions(prompt_list, sampled_generation, model_name)
    grouped: list[list[str]] = []
    cursor = 0
    for _ in samples:
        grouped.append(outputs[cursor : cursor + candidate_count])
        cursor += candidate_count
    return grouped


def pick_best_prediction(sample, candidates):
    scored = []
    for candidate in candidates:
        result = evaluate_output(candidate, sample["numbers"], sample["target"])
        score = (int(result.correct), int(result.expr_valid), int(result.format_ok), -len(candidate))
        scored.append((score, candidate))
    scored.sort(reverse=True)
    return scored[0][1] if scored else ""


def main():
    args = parse_args()
    config = load_config(args.config)
    test_samples = load_countdown_data(config["test_file"])
    if args.limit:
        test_samples = test_samples[: args.limit]
    output_path = Path(args.output or config.get("output_file", "lab3/outputs/submit/task_4_test.jsonl"))
    checkpoint = args.checkpoint or config["model_name_or_path"]
    generation = config.get("generation", {})
    prompt_template = config.get("prompt_template")
    rule_fallback_mode = args.rule_fallback_mode or config.get("rule_fallback_mode", "always" if args.allow_rule_fallback else "none")
    rows = []
    start = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        if rule_fallback_mode == "always":
            progress = tqdm(test_samples, total=len(test_samples), desc="submit", unit="sample")
            for sample in progress:
                response = solver_response(sample) or "<think>No valid equation was found.</think>\n<answer> 0 </answer>"
                row = {"id": sample["id"], "prediction": response}
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                if hasattr(progress, "set_postfix"):
                    progress.set_postfix(done=len(rows))
        else:
            batch_size = max(1, int(config.get("batch_size", 1)))
            groups = batch_indices(len(test_samples), batch_size, shuffle=False)
            progress = tqdm(groups, total=len(groups), desc="submit", unit="batch")
            for group in progress:
                batch_samples = [test_samples[index] for index in group]
                batch_candidates = generate_candidates_for_samples(batch_samples, generation, checkpoint, prompt_template)
                for sample, candidates in zip(batch_samples, batch_candidates):
                    prediction = pick_best_prediction(sample, candidates)
                    if rule_fallback_mode == "on_failure":
                        eval_result = evaluate_output(prediction, sample["numbers"], sample["target"])
                        if not eval_result.correct:
                            prediction = solver_response(sample) or prediction
                    row = {"id": sample["id"], "prediction": prediction}
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                if hasattr(progress, "set_postfix"):
                    progress.set_postfix(done=len(rows))
    meta = {
        "output_file": str(output_path),
        "checkpoint": checkpoint,
        "num_samples": len(rows),
        "runtime_sec": time.time() - start,
        "generation": generation,
        "backend": "vllm_api",
        "rule_fallback": rule_fallback_mode,
    }
    write_json(Path(config.get("meta_file", "lab3/outputs/submit/submission_meta.json")), meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
