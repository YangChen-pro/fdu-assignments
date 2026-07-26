import argparse
import json
import random
import time
from pathlib import Path

import torch
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_: dict):
        return iterable

from preprocess_utils import (
    build_prompt,
    corrupt_answer,
    evaluate_output,
    load_countdown_data,
    load_jsonl,
    make_preference_row,
    repair_output_with_solver,
    solver_answer_only_response,
    solver_response,
    split_samples,
    summarize_outputs,
    write_json,
    write_jsonl,
)
from train_utils import (
    assert_finite,
    batch_indices,
    collate_prompt_response,
    count_trainable_parameters,
    dpo_loss,
    init_swanlab,
    load_causal_lm,
    load_config,
    load_tokenizer,
    log_metrics,
    prepare_run,
    sequence_log_probs,
    set_seed,
    vllm_chat_completions,
)


def parse_args():
    parser = argparse.ArgumentParser(description="DPO 数据构造、训练和评测")
    parser.add_argument("--config", default="lab3/configs/plan_d_formal_swan.yaml")
    parser.add_argument("--mode", choices=["build-data", "train", "eval", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="快速跑时限制样本数")
    parser.add_argument("--dry-run-solver", action="store_true", help="本地检查流程时使用规则求解器")
    return parser.parse_args()


def teacher_chosen_prompt(sample):
    return (
        build_prompt(sample)
        + "\n\nReturn exactly this structure:\n"
        + "<think> concise step-by-step arithmetic reasoning </think>\n"
        + "<answer> one valid expression using every given number exactly once </answer>\n"
        + "Example format: <think>8-2=6 and 6*(7-3)=24.</think>\n"
        + "<answer> (8-2)*(7-3) </answer>\n"
        + "Before finalizing, verify the expression equals the target.\n"
        + "Do not include markdown fences, extra tags, or any final answer outside <answer>."
    )


def teacher_rejected_prompt(sample):
    return (
        build_prompt(sample)
        + "\n\nGenerate a plausible but clearly wrong response. It must still look like this:\n"
        + "<think> concise but flawed reasoning </think>\n"
        + "<answer> an invalid or incorrect expression </answer>\n"
        + "The answer should be wrong because of arithmetic, format, or number-use error."
    )

def build_preference_data(config, args):
    seed = int(config.get("seed", 42))
    samples = load_countdown_data(config["raw_train_file"])
    data_build = config.get("data_build", {})
    max_samples = args.limit if args.limit is not None else data_build.get("max_samples")
    train_samples, valid_samples = split_samples(
        samples,
        valid_ratio=float(config.get("valid_ratio", 0.05)),
        seed=seed,
        max_samples=max_samples,
    )
    output_dir = Path(config["output_dir"])
    source_plan = config.get("source_plan", "plan_b")
    rejected_reasons = data_build.get(
        "rejected_reasons",
        ["wrong_result", "number_usage_error", "missing_answer", "empty_answer"],
    )
    chosen_backend = data_build.get("chosen_backend", "solver" if args.dry_run_solver else "teacher")
    rejected_backend = data_build.get(
        "rejected_backend",
        "teacher" if source_plan == "plan_b" and chosen_backend == "teacher" else "corrupt",
    )
    teacher_backend = data_build.get("teacher_backend", "vllm_api")
    if chosen_backend not in {"teacher", "solver"}:
        raise ValueError(f"unsupported chosen_backend: {chosen_backend}")
    if rejected_backend not in {"corrupt", "teacher", "model"}:
        raise ValueError(f"unsupported rejected_backend: {rejected_backend}")
    if chosen_backend == "teacher" and teacher_backend != "vllm_api":
        raise ValueError("inference backend is refactored to vllm_api only")
    if rejected_backend in {"teacher", "model"} and teacher_backend != "vllm_api":
        raise ValueError("inference backend is refactored to vllm_api only")

    def build_rows(rows, split_name):
        built: list[dict] = []
        skipped: list[dict] = []
        repaired_count = 0
        batch_size = int(data_build.get("teacher_batch_size", 1))
        target_file = Path(config[f"{split_name}_file"])
        target_file.parent.mkdir(parents=True, exist_ok=True)
        groups = batch_indices(len(rows), batch_size, shuffle=False)
        progress = tqdm(groups, total=len(groups), desc=f"{source_plan}-{split_name}", unit="batch")
        with target_file.open("w", encoding="utf-8") as handle:
            for group in progress:
                batch = [rows[index] for index in group]
                if chosen_backend == "solver":
                    answer_only = bool(data_build.get("answer_only_chosen", False))
                    if answer_only:
                        chosen_outputs = [
                            solver_answer_only_response(sample) or "<answer> 0 </answer>" for sample in batch
                        ]
                    else:
                        chosen_outputs = [
                            solver_response(sample) or "<think>No solution found.</think><answer> 0 </answer>"
                            for sample in batch
                        ]
                else:
                    prompts = [teacher_chosen_prompt(sample) for sample in batch]
                    chosen_outputs = vllm_chat_completions(
                        prompts,
                        config.get("teacher_generation", {}),
                        config["teacher_model_name_or_path"],
                    )
                model_rejected_outputs = [None] * len(batch)
                if rejected_backend == "model":
                    rejected_model = data_build.get("rejected_model_name_or_path", config["model_name_or_path"])
                    rejected_generation = data_build.get("rejected_generation", config.get("generation", {}))
                    model_rejected_outputs = vllm_chat_completions(
                        [build_prompt(sample) for sample in batch],
                        rejected_generation,
                        rejected_model,
                    )
                for sample, chosen, model_rejected in zip(batch, chosen_outputs, model_rejected_outputs):
                    chosen_eval = evaluate_output(chosen, sample["numbers"], sample["target"])
                    if not chosen_eval.correct and chosen_backend == "teacher":
                        retries = int(data_build.get("teacher_retries", 2))
                        for retry in range(retries):
                            repair_prompt = (
                                teacher_chosen_prompt(sample)
                                + f"\n\nYour previous answer failed validation because `{chosen_eval.reason}`. "
                                + "Try again with a different expression and verify it exactly equals the target."
                            )
                            chosen = vllm_chat_completions(
                                [repair_prompt],
                                config.get("teacher_generation", {}),
                                config["teacher_model_name_or_path"],
                            )[0]
                            chosen_eval = evaluate_output(chosen, sample["numbers"], sample["target"])
                            if chosen_eval.correct:
                                break
                    if not chosen_eval.correct and data_build.get("solver_repair_on_fail", False):
                        repaired = repair_output_with_solver(chosen, sample)
                        if repaired is not None:
                            chosen = repaired
                            chosen_eval = evaluate_output(chosen, sample["numbers"], sample["target"])
                            if chosen_eval.correct:
                                repaired_count += 1
                    if not chosen_eval.correct:
                        skipped.append(
                            {
                                "sample_id": sample["id"],
                                "numbers": sample["numbers"],
                                "target": sample["target"],
                                "reason": chosen_eval.reason,
                                "chosen": chosen,
                            }
                        )
                        continue
                    reason = random.choice(rejected_reasons)
                    if rejected_backend == "teacher":
                        reject_prompt = teacher_rejected_prompt(sample)
                        rejected = vllm_chat_completions(
                            [reject_prompt],
                            config.get("teacher_generation", {}),
                            config["teacher_model_name_or_path"],
                        )[0]
                        rejected_eval = evaluate_output(rejected, sample["numbers"], sample["target"])
                        if rejected_eval.correct:
                            rejected = corrupt_answer(chosen, sample["numbers"], sample["target"], reason)
                    elif rejected_backend == "model":
                        rejected = model_rejected or ""
                        rejected_eval = evaluate_output(rejected, sample["numbers"], sample["target"])
                        if rejected_eval.correct:
                            rejected = corrupt_answer(chosen, sample["numbers"], sample["target"], reason)
                    else:
                        rejected = corrupt_answer(chosen, sample["numbers"], sample["target"], reason)
                    row = make_preference_row(sample, chosen, rejected, source_plan, reason)
                    built.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                if hasattr(progress, "set_postfix"):
                    progress.set_postfix(kept=len(built), skipped=len(skipped), repaired=repaired_count)
        summary = summarize_preference_rows(built)
        summary["skipped"] = skipped[:100]
        summary["num_skipped"] = len(skipped)
        summary["num_repaired"] = repaired_count
        summary["chosen_backend"] = chosen_backend
        summary["rejected_backend"] = rejected_backend
        write_json(output_dir / f"data_summary_{source_plan}_{split_name}.json", summary)
        return built

    train_rows = build_rows(train_samples, "train")
    valid_rows = build_rows(valid_samples, "valid")
    return {"train": len(train_rows), "valid": len(valid_rows)}


def summarize_preference_rows(rows: list[dict]) -> dict:
    chosen_rows = [
        {"prediction": row["chosen"], "numbers": row["numbers"], "target": row["target"], "sample_id": row["sample_id"]}
        for row in rows
    ]
    rejected_rows = [
        {"prediction": row["rejected"], "numbers": row["numbers"], "target": row["target"], "sample_id": row["sample_id"]}
        for row in rows
    ]
    return {
        "num_rows": len(rows),
        "chosen": summarize_outputs(chosen_rows),
        "rejected": summarize_outputs(rejected_rows),
    }


def evaluate_model(config, output_dir, checkpoint=None, limit=None):
    model_path = checkpoint or config.get("eval_model_name_or_path") or config["model_name_or_path"]
    rows = load_jsonl(config["valid_file"])
    if limit:
        rows = rows[:limit]
    predictions = []
    generation = config.get("eval_generation", config.get("generation", {}))
    for group in batch_indices(len(rows), int(config.get("eval_batch_size", 1)), shuffle=False):
        batch = [rows[index] for index in group]
        outputs = vllm_chat_completions([row["prompt"] for row in batch], generation, model_path)
        for row, output in zip(batch, outputs):
            predictions.append(
                {
                    "sample_id": row["sample_id"],
                    "numbers": row["numbers"],
                    "target": row["target"],
                    "prediction": output,
                }
            )
    summary = summarize_outputs(predictions)
    write_json(output_dir / "eval_metrics.json", summary)
    write_json(output_dir / "eval_samples.json", predictions[:100])
    return summary


def train_dpo(config_path: str, config: dict, args: argparse.Namespace) -> Path:
    set_seed(int(config.get("seed", 42)))
    output_dir = prepare_run(config_path, config)
    tracker = init_swanlab(config, Path(config_path).stem)
    tokenizer = load_tokenizer(config["model_name_or_path"])
    policy = load_causal_lm(config["model_name_or_path"], config, train=True)
    reference = load_causal_lm(config.get("ref_model_name_or_path", config["model_name_or_path"]), config, train=False)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    print(f"trainable parameters: {count_trainable_parameters(policy):,}")

    rows = load_jsonl(config["train_file"])
    if args.limit:
        rows = rows[: args.limit]
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(config.get("learning_rate", 5e-6)))
    max_steps = int(config.get("max_optimizer_steps", config.get("max_steps", 0)))
    batch_size = int(config.get("per_device_batch_size", 1))
    grad_accum = int(config.get("gradient_accumulation_steps", 1))
    dpo_chat_template_kwargs = config.get("dpo_chat_template_kwargs")
    dpo_system_prompt = config.get("dpo_system_prompt")
    global_step = 0
    micro_step = 0
    policy.train()
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(int(config.get("num_epochs", 1))):
        epoch_batches = batch_indices(len(rows), batch_size, shuffle=True)
        for batch_index, indices in enumerate(epoch_batches, 1):
            batch_rows = [rows[index] for index in indices]
            prompts = [row["prompt"] for row in batch_rows]
            chosen = [row["chosen"] for row in batch_rows]
            rejected = [row["rejected"] for row in batch_rows]
            chosen_batch = collate_prompt_response(
                tokenizer,
                prompts,
                chosen,
                int(config["max_prompt_length"]),
                int(config["max_response_length"]),
                chat_template_kwargs=dpo_chat_template_kwargs,
                system_prompt=dpo_system_prompt,
                append_response_eos=bool(config.get("append_response_eos", False)),
            )
            rejected_batch = collate_prompt_response(
                tokenizer,
                prompts,
                rejected,
                int(config["max_prompt_length"]),
                int(config["max_response_length"]),
                chat_template_kwargs=dpo_chat_template_kwargs,
                system_prompt=dpo_system_prompt,
                append_response_eos=bool(config.get("append_response_eos", False)),
            )
            policy_chosen = sequence_log_probs(policy, chosen_batch)
            policy_rejected = sequence_log_probs(policy, rejected_batch)
            with torch.no_grad():
                ref_chosen = sequence_log_probs(reference, chosen_batch)
                ref_rejected = sequence_log_probs(reference, rejected_batch)
            if config.get("length_normalize_logps", False):
                policy_chosen = policy_chosen / chosen_batch["response_lengths"].clamp_min(1)
                policy_rejected = policy_rejected / rejected_batch["response_lengths"].clamp_min(1)
                ref_chosen = ref_chosen / chosen_batch["response_lengths"].clamp_min(1)
                ref_rejected = ref_rejected / rejected_batch["response_lengths"].clamp_min(1)
            loss, stats = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, float(config.get("beta", 0.1)))
            loss = loss / grad_accum
            assert_finite("dpo_loss", loss)
            loss.backward()
            micro_step += 1
            should_step = (micro_step % grad_accum == 0) or (batch_index == len(epoch_batches))
            if should_step:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), float(config.get("max_grad_norm", 1.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % int(config.get("log_interval", 1)) == 0:
                    stats.update(
                        {
                            "step": global_step,
                            "micro_step": micro_step,
                            "epoch": epoch,
                            "runtime_sec": time.time() - start_time,
                        }
                    )
                    log_metrics(output_dir, stats, tracker)
                if global_step % int(config.get("save_interval", 100)) == 0:
                    save_dir = output_dir / f"checkpoint-{global_step}"
                    policy.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
            if max_steps and global_step >= max_steps:
                break
        if max_steps and global_step >= max_steps:
            break
    final_dir = output_dir / "final"
    policy.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    write_json(
        output_dir / "train_summary.json",
        {"global_step": global_step, "micro_step": micro_step, "runtime_sec": time.time() - start_time},
    )
    return final_dir


def main():
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in {"build-data", "all"}:
        print(json.dumps(build_preference_data(config, args), ensure_ascii=False, indent=2))
    final_dir = None
    if args.mode in {"train", "all"}:
        final_dir = train_dpo(args.config, config, args)
    if args.mode in {"eval", "all"}:
        print(json.dumps(evaluate_model(config, output_dir, str(final_dir) if final_dir else None, args.limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
