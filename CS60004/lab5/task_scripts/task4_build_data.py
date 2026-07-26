from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd

from utils import read_jsonl, write_json, write_jsonl


def prompt_key(text: str) -> str:
    return " ".join(str(text).split()).lower()


def message_list(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    messages = []
    for item in value:
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role and content:
            messages.append({"role": role, "content": content})
    return messages


def valid_messages(messages: list[dict[str, str]], max_chars: int) -> bool:
    if len(messages) < 2:
        return False
    if messages[0]["role"] != "user" or messages[-1]["role"] != "assistant":
        return False
    return sum(len(item["content"]) for item in messages) <= max_chars


def user_prompt(messages: list[dict[str, str]]) -> str:
    for item in messages:
        if item["role"] == "user":
            return item["content"]
    return ""


def make_record(source: str, idx: int, messages: list[dict[str, str]]) -> dict[str, Any]:
    return {"source": source, "id": f"{source}_{idx:06d}", "messages": messages}


def banned_prompts(paths: list[Path]) -> set[str]:
    banned = set()
    for path in paths:
        for row in read_jsonl(path):
            banned.add(prompt_key(row["prompt"]))
    return banned


def load_tulu(
    path: Path,
    banned: set[str],
    max_records: int,
    val_records: int,
    max_chars: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    if max_records <= 0 and val_records <= 0:
        return [], [], 0
    rows = pd.read_parquet(path).to_dict("records")
    rng.shuffle(rows)
    records = []
    overlaps = 0
    for row in rows:
        if prompt_key(row.get("prompt", "")) in banned:
            overlaps += 1
            continue
        messages = message_list(row.get("messages", []))
        if valid_messages(messages, max_chars):
            records.append(make_record("tulu3_if", len(records), messages))
        if len(records) >= max_records + val_records:
            break
    return records[val_records:], records[:val_records], overlaps


def alpaca_prompt(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    if input_text:
        return f"{instruction}\n\nInput:\n{input_text}"
    return instruction


def load_alpaca(
    path: Path,
    banned: set[str],
    max_records: int,
    val_records: int,
    max_chars: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    if max_records <= 0 and val_records <= 0:
        return [], [], 0
    rows = json.loads(path.read_text(encoding="utf-8"))
    rng.shuffle(rows)
    records = []
    overlaps = 0
    for row in rows:
        prompt = alpaca_prompt(row)
        if prompt_key(prompt) in banned:
            overlaps += 1
            continue
        output = str(row.get("output", "")).strip()
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": output}]
        if prompt and output and valid_messages(messages, max_chars):
            records.append(make_record("alpaca_gpt4_en", len(records), messages))
        if len(records) >= max_records + val_records:
            break
    return records[val_records:], records[:val_records], overlaps


def load_role(path: Path, repeats: int) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    records = []
    for repeat_idx in range(repeats):
        for row_idx, row in enumerate(rows):
            record = dict(row)
            record["source"] = "role_memory_replay"
            record["id"] = f"role_memory_replay_r{repeat_idx}_{row_idx:06d}"
            records.append(record)
    return records


def valid_rlvr_messages(messages: list[dict[str, str]], max_chars: int) -> bool:
    return len(messages) == 1 and messages[0]["role"] == "user" and len(messages[0]["content"]) <= max_chars


def load_rlvr_ifeval(
    path: Path,
    banned: set[str],
    max_records: int,
    val_records: int,
    max_chars: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    rows = pd.read_parquet(path).to_dict("records")
    rng.shuffle(rows)
    records = []
    overlaps = 0
    for row in rows:
        messages = message_list(row.get("messages", []))
        prompt = user_prompt(messages)
        if prompt_key(prompt) in banned:
            overlaps += 1
            continue
        if not valid_rlvr_messages(messages, max_chars):
            continue
        ground_truth = json.loads(row["ground_truth"])
        records.append(
            {
                "source": "rlvr_ifeval",
                "id": f"rlvr_ifeval_{len(records):06d}",
                "messages": messages,
                "ground_truth": ground_truth,
                "constraint_type": row.get("constraint_type", ""),
                "constraint": row.get("constraint", ""),
            }
        )
        if len(records) >= max_records + val_records:
            break
    return records[val_records:], records[:val_records], overlaps


def words(count: int, token: str = "word") -> str:
    return " ".join([token] * max(1, count))


def sentence_count_answer(count: int) -> str:
    return " ".join(["Done."] * max(1, count))


def choice_answer(options: str) -> str:
    if options == "yes/no/maybe":
        return "yes"
    if options == "I know or I don't know":
        return "I know"
    if options == "a), b), c), d)":
        return "a)"
    return re.split(r"[,/]", options or "yes")[0].strip() or "yes"


def relation_target(gt: dict[str, Any], at_most_default: int = 1) -> int:
    n = max(1, int(gt.get("N") or 1))
    return at_most_default if gt.get("quantifier") == "at most" else n


def synthetic_ifeval_answer(gt: dict[str, Any]) -> str:
    func = gt.get("func_name")
    n = relation_target(gt)

    if func == "validate_lowercase":
        return "this is a simple answer that follows the request."
    if func == "validate_uppercase":
        return "THIS IS A SIMPLE ANSWER THAT FOLLOWS THE REQUEST."
    if func == "validate_no_commas":
        return "This answer follows the request without using comma punctuation."
    if func == "validate_json_format":
        return '{"answer":"done"}'
    if func == "validate_quotation":
        return '"Here is the answer."'
    if func == "validate_title":
        return "<<Answer>>\nHere is the answer."
    if func == "validate_two_responses":
        return "First answer.\n******\nSecond answer."
    if func == "verify_keywords":
        return " ".join(str(item) for item in gt.get("keyword_list") or ["answer"])
    if func == "validate_forbidden_words":
        return "This response avoids the restricted vocabulary and answers safely."
    if func == "validate_placeholders":
        return " ".join(f"[item_{idx}]" for idx in range(1, n + 1))
    if func == "verify_bullet_points":
        return "\n".join(f"* point {idx}" for idx in range(1, n + 1))
    if func == "validate_highlighted_sections":
        return " ".join(f"*highlight {idx}*" for idx in range(1, n + 1))
    if func == "validate_sections":
        splitter = str(gt.get("section_splitter") or "Section:")
        return "\n".join(f"{splitter} {idx}\nContent {idx}." for idx in range(1, n + 1))
    if func == "verify_paragraph_count":
        return " *** ".join(f"Paragraph {idx}" for idx in range(1, n + 1))
    if func == "validate_paragraphs":
        nth = max(1, int(gt.get("i") or 1))
        first = str(gt.get("first_word") or "first")
        parts = [f"Paragraph {idx} content." for idx in range(1, n + 1)]
        if nth <= len(parts):
            parts[nth - 1] = f"{first} paragraph {nth} content."
        return "\n\n".join(parts)
    if func == "verify_postscript":
        return f"Here is the answer.\n{gt.get('postscript_marker') or 'P.S.'} noted."
    if func == "validate_end":
        return f"Here is the answer. {gt.get('end_phrase') or 'Done'}"
    if func == "validate_repeat_prompt":
        return f"{gt.get('original_prompt') or 'Repeat the request'}\nAnswer: done."
    if func == "verify_keyword_frequency":
        return words(int(gt.get("N") or 1), str(gt.get("word") or "word"))
    if func == "verify_letter_frequency":
        return words(int(gt.get("N") or 1), str(gt.get("letter") or "a").lower())
    if func == "verify_sentence_constraint":
        return sentence_count_answer(relation_target(gt))
    if func == "validate_word_constraint":
        return words(relation_target(gt))
    if func == "validate_frequency_capital_words":
        return words(relation_target(gt), "UP") if gt.get("quantifier") != "at most" else "simple answer"
    if func == "validate_choice":
        return choice_answer(str(gt.get("options") or "yes"))
    return "Here is the answer."


def build_sft_record(record: dict[str, Any], source: str) -> dict[str, Any]:
    answer = synthetic_ifeval_answer(record["ground_truth"])
    return {
        "source": source,
        "id": record["id"],
        "messages": record["messages"] + [{"role": "assistant", "content": answer}],
        "ground_truth": record["ground_truth"],
        "constraint_type": record.get("constraint_type", ""),
    }


class SafeFormatMap(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def constraint_clause(record: dict[str, Any]) -> str:
    gt = record.get("ground_truth") or {}
    keywords = gt.get("keyword_list") or []
    mapping = SafeFormatMap({k: v for k, v in gt.items() if v is not None})
    mapping.update(
        {
            "first word": gt.get("first_word"),
            "end phrase": gt.get("end_phrase"),
            "section splitter": gt.get("section_splitter"),
            "postscript marker": gt.get("postscript_marker"),
            "forbidden words": ", ".join(gt.get("forbidden_words") or []),
        }
    )
    for idx, keyword in enumerate(keywords, 1):
        mapping[f"keyword{idx}"] = keyword
    text = str(record.get("constraint") or "").replace("at least / around / at most", str(gt.get("quantifier") or "exactly"))
    return text.format_map(mapping)


FORMAT_FUNCS = {
    "validate_json_format",
    "validate_quotation",
    "validate_title",
    "validate_two_responses",
    "verify_bullet_points",
    "validate_highlighted_sections",
    "validate_sections",
    "verify_paragraph_count",
    "validate_paragraphs",
}


def compatible_constraints(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_func = (left.get("ground_truth") or {}).get("func_name")
    right_func = (right.get("ground_truth") or {}).get("func_name")
    pair = {left_func, right_func}
    if not left_func or not right_func or left_func == right_func:
        return False
    if pair == {"validate_lowercase", "validate_uppercase"}:
        return False
    if "validate_lowercase" in pair and "validate_frequency_capital_words" in pair:
        return False
    if "validate_json_format" in pair and len(pair & FORMAT_FUNCS) > 1:
        return False
    if "validate_choice" in pair and len(pair & (FORMAT_FUNCS | {"validate_word_constraint", "verify_sentence_constraint"})) > 0:
        return False
    return "validate_repeat_prompt" not in pair


def add_extra_constraint(record: dict[str, Any], extra: dict[str, Any], idx: int, source: str) -> dict[str, Any]:
    prompt = user_prompt(record["messages"])
    extra_text = constraint_clause(extra)
    return {
        "source": source,
        "id": f"{source}_{idx:06d}",
        "messages": [{"role": "user", "content": f"{prompt}\n\nAdditional requirement: {extra_text}"}],
        "ground_truth": [record["ground_truth"], extra["ground_truth"]],
        "constraint_type": f"{record.get('constraint_type', '')} + {extra.get('constraint_type', '')}",
        "constraint": [record.get("constraint", ""), extra_text],
    }


def make_multi_constraint_records(records: list[dict[str, Any]], rng: random.Random, source: str) -> list[dict[str, Any]]:
    pool = records[:]
    rng.shuffle(pool)
    cursor = 0
    output = []
    for record in records:
        for _ in range(len(pool)):
            extra = pool[cursor % len(pool)]
            cursor += 1
            if compatible_constraints(record, extra):
                output.append(add_extra_constraint(record, extra, len(output), source))
                break
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Task4 SFT or GRPO data with Lab5 dev/test prompt isolation.")
    parser.add_argument("--mode", choices=["sft_mix", "grpo_rlvr", "grpo_rlvr_multi", "grpo_hybrid", "sft_rlvr"], default="sft_mix")
    parser.add_argument("--output-dir", type=Path, default=Path("lab5/outputs/task4/data_instruction_role_mix_r1"))
    parser.add_argument("--tulu-parquet", type=Path, default=Path("lab5/outputs/external_data/tulu3_sft_if/data/train-00000-of-00001.parquet"))
    parser.add_argument("--alpaca-json", type=Path, default=Path("lab5/outputs/external_data/alpaca_gpt4_en/alpaca_gpt4_data_en.json"))
    parser.add_argument("--rlvr-parquet", type=Path, default=Path("lab5/outputs/external_data/rlvr_ifeval/data/train-00000-of-00001.parquet"))
    parser.add_argument("--role-train", type=Path, default=Path("lab5/outputs/task3/data_memory_sft_r2_voice_dedup_cleanmem/sft_train.jsonl"))
    parser.add_argument("--role-val", type=Path, default=Path("lab5/outputs/task3/data_memory_sft_r2_voice_dedup_cleanmem/sft_val.jsonl"))
    parser.add_argument("--hybrid-role-train", type=Path, default=Path("lab5/outputs/task3/data_memory_sft_r2_voice_dedup_cleanmem/grpo_train.jsonl"))
    parser.add_argument("--hybrid-role-val", type=Path, default=Path("lab5/outputs/task3/data_memory_sft_r2_voice_dedup_cleanmem/grpo_val.jsonl"))
    parser.add_argument("--hybrid-instruction-train", type=Path, default=Path("lab5/outputs/task4/data_instruction_grpo_rlvr_multi_r1/grpo_train.jsonl"))
    parser.add_argument("--hybrid-instruction-val", type=Path, default=Path("lab5/outputs/task4/data_instruction_grpo_rlvr_multi_r1/grpo_val.jsonl"))
    parser.add_argument("--lab5-dev", type=Path, default=Path("lab5/public_pack/data/instruction_following/dev.jsonl"))
    parser.add_argument("--lab5-test", type=Path, default=Path("lab5/public_pack/data/instruction_following/test.jsonl"))
    parser.add_argument("--tulu-max", type=int, default=16000)
    parser.add_argument("--alpaca-max", type=int, default=6000)
    parser.add_argument("--tulu-val", type=int, default=300)
    parser.add_argument("--alpaca-val", type=int, default=100)
    parser.add_argument("--role-repeats", type=int, default=8)
    parser.add_argument("--role-val-repeats", type=int, default=1)
    parser.add_argument("--grpo-max", type=int, default=12000)
    parser.add_argument("--grpo-val", type=int, default=500)
    parser.add_argument("--max-chars", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=20260531)
    return parser.parse_args()


def build_sft_mix(args: argparse.Namespace, rng: random.Random, banned: set[str]) -> dict[str, Any]:
    tulu_train, tulu_val, tulu_overlaps = load_tulu(args.tulu_parquet, banned, args.tulu_max, args.tulu_val, args.max_chars, rng)
    alpaca_train, alpaca_val, alpaca_overlaps = load_alpaca(args.alpaca_json, banned, args.alpaca_max, args.alpaca_val, args.max_chars, rng)
    role_train = load_role(args.role_train, args.role_repeats)
    role_val = load_role(args.role_val, args.role_val_repeats)

    train = role_train + tulu_train + alpaca_train
    val = role_val + tulu_val + alpaca_val
    rng.shuffle(train)
    rng.shuffle(val)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "sft_train.jsonl", train)
    write_jsonl(args.output_dir / "sft_val.jsonl", val)
    write_jsonl(args.output_dir / "data_samples.jsonl", train[:20])
    return {
        "purpose": "single-checkpoint fusion dataset for Lab5 Task4 role simulation and instruction following",
        "train_count": len(train),
        "val_count": len(val),
        "counts": {
            "role_memory_replay_train": len(role_train),
            "role_memory_replay_val": len(role_val),
            "tulu3_if_train": len(tulu_train),
            "tulu3_if_val": len(tulu_val),
            "alpaca_gpt4_en_train": len(alpaca_train),
            "alpaca_gpt4_en_val": len(alpaca_val),
        },
        "filtered_exact_prompt_overlaps_with_lab5_dev_test": {
            "tulu3_if": tulu_overlaps,
            "alpaca_gpt4_en": alpaca_overlaps,
        },
        "external_sources": [
            "ModelScope: LLM-Research/tulu-3-sft-personas-instruction-following, Apache License 2.0",
            "ModelScope: llamafactory/alpaca_gpt4_en, Apache License 2.0",
        ],
        "lab5_public_pack_policy": "read only; dev/test prompts used only for exact-overlap filtering, never as training targets",
        "seed": args.seed,
        "max_chars": args.max_chars,
    }


def build_grpo_rlvr(args: argparse.Namespace, rng: random.Random, banned: set[str]) -> dict[str, Any]:
    train, val, overlaps = load_rlvr_ifeval(args.rlvr_parquet, banned, args.grpo_max, args.grpo_val, args.max_chars, rng)
    rng.shuffle(train)
    rng.shuffle(val)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "grpo_train.jsonl", train)
    write_jsonl(args.output_dir / "grpo_val.jsonl", val)
    write_jsonl(args.output_dir / "data_samples.jsonl", train[:20])
    return {
        "purpose": "rule-reward GRPO dataset for Lab5 Task4 instruction-following capability",
        "train_count": len(train),
        "val_count": len(val),
        "counts": {"rlvr_ifeval_train": len(train), "rlvr_ifeval_val": len(val)},
        "filtered_exact_prompt_overlaps_with_lab5_dev_test": {"rlvr_ifeval": overlaps},
        "external_sources": ["ModelScope: LLM-Research/RLVR-IFeval, public IFEval-style RLVR data"],
        "lab5_public_pack_policy": "read only; dev/test prompts used only for exact-overlap filtering, never as training targets",
        "seed": args.seed,
        "max_chars": args.max_chars,
    }


def build_grpo_rlvr_multi(args: argparse.Namespace, rng: random.Random, banned: set[str]) -> dict[str, Any]:
    base_train, base_val, overlaps = load_rlvr_ifeval(args.rlvr_parquet, banned, args.grpo_max, args.grpo_val, args.max_chars, rng)
    train = make_multi_constraint_records(base_train, rng, "rlvr_ifeval_multi")
    val = make_multi_constraint_records(base_val, rng, "rlvr_ifeval_multi_val")
    rng.shuffle(train)
    rng.shuffle(val)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "grpo_train.jsonl", train)
    write_jsonl(args.output_dir / "grpo_val.jsonl", val)
    write_jsonl(args.output_dir / "data_samples.jsonl", train[:20])
    return {
        "purpose": "multi-constraint rule-reward GRPO dataset for Lab5 Task4 instruction following",
        "train_count": len(train),
        "val_count": len(val),
        "counts": {"rlvr_ifeval_multi_train": len(train), "rlvr_ifeval_multi_val": len(val)},
        "base_records_loaded": {"train": len(base_train), "val": len(base_val)},
        "filtered_exact_prompt_overlaps_with_lab5_dev_test": {"rlvr_ifeval": overlaps},
        "external_sources": ["ModelScope: LLM-Research/RLVR-IFeval, public IFEval-style RLVR data"],
        "cleaning": "exact prompt overlap filtering against Lab5 instruction dev/test; synthetic multi-constraint prompts combine public RLVR-IFeval constraint metadata only",
        "lab5_public_pack_policy": "read only; dev/test prompts used only for exact-overlap filtering, never as training targets",
        "seed": args.seed,
        "max_chars": args.max_chars,
    }


def repeat_rows(rows: list[dict[str, Any]], repeats: int, source_suffix: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for repeat_idx in range(max(1, repeats)):
        for row_idx, row in enumerate(rows):
            item = dict(row)
            item["source"] = f"{item.get('source', source_suffix)}:{source_suffix}"
            item["id"] = item.get("id") or f"{source_suffix}_r{repeat_idx}_{row_idx:06d}"
            output.append(item)
    return output


def build_grpo_hybrid(args: argparse.Namespace, rng: random.Random, banned: set[str]) -> dict[str, Any]:
    role_train = repeat_rows(read_jsonl(args.hybrid_role_train), args.role_repeats, "role_grpo")
    role_val = repeat_rows(read_jsonl(args.hybrid_role_val), args.role_val_repeats, "role_grpo_val")
    instruction_train = read_jsonl(args.hybrid_instruction_train)
    instruction_val = read_jsonl(args.hybrid_instruction_val)
    rng.shuffle(instruction_train)
    rng.shuffle(instruction_val)
    if args.grpo_max > 0:
        instruction_train = instruction_train[: args.grpo_max]
    if args.grpo_val > 0:
        instruction_val = instruction_val[: args.grpo_val]

    train = role_train + instruction_train
    val = role_val + instruction_val
    rng.shuffle(train)
    rng.shuffle(val)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "grpo_train.jsonl", train)
    write_jsonl(args.output_dir / "grpo_val.jsonl", val)
    write_jsonl(args.output_dir / "data_samples.jsonl", train[:20])
    return {
        "purpose": "hybrid GRPO dataset for one Lab5 model: role simulation retention plus instruction constraint following",
        "train_count": len(train),
        "val_count": len(val),
        "counts": {
            "role_train": len(role_train),
            "role_val": len(role_val),
            "instruction_train": len(instruction_train),
            "instruction_val": len(instruction_val),
        },
        "role_sources": [str(args.hybrid_role_train), str(args.hybrid_role_val)],
        "instruction_sources": [str(args.hybrid_instruction_train), str(args.hybrid_instruction_val)],
        "external_sources": [
            "Role data: previously generated non-held-out synthetic interview/memory data under lab5/outputs/task3",
            "Instruction data: ModelScope LLM-Research/RLVR-IFeval multi-constraint records filtered against Lab5 instruction dev/test prompts",
        ],
        "cleaning": "role data uses non-held-out synthetic prompts; instruction data inherits exact prompt overlap filtering against Lab5 dev/test; no test targets are used",
        "lab5_public_pack_policy": "read only; dev/test prompts used only for exact-overlap filtering or evaluation, never as training targets",
        "seed": args.seed,
        "max_chars": args.max_chars,
    }


def build_sft_rlvr(args: argparse.Namespace, rng: random.Random, banned: set[str]) -> dict[str, Any]:
    train, val, overlaps = load_rlvr_ifeval(args.rlvr_parquet, banned, args.grpo_max, args.grpo_val, args.max_chars, rng)
    train = [build_sft_record(row, "rlvr_ifeval_synthetic") for row in train]
    val = [build_sft_record(row, "rlvr_ifeval_synthetic") for row in val]
    rng.shuffle(train)
    rng.shuffle(val)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "sft_train.jsonl", train)
    write_jsonl(args.output_dir / "sft_val.jsonl", val)
    write_jsonl(args.output_dir / "data_samples.jsonl", train[:20])
    return {
        "purpose": "synthetic SFT dataset for instruction constraint following without using Lab5 dev/test targets",
        "train_count": len(train),
        "val_count": len(val),
        "counts": {"rlvr_ifeval_synthetic_train": len(train), "rlvr_ifeval_synthetic_val": len(val)},
        "filtered_exact_prompt_overlaps_with_lab5_dev_test": {"rlvr_ifeval": overlaps},
        "external_sources": ["ModelScope: LLM-Research/RLVR-IFeval prompts and public constraint metadata"],
        "cleaning": "exact prompt overlap filtering against Lab5 instruction dev/test; programmatic synthetic answers generated from public constraint metadata only",
        "lab5_public_pack_policy": "read only; dev/test prompts used only for exact-overlap filtering, never as training targets",
        "seed": args.seed,
        "max_chars": args.max_chars,
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    banned = banned_prompts([args.lab5_dev, args.lab5_test])

    if args.mode == "grpo_rlvr":
        manifest = build_grpo_rlvr(args, rng, banned)
    elif args.mode == "grpo_rlvr_multi":
        manifest = build_grpo_rlvr_multi(args, rng, banned)
    elif args.mode == "grpo_hybrid":
        manifest = build_grpo_hybrid(args, rng, banned)
    elif args.mode == "sft_rlvr":
        manifest = build_sft_rlvr(args, rng, banned)
    else:
        manifest = build_sft_mix(args, rng, banned)

    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
