from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import difflib
import json
from pathlib import Path
import random
from typing import Any

from tqdm import tqdm

from task2_build_data import build_system_message
from utils import (
    call_chat_completion,
    load_persona,
    parse_json_object,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
    write_jsonl_record,
)


FUZZY_OVERLAP_THRESHOLD = 0.74
GEN_SYSTEM_PROMPT = """You generate strictly non-held-out training data for a role-playing interview benchmark.
Return only valid JSON. Do not use markdown.
Use only the provided persona and evidence; create questions that are answerable from that evidence.
Each answer must be first-person, character-specific, grounded, and must not mention prompts, datasets, retrieved memories, or being an AI.
Prioritize factual restraint: if the evidence does not support a concrete anecdote, name, relationship, place, award, faith/family detail, or life event, the answer should stay character-specific but more general instead of inventing it.
Match the question language exactly; never mix Chinese/CJK into an English answer unless the evidence explicitly supports that language switch."""


def list_character_ids(data_root: Path) -> list[str]:
    character_dir = data_root / "individual_simulation_data" / "characters"
    return sorted(path.name.removeprefix("wiki_").removesuffix(".txt") for path in character_dir.glob("wiki_*.txt"))


def load_eval_questions(eval_path: Path) -> dict[str, list[str]]:
    questions: dict[str, list[str]] = {}
    for row in read_jsonl(eval_path):
        questions.setdefault(row["character_id"], []).append(row["question"])
    return questions


def normalize_question(question: str) -> str:
    return " ".join(question.lower().split())


def similar_to_heldout(question: str, heldout_questions: list[str]) -> bool:
    q = normalize_question(question)
    return any(
        difflib.SequenceMatcher(None, q, normalize_question(heldout)).ratio() >= FUZZY_OVERLAP_THRESHOLD
        for heldout in heldout_questions
    )


def compact_lines(texts: list[str], max_items: int, max_chars: int) -> str:
    lines = []
    for text in texts[:max_items]:
        line = " ".join(str(text).split())
        if line:
            lines.append(line[:max_chars])
    return "\n".join(f"- {line}" for line in lines)


def character_evidence(data_root: Path, character_id: str, max_items: int, seed: int) -> str:
    base = data_root / "individual_simulation_data"
    rng = random.Random(f"{seed}:{character_id}")
    snippets: list[str] = []

    scenes = read_json(base / "scene" / f"generated_agent_scene_{character_id}.json")
    rng.shuffle(scenes)
    for scene in scenes[: max_items // 2]:
        snippets.append(
            " | ".join(
                str(scene.get(key, ""))
                for key in ("location", "background", "profile")
                if scene.get(key)
            )
        )

    dialogues = read_json(base / "dialogue" / f"generated_agent_dialogue_{character_id}.json")
    rng.shuffle(dialogues)
    for item in dialogues[:max_items]:
        target_lines = [
            str(turn.get("content", ""))
            for turn in item.get("dialogue", [])
            if turn.get("role") == character_id and turn.get("content")
        ]
        if target_lines:
            snippets.append(" ".join(target_lines[:3]))
    return compact_lines(snippets, max_items=max_items, max_chars=420)


def build_generation_messages(character_id: str, persona: str, evidence: str, num_samples: int) -> list[dict[str, str]]:
    user_prompt = f"""Target character: {character_id}

[Persona]
{persona}

[Evidence from scenes/dialogues]
{evidence}

Create {num_samples} new interview-style training examples for this same character.
Questions should cover concrete background, relationships, values, habits, personality, work, humor, public responsibility, creative process, and memorable experiences.
Prefer questions that require details from the evidence instead of generic self-description, but do not imitate any held-out evaluation question.
Answers should be 45-120 words, first-person, natural, and character-specific.
Answers must use only supported facts. If asked for a specific story but the evidence only supports a general tendency, answer cautiously with that tendency instead of inventing an event.
Avoid generic self-help language, unsupported dramatic anecdotes, unsupported relationship claims, meta comments, and language mixing.

Return exactly this JSON object:
{{
  "items": [
    {{"question": "...", "answer": "..."}}
  ]
}}
"""
    return [
        {"role": "system", "content": GEN_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def generate_character(
    character_id: str,
    args: argparse.Namespace,
    heldout: dict[str, list[str]],
    heldout_all: set[str],
) -> list[dict[str, Any]]:
    persona = load_persona(args.data_root, character_id)
    evidence = character_evidence(args.data_root, character_id, args.evidence_items, args.seed)
    messages = build_generation_messages(character_id, persona, evidence, args.samples_per_character)

    raw = call_chat_completion(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        messages=messages,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    parsed = parse_json_object(raw)
    rows = []
    for index, item in enumerate(parsed.get("items", [])):
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue
        if normalize_question(question) in heldout_all or similar_to_heldout(question, heldout.get(character_id, [])):
            continue
        rows.append(
            {
                "id": f"synthetic_{character_id}_{index:03d}",
                "character_id": character_id,
                "question": question,
                "answer": answer,
                "source": "Qwen3.5-35B-A3B_synthetic_interview",
            }
        )
    return rows[: args.samples_per_character]


def to_sft_row(row: dict[str, Any], data_root: Path) -> dict[str, Any]:
    persona = load_persona(data_root, row["character_id"])
    return {
        "messages": [
            build_system_message(row["character_id"], persona),
            {"role": "user", "content": f"Interview question: {row['question']}"},
            {"role": "assistant", "content": row["answer"]},
        ],
        "meta": {key: row[key] for key in ("id", "character_id", "source")},
    }


def to_grpo_row(sft_row: dict[str, Any]) -> dict[str, Any]:
    messages = sft_row["messages"]
    reference_answer = messages[-1]["content"]
    meta = sft_row["meta"]
    return {
        "messages": messages[:-1],
        "reference_answer": reference_answer,
        "solution": reference_answer,
        "character_id": meta["character_id"],
        "source": meta.get("source", ""),
    }


def split_rows(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    val_size = max(1, round(len(rows) * val_ratio)) if rows else 0
    return rows[val_size:], rows[:val_size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate strict non-eval synthetic interview SFT data for Lab5 Task2.")
    parser.add_argument("--data-root", type=Path, default=Path("lab5/public_pack/data"))
    parser.add_argument("--eval-path", type=Path, default=Path("lab5/public_pack/data/role_interview_eval.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("lab5/outputs/task2/data_synthetic_interview_Qwen3.5-35B-A3B_noevalprompt"))
    parser.add_argument("--sample-path", type=Path, default=Path("lab5/outputs/task2/task2_synthetic_interview_Qwen3.5-35B-A3B_noevalprompt_samples.jsonl"))
    parser.add_argument("--model", default="Qwen3.5-35B-A3B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8318/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--samples-per-character", type=int, default=16)
    parser.add_argument("--evidence-items", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--characters", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    characters = [item.strip() for item in args.characters.split(",") if item.strip()] or list_character_ids(args.data_root)
    heldout = load_eval_questions(args.eval_path)
    heldout_all = {normalize_question(question) for questions in heldout.values() for question in questions}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated_path = args.output_dir / "generated_qa.jsonl"
    rows: list[dict[str, Any]] = []
    with generated_path.open("w", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {executor.submit(generate_character, character, args, heldout, heldout_all): character for character in characters}
        for future in tqdm(as_completed(futures), total=len(futures), desc="generate synthetic interview", unit="char"):
            character = futures[future]
            try:
                char_rows = future.result()
            except Exception as exc:
                write_jsonl_record(f, {"character_id": character, "status": "error", "error": repr(exc)})
                continue
            for row in char_rows:
                write_jsonl_record(f, {"status": "ok", **row})
            rows.extend(char_rows)

    sft_rows = [to_sft_row(row, args.data_root) for row in rows]
    train_rows, val_rows = split_rows(sft_rows, args.val_ratio, args.seed)
    write_jsonl(args.output_dir / "sft_train.jsonl", [{"messages": row["messages"]} for row in train_rows])
    write_jsonl(args.output_dir / "sft_val.jsonl", [{"messages": row["messages"]} for row in val_rows])
    write_jsonl(args.output_dir / "grpo_train.jsonl", [to_grpo_row(row) for row in train_rows])
    write_jsonl(args.output_dir / "grpo_val.jsonl", [to_grpo_row(row) for row in val_rows])
    write_jsonl(args.sample_path, sft_rows[:20])
    write_json(
        args.output_dir / "dataset_meta.json",
        {
            "strict_eval_isolation": True,
            "heldout_eval_file": str(args.eval_path),
            "num_characters": len(characters),
            "samples_per_character": args.samples_per_character,
            "num_samples": len(sft_rows),
            "num_train": len(train_rows),
            "num_val": len(val_rows),
            "global_exact_filter": True,
            "same_character_fuzzy_filter_threshold": FUZZY_OVERLAP_THRESHOLD,
            "source": "Qwen3.5-35B-A3B synthetic interview from persona/scenes/dialogues; held-out eval questions are used only for local filtering and are never shown to the generator",
            "generated_file": str(generated_path),
            "train_file": str(args.output_dir / "sft_train.jsonl"),
            "val_file": str(args.output_dir / "sft_val.jsonl"),
            "grpo_train_file": str(args.output_dir / "grpo_train.jsonl"),
            "grpo_val_file": str(args.output_dir / "grpo_val.jsonl"),
            "sample_file": str(args.sample_path),
        },
    )
    print(json.dumps({"num_samples": len(sft_rows), "train": len(train_rows), "val": len(val_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
