from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from utils import load_persona, read_json, read_jsonl, records_by_id, write_json, write_jsonl


SYSTEM_TEMPLATE = """You are role-playing as {character_id}. Answer in first person and stay consistent with this character's persona, memories, values, and speaking style. Do not mention that you are an AI.

Persona:
{persona}"""

DIALOGUE_USER_TEMPLATE = """Scene:
{scene}

Recent dialogue:
{history}

Continue the dialogue as {character_id}."""

INTERVIEW_USER_TEMPLATE = "Interview question: {question}"


def to_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item)
    return str(value or "")


def list_character_ids(data_root: Path) -> list[str]:
    character_dir = data_root / "individual_simulation_data" / "characters"
    return sorted(path.name.removeprefix("wiki_").removesuffix(".txt") for path in character_dir.glob("wiki_*.txt"))


def format_scene(item: dict[str, Any]) -> str:
    lines = [
        f"Setting: {to_text(item.get('setting'))}",
        f"Location: {to_text(item.get('location'))}",
        f"Background: {to_text(item.get('background'))}",
        f"Emotion: {to_text(item.get('emotion'))}",
        f"Topic: {to_text(item.get('topic'))}",
    ]
    return "\n".join(line for line in lines if not line.endswith(": "))


def format_history(turns: list[dict[str, Any]], history_turns: int) -> str:
    recent_turns = turns[-history_turns:] if history_turns > 0 else turns
    if not recent_turns:
        return "No previous dialogue."

    lines = []
    for turn in recent_turns:
        role = turn.get("role", "Unknown")
        action = turn.get("action", "")
        content = turn.get("content", "")
        prefix = f"{role} {action}".strip()
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def build_system_message(character_id: str, persona: str) -> dict[str, str]:
    return {"role": "system", "content": SYSTEM_TEMPLATE.format(character_id=character_id, persona=persona)}


def build_dialogue_messages(
    character_id: str,
    persona: str,
    item: dict[str, Any],
    history: list[dict[str, Any]],
    history_turns: int,
) -> list[dict[str, str]]:
    return [
        build_system_message(character_id, persona),
        {
            "role": "user",
            "content": DIALOGUE_USER_TEMPLATE.format(
                scene=format_scene(item),
                history=format_history(history, history_turns),
                character_id=character_id,
            ),
        },
    ]


def build_interview_messages(character_id: str, persona: str, question: str, answer: str) -> list[dict[str, str]]:
    return [
        build_system_message(character_id, persona),
        {"role": "user", "content": INTERVIEW_USER_TEMPLATE.format(question=question)},
        {"role": "assistant", "content": answer},
    ]


def build_dialogue_samples(data_root: Path, character_ids: list[str], history_turns: int) -> list[dict[str, Any]]:
    base = data_root / "individual_simulation_data"
    samples = []

    for character_id in character_ids:
        persona = load_persona(data_root, character_id)
        dialogue_path = base / "dialogue" / f"generated_agent_dialogue_{character_id}.json"
        for dialogue_index, item in enumerate(read_json(dialogue_path)):
            history: list[dict[str, Any]] = []
            for turn_index, turn in enumerate(item.get("dialogue", [])):
                is_target_turn = turn.get("role") == character_id and turn.get("action") == "(speaking)"
                content = str(turn.get("content") or "").strip()
                if is_target_turn and content:
                    messages = build_dialogue_messages(character_id, persona, item, history, history_turns)
                    messages.append({"role": "assistant", "content": content})
                    samples.append(
                        {
                            "messages": messages,
                            "meta": {
                                "character_id": character_id,
                                "dialogue_index": dialogue_index,
                                "turn_index": turn_index,
                                "source": "dialogue",
                            },
                        }
                    )
                history.append(turn)
    return samples


def build_distill_samples(
    data_root: Path,
    eval_path: Path,
    predictions_path: Path,
    source_name: str,
) -> list[dict[str, Any]]:
    predictions = records_by_id(predictions_path)
    samples = []
    for sample in read_jsonl(eval_path):
        prediction = predictions.get(sample["id"], {})
        answer = str(prediction.get("prediction") or "").strip()
        if not answer:
            continue
        character_id = sample["character_id"]
        persona = load_persona(data_root, character_id)
        samples.append(
            {
                "messages": build_interview_messages(character_id, persona, sample["question"], answer),
                "meta": {
                    "character_id": character_id,
                    "eval_id": sample["id"],
                    "question_index": sample.get("question_index"),
                    "source": f"interview_distill:{source_name}",
                },
            }
        )
    return samples


def build_samples(args: argparse.Namespace, character_ids: list[str]) -> list[dict[str, Any]]:
    samples = []
    if not args.only_distill:
        samples.extend(build_dialogue_samples(args.data_root, character_ids, args.history_turns))
    if args.distill_predictions:
        samples.extend(
            build_distill_samples(
                data_root=args.data_root,
                eval_path=args.eval_path,
                predictions_path=args.distill_predictions,
                source_name=args.distill_name,
            )
        )
    return samples


def split_samples(samples: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, round(len(shuffled) * val_ratio)) if shuffled else 0
    return shuffled[val_size:], shuffled[:val_size]


def sft_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"messages": sample["messages"]} for sample in samples]


def grpo_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        messages = sample["messages"]
        reference_answer = messages[-1]["content"]
        rows.append(
            {
                "messages": messages[:-1],
                "reference_answer": reference_answer,
                "solution": reference_answer,
                "character_id": sample["meta"]["character_id"],
                "source": sample["meta"].get("source", ""),
            }
        )
    return rows


def sample_rows(samples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return [{"meta": sample["meta"], "messages": sample["messages"]} for sample in samples[:limit]]


def count_sources(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        source = sample["meta"].get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Lab5 Task2 SFT/GRPO data from role dialogues and optional interview distillation.")
    parser.add_argument("--data-root", type=Path, default=Path("lab5/public_pack/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("lab5/outputs/task2/data"))
    parser.add_argument("--sample-path", type=Path, default=Path("lab5/outputs/task2/task2_sft_samples.jsonl"))
    parser.add_argument("--eval-path", type=Path, default=Path("lab5/public_pack/data/role_interview_eval.jsonl"))
    parser.add_argument("--distill-predictions", type=Path)
    parser.add_argument("--distill-name", default="memory")
    parser.add_argument("--only-distill", action="store_true")
    parser.add_argument("--characters", default="")
    parser.add_argument("--history-turns", type=int, default=6)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    character_ids = [item.strip() for item in args.characters.split(",") if item.strip()] or list_character_ids(args.data_root)
    samples = build_samples(args, character_ids)
    random.Random(args.seed).shuffle(samples)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "sft_train.jsonl"
    val_path = args.output_dir / "sft_val.jsonl"
    grpo_train_path = args.output_dir / "grpo_train.jsonl"
    grpo_val_path = args.output_dir / "grpo_val.jsonl"
    meta_path = args.output_dir / "dataset_meta.json"

    write_jsonl(train_path, sft_rows(train_samples))
    write_jsonl(val_path, sft_rows(val_samples))
    write_jsonl(grpo_train_path, grpo_rows(train_samples))
    write_jsonl(grpo_val_path, grpo_rows(val_samples))
    write_jsonl(args.sample_path, sample_rows(samples, args.sample_limit))
    write_json(
        meta_path,
        {
            "num_characters": len(character_ids),
            "characters": character_ids,
            "source_counts": count_sources(samples),
            "num_samples": len(samples),
            "num_train": len(train_samples),
            "num_val": len(val_samples),
            "history_turns": args.history_turns,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "distill_predictions": str(args.distill_predictions) if args.distill_predictions else "",
            "only_distill": args.only_distill,
            "sft_train_file": str(train_path),
            "sft_val_file": str(val_path),
            "grpo_train_file": str(grpo_train_path),
            "grpo_val_file": str(grpo_val_path),
            "sample_file": str(args.sample_path),
        },
    )
    print(f"wrote {len(train_samples)} train and {len(val_samples)} val samples")
    print(f"sources: {count_sources(samples)}")
    print(f"train: {train_path}")
    print(f"val: {val_path}")
    print(f"grpo train: {grpo_train_path}")
    print(f"grpo val: {grpo_val_path}")
    print(f"samples: {args.sample_path}")


if __name__ == "__main__":
    main()
