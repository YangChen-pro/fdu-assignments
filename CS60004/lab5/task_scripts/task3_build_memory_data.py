from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tqdm import tqdm

from task1_memory_agent import build_messages, retrieve_character_memories
from utils import load_persona, read_jsonl, write_json, write_jsonl


def clean_question(user_content: str) -> str:
    text = user_content.strip()
    prefix = "Interview question:"
    if text.startswith(prefix):
        return text[len(prefix) :].strip()
    return text


def convert_row(row: dict[str, Any], args: argparse.Namespace, embedding_model: Any) -> dict[str, Any]:
    character_id = row["character_id"]
    question = clean_question(row["messages"][-1]["content"])
    persona = load_persona(args.data_root, character_id)
    memories = retrieve_character_memories(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        embedding_model=embedding_model,
        character_id=character_id,
        question=question,
        args=args,
    )
    messages = build_messages(
        mode="memory",
        character_id=character_id,
        question=question,
        persona=persona,
        memories=memories,
        prompt_style=args.prompt_style,
    )
    return {
        "messages": messages + [{"role": "assistant", "content": row["reference_answer"]}],
        "meta": {
            "character_id": character_id,
            "source": row.get("source", ""),
            "memory_ids": [memory["memory_id"] for memory in memories],
        },
    }


def convert_split(
    path: Path,
    args: argparse.Namespace,
    embedding_model: Any,
    limit: int,
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if limit > 0:
        rows = rows[:limit]
    return [
        convert_row(row, args, embedding_model)
        for row in tqdm(rows, desc=f"memory data {path.name}", unit="sample")
    ]


def to_sft_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"messages": row["messages"]} for row in rows]


def to_grpo_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grpo_rows = []
    for row in rows:
        messages = row["messages"]
        grpo_rows.append(
            {
                "messages": messages[:-1],
                "reference_answer": messages[-1]["content"],
                "solution": messages[-1]["content"],
                "character_id": row["meta"]["character_id"],
                "source": row["meta"].get("source", ""),
            }
        )
    return grpo_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Task3 memory-aware SFT/GRPO data from isolated Task2 synthetic rows."
    )
    parser.add_argument("--data-root", type=Path, default=Path("lab5/public_pack/data"))
    parser.add_argument(
        "--input-train",
        type=Path,
        default=Path("lab5/outputs/task2/data_synthetic_interview_Qwen3.5-35B-A3B_noevalprompt/grpo_train.jsonl"),
    )
    parser.add_argument(
        "--input-val",
        type=Path,
        default=Path("lab5/outputs/task2/data_synthetic_interview_Qwen3.5-35B-A3B_noevalprompt/grpo_val.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("lab5/outputs/task3/data_memory_sft"))
    parser.add_argument("--sample-path", type=Path, default=Path("lab5/outputs/task3/data_memory_sft_samples.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path("lab5/outputs/task1/memory_cache"))
    parser.add_argument("--embedding-model", default="models/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-gpu-memory-utilization", type=float, default=0.20)
    parser.add_argument("--embedding-max-model-len", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--prompt-style", choices=("default", "grounded", "voice", "concise"), default="default")
    parser.add_argument("--memory-layout", choices=("default", "dedup"), default="default")
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from vllm import LLM

    embedding_model = LLM(
        model=args.embedding_model,
        gpu_memory_utilization=args.embedding_gpu_memory_utilization,
        max_model_len=args.embedding_max_model_len,
    )
    train_rows = convert_split(args.input_train, args, embedding_model, args.max_train_samples)
    val_rows = convert_split(args.input_val, args, embedding_model, args.max_val_samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "sft_train.jsonl"
    val_path = args.output_dir / "sft_val.jsonl"
    grpo_train_path = args.output_dir / "grpo_train.jsonl"
    grpo_val_path = args.output_dir / "grpo_val.jsonl"

    write_jsonl(train_path, to_sft_rows(train_rows))
    write_jsonl(val_path, to_sft_rows(val_rows))
    write_jsonl(grpo_train_path, to_grpo_rows(train_rows))
    write_jsonl(grpo_val_path, to_grpo_rows(val_rows))
    write_jsonl(args.sample_path, train_rows[: args.sample_limit])
    write_json(
        args.output_dir / "dataset_meta.json",
        {
            "strict_eval_isolation": True,
            "source": "Task2 synthetic no-eval-prompt GRPO rows with retrieved memories added to the prompt.",
            "heldout_eval_file": "lab5/public_pack/data/role_interview_eval.jsonl",
            "input_train": str(args.input_train),
            "input_val": str(args.input_val),
            "num_train": len(train_rows),
            "num_val": len(val_rows),
            "sft_train_file": str(train_path),
            "sft_val_file": str(val_path),
            "grpo_train_file": str(grpo_train_path),
            "grpo_val_file": str(grpo_val_path),
            "sample_file": str(args.sample_path),
            "top_k": args.top_k,
            "prompt_style": args.prompt_style,
            "memory_layout": args.memory_layout,
        },
    )
    print(f"wrote {len(train_rows)} train and {len(val_rows)} val memory-aware samples")


if __name__ == "__main__":
    main()
