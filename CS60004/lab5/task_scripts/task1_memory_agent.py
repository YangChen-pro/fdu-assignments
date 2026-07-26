from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from task0_memory import retrieve_memories
from utils import call_chat_completion, load_persona, read_json, read_jsonl, write_json, write_jsonl_record


EMBEDDING_TASK = (
    "Given an interview question for a target character, retrieve memories that help "
    "answer consistently with the character's facts, values, personality, and speaking style."
)


def clean_memory_field(value: Any) -> str:
    return " ".join(str(value).replace("<|NONSTOP|>", " ").split())


def format_memory_value(value: Any, sep: str = " ") -> str:
    if isinstance(value, list):
        return sep.join(clean_memory_field(item) for item in value)
    return clean_memory_field(value)


def build_memory_pool(data_root: Path, character_id: str, memory_layout: str = "default") -> list[dict[str, Any]]:
    base = data_root / "individual_simulation_data"
    memories: list[dict[str, Any]] = []
    if memory_layout != "dedup":
        memories.append(
            {
                "memory_id": f"{character_id}:persona",
                "type": "persona",
                "text": f"Persona: {load_persona(data_root, character_id)}",
                "importance": 0.95,
            }
        )

    scene_path = base / "scene" / f"generated_agent_scene_{character_id}.json"
    for idx, scene in enumerate(read_json(scene_path)):
        scene_lines = [
            f"Location: {format_memory_value(scene['location'])}",
            f"Background: {format_memory_value(scene['background'])}",
        ]
        if memory_layout != "dedup":
            scene_lines.append(f"Profile: {format_memory_value(scene['profile'])}")
        memories.append(
            {
                "memory_id": f"{character_id}:scene:{idx}",
                "type": "scene",
                "text": "\n".join(scene_lines),
                "importance": 0.65,
            }
        )

    dialogue_path = base / "dialogue" / f"generated_agent_dialogue_{character_id}.json"
    for idx, item in enumerate(read_json(dialogue_path)):
        target_turns = [
            clean_memory_field(turn["content"])
            for turn in item["dialogue"]
            if turn["role"] == character_id
        ]
        other_turns = [
            f"{turn['role']}: {clean_memory_field(turn['content'])}"
            for turn in item["dialogue"]
            if turn["role"] != character_id
        ]
        memories.append(
            {
                "memory_id": f"{character_id}:dialogue:{idx}",
                "type": "dialogue",
                "text": "\n".join(
                    [
                        f"Setting: {format_memory_value(item['setting'])}",
                        f"Emotion: {format_memory_value(item['emotion'])}",
                        f"Topic: {format_memory_value(item['topic'], sep=', ')}",
                        f"Location: {format_memory_value(item['location'])}",
                        f"Background: {format_memory_value(item['background'])}",
                        f"{character_id} said: {' '.join(target_turns)}",
                        f"Other context: {' '.join(other_turns[:4])}",
                    ]
                ),
                "importance": 0.80,
            }
        )

    return memories


def embed_texts(model: Any, texts: list[str], batch_size: int) -> np.ndarray:
    embeddings = []
    for start in range(0, len(texts), batch_size):
        outputs = model.embed(texts[start : start + batch_size])
        embeddings.extend(output.outputs.embedding for output in outputs)
    return np.asarray(embeddings, dtype=np.float32)


def load_or_encode_memories(
    data_root: Path,
    cache_dir: Path,
    embedding_model: Any,
    character_id: str,
    batch_size: int,
    memory_layout: str,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    memories = build_memory_pool(data_root, character_id, memory_layout)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"{character_id}.{memory_layout}"
    meta_path = cache_dir / f"{cache_key}.memories.json"
    emb_path = cache_dir / f"{cache_key}.embeddings.npy"

    if meta_path.exists() and emb_path.exists():
        cached_memories = read_json(meta_path)
        embeddings = np.load(emb_path)
        if cached_memories == memories:
            return cached_memories, embeddings

    embeddings = embed_texts(embedding_model, [memory["text"] for memory in memories], batch_size)
    write_json(meta_path, memories)
    np.save(emb_path, embeddings)
    return memories, embeddings


def retrieve_character_memories(
    data_root: Path,
    cache_dir: Path,
    embedding_model: Any,
    character_id: str,
    question: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    memories, memory_embeddings = load_or_encode_memories(
        data_root=data_root,
        cache_dir=cache_dir,
        embedding_model=embedding_model,
        character_id=character_id,
        batch_size=args.embedding_batch_size,
        memory_layout=args.memory_layout,
    )
    query_text = f"Instruct: {EMBEDDING_TASK}\nQuery:{question}"
    query_embedding = embed_texts(embedding_model, [query_text], batch_size=1)[0]
    candidates = []
    for idx, memory in enumerate(memories):
        item = dict(memory)
        item["embedding"] = memory_embeddings[idx]
        candidates.append(item)

    selected = retrieve_memories(
        query=question,
        query_embedding=query_embedding,
        memories=candidates,
        top_k=args.top_k,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )
    for item in selected:
        item.pop("embedding", None)
    return selected


def format_memories(memories: list[dict[str, Any]]) -> str:
    lines = []
    for rank, memory in enumerate(memories, start=1):
        lines.append(
            f"{rank}. [{memory['memory_id']}] {memory['text']} "
            f"(score={memory['score']:.4f})"
        )
    return "\n".join(lines)


def build_messages(
    mode: str,
    character_id: str,
    question: str,
    persona: str | None,
    memories: list[dict[str, Any]] | None,
    prompt_style: str = "default",
) -> list[dict[str, str]]:
    if mode == "question":
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the interview question directly.",
            },
            {"role": "user", "content": question},
        ]

    extra_rules = ""
    if prompt_style == "grounded":
        extra_rules = (
            "\n\nGrounding rules: answer the exact interview question in 1-3 natural sentences; "
            "match the question language; do not mix Chinese into an English answer; "
            "do not introduce unsupported concrete names, family details, places, ages, "
            "faith, languages, pets, medical facts, awards, or major life events. "
            "If evidence does not support a detail, stay character-specific but more general."
        )
    elif prompt_style == "voice":
        extra_rules = (
            "\n\nInterview answer rules: answer in the same language as the question; "
            "if the question is English, use English only. Speak naturally in the character's voice, "
            "using concrete persona or memory details only when they are supported. "
            "Answer directly and finish with a complete sentence. Do not mention prompts, persona, "
            "retrieved memories, or evaluation."
        )
    elif prompt_style == "concise":
        extra_rules = (
            "\n\nConcise grounding rules: answer the exact interview question in 1-3 sentences, "
            "usually under 90 words. If the question asks for a fixed format such as three words, "
            "obey that format exactly. Use concrete persona or memory details only when supported; "
            "prefer the character's known occupation, habits, relationships, values, or style over "
            "a new anecdote. For English questions use English only. Finish with a complete sentence "
            "and do not mention prompts, persona, retrieved memories, or evaluation."
        )
    if mode == "persona":
        system = (
            f"You are role-playing as {character_id}. Answer in first person, stay consistent "
            "with the persona, and do not mention that you are an AI."
            f"{extra_rules}\n\nPersona:\n{persona}"
        )
    else:
        system = (
            f"You are role-playing as {character_id}. Answer in first person, stay consistent "
            "with the persona and the retrieved memories. Use the memories only when relevant; "
            "do not invent facts that conflict with them, and do not mention retrieval."
            f"{extra_rules}\n\nPersona:\n{persona}\n\nRetrieved memories:\n{format_memories(memories or [])}"
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Interview question: {question}"},
    ]


def prepare_prompt_input(
    args: argparse.Namespace,
    data_root: Path,
    mode: str,
    sample: dict[str, Any],
    embedding_model: Any | None,
) -> dict[str, Any]:
    sample_started = time.time()
    character_id = sample["character_id"]
    question = sample["question"]
    persona = None if mode == "question" else load_persona(data_root, character_id)
    memories: list[dict[str, Any]] = []

    retrieval_elapsed = 0.0
    if mode == "memory":
        if embedding_model is None:
            raise ValueError("memory mode requires an embedding model")
        retrieval_started = time.time()
        memories = retrieve_character_memories(
            data_root=data_root,
            cache_dir=Path(args.output_dir) / "memory_cache",
            embedding_model=embedding_model,
            character_id=character_id,
            question=question,
            args=args,
        )
        retrieval_elapsed = time.time() - retrieval_started

    return {
        "sample": sample,
        "character_id": character_id,
        "question": question,
        "messages": build_messages(mode, character_id, question, persona, memories, args.prompt_style),
        "memories": memories,
        "retrieval_latency_s": retrieval_elapsed,
        "started_at": sample_started,
    }


def request_prediction(
    args: argparse.Namespace, messages: list[dict[str, str]]
) -> tuple[str, float, float]:
    chat_started = time.time()
    prediction = call_chat_completion(
        base_url=args.api_base,
        model=args.served_model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        timeout=args.timeout,
    )
    return prediction, time.time() - chat_started, time.time()


def write_prediction_records(
    pred_f: Any,
    detail_f: Any,
    mode: str,
    prepared: dict[str, Any],
    prediction: str,
    latency_s: float,
    chat_latency_s: float,
) -> None:
    sample = prepared["sample"]
    memories = prepared["memories"]
    pred_record = {
        "id": sample["id"],
        "character_id": prepared["character_id"],
        "prediction": prediction,
    }
    detail_record = {
        **pred_record,
        "question": prepared["question"],
        "mode": mode,
        "latency_s": latency_s,
        "chat_latency_s": chat_latency_s,
        "retrieval_latency_s": prepared["retrieval_latency_s"],
        "retrieved_memory_ids": [memory["memory_id"] for memory in memories],
        "retrieved_memories": memories,
    }
    write_jsonl_record(pred_f, pred_record)
    write_jsonl_record(detail_f, detail_record)


def run_prompt_setting(
    args: argparse.Namespace,
    mode: str,
    samples: list[dict[str, Any]],
    embedding_model: Any | None,
) -> None:
    data_root = Path(args.data_root)
    output_name = args.output_name or mode
    mode_output_dir = Path(args.output_dir) / output_name
    mode_output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = mode_output_dir / "predictions.jsonl"
    detail_path = mode_output_dir / "details.jsonl"
    metrics_path = mode_output_dir / "metrics.json"

    latencies = []
    chat_latencies = []
    retrieval_latencies = []
    concurrency = max(1, args.concurrency)
    started_at = time.time()

    with prediction_path.open("w", encoding="utf-8") as pred_f, detail_path.open(
        "w", encoding="utf-8"
    ) as detail_f, ThreadPoolExecutor(max_workers=concurrency) as executor, tqdm(
        total=len(samples), desc=mode, unit="sample"
    ) as progress:
        for start in range(0, len(samples), concurrency):
            batch = samples[start : start + concurrency]
            prepared_batch = [
                prepare_prompt_input(args, data_root, mode, sample, embedding_model)
                for sample in batch
            ]
            futures = [
                executor.submit(request_prediction, args, prepared["messages"])
                for prepared in prepared_batch
            ]

            for prepared, future in zip(prepared_batch, futures):
                prediction, chat_elapsed, completed_at = future.result()
                latency = completed_at - prepared["started_at"]
                latencies.append(latency)
                chat_latencies.append(chat_elapsed)
                if mode == "memory":
                    retrieval_latencies.append(prepared["retrieval_latency_s"])

                write_prediction_records(
                    pred_f=pred_f,
                    detail_f=detail_f,
                    mode=mode,
                    prepared=prepared,
                    prediction=prediction,
                    latency_s=latency,
                    chat_latency_s=chat_elapsed,
                )
                progress.update(1)

    metrics = {
        "mode": mode,
        "num_samples": len(samples),
        "concurrency": concurrency,
        "total_latency_s": time.time() - started_at,
        "avg_latency_s": float(np.mean(latencies)) if latencies else math.nan,
        "p50_latency_s": float(np.percentile(latencies, 50)) if latencies else math.nan,
        "p95_latency_s": float(np.percentile(latencies, 95)) if latencies else math.nan,
        "avg_chat_latency_s": float(np.mean(chat_latencies)) if chat_latencies else math.nan,
        "avg_retrieval_latency_s": float(np.mean(retrieval_latencies))
        if retrieval_latencies
        else 0.0,
        "prediction_file": str(prediction_path),
        "detail_file": str(detail_path),
        "prompt_style": args.prompt_style,
        "memory_layout": args.memory_layout,
    }
    write_json(metrics_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lab5 Task1 memory-augmented role agent.")
    parser.add_argument("--data-root", default="lab5/public_pack/data")
    parser.add_argument("--output-dir", default="lab5/outputs/task1")
    parser.add_argument("--output-name", default="")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--served-model", default="Qwen2.5-1.5B-Instruct")
    parser.add_argument("--embedding-model", default="models/Qwen3-Embedding-0.6B")
    parser.add_argument("--modes", default="question,persona,memory")
    parser.add_argument("--prompt-style", choices=("default", "grounded", "voice", "concise"), default="default")
    parser.add_argument("--memory-layout", choices=("default", "dedup"), default="default")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--character", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-gpu-memory-utilization", type=float, default=0.20)
    parser.add_argument("--embedding-max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    samples = read_jsonl(data_root / "role_interview_eval.jsonl")
    if args.character:
        samples = [sample for sample in samples if sample["character_id"] == args.character]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    invalid_modes = set(modes) - {"question", "persona", "memory"}
    if invalid_modes:
        raise ValueError(f"Unsupported modes: {sorted(invalid_modes)}")
    if args.output_name and len(modes) != 1:
        raise ValueError("--output-name can only be used with exactly one mode")

    embedding_model = None
    if "memory" in modes:
        from vllm import LLM

        embedding_model = LLM(
            model=str(Path(args.embedding_model)),
            gpu_memory_utilization=args.embedding_gpu_memory_utilization,
            max_model_len=args.embedding_max_model_len,
        )

    for mode in modes:
        run_prompt_setting(args, mode, samples, embedding_model)


if __name__ == "__main__":
    main()
