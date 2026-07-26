from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path, skip_invalid: bool = False) -> list[dict[str, Any]]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                if not skip_invalid:
                    raise
    return samples


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            write_jsonl_record(f, record, flush=False)


def write_jsonl_record(f: Any, record: dict[str, Any], flush: bool = True) -> None:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if flush:
        f.flush()


def records_by_field(path: Path, field: str, required_field: str | None = None) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path, skip_invalid=True):
        value = row.get(field)
        if value is None:
            continue
        if required_field is not None and row.get(required_field) is None:
            continue
        records[str(value)] = row
    return records


def records_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return records_by_field(path, "id")


def ordered_records_for_samples(
    samples: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    sample_field: str = "id",
    ok_only: bool = False,
    output_fields: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        record = records.get(str(sample[sample_field]))
        if not record or (ok_only and record.get("status") != "ok"):
            continue
        if output_fields is None:
            rows.append(record)
        else:
            rows.append({field: record.get(field, sample.get(field, "")) for field in output_fields})
    return rows


def load_persona(data_root: Path, character_id: str) -> str:
    path = data_root / "individual_simulation_data" / "characters" / f"wiki_{character_id}.txt"
    return path.read_text(encoding="utf-8").strip()


def call_chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
    api_key: str = "EMPTY",
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    extra_body: dict[str, Any] | None = None,
) -> str:
    """Call an OpenAI-compatible chat completion endpoint."""
    from openai import OpenAI

    request_args: dict[str, Any] = {"model": model, "messages": messages}
    if max_tokens is not None:
        request_args["max_tokens"] = max_tokens
    if temperature is not None:
        request_args["temperature"] = temperature
    if top_p is not None:
        request_args["top_p"] = top_p

    body = dict(extra_body or {})
    if repetition_penalty is not None:
        body["repetition_penalty"] = repetition_penalty
    if body:
        request_args["extra_body"] = body

    client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"), timeout=timeout)
    completion = client.chat.completions.create(**request_args)
    content = completion.choices[0].message.content
    return (content or "").strip()


DASHSCOPE_JUDGE_MODEL = "deepseek-v4-flash"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")


CJK_RE = re.compile(r"[\u3400-\u9fff]")


def contains_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator for a role-playing individual simulation benchmark.
Judge whether each candidate answer is consistent with the target character. Use only the
provided evidence. Do not reward generic helpfulness unless it also fits the character.

Score each dimension from 1 to 5:
- memorisation: correct use of character background, experiences, relationships, and memories.
- values: alignment with character values, preferences, motivations, and behavior tendencies.
- personality: consistency with character personality, voice, emotional tone, and speaking style.
- hallucination: avoidance of contradictions or unsupported important claims.
- answer_quality: directness, coherence, completeness, and first-person interview quality.

Rules:
- Penalize answers that say "as an AI" or refuse role-playing unless the character would say that.
- Penalize answers that mention prompts, persona, retrieved memories, or evaluation internals.
- If two answers are close, prefer the one grounded more clearly in character evidence.
- Return only valid JSON. Do not include markdown fences or extra commentary.
- Escape double quotes inside JSON string values.
- The winner value must be exactly one of the candidate answer names or tie; never include brackets.
"""


JUDGE_JSON_SCHEMA = """{
  "scores": {
    "<answer_name>": {
      "memorisation": 1,
      "values": 1,
      "personality": 1,
      "hallucination": 1,
      "answer_quality": 1,
      "overall": 1
    }
  },
  "winner": "<answer_name>|tie",
  "memory_helped": true,
  "memory_helped_reason": "...",
  "memory_value": {
    "is_relevant": true,
    "reason": "..."
  },
  "reason": "..."
}"""


def format_judge_memories(memories: list[dict[str, Any]]) -> str:
    """Format retrieved memories as evidence for the judge prompt."""
    if not memories:
        return "No retrieved memories were provided."

    lines = []
    for rank, memory in enumerate(memories, start=1):
        score = memory.get("score")
        score_text = f" score={score:.4f}" if isinstance(score, int | float) else ""
        lines.append(f"{rank}. [{memory.get('memory_id', 'unknown')}]{score_text}\n{memory.get('text', '')}")
    return "\n\n".join(lines)


def build_role_judge_messages(
    character_id: str,
    question: str,
    persona: str,
    answers: dict[str, str],
    retrieved_memories: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Build reusable LLM-as-a-Judge messages for role simulation outputs.

    Args:
        character_id: Target character name.
        question: Interview question.
        persona: Character persona text.
        answers: Mapping from run name to candidate answer.
        retrieved_memories: Optional memories used by memory-augmented systems.

    Returns:
        Chat messages that ask the judge model to score all candidate answers.
    """
    answer_blocks = []
    for name, answer in answers.items():
        answer_blocks.append(f"[{name}]\n{answer}")

    user_prompt = f"""Target character: {character_id}

[PERSONA]
{persona}

[RETRIEVED MEMORIES]
{format_judge_memories(retrieved_memories or [])}

Interview question:
{question}

Candidate answers:

{chr(10).join(answer_blocks)}

Return JSON using this schema. Keep the same answer names from Candidate answers:
{JUDGE_JSON_SCHEMA}
"""
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from a model response that should contain only JSON."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])
