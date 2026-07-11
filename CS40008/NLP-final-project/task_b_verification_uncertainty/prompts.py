"""Prompt templates and parsers for Task B verification experiments."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from task_a_robustness.data import ComveExample
from task_a_robustness.prompts import parse_label


TASK_A_OUTPUT_CONTRACT = (
    "Output contract: answer with exactly one digit, either 0 or 1. "
    "Do not include explanation or extra text."
)


@dataclass(frozen=True)
class PromptTemplate:
    """A named Task B prompt template."""

    name: str
    template: str

    def render(self, example: ComveExample) -> str:
        """Render the prompt for one ComVE example."""

        return self.template.format(sent0=example.sent0, sent1=example.sent1)


DIRECT_TEMPLATE = PromptTemplate(
    name="direct",
    template=(
        "You are solving a commonsense validation task. Two sentences are given. "
        "Exactly one sentence is against commonsense.\n"
        "Return only 0 if Sentence 0 is against commonsense, or 1 if Sentence 1 is against commonsense."
        "\n\nSentence 0: {sent0}\nSentence 1: {sent1}\nLabel:\n\n"
        f"{TASK_A_OUTPUT_CONTRACT}"
    ),
)

CONSTRAINT_TEMPLATE = PromptTemplate(
    name="constraint_first",
    template=(
        "You are solving a commonsense validation task. Two sentences are given. "
        "Exactly one sentence is against commonsense.\n"
        "First identify the commonsense constraint needed to judge the pair, then check "
        "each sentence against that constraint internally.\n"
        "Return only 0 if Sentence 0 is against commonsense, or 1 if Sentence 1 is against commonsense."
        "\n\nSentence 0: {sent0}\nSentence 1: {sent1}\nLabel:\n\n"
        f"{TASK_A_OUTPUT_CONTRACT}"
    ),
)

CANDIDATE_TEMPLATE = PromptTemplate(
    name="candidate",
    template=(
        "Generate one candidate answer for this commonsense validation task. Explain the "
        "commonsense rule, then choose the sentence that violates it. Return a JSON object "
        "with keys label and reason. The label must be 0 or 1.\n\n"
        "Sentence 0: {sent0}\nSentence 1: {sent1}\nJSON:"
    ),
)


def verifier_prompt(sent0: str, sent1: str, candidate_label: int, candidate_reason: str) -> str:
    """Build a verifier prompt for one generated candidate."""

    return (
        "You are a strict commonsense verifier. Given two sentences and a proposed answer, "
        "judge whether the explanation correctly supports the label. Return a JSON object "
        "with keys score, final_label, and reason. Score is an integer from 1 to 5, where "
        "5 means the explanation is correct and strongly supports the label.\n\n"
        f"Sentence 0: {sent0}\n"
        f"Sentence 1: {sent1}\n"
        f"Proposed label: {candidate_label}\n"
        f"Proposed reason: {candidate_reason}\n"
        "JSON:"
    )


def extract_response_text(payload: dict[str, object]) -> str:
    """Extract a response string from common batch-output field names."""

    response = payload.get("response") or payload.get("output_text") or payload.get("text")
    if response is None:
        raise ValueError("payload has no response/output_text/text field")
    return str(response)


def parse_reason(text: str) -> str:
    """Extract a short reason from JSON-like or free-form LLM text."""

    if text.strip() in {"0", "1"}:
        return ""
    parsed = _try_json_object(text)
    if parsed:
        for key in ("reason", "explanation", "rationale"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    match = re.search(r"(?:reason|explanation|rationale)\s*[:：]\s*(.+)", text, re.I | re.S)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_score(text: str) -> int | None:
    """Extract a verifier score in [1, 5]."""

    parsed = _try_json_object(text)
    if parsed:
        value = parsed.get("score")
        if isinstance(value, int) and 1 <= value <= 5:
            return value
        if isinstance(value, str) and value.strip().isdigit():
            score = int(value.strip())
            if 1 <= score <= 5:
                return score
    match = re.search(r"score\s*[:：]?\s*([1-5])\b", text, re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"\b([1-5])\s*/\s*5\b", text)
    if match:
        return int(match.group(1))
    return None


def parse_final_label(text: str) -> int | None:
    """Extract final_label if present, otherwise fall back to generic label parsing."""

    parsed = _try_json_object(text)
    if parsed:
        for key in ("final_label", "label", "answer"):
            value = parsed.get(key)
            if value in (0, 1):
                return int(value)
            if isinstance(value, str) and value.strip() in {"0", "1"}:
                return int(value.strip())
    return parse_label(text)


def write_prompt_batch(
    examples: list[ComveExample],
    output_path: Path,
    template: PromptTemplate,
    method_prefix: str,
    repeat: int = 1,
) -> None:
    """Write JSONL prompts for direct, constraint-first, or candidate generation."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for example in examples:
            for sample_index in range(repeat):
                payload = {
                    "id": example.id,
                    "method": f"{method_prefix}:{template.name}",
                    "sample_index": sample_index,
                    "gold": example.gold,
                    "sent0": example.sent0,
                    "sent1": example.sent1,
                    "prompt": template.render(example),
                }
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _try_json_object(text: str) -> dict[str, object] | None:
    """Parse the first JSON object found in text, returning None on failure."""

    stripped = text.strip()
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.insert(0, stripped[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
