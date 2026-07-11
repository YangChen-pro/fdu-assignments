"""Prompt templates and batch builders for LLM Task A experiments."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .data import ComveExample


@dataclass(frozen=True)
class PromptTemplate:
    """A named prompt template for ComVE Task A."""

    name: str
    template: str

    def render(self, example: ComveExample, demonstrations: list[ComveExample] | None = None) -> str:
        """Render the prompt with optional few-shot demonstrations."""

        demo_text = ""
        if demonstrations:
            demo_text = "\n\nExamples:\n" + "\n".join(format_demo(item) for item in demonstrations)
        return self.template.format(sent0=example.sent0, sent1=example.sent1, demos=demo_text)


ZERO_SHOT_TEMPLATES = [
    PromptTemplate(
        name="direct",
        template=(
            "You are solving a commonsense validation task. Two sentences are given. "
            "Exactly one sentence is against commonsense.\n"
            "Return only 0 if Sentence 0 is against commonsense, or 1 if Sentence 1 is against commonsense."
            "{demos}\n\nSentence 0: {sent0}\nSentence 1: {sent1}\nLabel:"
        ),
    ),
    PromptTemplate(
        name="judge",
        template=(
            "Decide which of the following two statements is less plausible in the real world. "
            "Answer with a single digit: 0 or 1."
            "{demos}\n\n0. {sent0}\n1. {sent1}\nAnswer:"
        ),
    ),
    PromptTemplate(
        name="minimal",
        template=(
            "Commonsense check. Pick the impossible or nonsensical sentence. "
            "Output only its index, 0 or 1."
            "{demos}\n\n[0] {sent0}\n[1] {sent1}\nIndex:"
        ),
    ),
]


def format_demo(example: ComveExample) -> str:
    """Format one demonstration example."""

    return f"Sentence 0: {example.sent0}\nSentence 1: {example.sent1}\nLabel: {example.gold}"


def parse_label(text: str) -> int | None:
    """Extract a binary label from an LLM response."""

    stripped = text.strip()
    if stripped in {"0", "1"}:
        return int(stripped)
    match = re.search(r"(?:label|answer|index)\s*[:：]?\s*([01])\b", stripped, re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"\b([01])\b", stripped)
    return int(match.group(1)) if match else None


def write_prompt_batch(
    examples: list[ComveExample],
    output_path: Path,
    templates: list[PromptTemplate] | None = None,
    demonstrations: list[ComveExample] | None = None,
    method_prefix: str = "llm_zero_shot",
    repeat: int = 1,
) -> None:
    """Write JSONL prompts for later API or manual LLM runs."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_templates = templates or ZERO_SHOT_TEMPLATES
    with output_path.open("w", encoding="utf-8") as file:
        for example in examples:
            for template in prompt_templates:
                for sample_index in range(repeat):
                    payload = {
                        "id": example.id,
                        "method": f"{method_prefix}:{template.name}",
                        "sample_index": sample_index,
                        "gold": example.gold,
                        "sent0": example.sent0,
                        "sent1": example.sent1,
                        "prompt": template.render(example, demonstrations),
                    }
                    file.write(json.dumps(payload, ensure_ascii=False) + "\n")
