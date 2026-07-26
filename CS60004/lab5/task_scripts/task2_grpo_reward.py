from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from openai import OpenAI
from swift.rewards.orm import ORM, orms


DEFAULT_BASE_URL = "http://127.0.0.1:8318/v1"
DEFAULT_MODEL = "Qwen3.5-35B-A3B"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_TIMEOUT = 300
DEFAULT_MAX_TOKENS = 1024
DEFAULT_CONCURRENCY = 16
DEFAULT_RETRIES = 2

SYSTEM_PROMPT = """You are an impartial reward model for a role-playing individual simulation task.
Score the candidate answer using the target persona, the full prompt evidence, retrieved memories inside the prompt, and a non-binding reference answer.
Reward answers that answer directly, stay in first person, preserve the target character's occupation/domain, values, relationships, and voice, and use only evidence-grounded details.
Do not require copying the reference answer; valid alternative wording should score well if it is faithful to the persona and prompt evidence.
Penalize generic self-help language, role/domain drift, unsupported names, locations, events, abilities, languages, relationships, faith/family details, meta AI disclaimers, excessive verbosity, evasive answers, and language mixing.
If the user prompt is in English, Chinese/CJK text in the answer is a serious stability error unless explicitly justified by the character evidence.
A vivid answer with invented specifics should score lower than a plainer answer that is faithful to the persona and retrieved memories.
Return only valid JSON without markdown.
"""

USER_TEMPLATE = """Target character: {character_id}

[PERSONA]
{persona}

[PROMPT]
{prompt}

[REFERENCE ANSWER - useful example, not a gold string to copy]
{reference_answer}

[CANDIDATE ANSWER]
{candidate_answer}

Score each dimension from 1 to 5:
- memorisation: correct use of character background, occupation/domain, relationships, memories, and prompt evidence.
- values: alignment with the character's values, motivations, preferences, and behavior tendencies.
- personality: matches the character's voice, emotional tone, humor level, and speaking style.
- hallucination: avoids contradictions, unsupported important claims, role/domain drift, and meta AI disclaimers.
- answer_quality: directly answers the interview question, is coherent, complete, concise, and natural.

Hard caps for overall:
- overall <= 2 if the answer's main scenario drifts away from the character's established occupation/domain.
- overall <= 2 if it adds unsupported concrete personal facts, names, relationships, locations, faith/family details, or events that matter to the answer.
- overall <= 2 if it contains Chinese/CJK text for an English prompt without clear support from the evidence.
- overall <= 3 if it is mostly generic advice or self-reflection that could fit many characters.
- overall <= 4 if it is character-consistent but less specific than the persona, retrieved memories, or reference answer support.

Set overall to the holistic score after applying the hard caps. Prefer concise, first-person, character-specific, evidence-faithful answers over verbose or vivid answers that add unsupported facts.

Return this JSON schema exactly:
{{
  "memorisation": 1,
  "values": 1,
  "personality": 1,
  "hallucination": 1,
  "answer_quality": 1,
  "overall": 1,
  "reason": "..."
}}
"""



def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _parse_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _extract_score(result: dict[str, Any], key: str, default: float = 3.0) -> float:
    try:
        value = float(result.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(1.0, min(5.0, value))


def _has_cjk(text: str) -> bool:
    return re.search(r"[\u3400-\u9fff]", text) is not None


def _looks_english_prompt(text: str) -> bool:
    letters = sum(ch.isalpha() and ch.isascii() for ch in text)
    return letters >= 20 and not _has_cjk(text[:1200])


def _score_to_reward(result: dict[str, Any], prompt: str, completion: str) -> float:
    weights = {
        "overall": 0.45,
        "memorisation": 0.20,
        "hallucination": 0.15,
        "values": 0.10,
        "personality": 0.05,
        "answer_quality": 0.05,
    }
    score = sum(_extract_score(result, key) * weight for key, weight in weights.items())
    lower = completion.lower()
    word_count = len(completion.split())

    if not completion.strip() or word_count < 3:
        score = min(score, 2.0)
    if any(phrase in lower for phrase in ("as an ai", "language model", "i don't have personal")):
        score = min(score, 2.0)
    if _looks_english_prompt(prompt) and _has_cjk(completion):
        score = min(score, 2.0)
    if word_count > 180:
        score = min(score, 3.5)

    return (score - 3.0) / 2.0

def _load_persona(character_id: str) -> str:
    data_root = Path(os.getenv("LAB5_DATA_ROOT", "lab5/public_pack/data"))
    path = data_root / "individual_simulation_data" / "characters" / f"wiki_{character_id}.txt"
    return path.read_text(encoding="utf-8").strip()


def _prompt_text(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"[{item.get('role', 'unknown')}]\n{item.get('content', '')}" for item in messages)


def _judge_one(client: OpenAI, item: dict[str, Any], completion: str) -> float:
    messages = item.get("messages", [])
    character_id = item.get("character_id") or "Unknown"
    user_prompt = USER_TEMPLATE.format(
        character_id=character_id,
        persona=_load_persona(character_id),
        prompt=_prompt_text(messages),
        reference_answer=item.get("reference_answer") or item.get("solution") or "",
        candidate_answer=completion,
    )
    request_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    raw = ""
    for attempt in range(_env_int("GRPO_JUDGE_MAX_RETRIES", DEFAULT_RETRIES) + 1):
        try:
            response = client.chat.completions.create(
                model=os.getenv("GRPO_JUDGE_MODEL", DEFAULT_MODEL),
                messages=request_messages,
                max_tokens=_env_int("GRPO_JUDGE_MAX_TOKENS", DEFAULT_MAX_TOKENS),
            )
            raw = (response.choices[0].message.content or "").strip()
            result = _parse_json(raw)
            return _score_to_reward(result, user_prompt, completion)
        except Exception:
            if attempt >= _env_int("GRPO_JUDGE_MAX_RETRIES", DEFAULT_RETRIES):
                break
            time.sleep(1.0 + attempt)
    return 0.0


class LLMJudgeReward(ORM):
    """External LLM-as-a-Judge reward for Lab5 Task2 GRPO."""

    def __init__(self, args: Any | None = None, **kwargs: Any) -> None:
        super().__init__(args, **kwargs)
        self.client = OpenAI(
            api_key=os.getenv("GRPO_JUDGE_API_KEY", DEFAULT_API_KEY),
            base_url=os.getenv("GRPO_JUDGE_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            timeout=_env_int("GRPO_JUDGE_TIMEOUT", DEFAULT_TIMEOUT),
        )
        self.concurrency = _env_int("GRPO_JUDGE_CONCURRENCY", DEFAULT_CONCURRENCY)

    def __call__(self, completions: list[str], **kwargs: Any) -> list[float]:
        rows = _rows_from_kwargs(len(completions), kwargs)
        with ThreadPoolExecutor(max_workers=max(1, self.concurrency)) as executor:
            futures = [executor.submit(_judge_one, self.client, row, completion) for row, completion in zip(rows, completions)]
            return [future.result() for future in futures]


def _rows_from_kwargs(count: int, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        row = {}
        for key, value in kwargs.items():
            if key == "trainer_state":
                continue
            if isinstance(value, list) and len(value) == count:
                row[key] = value[idx]
        rows.append(row)
    return rows


_PUBLIC_CHECKS = {
    "validate_lowercase": ("change_case:english_lowercase", lambda gt: {}),
    "validate_uppercase": ("change_case:english_capital", lambda gt: {}),
    "validate_no_commas": ("punctuation:no_comma", lambda gt: {}),
    "validate_json_format": ("detectable_format:json_format", lambda gt: {}),
    "validate_quotation": ("startend:quotation", lambda gt: {}),
    "validate_title": ("detectable_format:title", lambda gt: {}),
    "validate_two_responses": ("combination:two_responses", lambda gt: {}),
    "verify_keywords": ("keywords:existence", lambda gt: {"keywords": gt.get("keyword_list") or []}),
    "validate_forbidden_words": ("keywords:forbidden_words", lambda gt: {"forbidden_words": gt.get("forbidden_words") or []}),
    "validate_placeholders": ("detectable_content:number_placeholders", lambda gt: {"num_placeholders": gt.get("N")}),
    "verify_bullet_points": ("detectable_format:number_bullet_lists", lambda gt: {"num_bullets": gt.get("N")}),
    "validate_highlighted_sections": ("detectable_format:number_highlighted_sections", lambda gt: {"num_highlights": gt.get("N")}),
    "validate_sections": ("detectable_format:multiple_sections", lambda gt: {"section_spliter": gt.get("section_splitter"), "num_sections": gt.get("N")}),
    "verify_paragraph_count": ("length_constraints:number_paragraphs", lambda gt: {"num_paragraphs": gt.get("N")}),
    "validate_paragraphs": ("length_constraints:nth_paragraph_first_word", lambda gt: {"num_paragraphs": gt.get("N"), "nth_paragraph": gt.get("i"), "first_word": gt.get("first_word")}),
    "verify_postscript": ("detectable_content:postscript", lambda gt: {"postscript_marker": gt.get("postscript_marker")}),
    "validate_end": ("startend:end_checker", lambda gt: {"end_phrase": gt.get("end_phrase")}),
    "validate_repeat_prompt": ("combination:repeat_prompt", lambda gt: {"prompt_to_repeat": gt.get("original_prompt")}),
}


def _instruction_eval_modules():
    code_root = Path(__file__).resolve().parents[1] / "public_pack" / "code"
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from instruction_eval import instructions_registry, instructions_util

    return instructions_registry, instructions_util


def _ground_truths(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _public_instruction_passed(instruction_id: str, kwargs: dict[str, Any], completion: str) -> bool:
    registry, _ = _instruction_eval_modules()
    instruction = registry.INSTRUCTION_DICT[instruction_id](instruction_id)
    instruction.build_description(**kwargs)
    return bool(completion.strip() and instruction.check_following(completion))


def _ratio_score(actual: int, target: Any, relation: str | None = None) -> float:
    if target is None:
        return 0.0
    target = max(1, int(target))
    if relation == "at least":
        return min(1.0, actual / target)
    if relation == "at most":
        return 1.0 if actual <= target else max(0.0, 1.0 - (actual - target) / target)
    if relation == "around":
        return max(0.0, 1.0 - abs(actual - target) / target)
    return 1.0 if actual == target else max(0.0, 1.0 - abs(actual - target) / target)


def _choice_passed(options: str, completion: str) -> bool:
    lower = completion.strip().lower().strip(' \"\'.,;:')
    if options == "yes/no/maybe":
        choices = ["yes", "no", "maybe"]
    elif options == "I know or I don't know":
        choices = ["i know", "i don't know"]
    elif options == "a), b), c), d)":
        choices = ["a)", "b)", "c)", "d)"]
    else:
        choices = [item.strip().lower() for item in re.split(r"[,/]", options or "") if item.strip()]
    return any(lower == choice or lower.startswith(choice + " ") for choice in choices)


def _score_single_ifeval_rule(gt: dict[str, Any], completion: str) -> float:
    func = gt.get("func_name")
    try:
        if func in _PUBLIC_CHECKS:
            instruction_id, kwargs_fn = _PUBLIC_CHECKS[func]
            score = 1.0 if _public_instruction_passed(instruction_id, kwargs_fn(gt), completion) else 0.0
        elif func == "verify_keyword_frequency":
            count = len(re.findall(re.escape(str(gt.get("word") or "")), completion, flags=re.IGNORECASE))
            score = _ratio_score(count, gt.get("N"))
        elif func == "verify_letter_frequency":
            count = completion.lower().count(str(gt.get("letter") or "").lower())
            score = _ratio_score(count, gt.get("N"))
        elif func == "verify_sentence_constraint":
            _, util = _instruction_eval_modules()
            score = _ratio_score(util.count_sentences(completion), gt.get("N"), gt.get("quantifier"))
        elif func == "validate_word_constraint":
            _, util = _instruction_eval_modules()
            score = _ratio_score(util.count_words(completion), gt.get("N"), gt.get("quantifier"))
        elif func == "validate_frequency_capital_words":
            count = len(re.findall(r"\b[A-Z]+\b", completion))
            score = _ratio_score(count, gt.get("N"), gt.get("quantifier"))
        elif func == "validate_choice":
            score = 1.0 if _choice_passed(str(gt.get("options") or ""), completion) else 0.0
        else:
            return 0.0
        return 2.0 * max(0.0, min(1.0, score)) - 1.0
    except Exception:
        return 0.0


def _score_ifeval_rule(ground_truth: Any, completion: str) -> float:
    rewards = [_score_single_ifeval_rule(gt, completion) for gt in _ground_truths(ground_truth)]
    if not rewards:
        return 0.0
    if len(rewards) == 1:
        return rewards[0]
    pass_scores = [(reward + 1.0) / 2.0 for reward in rewards]
    strict_score = 0.7 * min(pass_scores) + 0.3 * (sum(pass_scores) / len(pass_scores))
    return 2.0 * strict_score - 1.0


class IFEvalRuleReward(ORM):
    """Local rule reward for RLVR-IFeval instruction-following constraints."""

    def __call__(self, completions: list[str], **kwargs: Any) -> list[float]:
        rows = _rows_from_kwargs(len(completions), kwargs)
        return [_score_ifeval_rule(row.get("ground_truth"), completion) for row, completion in zip(rows, completions)]


class Lab5HybridReward(ORM):
    """Route Lab5 mixed GRPO rows to the matching reward without changing inference."""

    def __init__(self, args: Any | None = None, **kwargs: Any) -> None:
        super().__init__(args, **kwargs)
        self.client = OpenAI(
            api_key=os.getenv("GRPO_JUDGE_API_KEY", DEFAULT_API_KEY),
            base_url=os.getenv("GRPO_JUDGE_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            timeout=_env_int("GRPO_JUDGE_TIMEOUT", DEFAULT_TIMEOUT),
        )
        self.concurrency = _env_int("GRPO_JUDGE_CONCURRENCY", DEFAULT_CONCURRENCY)

    def __call__(self, completions: list[str], **kwargs: Any) -> list[float]:
        rows = _rows_from_kwargs(len(completions), kwargs)
        rewards: list[float | None] = [None] * len(completions)
        role_jobs: list[tuple[int, dict[str, Any], str]] = []
        for idx, (row, completion) in enumerate(zip(rows, completions)):
            if row.get("ground_truth") is not None:
                rewards[idx] = _score_ifeval_rule(row.get("ground_truth"), completion)
            elif row.get("character_id"):
                role_jobs.append((idx, row, completion))
            else:
                rewards[idx] = 0.0

        with ThreadPoolExecutor(max_workers=max(1, self.concurrency)) as executor:
            futures = [executor.submit(_judge_one, self.client, row, completion) for _, row, completion in role_jobs]
            for (idx, _, _), future in zip(role_jobs, futures):
                rewards[idx] = future.result()
        return [float(reward) if reward is not None else 0.0 for reward in rewards]


orms["llm_judge"] = LLMJudgeReward
orms["ifeval_rule"] = IFEvalRuleReward
orms["lab5_hybrid"] = Lab5HybridReward
