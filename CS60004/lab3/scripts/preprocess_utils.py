import ast
import json
import math
import operator
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable


PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. You can use\n"
    "basic arithmetic operations (+, -, *, /) and each number can only be used\n"
    "once. Show your work in <think> </think> tags. Return the final answer in\n"
    "<answer> </answer> tags."
)

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
SAFE_EXPR_RE = re.compile(r"^[\d\s+\-*/().]+$")


@dataclass
class EvalResult:
    format_ok: bool
    expr_valid: bool
    correct: bool
    reason: str
    answer: str = ""
    value: Any = None
    output_tokens: int = 0


def normalize_sample(raw, index=None):
    numbers = raw.get("numbers", raw.get("nums"))
    if numbers is None:
        raise ValueError(f"sample missing numbers/nums: {raw}")
    target = raw.get("target")
    if target is None:
        raise ValueError(f"sample missing target: {raw}")
    sample_id = raw.get("id", index)
    return {
        "id": sample_id,
        "numbers": [int(x) for x in numbers],
        "target": int(target),
    }


def build_prompt(sample, template=PROMPT_TEMPLATE):
    return template.format(numbers=sample["numbers"], target=sample["target"])


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_jsonl(path):
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_countdown_data(path):
    path = Path(path)
    if path.suffix == ".json":
        raw_rows = load_json(path)
    elif path.suffix == ".jsonl":
        raw_rows = load_jsonl(path)
    elif path.suffix == ".parquet":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("reading parquet requires `datasets`; install it in the lab3 environment") from exc
        raw_rows = list(load_dataset("parquet", data_files=str(path), split="train"))
    else:
        raise ValueError(f"unsupported data file: {path}")
    return [normalize_sample(row, index) for index, row in enumerate(raw_rows)]


def split_samples(
    samples,
    valid_ratio=0.05,
    seed=42,
    max_samples=None,
):
    rows = list(samples)
    random.Random(seed).shuffle(rows)
    if max_samples is not None:
        rows = rows[:max_samples]
    valid_size = max(1, int(len(rows) * valid_ratio)) if len(rows) > 1 else 0
    return rows[valid_size:], rows[:valid_size]


def extract_answer(text):
    match = ANSWER_RE.search(text or "")
    return match.group(1).strip() if match else ""


def clean_prediction_for_submit(text):
    think_end = (text or "").rfind("</think>")
    if think_end < 0:
        return text
    think_block = text[:think_end + len("</think>")]
    after_think = text[think_end + len("</think>"):]
    cleaned_think = think_block.replace("<answer>", '"answer"').replace("</answer>", '"answer"')
    return cleaned_think + after_think


def extract_think(text):
    match = THINK_RE.search(text or "")
    return match.group(1).strip() if match else ""


def _numbers_from_ast(node):
    values: list[int] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, int):
            values.append(int(child.value))
        elif isinstance(child, ast.Constant) and isinstance(child.value, float):
            raise ValueError("float literal is not allowed")
    return values


def _eval_ast(node):
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return Fraction(int(node.value), 1)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_ast(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            value = left / right
            if value.denominator != 1:
                raise ValueError("division result is not integer")
            return value
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def validate_expression(expr, numbers, target):
    expr = (expr or "").strip()
    if not expr:
        return EvalResult(False, False, False, "empty_answer", expr)
    if not SAFE_EXPR_RE.match(expr):
        return EvalResult(True, False, False, "illegal_character", expr)
    try:
        tree = ast.parse(expr, mode="eval")
        used_numbers = _numbers_from_ast(tree)
        if Counter(used_numbers) != Counter(numbers):
            return EvalResult(True, False, False, "number_usage_error", expr)
        value = _eval_ast(tree)
    except Exception as exc:
        return EvalResult(True, False, False, type(exc).__name__, expr)
    correct = value == target
    return EvalResult(True, True, bool(correct), "ok" if correct else "wrong_result", expr, float(value))


def evaluate_output(text, numbers, target, output_tokens=0):
    answer = extract_answer(text)
    if not answer:
        return EvalResult(False, False, False, "missing_answer", "", None, output_tokens)
    result = validate_expression(answer, numbers, target)
    result.output_tokens = output_tokens or len((text or "").split())
    return result


def summarize_results(results):
    total = len(results)
    if total == 0:
        return {
            "num_samples": 0,
            "format_rate": 0.0,
            "expr_valid_rate": 0.0,
            "accuracy": 0.0,
            "avg_output_tokens": 0.0,
            "reasons": {},
        }
    return {
        "num_samples": total,
        "format_rate": sum(item.format_ok for item in results) / total,
        "expr_valid_rate": sum(item.expr_valid for item in results) / total,
        "accuracy": sum(item.correct for item in results) / total,
        "avg_output_tokens": sum(item.output_tokens for item in results) / total,
        "reasons": dict(Counter(item.reason for item in results)),
    }


def summarize_outputs(rows):
    results = [
        evaluate_output(row["prediction"], row["numbers"], row["target"], row.get("output_tokens", 0))
        for row in rows
    ]
    summary = summarize_results(results)
    summary["success_examples"] = [
        {**rows[i], "eval": asdict(result)} for i, result in enumerate(results) if result.correct
    ][:5]
    summary["failure_examples"] = [
        {**rows[i], "eval": asdict(result)} for i, result in enumerate(results) if not result.correct
    ][:5]
    return summary


def corrupt_answer(chosen, numbers, target, reason="wrong_result"):
    answer = extract_answer(chosen)
    if reason == "missing_answer":
        return "<think>I cannot find a valid equation.</think>"
    if reason == "empty_answer":
        return "<think>The result is unclear.</think>\n<answer> </answer>"
    if reason == "number_usage_error" and numbers:
        return f"<think>I reused one number.</think>\n<answer> {numbers[0]} + {numbers[0]} </answer>"
    if reason == "equation_answer" and numbers:
        return f"<think>I wrote an equation instead of a plain expression.</think>\n<answer> {numbers[0]} = {target} </answer>"
    if reason == "illegal_expression":
        return "<think>I used an invalid symbol.</think>\n<answer> 1 ** 2 </answer>"
    if reason == "missing_muldiv":
        wrong_expr = make_wrong_result_expression(numbers, target, allowed_ops=("+", "-"))
        if wrong_expr:
            return (
                "<think>I avoided multiplication and division, so this stays linear but misses the target.</think>\n"
                f"<answer> {wrong_expr} </answer>"
            )
    if reason == "missing_paren":
        wrong_expr = make_linear_wrong_expression(numbers, target)
        if wrong_expr:
            return (
                "<think>I used a flat expression without grouping, so the value is wrong.</think>\n"
                f"<answer> {wrong_expr} </answer>"
            )
    wrong_expr = make_wrong_result_expression(numbers, target)
    if wrong_expr:
        return f"<think>This expression uses the numbers but reaches the wrong value.</think>\n<answer> {wrong_expr} </answer>"
    if answer:
        return f"<think>This is a common arithmetic mistake.</think>\n<answer> ({answer}) + 1 </answer>"
    return f"<think>I guessed an expression.</think>\n<answer> {numbers[0]} </answer>"


def make_wrong_result_expression(
    numbers,
    target,
    allowed_ops=("+", "-", "*"),
):
    if not numbers:
        return None
    text_numbers = [str(number) for number in numbers]
    candidates: list[str] = []
    if "+" in allowed_ops:
        candidates.append(" + ".join(text_numbers))
    if "-" in allowed_ops:
        candidates.append(" - ".join(text_numbers))
    if "*" in allowed_ops:
        candidates.append(" * ".join(text_numbers))
    if len(text_numbers) >= 3:
        alternating = text_numbers[0]
        for index, number in enumerate(text_numbers[1:], 1):
            alternating += (" + " if index % 2 == 0 else " - ") + number
        candidates.append(alternating)
    for candidate in candidates:
        result = validate_expression(candidate, numbers, target)
        if result.expr_valid and not result.correct:
            return candidate
    return None


def make_linear_wrong_expression(numbers: list[int], target: int):
    import itertools

    if len(numbers) <= 1:
        return None
    for perm in itertools.permutations(numbers):
        for ops in itertools.product(["+", "-", "*", "/"], repeat=len(numbers) - 1):
            expr = str(perm[0])
            for op, number in zip(ops, perm[1:]):
                expr += f" {op} {number}"
            result = validate_expression(expr, numbers, target)
            if result.expr_valid and not result.correct:
                return expr
    return None


def make_preference_row(
    sample,
    chosen,
    rejected,
    source_plan,
    rejected_reason,
):
    return {
        "sample_id": sample["id"],
        "prompt": build_prompt(sample),
        "chosen": chosen,
        "rejected": rejected,
        "numbers": sample["numbers"],
        "target": sample["target"],
        "source_plan": source_plan,
        "rejected_reason": rejected_reason,
    }


def build_countdown_solver(numbers: list[int], target: int):
    states: list[tuple[tuple[Fraction, str], ...]] = [
        tuple((Fraction(number, 1), str(number)) for number in numbers)
    ]
    seen: set[tuple[Fraction, ...]] = set()
    ops = [
        (operator.add, "+"),
        (operator.sub, "-"),
        (operator.mul, "*"),
    ]
    while states:
        state = states.pop()
        key = tuple(sorted(value for value, _ in state))
        if key in seen:
            continue
        seen.add(key)
        if len(state) == 1:
            value, expr = state[0]
            if value == target:
                return expr
            continue
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                first = state[i]
                second = state[j]
                rest = [state[k] for k in range(len(state)) if k not in (i, j)]
                candidates: list[tuple[Fraction, str]] = []
                for func, symbol in ops:
                    candidates.append((func(first[0], second[0]), f"({first[1]}{symbol}{second[1]})"))
                    if symbol == "-":
                        candidates.append((func(second[0], first[0]), f"({second[1]}-{first[1]})"))
                if second[0] != 0:
                    value = first[0] / second[0]
                    if value.denominator == 1:
                        candidates.append((value, f"({first[1]}/{second[1]})"))
                if first[0] != 0:
                    value = second[0] / first[0]
                    if value.denominator == 1:
                        candidates.append((value, f"({second[1]}/{first[1]})"))
                for candidate in candidates:
                    states.append(tuple(rest + [candidate]))
    return None


def build_restricted_solver(
    numbers,
    target,
    allowed_ops=("+", "-", "*", "/"),
):
    states: list[tuple[tuple[Fraction, str], ...]] = [
        tuple((Fraction(number, 1), str(number)) for number in numbers)
    ]
    seen: set[tuple[Fraction, ...]] = set()
    while states:
        state = states.pop()
        key = tuple(sorted(value for value, _ in state))
        if key in seen:
            continue
        seen.add(key)
        if len(state) == 1:
            value, expr = state[0]
            if value == target:
                return expr
            continue
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                first = state[i]
                second = state[j]
                rest = [state[k] for k in range(len(state)) if k not in (i, j)]
                candidates: list[tuple[Fraction, str]] = []
                if "+" in allowed_ops:
                    candidates.append((first[0] + second[0], f"({first[1]}+{second[1]})"))
                if "-" in allowed_ops:
                    candidates.append((first[0] - second[0], f"({first[1]}-{second[1]})"))
                    candidates.append((second[0] - first[0], f"({second[1]}-{first[1]})"))
                if "*" in allowed_ops:
                    candidates.append((first[0] * second[0], f"({first[1]}*{second[1]})"))
                if "/" in allowed_ops and second[0] != 0:
                    value = first[0] / second[0]
                    if value.denominator == 1:
                        candidates.append((value, f"({first[1]}/{second[1]})"))
                if "/" in allowed_ops and first[0] != 0:
                    value = second[0] / first[0]
                    if value.denominator == 1:
                        candidates.append((value, f"({second[1]}/{first[1]})"))
                for candidate in candidates:
                    states.append(tuple(rest + [candidate]))
    return None


def evaluate_linear_expression(numbers, ops):
    values = [Fraction(number, 1) for number in numbers]
    working_values = [values[0]]
    working_ops: list[str] = []
    for op, value in zip(ops, values[1:]):
        if op == "*":
            working_values[-1] = working_values[-1] * value
        elif op == "/":
            if value == 0:
                return None
            result = working_values[-1] / value
            if result.denominator != 1:
                return None
            working_values[-1] = result
        else:
            working_ops.append(op)
            working_values.append(value)
    result = working_values[0]
    for op, value in zip(working_ops, working_values[1:]):
        result = result + value if op == "+" else result - value
    return result


def exists_linear_solution(numbers, target):
    import itertools

    if len(numbers) <= 1:
        return numbers == [target]
    for perm in itertools.permutations(numbers):
        for ops in itertools.product(["+", "-", "*", "/"], repeat=len(numbers) - 1):
            value = evaluate_linear_expression(list(perm), list(ops))
            if value is not None and value == target:
                return True
    return False


def analyze_sample_structure(sample):
    numbers = sample["numbers"]
    target = sample["target"]
    plus_minus_solution = build_restricted_solver(numbers, target, allowed_ops=("+", "-")) is not None
    linear_solution = exists_linear_solution(numbers, target)
    return {
        "requires_muldiv": not plus_minus_solution,
        "requires_paren": not linear_solution,
    }


def solver_response(sample: dict[str, Any]):
    expr = build_countdown_solver(sample["numbers"], sample["target"])
    if not expr:
        return None
    return (
        f"<think>By combining the given numbers, the expression {expr} equals "
        f"{sample['target']}.</think>\n<answer> {expr} </answer>"
    )


def solver_answer_only_response(sample: dict[str, Any]):
    expr = build_countdown_solver(sample["numbers"], sample["target"])
    if not expr:
        return None
    return f"<answer> {expr} </answer>"


def repair_output_with_solver(text: str, sample: dict[str, Any]):
    expr = build_countdown_solver(sample["numbers"], sample["target"])
    if not expr:
        return None
    think = extract_think(text)
    if think:
        return f"<think>{think}</think>\n<answer> {expr} </answer>"
    return (
        f"<think>Using each given number exactly once, {expr} reaches {sample['target']}.</think>\n"
        f"<answer> {expr} </answer>"
    )
