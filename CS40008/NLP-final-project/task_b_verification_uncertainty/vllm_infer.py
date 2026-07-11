"""Run Task B verification and uncertainty experiments with local vLLM models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from task_a_robustness.data import load_examples
from task_a_robustness.prompts import parse_label
from task_a_robustness.vllm_config import DEFAULT_MAX_TOKENS_BY_MODE, MODEL_SPECS
from task_a_robustness.vllm_helpers import (
    apply_chat_template,
    build_llm_kwargs,
    clean_output,
    make_sampling_params,
    settings_dict,
)

from .cli import candidate_identifier
from .paths import OUTPUTS_DIR, ensure_runtime_dirs
from .prompts import CANDIDATE_TEMPLATE, CONSTRAINT_TEMPLATE, DIRECT_TEMPLATE
from .prompts import extract_response_text, parse_reason, verifier_prompt


TEMPLATES = {
    "direct": (DIRECT_TEMPLATE, "task_b_direct"),
    "constraint": (CONSTRAINT_TEMPLATE, "task_b_constraint_first"),
    "candidate": (CANDIDATE_TEMPLATE, "task_b_candidate"),
}


def response_output_path(args: argparse.Namespace, spec_name: str) -> Path:
    """Return the default response JSONL path for a run."""

    if args.output is not None:
        return args.output
    suffix = f"{spec_name}_{args.thinking_mode}_{args.task_mode}_{args.split}"
    if args.sample_start is not None or args.sample_end is not None:
        start = 0 if args.sample_start is None else args.sample_start
        if args.sample_end is None:
            suffix += f"_s{start}"
        else:
            suffix += f"_s{start}_e{args.sample_end}"
    if args.task_mode == "candidate" and args.repeat > 1:
        suffix += f"_k{args.repeat}"
    if args.max_samples:
        suffix += f"_n{args.max_samples}"
    return OUTPUTS_DIR / "vllm" / f"{suffix}_responses.jsonl"


def run_model(args: argparse.Namespace) -> None:
    """Run one local Qwen model for one Task B prompt mode."""

    spec = MODEL_SPECS[args.model]
    settings = spec.modes[args.thinking_mode]
    if not spec.path.exists():
        raise FileNotFoundError(f"model path does not exist: {spec.path}")
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")

    ensure_runtime_dirs()
    output_path = response_output_path(args, spec.name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from vllm import LLM

    llm = LLM(**build_llm_kwargs(args, spec))
    tokenizer = llm.get_tokenizer()
    max_tokens = args.max_tokens or DEFAULT_MAX_TOKENS_BY_MODE[args.thinking_mode]
    sampling_params = make_sampling_params(settings, max_tokens)
    used_settings = settings_dict(settings, max_tokens)

    if args.task_mode == "verifier":
        if args.candidates is None:
            raise ValueError("--candidates is required when --task-mode verifier")
        items = build_verifier_items(args, tokenizer, settings, spec.name, used_settings)
    else:
        items = build_task_items(args, tokenizer, settings, spec.name, used_settings)

    completed = load_completed_response_keys(output_path) if args.resume else set()
    remaining = [item for item in items if response_key(item) not in completed]
    rows_written = len(completed)
    with output_path.open("a" if args.resume else "w", encoding="utf-8") as file:
        for start in range(0, len(remaining), args.batch_size):
            batch = remaining[start : start + args.batch_size]
            print_progress("batch_start", spec.name, args, rows_written, len(items), output_path)
            outputs = llm.generate([str(item["prompt"]) for item in batch], sampling_params)
            for item, output in zip(batch, outputs):
                row = build_response_row(item, output.outputs[0].text)
                file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                rows_written += 1
            file.flush()
            print_progress("batch_done", spec.name, args, rows_written, len(items), output_path)

    payload = {
        "model": spec.name,
        "task_mode": args.task_mode,
        "thinking_mode": args.thinking_mode,
        "responses": str(output_path),
        "rows": rows_written,
        "settings": used_settings,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_task_items(args, tokenizer, settings, spec_name: str, used_settings: dict[str, object]):
    """Build direct, constraint-first, or candidate prompt items."""

    template, method_prefix = TEMPLATES[args.task_mode]
    examples = load_examples(args.split)
    if args.sample_start is not None or args.sample_end is not None:
        start = 0 if args.sample_start is None else args.sample_start
        end = args.sample_end
        examples = examples[start:end]
    if args.max_samples:
        examples = examples[: args.max_samples]

    items: list[dict[str, object]] = []
    for example in examples:
        user_prompt = template.render(example)
        prompt = apply_chat_template(tokenizer, user_prompt, settings)
        for sample_index in range(args.repeat):
            items.append(
                {
                    "id": example.id,
                    "method": f"{spec_name}:{args.thinking_mode}:{method_prefix}:{template.name}",
                    "gold": example.gold,
                    "sent0": example.sent0,
                    "sent1": example.sent1,
                    "sample_index": sample_index,
                    "prompt": prompt,
                    "run_settings": used_settings,
                }
            )
    return items


def build_verifier_items(args, tokenizer, settings, spec_name: str, used_settings: dict[str, object]):
    """Build verifier prompt items from candidate response JSONL."""

    items: list[dict[str, object]] = []
    with args.candidates.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            payload = json.loads(line)
            response = extract_response_text(payload)
            candidate_label = parse_label(response)
            candidate_id = candidate_identifier(payload, line_number)
            user_prompt = verifier_prompt(
                sent0=str(payload["sent0"]),
                sent1=str(payload["sent1"]),
                candidate_label=candidate_label if candidate_label in (0, 1) else -1,
                candidate_reason=parse_reason(response),
            )
            items.append(
                {
                    "candidate_id": candidate_id,
                    "id": payload["id"],
                    "method": f"{spec_name}:{args.thinking_mode}:task_b_verifier",
                    "gold": payload["gold"],
                    "sent0": payload["sent0"],
                    "sent1": payload["sent1"],
                    "sample_index": payload.get("sample_index", 0),
                    "candidate_method": payload.get("method", "task_b_candidate"),
                    "candidate_response": response,
                    "prompt": apply_chat_template(tokenizer, user_prompt, settings),
                    "run_settings": used_settings,
                }
            )
            if args.max_samples and len(items) >= args.max_samples:
                break
    return items


def build_response_row(item: dict[str, object], raw_text: str) -> dict[str, object]:
    """Build one response row compatible with existing Task B parsers."""

    row = {key: value for key, value in item.items() if key != "prompt"}
    row["response"] = clean_output(raw_text)
    row["raw_output"] = raw_text
    return row


def response_key(row: dict[str, object]) -> tuple[str, str, int]:
    """Return a stable resume key for a response row."""

    item_id = str(row.get("candidate_id") or row["id"])
    return (item_id, str(row["method"]), int(row.get("sample_index", 0)))


def load_completed_response_keys(path: Path) -> set[tuple[str, str, int]]:
    """Load completed response keys for resumable runs."""

    if not path.exists():
        return set()
    completed: set[tuple[str, str, int]] = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                completed.add(response_key(json.loads(line)))
    return completed


def print_progress(
    event: str, model_name: str, args: argparse.Namespace, completed: int, total: int, output_path: Path
) -> None:
    """Print machine-readable progress for long runs."""

    payload = {
        "event": event,
        "model": model_name,
        "task_mode": args.task_mode,
        "thinking_mode": args.thinking_mode,
        "completed": completed,
        "total": total,
        "responses": str(output_path),
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--thinking-mode", choices=["non_thinking", "thinking"], required=True)
    parser.add_argument("--task-mode", choices=["direct", "constraint", "candidate", "verifier"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--sample-start", type=int, default=None)
    parser.add_argument("--sample-end", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(func=run_model)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
