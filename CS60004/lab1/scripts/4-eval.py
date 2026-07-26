import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from preprocess_utils import build_input, build_instruction, read_jsonl, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Task 4 使用 vLLM 接口进行统一评测")
    parser.add_argument(
        "--input_path",
        type=str,
        default="datasets/provide_to_students/val.jsonl",
        help="输入 jsonl 文件路径。",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="vLLM 服务中的模型名称，或用于标识模型的路径字符串。",
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        default=None,
        help="可选的模型标签；不传时根据 model_name 自动推导。",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:12345/v1",
        help="vLLM OpenAI 兼容接口地址。",
    )
    parser.add_argument(
        "--plan",
        type=str,
        nargs="+",
        default=["A"],
        choices=["A", "B", "C"],
        help="评测使用的输出格式方案，可同时传多个。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="生成温度。",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="最大生成 token 数。",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="并发请求数量。",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/task4",
        help="评测结果输出目录。",
    )
    return parser.parse_args()


def sanitize_name(text):
    safe = []
    for ch in str(text):
        if ch.isalnum() or ch in ["-", "_", "."]:
            safe.append(ch)
        else:
            safe.append("_")
    sanitized = "".join(safe).strip("_")
    return sanitized or "unknown"


def infer_model_tag(model_name, model_tag=None):
    if model_tag:
        return sanitize_name(model_tag)

    model_path = Path(model_name)
    if model_path.is_absolute() or "/" in model_name:
        parent_name = sanitize_name(model_path.parent.name)
        current_name = sanitize_name(model_path.name)
        if parent_name and current_name:
            return f"{parent_name}_{current_name}"
        return current_name or parent_name or "model"

    return sanitize_name(model_name)


def build_messages(sample, plan):
    instruction = build_instruction(plan=plan)
    input_text = build_input(sample, plan=plan)
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text},
    ]


def parse_prediction(raw_text):
    try:
        parsed_obj = json.loads(raw_text)
    except Exception:
        return {
            "parsed": False,
            "answer": None,
        }

    if not isinstance(parsed_obj, dict):
        return {
            "parsed": False,
            "answer": None,
        }

    answer = parsed_obj.get("answer")
    if answer is not None:
        answer = str(answer).strip().upper()
        if answer not in ["A", "B", "C", "D", "E"]:
            answer = None

    return {
        "parsed": True,
        "answer": answer,
    }


def parse_gold_answer(gold_value):
    if gold_value is None:
        return None

    if isinstance(gold_value, str):
        text = gold_value.strip()
        if text.upper() in ["A", "B", "C", "D", "E"]:
            return text.upper()
        try:
            gold_obj = json.loads(text)
        except Exception:
            return None
        answer = gold_obj.get("answer")
        if answer is None:
            return None
        answer = str(answer).strip().upper()
        if answer in ["A", "B", "C", "D", "E"]:
            return answer
        return None

    if isinstance(gold_value, dict):
        answer = gold_value.get("answer")
        if answer is None:
            return None
        answer = str(answer).strip().upper()
        if answer in ["A", "B", "C", "D", "E"]:
            return answer

    return None


def compute_metrics(results):
    total = len(results)
    parsed_count = sum(1 for item in results if item["parsed"])
    metrics = {
        "num_samples": total,
        "json_parse_rate": parsed_count / total if total > 0 else 0.0,
    }

    has_gold = any("gold_answer" in item for item in results)
    if has_gold:
        correct_count = sum(1 for item in results if item.get("correct", False))
        metrics["answer_accuracy"] = correct_count / total if total > 0 else 0.0

    return metrics


def run_one_sample(client, sample, model_name, plan, max_tokens, temperature):
    messages = build_messages(sample, plan)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
    )
    raw_text = (response.choices[0].message.content or "").strip()
    parsed_result = parse_prediction(raw_text)

    result = {
        "question": sample["question"],
        "option": sample["option"],
        "answer": raw_text,
        "parsed": parsed_result["parsed"],
    }

    if "sample_id" in sample:
        result["sample_id"] = sample["sample_id"]

    if "answer" in sample:
        gold_answer = parse_gold_answer(sample.get("answer"))
        result["gold_answer"] = gold_answer
        result["correct"] = parsed_result["answer"] == gold_answer if gold_answer is not None else False

    return result


def main():
    args = parse_args()
    input_tag = Path(args.input_path).stem
    model_tag = infer_model_tag(args.model_name, args.model_tag)

    output_dir = args.output_root
    os.makedirs(output_dir, exist_ok=True)

    client = OpenAI(
        api_key="EMPTY",
        base_url=args.base_url,
    )

    samples = read_jsonl(args.input_path)
    for plan in args.plan:
        plan_tag = plan.lower()
        output_path = os.path.join(output_dir, f"{model_tag}_{input_tag}_plan_{plan_tag}.jsonl")
        metrics_path = os.path.join(output_dir, f"{model_tag}_{input_tag}_plan_{plan_tag}_metrics.json")
        results = []

        if args.num_workers <= 1:
            for sample in tqdm(samples, desc=f"Evaluating Plan {plan}", ncols=80):
                result = run_one_sample(
                    client=client,
                    sample=sample,
                    model_name=args.model_name,
                    plan=plan,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                results.append(result)
        else:
            indexed_results = []
            pbar = tqdm(total=len(samples), desc=f"Evaluating Plan {plan}", ncols=80)
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = []
                for idx, sample in enumerate(samples):
                    future = executor.submit(
                        run_one_sample,
                        client,
                        sample,
                        args.model_name,
                        plan,
                        args.max_tokens,
                        args.temperature,
                    )
                    futures.append((idx, future))

                future_to_idx = {future: idx for idx, future in futures}

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    result = future.result()
                    indexed_results.append((idx, result))
                    pbar.update(1)

            pbar.close()
            indexed_results.sort(key=lambda x: x[0])
            results = [result for _, result in indexed_results]

        write_jsonl(output_path, results)

        metrics = compute_metrics(results)
        with open(metrics_path, "w", encoding="utf-8") as file_obj:
            json.dump(metrics, file_obj, ensure_ascii=False, indent=2)

        print(f"Input file       : {args.input_path}")
        print(f"Model tag        : {model_tag}")
        print(f"Plan             : {plan}")
        print(f"Output file      : {output_path}")
        print(f"Metrics file     : {metrics_path}")
        print(f"Number of samples: {metrics['num_samples']}")
        print(f"JSON parse rate  : {metrics['json_parse_rate']:.4f}")
        if "answer_accuracy" in metrics:
            print(f"Answer accuracy  : {metrics['answer_accuracy']:.4f}")


if __name__ == "__main__":
    main()


# 训练前
# python scripts/4-eval.py --model_name qwen2.5-1.5b-instruct --plan A B C
# 训练后
# python scripts/4-eval.py --model_name /root/autodl-tmp/CS60004/lab1/outputs/task2/v5-0324223414/latest --plan A B C
