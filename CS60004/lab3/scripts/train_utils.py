from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import random
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import torch


def load_config(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(f"{path} is not JSON-compatible YAML and PyYAML is not installed") from exc
        return yaml.safe_load(text)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_run(config_path, config):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / Path(config_path).name)
    save_json(output_dir / "run_config.json", config)
    return output_dir


def get_torch_dtype(name):
    if not name:
        return None
    lookup = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in lookup:
        raise ValueError(f"unsupported dtype: {name}")
    return lookup[name]


def load_tokenizer(model_name_or_path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_causal_lm(model_name_or_path, config, train=True):
    from transformers import AutoModelForCausalLM

    dtype = get_torch_dtype(config.get("torch_dtype", "bfloat16" if torch.cuda.is_available() else "float32"))
    kwargs = {"trust_remote_code": True}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if config.get("device_map"):
        kwargs["device_map"] = config["device_map"]
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    if not config.get("device_map"):
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if train:
        model.train()
        if config.get("gradient_checkpointing"):
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
    else:
        model.eval()
    return model


def init_swanlab(config, run_name):
    swan_config = config.get("swanlab", {})
    if not swan_config.get("enabled", False):
        return None
    try:
        import swanlab
    except ImportError:
        print("swanlab is not installed; metrics will be written locally only")
        return None
    try:
        return swanlab.init(
            project=swan_config.get("project", "cs60004-lab3"),
            experiment_name=swan_config.get("experiment_name", run_name),
            config=config,
            tags=swan_config.get("tags", []),
        )
    except Exception as exc:
        print(f"swanlab init failed; metrics will be written locally only: {exc}")
        return None


def log_metrics(output_dir, metrics, tracker=None):
    metrics = dict(metrics)
    metrics.setdefault("time", time.time())
    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
    if tracker is not None:
        tracker.log(metrics)


def encode_prompt_response(
    tokenizer,
    prompt,
    response,
    max_prompt_length,
    max_response_length,
    chat_template_kwargs=None,
    system_prompt=None,
    append_response_eos=False,
):
    if chat_template_kwargs is None:
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids[-max_prompt_length:]
    else:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        if hasattr(prompt_ids, "input_ids"):
            prompt_ids = prompt_ids.input_ids
        prompt_ids = prompt_ids[-max_prompt_length:]
    response_ids = tokenizer(response, add_special_tokens=False).input_ids
    if append_response_eos and tokenizer.eos_token_id is not None:
        if not response_ids or response_ids[-1] != tokenizer.eos_token_id:
            response_ids = response_ids + [tokenizer.eos_token_id]
    response_ids = response_ids[:max_response_length]
    if tokenizer.bos_token_id is not None:
        input_ids = [tokenizer.bos_token_id] + prompt_ids + response_ids
        response_start = 1 + len(prompt_ids)
    else:
        input_ids = prompt_ids + response_ids
        response_start = len(prompt_ids)
    labels = [-100] * len(input_ids)
    labels[response_start:] = input_ids[response_start:]
    return input_ids, labels, len(response_ids)


def collate_prompt_response(
    tokenizer,
    prompts,
    responses: list[str],
    max_prompt_length,
    max_response_length,
    chat_template_kwargs=None,
    system_prompt=None,
    append_response_eos=False,
):
    encoded = [
        encode_prompt_response(
            tokenizer,
            prompt,
            response,
            max_prompt_length,
            max_response_length,
            chat_template_kwargs=chat_template_kwargs,
            system_prompt=system_prompt,
            append_response_eos=append_response_eos,
        )
        for prompt, response in zip(prompts, responses)
    ]
    max_len = max(len(input_ids) for input_ids, _, _ in encoded)
    pad_id = tokenizer.pad_token_id
    input_batch, label_batch, attention_batch = [], [], []
    lengths = []
    for input_ids, labels, response_len in encoded:
        pad = max_len - len(input_ids)
        input_batch.append([pad_id] * pad + input_ids)
        label_batch.append([-100] * pad + labels)
        attention_batch.append([0] * pad + [1] * len(input_ids))
        lengths.append(response_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "input_ids": torch.tensor(input_batch, dtype=torch.long, device=device),
        "labels": torch.tensor(label_batch, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_batch, dtype=torch.long, device=device),
        "response_lengths": torch.tensor(lengths, dtype=torch.float, device=device),
    }


def sequence_log_probs(model, batch):
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~mask, 0)
    token_logps = torch.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen,
    policy_rejected,
    ref_chosen,
    ref_rejected,
    beta,
):
    chosen_rewards = policy_chosen - ref_chosen
    rejected_rewards = policy_rejected - ref_rejected
    logits = beta * (chosen_rewards - rejected_rewards)
    loss = -torch.nn.functional.logsigmoid(logits).mean()
    with torch.no_grad():
        stats = {
            "loss": float(loss.detach().cpu()),
            "chosen_logp_mean": float(policy_chosen.mean().detach().cpu()),
            "rejected_logp_mean": float(policy_rejected.mean().detach().cpu()),
            "reward_margin": float((chosen_rewards - rejected_rewards).mean().detach().cpu()),
        }
    return loss, stats


def grpo_clip_loss(new_logps, old_logps, advantages, clip_range):
    ratio = torch.exp(new_logps - old_logps)
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    objective = torch.minimum(ratio * advantages, clipped * advantages)
    return -objective.mean()


def assert_finite(name, value):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or inf")


def batch_indices(total, batch_size, shuffle=True):
    indices = list(range(total))
    if shuffle:
        random.shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, total, batch_size)]


def count_trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def estimate_entropy_from_logits(logits, labels):
    mask = labels[:, 1:].ne(-100)
    probs = torch.softmax(logits[:, :-1, :], dim=-1)
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    denom = mask.sum().clamp_min(1)
    return (entropy * mask).sum() / denom


def vllm_chat_completions(
    prompts,
    generation,
    default_model,
    system_prompt=None,
) -> list[str]:
    api_base = generation.get("api_base", "http://localhost:8000/v1")
    endpoint = api_base.rstrip("/") + "/chat/completions"
    model = generation.get("model", default_model)
    timeout = float(generation.get("timeout", 600))
    request_concurrency = max(1, int(generation.get("request_concurrency", 1)))
    system_prompt = system_prompt or generation.get(
        "system_prompt",
        "You are a careful arithmetic solver. The <answer> block must contain exactly one final expression, "
        "with no equals sign, no explanation, and no extra text.",
    )
    def request_one(prompt):
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(generation.get("temperature", 0.7)),
            "top_p": float(generation.get("top_p", 0.8)),
            "top_k": int(generation.get("top_k", 20)),
            "max_tokens": min(int(generation.get("max_tokens", generation.get("max_new_tokens", 1024))), 1024),
            "presence_penalty": float(generation.get("presence_penalty", 1.5)),
            "chat_template_kwargs": {"enable_thinking": bool(generation.get("enable_thinking", False))},
        }
        min_p = float(generation.get("min_p", 0.0))
        if min_p > 0:
            payload["min_p"] = min_p
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"vLLM API request failed: {endpoint}") from exc
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"vLLM API returned no choices: {data}")
        message = choices[0].get("message") or {}
        return message.get("content", "")

    if request_concurrency == 1 or len(prompts) <= 1:
        return [request_one(prompt) for prompt in prompts]

    outputs = [""] * len(prompts)
    with ThreadPoolExecutor(max_workers=min(request_concurrency, len(prompts))) as executor:
        future_to_index = {executor.submit(request_one, prompt): index for index, prompt in enumerate(prompts)}
        for future in as_completed(future_to_index):
            outputs[future_to_index[future]] = future.result()
    return outputs
