import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import torch

from preprocess_utils import (
    analyze_sample_structure,
    build_prompt,
    evaluate_output,
    extract_answer,
    load_countdown_data,
    summarize_results,
    write_json,
)
from train_utils import (
    assert_finite,
    batch_indices,
    collate_prompt_response,
    estimate_entropy_from_logits,
    grpo_clip_loss,
    init_swanlab,
    load_causal_lm,
    load_config,
    load_tokenizer,
    log_metrics,
    prepare_run,
    sequence_log_probs,
    set_seed,
    vllm_chat_completions,
)


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO 训练")
    parser.add_argument("--config", default="lab3/configs/thinking_grpo_r3.yaml")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def generate_rollouts_vllm(prompts, group_size, config, model_name):
    expanded = [prompt for prompt in prompts for _ in range(group_size)]
    generation = config.get("generation", {})
    sampled_generation = dict(generation)
    sampled_generation["do_sample"] = True
    sampled_generation["temperature"] = float(config.get("rollout_temperature", generation.get("temperature", 0.7)))
    sampled_generation["top_p"] = float(config.get("rollout_top_p", generation.get("top_p", 0.8)))
    abs_model = str(Path(model_name).resolve())
    for attempt in range(3):
        try:
            responses = vllm_chat_completions(expanded, sampled_generation, abs_model)
            return [{"prompt": prompt, "response": response} for prompt, response in zip(expanded, responses)]
        except Exception as exc:
            if attempt < 2:
                print(f"  vLLM rollout failed (attempt {attempt+1}), retrying in 10s: {exc}")
                time.sleep(10)
            else:
                raise


def generate_rollouts_local(tokenizer, model, prompts, group_size, config):
    expanded = [prompt for prompt in prompts for _ in range(group_size)]
    generation = config.get("generation", {})
    enable_thinking = bool(generation.get("enable_thinking", False))
    use_chat_template = bool(generation.get("use_chat_template", False)) or enable_thinking
    system_prompt = generation.get(
        "system_prompt",
        "You are a careful arithmetic solver. The <answer> block must contain exactly one final expression, "
        "with no equals sign, no explanation, and no extra text.",
    )
    if use_chat_template:
        chat_texts = []
        for prompt in expanded:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            ct_kwargs = {"enable_thinking": True} if enable_thinking else {}
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **ct_kwargs
            )
            chat_texts.append(text)
    else:
        chat_texts = None

    gen_batch_size = int(config.get("generate_batch_size", 8))
    max_new_tokens = min(int(generation.get("max_new_tokens", config["max_response_length"])), 1024)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=float(config.get("rollout_temperature", generation.get("temperature", 0.7))),
        top_p=float(config.get("rollout_top_p", generation.get("top_p", 0.8))),
        top_k=int(generation.get("top_k", 20)),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    was_training = model.training
    old_use_cache = getattr(model.config, "use_cache", None)
    model.eval()
    if old_use_cache is not None:
        model.config.use_cache = True
    all_responses = []
    try:
        with torch.no_grad():
            for si in range(0, len(expanded), gen_batch_size):
                sj = min(si + gen_batch_size, len(expanded))
                if chat_texts is not None:
                    encoded = tokenizer(
                        chat_texts[si:sj], padding=True, truncation=True,
                        max_length=int(config.get("max_prompt_length", 256)),
                        return_tensors="pt",
                    ).to(model.device)
                else:
                    encoded = tokenizer(
                        expanded[si:sj], add_special_tokens=True, padding=True, truncation=True,
                        max_length=int(config["max_prompt_length"]),
                        return_tensors="pt",
                    ).to(model.device)
                outputs = model.generate(**encoded, **gen_kwargs)
                prompt_len = encoded["input_ids"].shape[1]
                batch_responses = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
                all_responses.extend(batch_responses)
                del encoded, outputs
                torch.cuda.empty_cache()
    finally:
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache
        if was_training:
            model.train()
    return [{"prompt": prompt, "response": response} for prompt, response in zip(expanded, all_responses)]


def generate_rollouts_vllm_exclusive(
    prompts,
    group_size,
    config,
    model_weights_dir,
):
    expanded = [prompt for prompt in prompts for _ in range(group_size)]
    generation = config.get("generation", {})

    _kill_vllm()
    port = int(config.get("vllm_port", 8001))
    gpu_util = float(config.get("vllm_gpu_memory_utilization", 0.90))
    python = "/root/autodl-tmp/conda_envs/lab3/bin/python3"
    abs_model = str(Path(model_weights_dir).resolve())
    enable_thinking = bool(generation.get("enable_thinking", False))

    cmd = [
        python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", abs_model,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--max-model-len", "2048",
        "--gpu-memory-utilization", str(gpu_util),
        "--dtype", "bfloat16",
        "--trust-remote-code",
    ]
    log_path = Path(config.get("output_dir", ".")) / "vllm_server.log"
    log_handle = log_path.open("a")
    proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, preexec_fn=os.setsid)

    import urllib.request
    endpoint = f"http://127.0.0.1:{port}/v1/models"
    vllm_ready = False
    for attempt in range(60):
        time.sleep(3)
        try:
            urllib.request.urlopen(endpoint, timeout=5)
            print(f"  vLLM ready (attempt {attempt+1}, {(attempt+1)*3}s)")
            vllm_ready = True
            break
        except Exception:
            if proc.poll() is not None:
                log_handle.close()
                err = log_path.read_text()[-1000:]
                raise RuntimeError(f"vLLM died (exit {proc.returncode}): {err}")
    if not vllm_ready:
        proc.kill()
        log_handle.close()
        raise RuntimeError("vLLM failed to start within 180s")

    try:
        sampled_generation = dict(generation)
        sampled_generation["temperature"] = float(config.get("rollout_temperature", generation.get("temperature", 0.7)))
        sampled_generation["top_p"] = float(config.get("rollout_top_p", generation.get("top_p", 0.8)))
        sampled_generation["api_base"] = f"http://127.0.0.1:{port}/v1"
        sampled_generation["model"] = abs_model
        sampled_generation["request_concurrency"] = int(config.get("vllm_request_concurrency", 32))
        responses = vllm_chat_completions(expanded, sampled_generation, abs_model)
    finally:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(3)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        log_handle.close()
        _kill_vllm()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for _ in range(15):
                free, total = torch.cuda.mem_get_info()
                used_mb = (total - free) / (1024 * 1024)
                if used_mb < 500:
                    break
                time.sleep(2)
            else:
                print(f"  Warning: GPU still has {used_mb:.0f}MB used after vLLM cleanup")
        time.sleep(2)

    return [{"prompt": prompt, "response": response} for prompt, response in zip(expanded, responses)]


def generate_rollouts(
    tokenizer,
    model,
    prompts,
    group_size,
    config,
    model_name,
):
    backend = config.get("rollout_backend", "local")
    if backend == "vllm_exclusive":
        return generate_rollouts_vllm_exclusive(prompts, group_size, config, model_name)
    if backend == "vllm_api":
        return generate_rollouts_vllm(prompts, group_size, config, model_name)
    return generate_rollouts_local(tokenizer, model, prompts, group_size, config)


def collate_kwargs_from_config(config):
    generation = config.get("generation", {})
    if not (generation.get("use_chat_template", False) or generation.get("enable_thinking", False)):
        return {}
    system_prompt = generation.get(
        "system_prompt",
        "You are a careful arithmetic solver. The <answer> block must contain exactly one final expression, "
        "with no equals sign, no explanation, and no extra text.",
    )
    template_kwargs = {"enable_thinking": True} if generation.get("enable_thinking", False) else {}
    return {"chat_template_kwargs": template_kwargs, "system_prompt": system_prompt}


import re

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def _extract_final_answer(text):
    think_end = text.rfind("</think>")
    if think_end >= 0:
        after_think = text[think_end:]
        matches = list(ANSWER_RE.finditer(after_think))
        if matches:
            return matches[-1].group(1).strip()
    return extract_answer(text)


def _evaluate_thinking_output(text, numbers, target):
    answer = _extract_final_answer(text)
    if not answer:
        return evaluate_output(text, numbers, target)
    wrapped = f"<answer>{answer}</answer>"
    return evaluate_output(wrapped, numbers, target)


def rewards_for_rollouts(rollouts, samples, group_size, reward_weights):
    evals = []
    rewards = []
    format_weight = float(reward_weights.get("format", 0.2))
    expr_weight = float(reward_weights.get("expr_valid", 0.2))
    correct_weight = float(reward_weights.get("correct", 1.0))
    truncation_penalty = float(reward_weights.get("truncation_penalty", 0.5))
    brevity_bonus = float(reward_weights.get("brevity_bonus", 0.0))
    value_weight = float(reward_weights.get("value_closeness", 0.0))
    missing_muldiv_penalty = float(reward_weights.get("missing_muldiv_penalty", 0.0))
    missing_paren_penalty = float(reward_weights.get("missing_paren_penalty", 0.0))
    subset_penalty = float(reward_weights.get("subset_number_penalty", 0.0))
    equation_penalty = float(reward_weights.get("equation_penalty", 0.0))
    for index, rollout in enumerate(rollouts):
        sample = samples[index // group_size]
        response = rollout["response"]
        result = _evaluate_thinking_output(response, sample["numbers"], sample["target"])
        answer = _extract_final_answer(response) or extract_answer(response)
        structure = sample.get("structure")
        if structure is None:
            structure = analyze_sample_structure(sample)
            sample["structure"] = structure

        is_truncated = "</think>" not in response

        if is_truncated:
            reward = -truncation_penalty
        else:
            reward = (
                correct_weight * float(result.correct)
                + expr_weight * float(result.expr_valid)
                + format_weight * float(result.format_ok)
            )
            if result.correct and brevity_bonus > 0:
                resp_len = len(response)
                length_ratio = max(0.0, 1.0 - resp_len / 2000.0)
                reward += brevity_bonus * length_ratio
            if result.expr_valid and result.value is not None and not result.correct and value_weight > 0:
                reward += value_weight / (1.0 + abs(float(result.value) - sample["target"]))
        if structure.get("requires_muldiv") and ("*" not in answer and "/" not in answer):
            reward -= missing_muldiv_penalty
        if structure.get("requires_paren") and ("(" not in answer or ")" not in answer):
            reward -= missing_paren_penalty
        if result.reason == "number_usage_error":
            reward -= subset_penalty
        if "=" in answer:
            reward -= equation_penalty
        evals.append(result)
        rewards.append(reward)
    return torch.tensor(rewards, dtype=torch.float, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), evals


def group_advantages(rewards, group_size):
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False)
    return ((grouped - mean) / (std + 1e-8)).view(-1)


def _kill_vllm():
    for pattern in ["vllm.entrypoints", "vllm.v1.engine", "vllm.worker"]:
        for sig_name in ["-TERM", "-KILL"]:
            try:
                subprocess.run(["pkill", sig_name, "-f", pattern], capture_output=True, timeout=5)
            except Exception:
                pass
        time.sleep(0.5)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for _ in range(10):
            free, total = torch.cuda.mem_get_info()
            used_mb = (total - free) / (1024 * 1024)
            if used_mb < 500:
                break
            time.sleep(2)
    time.sleep(2)


def _start_vllm(model_path, config):
    port = int(config.get("vllm_port", 8001))
    gpu_util = float(config.get("vllm_gpu_memory_utilization", 0.4))
    python = "/root/autodl-tmp/conda_envs/lab3/bin/python3"
    if not os.path.exists(python):
        python = os.path.join(os.environ.get("CONDA_PREFIX", ""), "bin", "python3")
    if not os.path.exists(python):
        import sys
        python = sys.executable
    abs_model = str(Path(model_path).resolve())
    cmd = [
        python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", abs_model,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--max-model-len", "2048",
        "--gpu-memory-utilization", str(gpu_util),
        "--dtype", "bfloat16",
        "--trust-remote-code",
    ]
    log_path = Path(config.get("output_dir", ".")) / "vllm_server.log"
    log_handle = log_path.open("w")
    proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
    import urllib.request
    endpoint = f"http://127.0.0.1:{port}/v1/models"
    for attempt in range(90):
        time.sleep(2)
        try:
            urllib.request.urlopen(endpoint, timeout=5)
            print(f"  vLLM ready on port {port} (attempt {attempt+1})")
            return proc
        except Exception:
            if proc.poll() is not None:
                log_handle.close()
                err = log_path.read_text()[-500:]
                raise RuntimeError(f"vLLM process died (exit {proc.returncode}): {err}")
    log_handle.close()
    raise RuntimeError("vLLM failed to start within 180s")


def _sync_vllm(policy, tokenizer, sync_dir, config, vllm_proc):
    sync_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(sync_dir)
    tokenizer.save_pretrained(sync_dir)
    if vllm_proc is not None:
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()
    time.sleep(3)
    proc = _start_vllm(str(sync_dir), config)
    import urllib.request
    port = int(config.get("vllm_port", 8001))
    abs_model = str(sync_dir.resolve())
    warmup_payload = json.dumps({
        "model": abs_model,
        "messages": [{"role": "user", "content": "1+1=?"}],
        "max_tokens": 16,
        "temperature": 0.1,
    }).encode()
    endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"
    for attempt in range(5):
        try:
            req = urllib.request.Request(endpoint, data=warmup_payload, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=30)
            print(f"  vLLM warmup OK (attempt {attempt+1})")
            return proc
        except Exception:
            time.sleep(5)
    print("  Warning: vLLM warmup failed, proceeding anyway")
    return proc


def _get_memory_mb():
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    cgroup_usage = -1
    try:
        cgroup_usage = int(Path("/sys/fs/cgroup/memory.current").read_text().strip()) / (1024 * 1024)
    except Exception:
        pass
    return {"rss_mb": rss, "cgroup_mb": cgroup_usage}


_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    print(f"\n[SIGNAL] Received signal {signum}, will save checkpoint and exit after current step...")
    _shutdown_requested = True


def train_grpo(config_path, config, limit=None):
    global _shutdown_requested
    _shutdown_requested = False

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGUSR1, _signal_handler)

    set_seed(int(config.get("seed", 42)))
    output_dir = prepare_run(config_path, config)
    tracker = init_swanlab(config, Path(config_path).stem)
    tokenizer = load_tokenizer(config["model_name_or_path"])
    policy = load_causal_lm(config["model_name_or_path"], config, train=True)
    reference = None
    if config.get("use_kl", False):
        reference = load_causal_lm(config.get("ref_model_name_or_path", config["model_name_or_path"]), config, train=False)
        for parameter in reference.parameters():
            parameter.requires_grad_(False)

    samples = load_countdown_data(config["train_file"])
    valid_samples = load_countdown_data(config["valid_file"]) if config.get("valid_file") else samples[:64]
    if limit:
        samples = samples[:limit]
        valid_samples = valid_samples[:limit]
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(config.get("learning_rate", 1e-6)))
    group_size = int(config.get("rollout_group_size", 4))
    batch_size = int(config.get("per_device_batch_size", 1))
    max_steps = int(config.get("max_steps", 100))
    save_interval = int(config.get("save_interval", 50))
    vllm_sync_interval = int(config.get("vllm_sync_interval", 0))
    vllm_proc = None
    sync_dir = output_dir / "_vllm_sync"

    resume_step = 0
    resume_ckpt = config.get("resume_from_step")
    if resume_ckpt:
        ckpt_dir = output_dir / f"checkpoint-{resume_ckpt}"
        if ckpt_dir.exists():
            print(f"Resuming from checkpoint at step {resume_ckpt}...")
            policy = load_causal_lm(str(ckpt_dir), config, train=True)
            optimizer = torch.optim.AdamW(policy.parameters(), lr=float(config.get("learning_rate", 1e-6)))
            resume_step = int(resume_ckpt)

    if config.get("rollout_backend") == "vllm_api" and vllm_sync_interval > 0:
        _kill_vllm()
        print("Starting initial vLLM server...")
        vllm_proc = _start_vllm(config["model_name_or_path"], config)

    global_step = resume_step
    vllm_model_name = config.get("rollout_model_name_or_path", config["model_name_or_path"])
    start_time = time.time()
    mem_info = _get_memory_mb()
    print(f"Starting training: group_size={group_size}, batch_size={batch_size}, max_steps={max_steps}, "
          f"resume_step={resume_step}, cgroup_mb={mem_info['cgroup_mb']:.0f}")
    try:
        while global_step < max_steps:
            if _shutdown_requested:
                print(f"[SHUTDOWN] Saving emergency checkpoint at step {global_step}...")
                ckpt = output_dir / f"checkpoint-{global_step}"
                policy.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                print(f"[SHUTDOWN] Saved to {ckpt}")
                break
            for indices in batch_indices(len(samples), batch_size, shuffle=True):
                if global_step >= max_steps or _shutdown_requested:
                    break
                batch_samples = [samples[index] for index in indices]
                prompts = [build_prompt(sample) for sample in batch_samples]

                use_vllm_exclusive = config.get("rollout_backend") == "vllm_exclusive"

                if use_vllm_exclusive:
                    policy.cpu()
                    if reference is not None:
                        reference.cpu()
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    for _wait in range(15):
                        free, total = torch.cuda.mem_get_info()
                        used_mb = (total - free) / (1024 * 1024)
                        if used_mb < 200:
                            break
                        time.sleep(1)
                    else:
                        print(f"  Warning: GPU still has {used_mb:.0f}MB before vLLM start")
                    weights_dir = output_dir / "_vllm_weights"
                    weights_dir.mkdir(parents=True, exist_ok=True)
                    policy.save_pretrained(weights_dir)
                    tokenizer.save_pretrained(weights_dir)
                    vllm_model_name = str(weights_dir)

                rollout_start = time.time()
                rollouts = generate_rollouts(
                    tokenizer,
                    policy,
                    prompts,
                    group_size,
                    config,
                    vllm_model_name,
                )
                rollout_time = time.time() - rollout_start

                if use_vllm_exclusive:
                    policy.to(torch.device("cuda"))
                    if reference is not None:
                        reference.to(torch.device("cuda"))
                    policy.train()
                    if config.get("gradient_checkpointing"):
                        policy.gradient_checkpointing_enable()
                        policy.config.use_cache = False

                reward_start = time.time()
                rewards, evals = rewards_for_rollouts(rollouts, batch_samples, group_size, config.get("reward_weights", {}))
                advantages = group_advantages(rewards, group_size)
                reward_time = time.time() - reward_start

                responses = [row["response"] for row in rollouts]
                rollout_prompts = [row["prompt"] for row in rollouts]

                fwd_chunk = int(config.get("forward_chunk_size", 4))
                n_rollouts = len(responses)

                old_logps_list = []
                collate_kwargs = collate_kwargs_from_config(config)
                with torch.no_grad():
                    for ci in range(0, n_rollouts, fwd_chunk):
                        chunk_batch = collate_prompt_response(
                            tokenizer, rollout_prompts[ci:ci+fwd_chunk], responses[ci:ci+fwd_chunk],
                            int(config["max_prompt_length"]), int(config["max_response_length"]),
                            **collate_kwargs,
                        )
                        old_logps_list.append(sequence_log_probs(policy, chunk_batch))
                        del chunk_batch
                old_logps = torch.cat(old_logps_list)
                del old_logps_list

                update_start = time.time()
                optimizer.zero_grad(set_to_none=True)
                total_loss = torch.tensor(0.0, device=old_logps.device)
                total_entropy = torch.tensor(0.0, device=old_logps.device)
                n_chunks = (n_rollouts + fwd_chunk - 1) // fwd_chunk
                for ci in range(0, n_rollouts, fwd_chunk):
                    cj = min(ci + fwd_chunk, n_rollouts)
                    chunk_batch = collate_prompt_response(
                        tokenizer, rollout_prompts[ci:cj], responses[ci:cj],
                        int(config["max_prompt_length"]), int(config["max_response_length"]),
                        **collate_kwargs,
                    )
                    chunk_outputs = policy(input_ids=chunk_batch["input_ids"], attention_mask=chunk_batch["attention_mask"])
                    chunk_new_logps = sequence_log_probs(policy, chunk_batch)
                    chunk_loss = grpo_clip_loss(chunk_new_logps, old_logps[ci:cj], advantages[ci:cj], float(config.get("clip_range", 0.2)))
                    if reference is not None:
                        with torch.no_grad():
                            chunk_ref_logps = sequence_log_probs(reference, chunk_batch)
                        chunk_loss = chunk_loss + float(config.get("kl_coef", 0.0)) * (chunk_new_logps - chunk_ref_logps).mean()
                    (chunk_loss / n_chunks).backward()
                    total_loss = total_loss + chunk_loss.detach() / n_chunks
                    total_entropy = total_entropy + estimate_entropy_from_logits(chunk_outputs.logits.detach(), chunk_batch["labels"]).detach() / n_chunks
                    del chunk_batch, chunk_outputs
                assert_finite("grpo_loss", total_loss)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), float(config.get("max_grad_norm", 1.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                update_time = time.time() - update_start
                global_step += 1

                summary = summarize_results(evals)
                mem_info = _get_memory_mb()
                metrics = {
                    "step": global_step,
                    "loss": float(total_loss.cpu()),
                    "reward_mean": float(rewards.mean().detach().cpu()),
                    "reward_correct_mean": summary["accuracy"],
                    "reward_format_mean": summary["format_rate"],
                    "accuracy": summary["accuracy"],
                    "format_rate": summary["format_rate"],
                    "entropy": float(total_entropy.cpu()),
                    "rollout_time_sec": rollout_time,
                    "reward_time_sec": reward_time,
                    "update_time_sec": update_time,
                    "step_time_sec": rollout_time + reward_time + update_time,
                    "runtime_sec": time.time() - start_time,
                    "cgroup_mb": mem_info["cgroup_mb"],
                    "gpu_mem_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0,
                }
                log_metrics(output_dir, metrics, tracker)
                if global_step % int(config.get("eval_interval", 20)) == 0:
                    write_json(output_dir / f"eval_step_{global_step}.json", evaluate_validation(tokenizer, policy, valid_samples, config))
                if global_step % save_interval == 0:
                    checkpoint = output_dir / f"checkpoint-{global_step}"
                    policy.save_pretrained(checkpoint)
                    tokenizer.save_pretrained(checkpoint)
                    print(f"  [SAVE] checkpoint-{global_step} (cgroup={mem_info['cgroup_mb']:.0f}MB, gpu={metrics['gpu_mem_mb']:.0f}MB)")
                    total_save = int(config.get("total_save", 0))
                    if total_save > 0:
                        import re as _re
                        existing = sorted(
                            [d for d in output_dir.iterdir() if d.is_dir() and _re.match(r"checkpoint-\d+$", d.name)],
                            key=lambda p: int(p.name.split("-")[1]),
                        )
                        while len(existing) > total_save:
                            old = existing.pop(0)
                            shutil.rmtree(old)
                            print(f"  [CLEAN] removed {old.name} (keeping {total_save} latest)")
                    if vllm_proc is not None and vllm_sync_interval > 0 and global_step % vllm_sync_interval == 0:
                        print(f"  Syncing vLLM weights at step {global_step}...")
                        vllm_proc = _sync_vllm(policy, tokenizer, sync_dir, config, vllm_proc)
                        vllm_model_name = str(sync_dir)
    finally:
        if vllm_proc is not None:
            print("Stopping vLLM server...")
            vllm_proc.terminate()
            try:
                vllm_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                vllm_proc.kill()
    final_dir = output_dir / "final"
    policy.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    write_json(output_dir / "train_summary.json", {"global_step": global_step, "runtime_sec": time.time() - start_time})
    return final_dir


def evaluate_validation(tokenizer, model, samples, config):
    results = []
    for group in batch_indices(len(samples), int(config.get("eval_batch_size", 1)), shuffle=False):
        batch_samples = [samples[index] for index in group]
        prompts = [build_prompt(sample) for sample in batch_samples]
        rollouts = generate_rollouts(
            tokenizer,
            model,
            prompts,
            1,
            config,
            config.get("eval_model_name_or_path", config.get("rollout_model_name_or_path", config["model_name_or_path"])),
        )
        results.extend(evaluate_output(row["response"], sample["numbers"], sample["target"]) for row, sample in zip(rollouts, batch_samples))
    return summarize_results(results)


def main():
    args = parse_args()
    config = load_config(args.config)
    final_dir = train_grpo(args.config, config, args.limit)
    print(json.dumps({"final_dir": str(final_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
