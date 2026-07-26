import argparse
import math
import os
import time

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.optim import Optimizer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

from train_utils import (
    build_dataloaders,
    build_run_dir,
    evaluate_loss,
    load_yaml_config,
    save_config_snapshot,
    setup_seed,
)


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Task 3 参数高效监督微调训练")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML 配置文件路径。",
    )
    return parser.parse_args()


def apply_task3_defaults(config):
    config.setdefault("experiment", {})
    config.setdefault("model", {})
    config.setdefault("data", {})
    config.setdefault("train", {})
    config.setdefault("logging", {})
    config.setdefault("output", {})

    peft_config = config.setdefault("peft", {})
    peft_config.setdefault("method", "lora")
    peft_config.setdefault("r", 8)
    peft_config.setdefault("lora_alpha", 16)
    peft_config.setdefault("lora_dropout", 0.05)
    peft_config.setdefault("bias", "none")
    peft_config.setdefault("target_modules", list(DEFAULT_TARGET_MODULES))
    peft_config.setdefault("modules_to_save", None)

    quantization_config = config["model"].setdefault("quantization", {})
    quantization_config.setdefault("enabled", False)
    quantization_config.setdefault("load_in_4bit", False)
    quantization_config.setdefault("bnb_4bit_quant_type", "nf4")
    quantization_config.setdefault("bnb_4bit_compute_dtype", "bfloat16")
    quantization_config.setdefault("bnb_4bit_use_double_quant", True)

    rft_config = config.setdefault("rft", {})
    rft_config.setdefault("train_backend", "lora")
    rft_config.setdefault("candidate_count", 4)
    rft_config.setdefault("source_paths", [])

    return config


def get_effective_backend(config):
    method = config["peft"]["method"].lower()
    if method == "rft":
        return config["rft"]["train_backend"].lower()
    return method


def validate_task3_config(config):
    method = config["peft"]["method"].lower()
    if method not in ["lora", "qlora", "rft"]:
        raise ValueError(f"Unsupported peft.method: {method}")

    backend = get_effective_backend(config)
    if backend not in ["lora", "qlora"]:
        raise ValueError(f"Unsupported training backend: {backend}")

    quantization_enabled = config["model"]["quantization"]["enabled"]
    if backend == "qlora" and not quantization_enabled:
        raise ValueError(
            "QLoRA requires model.quantization.enabled=true and 4-bit quantization settings."
        )
    if backend != "qlora" and quantization_enabled:
        raise ValueError(
            "Quantization is only supported for the QLoRA backend in this script."
        )


def maybe_init_swanlab(config, experiment_name):
    if not config["logging"]["use_swanlab"]:
        return None

    import swanlab as wandb

    wandb.init(
        project=config["logging"]["swanlab_project"],
        name=experiment_name,
        config=config,
    )
    return wandb


class AdamWFP32State(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad_fp32 = grad.float()

                state["step"] += 1
                step = state["step"]

                exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1

                denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                denom.add_(eps)

                param_fp32 = param.float()
                if weight_decay != 0.0:
                    param_fp32.mul_(1.0 - lr * weight_decay)
                param_fp32.addcdiv_(exp_avg, denom, value=-step_size)
                param.copy_(param_fp32.to(param.dtype))

        return loss


def print_first_optimizer_state(model, optimizer):
    first_param = next(param for param in model.parameters() if param.requires_grad)
    state = optimizer.state[first_param]
    print(f"param dtype      : {first_param.dtype}")
    print(f"grad dtype       : {first_param.grad.dtype}")
    print(f"exp_avg dtype    : {state['exp_avg'].dtype}")
    print(f"exp_avg_sq dtype : {state['exp_avg_sq'].dtype}")


def summarize_parameters(model):
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    ratio = 100.0 * trainable_params / max(total_params, 1)
    return total_params, trainable_params, ratio


def save_peft_checkpoint(save_dir, model, tokenizer):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def get_quantization_compute_dtype(name):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported bnb_4bit_compute_dtype: {name}")
    return mapping[name]


def build_method_display_name(config):
    method = config["peft"]["method"].lower()
    if method == "rft":
        return f"rft({config['rft']['train_backend'].lower()})"
    return method


def build_lora_config(config):
    peft_config = config["peft"]
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=peft_config["r"],
        lora_alpha=peft_config["lora_alpha"],
        lora_dropout=peft_config["lora_dropout"],
        bias=peft_config["bias"],
        target_modules=peft_config["target_modules"],
        modules_to_save=peft_config["modules_to_save"],
    )


def build_quantization_config(config):
    quantization_config = config["model"]["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=quantization_config["load_in_4bit"],
        bnb_4bit_quant_type=quantization_config["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=get_quantization_compute_dtype(
            quantization_config["bnb_4bit_compute_dtype"]
        ),
        bnb_4bit_use_double_quant=quantization_config["bnb_4bit_use_double_quant"],
    )


def build_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_peft_model(config, device):
    model_name_or_path = config["model"]["model_name_or_path"]
    use_bf16 = config["model"]["bf16"]
    backend = get_effective_backend(config)

    tokenizer = build_tokenizer(model_name_or_path)
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

    if backend == "qlora":
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            quantization_config=build_quantization_config(config),
            device_map={"": device.index if device.index is not None else 0},
            attn_implementation="flash_attention_2",
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.to(device)

    model = get_peft_model(model, build_lora_config(config))
    return tokenizer, model, backend


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    config = apply_task3_defaults(config)
    validate_task3_config(config)

    experiment_name, run_dir = build_run_dir(config["output"]["output_root"])
    save_config_snapshot(args.config, run_dir)

    setup_seed(config["train"]["seed"])

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Task3 training requires a CUDA device.")

    run = maybe_init_swanlab(config, experiment_name)

    tokenizer, model, backend = build_peft_model(config, device)
    method_name = build_method_display_name(config)
    total_params, trainable_params, trainable_ratio = summarize_parameters(model)

    train_loader, dev_loader, source_stats, total_samples, train_size, dev_size = build_dataloaders(
        config=config,
        tokenizer=tokenizer,
    )

    print(f"Experiment       : {experiment_name}")
    print(f"Run dir          : {run_dir}")
    print(f"Total samples    : {total_samples}")
    print(f"Source stats     : {source_stats}")
    print(f"Split            : train={train_size}, dev={dev_size}")
    print(f"Max length       : {config['model']['max_length']}")
    print(f"PEFT method      : {method_name}")
    print(f"Train backend    : {backend}")
    print(f"Trainable params : {trainable_params:,} / {total_params:,} ({trainable_ratio:.4f}%)")
    print(f"Target modules   : {config['peft']['target_modules']}")

    if backend == "qlora":
        print(f"4-bit quant type : {config['model']['quantization']['bnb_4bit_quant_type']}")
        print(
            "4-bit compute dt : "
            f"{config['model']['quantization']['bnb_4bit_compute_dtype']}"
        )
        print(
            "Double quant     : "
            f"{config['model']['quantization']['bnb_4bit_use_double_quant']}"
        )

    if config["peft"]["method"].lower() == "rft":
        print(f"RFT sources      : {config['rft']['source_paths']}")
        print(f"RFT candidates   : {config['rft']['candidate_count']}")

    optimizer = AdamWFP32State(
        [param for param in model.parameters() if param.requires_grad],
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )

    accumulation_steps = config["train"]["gradient_accumulation_steps"]
    num_epochs = config["train"]["num_epochs"]
    num_update_steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(num_training_steps * config["train"]["warmup_ratio"])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    use_bf16 = config["model"]["bf16"]
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
    global_step = 0
    optimizer_step = 0
    best_dev_loss = None
    log_steps = config["logging"]["log_steps"]
    save_steps = config["logging"]["save_steps"]
    grad_clip = config["train"]["grad_clip"]
    printed_optimizer_state = False
    last_grad_norm = 0.0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_start_time = time.time()
        train_loss_sum = 0.0
        train_loss_count = 0
        num_train_batches = len(train_loader)
        final_window_size = num_train_batches % accumulation_steps or accumulation_steps
        final_window_start = num_train_batches - final_window_size + 1

        for step, batch in enumerate(train_loader, start=1):
            global_step += 1
            batch = {key: value.to(device) for key, value in batch.items()}
            current_accumulation_steps = (
                final_window_size if step >= final_window_start else accumulation_steps
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16 if use_bf16 else torch.float16,
                enabled=True,
            ):
                outputs = model(**batch)
                loss = outputs.loss
                scaled_loss = loss / current_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            train_loss_sum += loss.item()
            train_loss_count += 1

            if step % accumulation_steps == 0 or step == num_train_batches:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    grad_clip,
                )
                last_grad_norm = float(grad_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if not printed_optimizer_state:
                    print_first_optimizer_state(model, optimizer)
                    printed_optimizer_state = True

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                if save_steps > 0 and optimizer_step % save_steps == 0:
                    latest_path = os.path.join(run_dir, "latest")
                    save_peft_checkpoint(
                        save_dir=latest_path,
                        model=model,
                        tokenizer=tokenizer,
                    )

            if step % log_steps == 0 or step == len(train_loader):
                spend_time = time.time() - epoch_start_time
                current_loss = train_loss_sum / max(train_loss_count, 1)
                current_lr = optimizer.param_groups[-1]["lr"]
                eta_min = spend_time / step * len(train_loader) / 60.0 - spend_time / 60.0
                print(
                    f"Epoch:[{epoch + 1}/{num_epochs}]"
                    f"({step}/{len(train_loader)}), "
                    f"loss: {current_loss:.4f}, "
                    f"lr: {current_lr:.8f}, "
                    f"grad_norm: {last_grad_norm:.4f}, "
                    f"epoch_time: {eta_min:.1f}min"
                )
                if run is not None:
                    run.log(
                        {
                            "train/loss": current_loss,
                            "train/lr": current_lr,
                            "train/epoch": epoch + 1,
                            "train/global_step": global_step,
                            "train/optimizer_step": optimizer_step,
                            "train/trainable_params": trainable_params,
                            "train/grad_norm": last_grad_norm,
                            "train/method": method_name,
                        }
                    )

        epoch_train_loss = train_loss_sum / max(train_loss_count, 1)
        print(
            f"Epoch {epoch + 1} finished, "
            f"train_loss={epoch_train_loss:.4f}, "
            f"time={(time.time() - epoch_start_time) / 60.0:.1f}min"
        )

        if dev_loader is not None:
            dev_loss = evaluate_loss(
                model=model,
                dataloader=dev_loader,
                device=device,
                use_bf16=use_bf16,
            )
            improved = best_dev_loss is None or dev_loss < best_dev_loss
            if improved:
                best_dev_loss = dev_loss
                best_path = os.path.join(run_dir, "best")
                save_peft_checkpoint(
                    save_dir=best_path,
                    model=model,
                    tokenizer=tokenizer,
                )

            print(
                f"Eval: epoch={epoch + 1}, "
                f"dev_loss={dev_loss:.4f}, "
                f"best_dev_loss={best_dev_loss:.4f}"
            )
            if run is not None:
                run.log(
                    {
                        "eval/dev_loss": dev_loss,
                        "eval/best_dev_loss": best_dev_loss,
                        "eval/epoch": epoch + 1,
                        "eval/global_step": global_step,
                    }
                )

        latest_path = os.path.join(run_dir, "latest")
        save_peft_checkpoint(
            save_dir=latest_path,
            model=model,
            tokenizer=tokenizer,
        )

    print("Training finished.")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()


# export CUDA_VISIBLE_DEVICES='0'
# export SWANLAB_API_KEY='your_key'
# python3 scripts/3-trian_lora_sft.py --config configs/task3_lora.yaml
# python3 scripts/3-trian_lora_sft.py --config configs/task3_qlora.yaml
# python3 scripts/3-trian_lora_sft.py --config configs/task3_rft.yaml
