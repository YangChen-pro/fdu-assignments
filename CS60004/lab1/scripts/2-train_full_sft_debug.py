import argparse
import math
import os
import time

import torch
from torch.optim import Optimizer
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from train_utils import (
    build_dataloaders,
    build_run_dir,
    evaluate_loss,
    load_yaml_config,
    save_checkpoint,
    save_config_snapshot,
    setup_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Task 2 全参数监督微调训练")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML 配置文件路径。",
    )
    return parser.parse_args()

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
    first_param = next(model.parameters())
    state = optimizer.state[first_param]
    print(f"param dtype      : {first_param.dtype}")
    print(f"grad dtype       : {first_param.grad.dtype}")
    print(f"exp_avg dtype    : {state['exp_avg'].dtype}")
    print(f"exp_avg_sq dtype : {state['exp_avg_sq'].dtype}")


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    experiment_name, run_dir = build_run_dir(config["output"]["output_root"])
    save_config_snapshot(args.config, run_dir)

    setup_seed(config["train"]["seed"])

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Task2 training requires a CUDA device.")

    run = maybe_init_swanlab(config, experiment_name)

    model_name_or_path = config["model"]["model_name_or_path"]
    max_length = config["model"]["max_length"]
    use_bf16 = config["model"]["bf16"]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.to(device)

    train_loader, dev_loader, source_stats, total_samples, train_size, dev_size = build_dataloaders(
        config=config,
        tokenizer=tokenizer,
    )

    print(f"Experiment    : {experiment_name}")
    print(f"Run dir       : {run_dir}")
    print(f"Total samples : {total_samples}")
    print(f"Source stats  : {source_stats}")
    print(f"Split         : train={train_size}, dev={dev_size}")
    print(f"Max length    : {max_length}")

    optimizer = AdamWFP32State(
        model.parameters(),
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

    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
    global_step = 0
    optimizer_step = 0
    best_dev_loss = None
    log_steps = config["logging"]["log_steps"]
    save_steps = config["logging"]["save_steps"]
    grad_clip = config["train"]["grad_clip"]
    printed_optimizer_state = False

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

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
                    save_checkpoint(
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
                save_checkpoint(
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
        save_checkpoint(
            save_dir=latest_path,
            model=model,
            tokenizer=tokenizer,
        )

    print("Training finished.")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()


# export CUDA_VISIBLE_DEVICES='3'
# export SWANLAB_API_KEY='your_key'
# python scripts/2-train_full_sft_debug.py --config configs/task2_plan_a.yaml
# python scripts/2-train_full_sft_debug.py --config configs/task2_plan_b.yaml
# python scripts/2-train_full_sft_debug.py --config configs/task2_plan_ab_clean.yaml
