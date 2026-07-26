import os
import random
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from preprocess_utils import read_jsonl


def load_yaml_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def infer_source_name(path):
    filename = Path(path).name.lower()
    if filename == "plan_a_train.jsonl":
        return "a"
    if filename == "plan_b_train.jsonl":
        return "b"
    if filename == "plan_c_train.jsonl":
        return "c"
    return Path(path).stem


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_run_dir(output_root):
    os.makedirs(output_root, exist_ok=True)

    max_version = -1
    for name in os.listdir(output_root):
        match = re.match(r"^v(\d+)[_-]", name)
        if match is None:
            continue
        max_version = max(max_version, int(match.group(1)))

    next_version = max_version + 1
    timestamp = datetime.now().strftime("%m%d%H%M%S")
    run_name = f"v{next_version}-{timestamp}"
    run_dir = os.path.join(output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_name, run_dir


def save_config_snapshot(config_path, run_dir):
    snapshot_path = os.path.join(run_dir, "config.yaml")
    shutil.copy2(config_path, snapshot_path)


def load_samples_from_paths(train_paths):
    merged_samples = []
    for train_path in train_paths:
        source_name = infer_source_name(train_path)
        samples = read_jsonl(train_path)
        for sample in samples:
            new_sample = dict(sample)
            new_sample["_source"] = source_name
            merged_samples.append(new_sample)
    return merged_samples


def split_train_dev(samples, dev_ratio, seed):
    shuffled = list(samples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    if dev_ratio == 0.0:
        return shuffled, []

    dev_size = int(len(shuffled) * dev_ratio)
    if dev_size == 0:
        return shuffled, []

    train_samples = shuffled[:-dev_size]
    dev_samples = shuffled[-dev_size:]
    return train_samples, dev_samples


def encode_sample(sample, tokenizer, max_length):
    messages = [
        {"role": "system", "content": sample["instruction"]},
        {"role": "user", "content": sample["input"]},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    output_ids = tokenizer(sample["output"], add_special_tokens=False).input_ids
    eos_id = tokenizer.eos_token_id

    input_ids = prompt_ids + output_ids + [eos_id]
    labels = [-100] * len(prompt_ids) + output_ids + [eos_id]

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def encode_samples(samples, tokenizer, max_length):
    return [
        encode_sample(
            sample=sample,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        for sample in samples
    ]


class FullSFTDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


class FullSFTCollator:
    def __init__(self, tokenizer, max_length):
        self.pad_token_id = tokenizer.pad_token_id
        self.max_length = max_length

    def __call__(self, features):
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        batch_max_length = max(len(feature["input_ids"]) for feature in features)
        target_length = min(self.max_length, batch_max_length)

        for feature in features:
            input_ids = feature["input_ids"][:target_length]
            labels = feature["labels"][:target_length]
            attention_mask = feature["attention_mask"][:target_length]

            pad_length = target_length - len(input_ids)
            batch_input_ids.append(input_ids + [self.pad_token_id] * pad_length)
            batch_labels.append(labels + [-100] * pad_length)
            batch_attention_mask.append(attention_mask + [0] * pad_length)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }


def build_dataloaders(config, tokenizer):
    train_paths = config["data"]["train_paths"]
    dev_ratio = config["data"]["dev_ratio"]
    max_length = config["model"]["max_length"]
    seed = config["train"]["seed"]
    batch_size = config["train"]["batch_size"]
    num_workers = config["train"]["num_workers"]

    all_samples = load_samples_from_paths(train_paths)
    source_stats = dict(Counter(sample["_source"] for sample in all_samples))
    train_samples, dev_samples = split_train_dev(
        samples=all_samples,
        dev_ratio=dev_ratio,
        seed=seed,
    )

    train_features = encode_samples(
        samples=train_samples,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    dev_features = encode_samples(
        samples=dev_samples,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    collator = FullSFTCollator(tokenizer=tokenizer, max_length=max_length)
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        FullSFTDataset(train_features),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
        generator=generator,
    )

    dev_loader = None
    if dev_features:
        dev_loader = DataLoader(
            FullSFTDataset(dev_features),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collator,
        )

    return train_loader, dev_loader, source_stats, len(all_samples), len(train_samples), len(dev_samples)


@torch.no_grad()
def evaluate_loss(model, dataloader, device, use_bf16):
    model.eval()
    device_type = "cuda" if device.type == "cuda" else "cpu"
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    total_loss = 0.0
    total_steps = 0

    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(
            device_type=device_type,
            dtype=autocast_dtype,
            enabled=(device_type == "cuda"),
        ):
            outputs = model(**batch)
        total_loss += outputs.loss.item()
        total_steps += 1

    model.train()
    if total_steps == 0:
        return 0.0
    return total_loss / total_steps

def save_checkpoint(save_dir, model, tokenizer):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
