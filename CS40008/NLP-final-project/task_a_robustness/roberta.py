"""Train and run a RoBERTa baseline for ComVE Task A."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .data import Prediction, json_note, load_examples, write_predictions
from .paths import CHECKPOINT_DIR, HF_CACHE_DIR, TASK_A_ROBERTA_RESULTS_DIR, ensure_runtime_dirs


DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def configure_hf_endpoint(endpoint: str | None) -> None:
    """Set Hugging Face endpoint before importing transformers."""

    if endpoint:
        os.environ.setdefault("HF_ENDPOINT", endpoint)


class ComveTorchDataset:
    """Torch dataset wrapper around tokenized ComVE examples."""

    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index: int) -> dict[str, object]:
        import torch

        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def __len__(self) -> int:
        return len(self.labels)


def tokenize_examples(examples, tokenizer, max_length: int) -> ComveTorchDataset:
    """Tokenize sent0/sent1 pairs for sequence classification."""

    encodings = tokenizer(
        [item.sent0 for item in examples],
        [item.sent1 for item in examples],
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    return ComveTorchDataset(encodings, [item.gold for item in examples])


def order_augment_examples(examples):
    """Add swapped-order copies so the classifier learns both label positions."""

    augmented = []
    for item in examples:
        augmented.append(item)
        augmented.append(type(item)(
            id=f"{item.id}__train_swap",
            sent0=item.sent1,
            sent1=item.sent0,
            gold=1 - item.gold,
        ))
    return augmented


def train(args: argparse.Namespace) -> None:
    """Fine-tune RoBERTa and save it to the configured checkpoint directory."""

    configure_hf_endpoint(args.hf_endpoint)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    ensure_runtime_dirs()
    train_examples = load_examples(args.train_split)
    if args.order_augment:
        train_examples = order_augment_examples(train_examples)
    eval_examples = load_examples(args.eval_split)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        num_labels=2,
    )
    train_dataset = tokenize_examples(train_examples, tokenizer, args.max_length)
    eval_dataset = tokenize_examples(eval_examples, tokenizer, args.max_length)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


def predict(args: argparse.Namespace) -> None:
    """Load a fine-tuned checkpoint and write prediction CSV."""

    configure_hf_endpoint(args.hf_endpoint)
    import numpy as np
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    ensure_runtime_dirs()
    examples = load_examples(args.split)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, cache_dir=args.cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir,
        cache_dir=args.cache_dir,
    )
    dataset = tokenize_examples(examples, tokenizer, args.max_length)
    trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(args.tmp_dir), report_to=[]))
    raw_output = trainer.predict(dataset)
    labels = np.argmax(raw_output.predictions, axis=1).tolist()
    predictions = [
        Prediction(
            id=example.id,
            method=args.method,
            pred=int(label),
            gold=example.gold,
            correct=int(label == example.gold),
            notes=json_note(split=args.split, model_dir=str(args.model_dir)),
        )
        for example, label in zip(examples, labels)
    ]
    write_predictions(predictions, args.output)


def _compute_metrics(eval_pred) -> dict[str, float]:
    import numpy as np

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": float((predictions == labels).mean())}


def build_parser() -> argparse.ArgumentParser:
    """Build the RoBERTa command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model-name", default="roberta-base")
    train_parser.add_argument("--hf-endpoint", default=DEFAULT_HF_ENDPOINT)
    train_parser.add_argument("--cache-dir", type=Path, default=HF_CACHE_DIR)
    train_parser.add_argument("--output-dir", type=Path, default=CHECKPOINT_DIR)
    train_parser.add_argument("--train-split", default="train")
    train_parser.add_argument("--eval-split", default="dev")
    train_parser.add_argument("--epochs", type=float, default=3)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--max-length", type=int, default=128)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--no-order-augment", action="store_false", dest="order_augment")
    train_parser.set_defaults(func=train)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model-dir", type=Path, default=CHECKPOINT_DIR)
    predict_parser.add_argument("--hf-endpoint", default=DEFAULT_HF_ENDPOINT)
    predict_parser.add_argument("--cache-dir", type=Path, default=HF_CACHE_DIR)
    predict_parser.add_argument("--split", default="test")
    predict_parser.add_argument("--method", default="roberta_finetuned")
    predict_parser.add_argument("--output", type=Path, default=TASK_A_ROBERTA_RESULTS_DIR / "predictions.csv")
    predict_parser.add_argument("--tmp-dir", type=Path, default=CHECKPOINT_DIR / "predict_tmp")
    predict_parser.add_argument("--max-length", type=int, default=128)
    predict_parser.set_defaults(func=predict)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
