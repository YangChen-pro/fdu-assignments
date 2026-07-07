"""训练、评估与搜索流程。"""

from __future__ import annotations

import csv
import json
from itertools import product
from dataclasses import replace
from datetime import datetime

import numpy as np

from .backend import get_array_module, seed_everything, to_numpy
from .config import SearchConfig, TrainConfig
from .data import DataSplit, iterate_minibatches, load_dataset, normalize_images
from .metrics import accuracy_score, confusion_matrix, per_class_accuracy
from .model import ThreeLayerMLP
from .visualization import (
    plot_confusion_matrix,
    plot_first_layer_weights,
    plot_misclassified_examples,
    plot_training_curves,
)


def train_model(
    config: TrainConfig,
    run_name: str | None = None,
    generate_reports: bool = True,
) -> dict:
    """训练模型并保存实验产物。"""
    xp = get_array_module()
    seed_everything(config.seed)

    dataset = load_dataset(
        data_dir=config.data_dir,
        output_dir=config.output_dir,
        seed=config.seed,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        limit_per_class=config.limit_per_class,
        force_rebuild=config.force_rebuild_cache,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.output_dir / "runs" / (run_name or f"{timestamp}_cupy_{config.activation}_h{config.hidden_dim}")
    run_dir.mkdir(parents=True, exist_ok=True)

    model = ThreeLayerMLP(
        input_dim=dataset.train.images.shape[1],
        hidden_dim=config.hidden_dim,
        hidden_dim2=config.resolved_hidden_dim2(),
        output_dim=len(dataset.class_names),
        activation=config.activation,
        xp=xp,
        seed=config.seed,
        dropout_rate=config.dropout_rate,
    )
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    best_checkpoint = run_dir / "best_model.npz"
    best_val_accuracy = -1.0
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        learning_rate = config.learning_rate / (1.0 + config.lr_decay * (epoch - 1))
        # 梯度与参数更新都在这里手写完成，不依赖任何现成训练框架。
        for batch_images, batch_labels in iterate_minibatches(
            dataset.train.images,
            dataset.train.labels,
            batch_size=config.batch_size,
            seed=config.seed + epoch,
            shuffle=True,
        ):
            features = normalize_images(batch_images, dataset.mean, dataset.std)
            batch_x = xp.asarray(features)
            batch_y = xp.asarray(batch_labels)
            model.loss_and_backward(batch_x, batch_y, config.weight_decay)
            model.step(learning_rate, grad_clip=config.grad_clip)

        train_metrics = evaluate_split(model, dataset.train, dataset.mean, dataset.std, config.eval_batch_size)
        val_metrics = evaluate_split(model, dataset.val, dataset.mean, dataset.std, config.eval_batch_size)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["learning_rate"].append(learning_rate)
        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            model.save(best_checkpoint)

    best_model = ThreeLayerMLP.load(best_checkpoint, xp)
    test_metrics = evaluate_split(
        best_model,
        dataset.test,
        dataset.mean,
        dataset.std,
        config.eval_batch_size,
        return_predictions=True,
    )
    matrix = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], len(dataset.class_names))
    summary = {
        "backend": "cupy",
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "class_names": dataset.class_names,
        "per_class_accuracy": per_class_accuracy(matrix).tolist(),
        "train_samples": int(dataset.train.images.shape[0]),
        "val_samples": int(dataset.val.images.shape[0]),
        "test_samples": int(dataset.test.images.shape[0]),
        "config": config.to_dict(),
    }

    (run_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "confusion_matrix.json").write_text(
        json.dumps(matrix.tolist(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    if generate_reports:
        plot_training_curves(history, run_dir / "training_curves.png")
        plot_confusion_matrix(matrix, dataset.class_names, run_dir / "confusion_matrix.png")
        plot_first_layer_weights(to_numpy(best_model.w1), dataset.image_shape, run_dir / "first_layer_weights.png")
        plot_misclassified_examples(
            dataset.test.images,
            test_metrics["y_true"],
            test_metrics["y_pred"],
            dataset.class_names,
            dataset.image_shape,
            run_dir / "misclassified_examples.png",
        )

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print("Confusion matrix:")
    print(matrix)
    return {
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_checkpoint),
        "summary": summary,
        "history": history,
        "confusion_matrix": matrix,
    }


def evaluate_model(
    config: TrainConfig,
    checkpoint_path: Path,
    output_dir: Path | None = None,
) -> dict:
    """在测试集上评估已保存的模型。"""
    xp = get_array_module()
    dataset = load_dataset(
        data_dir=config.data_dir,
        output_dir=config.output_dir,
        seed=config.seed,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        limit_per_class=config.limit_per_class,
        force_rebuild=config.force_rebuild_cache,
    )
    model = ThreeLayerMLP.load(checkpoint_path, xp)
    metrics = evaluate_split(
        model,
        dataset.test,
        dataset.mean,
        dataset.std,
        config.eval_batch_size,
        return_predictions=True,
    )
    matrix = confusion_matrix(metrics["y_true"], metrics["y_pred"], len(dataset.class_names))
    destination = output_dir or checkpoint_path.parent
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "evaluation_summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "accuracy": metrics["accuracy"],
                "loss": metrics["loss"],
                "class_names": dataset.class_names,
                "per_class_accuracy": per_class_accuracy(matrix).tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_confusion_matrix(matrix, dataset.class_names, destination / "evaluation_confusion_matrix.png")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print("Confusion matrix:")
    print(matrix)
    return {
        "accuracy": metrics["accuracy"],
        "loss": metrics["loss"],
        "confusion_matrix": matrix,
    }


def run_search(config: SearchConfig) -> dict:
    """执行带确定性采样的网格超参数搜索。"""
    candidates = build_search_candidates(config)

    search_dir = config.train_config.output_dir / "search" / datetime.now().strftime("%Y%m%d_%H%M%S")
    search_dir.mkdir(parents=True, exist_ok=True)
    (search_dir / "search_config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows: list[dict] = []
    best_row: dict | None = None
    for trial_id, candidate in enumerate(candidates, start=1):
        trial_config = replace(
            config.train_config,
            learning_rate=candidate["learning_rate"],
            hidden_dim=candidate["hidden_dim"],
            hidden_dim2=candidate["hidden_dim2"],
            weight_decay=candidate["weight_decay"],
            lr_decay=candidate["lr_decay"],
            grad_clip=candidate["grad_clip"],
            activation=candidate["activation"],
        )
        run_result = train_model(
            config=trial_config,
            run_name=f"trial_{trial_id:02d}",
            generate_reports=False,
        )
        row = {
            "trial": trial_id,
            **candidate,
            "best_val_accuracy": run_result["summary"]["best_val_accuracy"],
            "test_accuracy": run_result["summary"]["test_accuracy"],
            "run_dir": run_result["run_dir"],
        }
        rows.append(row)
        if best_row is None or row["best_val_accuracy"] > best_row["best_val_accuracy"]:
            best_row = row
        print(
            f"[Trial {trial_id:02d}] "
            f"lr={row['learning_rate']:.4f} "
            f"hidden=({row['hidden_dim']},{row['hidden_dim2']}) "
            f"decay={row['lr_decay']:.4f} "
            f"wd={row['weight_decay']:.4e} "
            f"clip={row['grad_clip']:.1f} "
            f"act={row['activation']} "
            f"val_acc={row['best_val_accuracy']:.4f}"
        )

    with (search_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (search_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if best_row is not None:
        (search_dir / "best_result.json").write_text(json.dumps(best_row, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "search_dir": str(search_dir),
        "results": rows,
        "best_result": best_row,
    }


def build_search_candidates(config: SearchConfig) -> list[dict]:
    """构造一个能覆盖整体搜索空间的确定性候选子集。"""
    all_candidates = [
        {
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "hidden_dim2": hidden_dim2,
            "weight_decay": weight_decay,
            "lr_decay": lr_decay,
            "grad_clip": grad_clip,
            "activation": activation,
        }
        for learning_rate, hidden_dim, hidden_dim2, weight_decay, lr_decay, grad_clip, activation in product(
            config.learning_rates,
            config.hidden_dims,
            config.hidden_dims2,
            config.weight_decays,
            config.lr_decays,
            config.grad_clips,
            config.activations,
        )
    ]
    if config.max_trials >= len(all_candidates):
        return all_candidates

    rng = np.random.default_rng(config.train_config.seed)
    shuffled_indices = rng.permutation(len(all_candidates))
    shuffled_candidates = [all_candidates[int(index)] for index in shuffled_indices]

    selected: list[dict] = []
    seen: set[tuple] = set()

    def add_candidate(candidate: dict) -> bool:
        if len(selected) >= config.max_trials:
            return False
        key = tuple(candidate.items())
        if key in seen:
            return False
        selected.append(candidate)
        seen.add(key)
        return True

    def cover_dimension(key: str, values: tuple) -> None:
        for value in values:
            for candidate in shuffled_candidates:
                if candidate[key] == value and add_candidate(candidate):
                    break

    # 先保证作业里最关键的几类超参数在候选集合里都有覆盖。
    cover_dimension("learning_rate", config.learning_rates)
    cover_dimension("hidden_dim", config.hidden_dims)
    cover_dimension("hidden_dim2", config.hidden_dims2)
    cover_dimension("weight_decay", config.weight_decays)

    for candidate in shuffled_candidates:
        add_candidate(candidate)
        if len(selected) >= config.max_trials:
            break
    return selected


def evaluate_split(
    model: ThreeLayerMLP,
    split: DataSplit,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    return_predictions: bool = False,
) -> dict:
    """按小批量方式评估一个数据划分。"""
    total_loss = 0.0
    total_samples = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch_images, batch_labels in iterate_minibatches(
        split.images,
        split.labels,
        batch_size=batch_size,
        seed=0,
        shuffle=False,
    ):
        features = normalize_images(batch_images, mean, std)
        batch_x = model.xp.asarray(features)
        batch_y = model.xp.asarray(batch_labels)
        batch_size_actual = int(batch_labels.shape[0])
        total_loss += float(model.compute_loss(batch_x, batch_y)) * batch_size_actual
        total_samples += batch_size_actual
        preds = to_numpy(model.predict(batch_x))
        predictions.append(preds.astype(np.int64))
        targets.append(batch_labels.astype(np.int64))

    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(targets)
    result = {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if return_predictions:
        result["y_true"] = y_true
        result["y_pred"] = y_pred
    return result
