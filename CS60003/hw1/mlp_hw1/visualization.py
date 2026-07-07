"""用于曲线与错误分析的可视化工具。"""

from __future__ import annotations

import matplotlib
import numpy as np
from pathlib import Path


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curves(history: dict, output_path: Path) -> None:
    """绘制训练/验证损失与验证准确率曲线。"""
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(epochs, history["val_loss"], label="val", color="tab:orange")
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")

    axes[2].plot(epochs, history["val_accuracy"], label="val acc", color="tab:green")
    axes[2].set_title("Validation Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0.0, 1.0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], output_path: Path) -> None:
    """绘制混淆矩阵热力图。"""
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black", fontsize=7)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_first_layer_weights(
    weights: np.ndarray,
    image_shape: tuple[int, int, int],
    output_path: Path,
    max_filters: int = 16,
) -> None:
    """将第一层权重可视化为 RGB 图像模式。"""
    count = min(max_filters, weights.shape[1])
    cols = 4
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for index in range(rows * cols):
        ax = axes[index // cols, index % cols]
        ax.axis("off")
        if index >= count:
            continue
        kernel = weights[:, index].reshape(image_shape)
        kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
        ax.imshow(kernel)
        ax.set_title(f"Neuron {index}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_misclassified_examples(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    image_shape: tuple[int, int, int],
    output_path: Path,
    max_examples: int = 12,
) -> None:
    """保存一组测试集误分类样本的网格图。"""
    wrong_indices = np.where(y_true != y_pred)[0][:max_examples]
    if wrong_indices.size == 0:
        return
    cols = 4
    rows = int(np.ceil(len(wrong_indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for index in range(rows * cols):
        ax = axes[index // cols, index % cols]
        ax.axis("off")
        if index >= len(wrong_indices):
            continue
        sample_id = wrong_indices[index]
        ax.imshow(images[sample_id].reshape(image_shape))
        ax.set_title(
            f"T:{class_names[int(y_true[sample_id])]}\nP:{class_names[int(y_pred[sample_id])]}",
            fontsize=8,
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
