"""EuroSAT 数据加载与预处理。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SHAPE = (64, 64, 3)


@dataclass
class DataSplit:
    """存放在 CPU 内存中的一个数据划分。"""

    images: np.ndarray
    labels: np.ndarray


@dataclass
class DatasetBundle:
    """训练、验证、测试划分及归一化统计量。"""

    train: DataSplit
    val: DataSplit
    test: DataSplit
    mean: np.ndarray
    std: np.ndarray
    class_names: list[str]
    image_shape: tuple[int, int, int]


def load_dataset(
    data_dir: Path,
    output_dir: Path,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    limit_per_class: int | None = None,
    force_rebuild: bool = False,
) -> DatasetBundle:
    """从缓存加载 EuroSAT 数据，或重新构建缓存。"""
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_stem = build_cache_stem(seed, val_ratio, test_ratio, limit_per_class)
    cache_path = cache_dir / f"{cache_stem}.npz"
    meta_path = cache_dir / f"{cache_stem}.json"

    if cache_path.exists() and meta_path.exists() and not force_rebuild:
        return _load_cached_bundle(cache_path, meta_path)

    bundle = _build_dataset(data_dir, seed, val_ratio, test_ratio, limit_per_class)
    np.savez(
        cache_path,
        train_images=bundle.train.images,
        train_labels=bundle.train.labels,
        val_images=bundle.val.images,
        val_labels=bundle.val.labels,
        test_images=bundle.test.images,
        test_labels=bundle.test.labels,
        mean=bundle.mean,
        std=bundle.std,
    )
    meta_path.write_text(
        json.dumps(
            {
                "class_names": bundle.class_names,
                "image_shape": list(bundle.image_shape),
                "seed": seed,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "limit_per_class": limit_per_class,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return bundle


def normalize_images(images: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """将 uint8 图像批次归一化为 float32 特征向量。"""
    features = images.astype(np.float32) / 255.0
    return (features - mean) / std


def build_cache_stem(
    seed: int,
    val_ratio: float,
    test_ratio: float,
    limit_per_class: int | None,
) -> str:
    """构建会随数据划分方案变化的缓存文件名前缀。"""
    limit_tag = "full" if limit_per_class is None else f"limit{limit_per_class}"
    val_tag = _format_ratio_tag("val", val_ratio)
    test_tag = _format_ratio_tag("test", test_ratio)
    return f"eurosat_seed{seed}_{val_tag}_{test_tag}_{limit_tag}"


def iterate_minibatches(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
):
    """从 NumPy 数组中按批次生成小批量数据。"""
    indices = np.arange(images.shape[0])
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield images[batch_indices], labels[batch_indices]


def _load_cached_bundle(cache_path: Path, meta_path: Path) -> DatasetBundle:
    """加载缓存的数据集数组。"""
    payload = np.load(cache_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return DatasetBundle(
        train=DataSplit(payload["train_images"], payload["train_labels"]),
        val=DataSplit(payload["val_images"], payload["val_labels"]),
        test=DataSplit(payload["test_images"], payload["test_labels"]),
        mean=payload["mean"],
        std=payload["std"],
        class_names=list(meta["class_names"]),
        image_shape=tuple(meta["image_shape"]),
    )


def _format_ratio_tag(name: str, ratio: float) -> str:
    """将划分比例转换为稳定且适合缓存命名的标签。"""
    scaled = int(round(ratio * 1000))
    return f"{name}{scaled:03d}"


def _build_dataset(
    data_dir: Path,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    limit_per_class: int | None,
) -> DatasetBundle:
    """从图像目录构建数据集。"""
    class_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    class_names = [path.name for path in class_dirs]

    image_paths: list[Path] = []
    labels: list[int] = []
    for class_index, class_dir in enumerate(class_dirs):
        files = sorted(path for path in class_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if limit_per_class is not None:
            files = files[:limit_per_class]
        image_paths.extend(files)
        labels.extend([class_index] * len(files))

    images = np.empty((len(image_paths), IMAGE_SHAPE[0] * IMAGE_SHAPE[1] * IMAGE_SHAPE[2]), dtype=np.uint8)
    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            rgb = image.convert("RGB").resize((IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
        images[index] = np.asarray(rgb, dtype=np.uint8).reshape(-1)
    labels_array = np.asarray(labels, dtype=np.int64)

    train_indices, val_indices, test_indices = _stratified_split(labels_array, val_ratio, test_ratio, seed)
    train_images = images[train_indices]
    train_labels = labels_array[train_indices]
    val_images = images[val_indices]
    val_labels = labels_array[val_indices]
    test_images = images[test_indices]
    test_labels = labels_array[test_indices]

    mean, std = _compute_mean_std(train_images)
    return DatasetBundle(
        train=DataSplit(train_images, train_labels),
        val=DataSplit(val_images, val_labels),
        test=DataSplit(test_images, test_labels),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        class_names=class_names,
        image_shape=IMAGE_SHAPE,
    )


def _compute_mean_std(images: np.ndarray, chunk_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """分块计算训练集的归一化统计量。"""
    total = images.shape[0]
    channel_sum = np.zeros(IMAGE_SHAPE[2], dtype=np.float64)
    channel_sum_squares = np.zeros(IMAGE_SHAPE[2], dtype=np.float64)
    pixel_count = total * IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
    for start in range(0, total, chunk_size):
        batch = images[start : start + chunk_size].astype(np.float32).reshape(-1, *IMAGE_SHAPE) / 255.0
        channel_sum += batch.sum(axis=(0, 1, 2), dtype=np.float64)
        channel_sum_squares += np.square(batch, dtype=np.float64).sum(axis=(0, 1, 2), dtype=np.float64)
    channel_mean = channel_sum / pixel_count
    channel_variance = np.maximum(channel_sum_squares / pixel_count - np.square(channel_mean), 1e-4)
    channel_std = np.sqrt(channel_variance)
    repeat_count = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
    mean = np.tile(channel_mean, repeat_count)
    std = np.tile(channel_std, repeat_count)
    return mean, std


def _stratified_split(
    labels: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在保持类别平衡的前提下划分数据索引。"""
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for class_id in np.unique(labels):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)
        class_count = len(class_indices)
        test_count = max(1, int(round(class_count * test_ratio)))
        val_count = max(1, int(round(class_count * val_ratio)))
        if test_count + val_count >= class_count:
            val_count = max(1, class_count // 6)
            test_count = max(1, class_count // 6)
        test_parts.append(class_indices[:test_count])
        val_parts.append(class_indices[test_count : test_count + val_count])
        train_parts.append(class_indices[test_count + val_count :])
    return (
        np.concatenate(train_parts),
        np.concatenate(val_parts),
        np.concatenate(test_parts),
    )
