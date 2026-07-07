"""Stanford Background dataset loading for U-Net segmentation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import CLASS_NAMES, IGNORE_INDEX

_RESAMPLING = getattr(Image, "Resampling", Image)
_TRANSPOSE = getattr(Image, "Transpose", Image)


class StanfordBackgroundDataset(Dataset):
    """Read Stanford Background images and semantic `.regions.txt` masks."""

    def __init__(self, data_config: dict[str, Any], split: str, train: bool) -> None:
        self.root = Path(data_config["root"])
        self.split_file = Path(data_config["split_dir"]) / f"{split}.txt"
        self.image_size = tuple(int(v) for v in data_config.get("image_size", [240, 320]))
        self.ignore_index = int(data_config.get("ignore_index", IGNORE_INDEX))
        self.mean = torch.tensor(data_config.get("mean", [0.485, 0.456, 0.406]), dtype=torch.float32)
        self.std = torch.tensor(data_config.get("std", [0.229, 0.224, 0.225]), dtype=torch.float32)
        self.augment = data_config.get("augment", {}) if train else {}
        self.ids = _read_split(self.split_file)
        if not self.ids:
            raise ValueError(f"Empty split file: {self.split_file}")
        self._validate_paths()

    def __len__(self) -> int:
        """Return the number of images in this split."""
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Return normalized image tensor, target mask tensor and image id."""
        image_id = self.ids[index]
        image = Image.open(self.root / "images" / f"{image_id}.jpg").convert("RGB")
        mask = _load_mask(self.root / "labels" / f"{image_id}.regions.txt", self.ignore_index)
        scale_range = self.augment.get("random_scale")
        if scale_range:
            image, mask = _random_resized_crop_pair(
                image,
                mask,
                self.image_size,
                scale_range,
                self.ignore_index,
                self.augment.get("rare_class_crop"),
            )
        else:
            image, mask = _resize_pair(image, mask, self.image_size)
        image, mask = self._augment(image, mask)
        return _image_to_tensor(image, self.mean, self.std), torch.from_numpy(mask.astype(np.int64)), image_id

    def _augment(self, image: Image.Image, mask: np.ndarray) -> tuple[Image.Image, np.ndarray]:
        flip_prob = float(self.augment.get("horizontal_flip", 0.0))
        if flip_prob > 0 and random.random() < flip_prob:
            image = image.transpose(_TRANSPOSE.FLIP_LEFT_RIGHT)
            mask = np.fliplr(mask).copy()

        jitter = float(self.augment.get("color_jitter", 0.0))
        if jitter > 0:
            image = _apply_color_jitter(image, jitter)
        return image, mask

    def _validate_paths(self) -> None:
        missing: list[str] = []
        for image_id in self.ids[:10]:
            if not (self.root / "images" / f"{image_id}.jpg").is_file():
                missing.append(f"images/{image_id}.jpg")
            if not (self.root / "labels" / f"{image_id}.regions.txt").is_file():
                missing.append(f"labels/{image_id}.regions.txt")
        if missing:
            raise FileNotFoundError(f"Missing Stanford Background files: {missing[:3]}")


def build_loaders(data_config: dict[str, Any], device: torch.device) -> dict[str, DataLoader]:
    """Build train and validation DataLoaders."""
    split_dir = Path(data_config["split_dir"])
    create_splits(Path(data_config["root"]), split_dir, float(data_config.get("train_ratio", 0.8)))
    batch_size = int(data_config.get("batch_size", 8))
    num_workers = int(data_config.get("num_workers", 4))
    pin_memory = bool(data_config.get("pin_memory", True)) and device.type == "cuda"
    train_set = StanfordBackgroundDataset(data_config, split="train", train=True)
    val_set = StanfordBackgroundDataset(data_config, split="val", train=False)
    train_sampler = _build_train_sampler(train_set, data_config)
    return {
        "train": DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
        "val": DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
    }


def create_splits(data_root: Path, split_dir: Path, train_ratio: float, seed: int = 42) -> None:
    """Create deterministic train/val split files if they are missing."""
    train_path = split_dir / "train.txt"
    val_path = split_dir / "val.txt"
    if train_path.is_file() and val_path.is_file():
        return
    image_ids = sorted(path.stem for path in (data_root / "images").glob("*.jpg"))
    if not image_ids:
        raise FileNotFoundError(f"No Stanford Background images found under {data_root / 'images'}")
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    train_count = int(round(len(image_ids) * train_ratio))
    train_ids = sorted(image_ids[:train_count])
    val_ids = sorted(image_ids[train_count:])
    split_dir.mkdir(parents=True, exist_ok=True)
    train_path.write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(val_ids) + "\n", encoding="utf-8")


def validate_dataset(data_root: str | Path, split_dir: str | Path) -> dict[str, Any]:
    """Return lightweight dataset statistics used in experiment records."""
    root = Path(data_root)
    split_path = Path(split_dir)
    image_ids = sorted(path.stem for path in (root / "images").glob("*.jpg"))
    region_ids = sorted(path.name.replace(".regions.txt", "") for path in (root / "labels").glob("*.regions.txt"))
    train_ids = _read_split(split_path / "train.txt") if (split_path / "train.txt").is_file() else []
    val_ids = _read_split(split_path / "val.txt") if (split_path / "val.txt").is_file() else []
    return {
        "num_images": len(image_ids),
        "num_region_labels": len(region_ids),
        "matched_pairs": len(set(image_ids) & set(region_ids)),
        "train_size": len(train_ids),
        "val_size": len(val_ids),
        "classes": CLASS_NAMES,
        "ignore_index": IGNORE_INDEX,
    }


def compute_class_weights(
    data_config: dict[str, Any],
    num_classes: int,
    method: str = "inverse_sqrt",
    max_weight: float = 2.5,
) -> list[float]:
    """Compute normalized class weights from the training masks."""
    split_file = Path(data_config["split_dir"]) / "train.txt"
    ids = _read_split(split_file)
    if not ids:
        raise ValueError(f"Cannot compute class weights from empty split: {split_file}")
    counts = np.zeros(num_classes, dtype=np.float64)
    root = Path(data_config["root"])
    ignore_index = int(data_config.get("ignore_index", IGNORE_INDEX))
    for image_id in ids:
        mask = _load_mask(root / "labels" / f"{image_id}.regions.txt", ignore_index)
        valid = mask != ignore_index
        counts += np.bincount(mask[valid].astype(np.int64), minlength=num_classes)[:num_classes]
    freqs = counts / counts.sum().clip(min=1.0)
    if method == "inverse":
        weights = 1.0 / np.maximum(freqs, 1.0e-8)
    elif method == "inverse_sqrt":
        weights = 1.0 / np.sqrt(np.maximum(freqs, 1.0e-8))
    elif method == "median_frequency":
        weights = np.median(freqs[freqs > 0]) / np.maximum(freqs, 1.0e-8)
    else:
        raise ValueError(f"Unsupported class weight method: {method}")
    weights = weights / weights.mean().clip(min=1.0e-8)
    if max_weight > 0:
        weights = np.minimum(weights, max_weight)
        weights = weights / weights.mean().clip(min=1.0e-8)
    return [float(value) for value in weights]


def _build_train_sampler(
    dataset: StanfordBackgroundDataset,
    data_config: dict[str, Any],
) -> WeightedRandomSampler | None:
    sampler_config = data_config.get("sampler", {})
    if not isinstance(sampler_config, dict) or not bool(sampler_config.get("enabled", False)):
        return None
    class_id = int(sampler_config.get("rare_class_id", 6))
    positive_weight = float(sampler_config.get("rare_class_weight", 2.0))
    epoch_multiplier = float(sampler_config.get("epoch_multiplier", 1.0))
    weights: list[float] = []
    for image_id in dataset.ids:
        mask = _load_mask(dataset.root / "labels" / f"{image_id}.regions.txt", dataset.ignore_index)
        weights.append(positive_weight if np.any(mask == class_id) else 1.0)
    num_samples = max(len(weights), int(round(len(weights) * epoch_multiplier)))
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=num_samples, replacement=True)


def _read_split(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_mask(path: Path, ignore_index: int) -> np.ndarray:
    mask = np.loadtxt(path, dtype=np.int64)
    return np.where(mask < 0, ignore_index, mask).astype(np.uint8)


def _resize_pair(image: Image.Image, mask: np.ndarray, image_size: tuple[int, int]) -> tuple[Image.Image, np.ndarray]:
    height, width = image_size
    image = image.resize((width, height), _RESAMPLING.BILINEAR)
    mask_image = Image.fromarray(mask).resize((width, height), _RESAMPLING.NEAREST)
    return image, np.asarray(mask_image, dtype=np.uint8)


def _random_resized_crop_pair(
    image: Image.Image,
    mask: np.ndarray,
    image_size: tuple[int, int],
    scale_range: list[float] | tuple[float, float],
    ignore_index: int,
    rare_class_crop: dict[str, Any] | None = None,
) -> tuple[Image.Image, np.ndarray]:
    target_h, target_w = image_size
    min_scale, max_scale = float(scale_range[0]), float(scale_range[1])
    scale = random.uniform(min_scale, max_scale)
    resized_h = max(1, int(round(target_h * scale)))
    resized_w = max(1, int(round(target_w * scale)))
    image = image.resize((resized_w, resized_h), _RESAMPLING.BILINEAR)
    mask_image = Image.fromarray(mask).resize((resized_w, resized_h), _RESAMPLING.NEAREST)

    canvas_w = max(target_w, resized_w)
    canvas_h = max(target_h, resized_h)
    if resized_w < target_w or resized_h < target_h:
        left = random.randint(0, canvas_w - resized_w)
        top = random.randint(0, canvas_h - resized_h)
        image_canvas = Image.new("RGB", (canvas_w, canvas_h), color=(0, 0, 0))
        mask_canvas = Image.new("L", (canvas_w, canvas_h), color=ignore_index)
        image_canvas.paste(image, (left, top))
        mask_canvas.paste(mask_image, (left, top))
        image, mask_image = image_canvas, mask_canvas

    box = _sample_crop_box(mask_image, (target_h, target_w), rare_class_crop)
    return image.crop(box), np.asarray(mask_image.crop(box), dtype=np.uint8)


def _sample_crop_box(
    mask_image: Image.Image,
    image_size: tuple[int, int],
    rare_class_crop: dict[str, Any] | None,
) -> tuple[int, int, int, int]:
    target_h, target_w = image_size
    width, height = mask_image.size
    if rare_class_crop and random.random() < float(rare_class_crop.get("probability", 0.0)):
        mask = np.asarray(mask_image, dtype=np.uint8)
        class_id = _choose_rare_class(rare_class_crop)
        min_pixels = int(rare_class_crop.get("min_pixels", 64))
        attempts = int(rare_class_crop.get("attempts", 12))
        coords = np.argwhere(mask == class_id)
        best_box: tuple[int, int, int, int] | None = None
        best_pixels = 0
        for _ in range(max(attempts, 1)):
            if coords.size == 0:
                break
            y, x = coords[random.randrange(len(coords))]
            left = _sample_offset_containing_point(int(x), target_w, width)
            top = _sample_offset_containing_point(int(y), target_h, height)
            box = (left, top, left + target_w, top + target_h)
            rare_pixels = int(np.count_nonzero(mask[top : top + target_h, left : left + target_w] == class_id))
            if rare_pixels >= min_pixels:
                return box
            if rare_pixels > best_pixels:
                best_pixels = rare_pixels
                best_box = box
        if best_box is not None and best_pixels > 0:
            return best_box
    left = random.randint(0, width - target_w)
    top = random.randint(0, height - target_h)
    return left, top, left + target_w, top + target_h


def _sample_offset_containing_point(point: int, crop_size: int, full_size: int) -> int:
    min_offset = max(0, point - crop_size + 1)
    max_offset = min(point, full_size - crop_size)
    if min_offset > max_offset:
        return random.randint(0, full_size - crop_size)
    return random.randint(min_offset, max_offset)


def _choose_rare_class(config: dict[str, Any]) -> int:
    class_ids = config.get("class_ids")
    if isinstance(class_ids, list) and class_ids:
        weights = config.get("class_weights")
        if isinstance(weights, list) and len(weights) == len(class_ids):
            return int(random.choices(class_ids, weights=[float(weight) for weight in weights], k=1)[0])
        return int(random.choice(class_ids))
    return int(config.get("class_id", 6))


def _image_to_tensor(image: Image.Image, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - mean[:, None, None]) / std[:, None, None]


def _apply_color_jitter(image: Image.Image, strength: float) -> Image.Image:
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = 1.0 + random.uniform(-strength, strength)
        image = enhancer(image).enhance(factor)
    return image
