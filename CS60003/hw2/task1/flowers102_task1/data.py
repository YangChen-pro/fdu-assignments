"""Flowers102 dataset loading with the official train/val/test split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from scipy.io import loadmat
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


SPLIT_KEYS = {
    "train": "trnid",
    "val": "valid",
    "test": "tstid",
}

EXPECTED_SPLIT_SIZES = {
    "train": 1020,
    "val": 1020,
    "test": 6149,
}


@dataclass(frozen=True)
class Flowers102Sample:
    """Single image sample resolved from the official split metadata."""

    image_id: int
    path: Path
    target: int


class Flowers102Dataset(Dataset[tuple[torch.Tensor, int]]):
    """PyTorch dataset for Oxford Flowers102 using local official files."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        if split not in SPLIT_KEYS:
            raise ValueError(f"Unsupported split '{split}', expected one of {sorted(SPLIT_KEYS)}")

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = load_split_samples(self.root, split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, sample.target


def load_split_samples(root: str | Path, split: str) -> list[Flowers102Sample]:
    """Read Flowers102 samples for a split from `imagelabels.mat` and `setid.mat`."""
    root_path = Path(root)
    labels_path = root_path / "imagelabels.mat"
    split_path = root_path / "setid.mat"
    image_dir = root_path / "jpg"

    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    labels = np.asarray(loadmat(labels_path)["labels"]).reshape(-1)
    split_ids = np.asarray(loadmat(split_path)[SPLIT_KEYS[split]]).reshape(-1)

    samples: list[Flowers102Sample] = []
    for raw_id in split_ids:
        image_id = int(raw_id)
        label_index = int(labels[image_id - 1]) - 1
        image_path = image_dir / f"image_{image_id:05d}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image file: {image_path}")
        samples.append(Flowers102Sample(image_id=image_id, path=image_path, target=label_index))
    return samples


def validate_dataset(root: str | Path) -> dict[str, int]:
    """Validate official split sizes and label ranges."""
    stats: dict[str, int] = {}
    for split, expected_size in EXPECTED_SPLIT_SIZES.items():
        samples = load_split_samples(root, split)
        if len(samples) != expected_size:
            raise ValueError(f"{split} split has {len(samples)} samples, expected {expected_size}")
        targets = [sample.target for sample in samples]
        if min(targets) < 0 or max(targets) >= 102:
            raise ValueError(f"{split} targets are outside [0, 101]")
        stats[split] = len(samples)
    return stats


def build_transforms(data_config: dict, train: bool) -> transforms.Compose:
    """Create ImageNet-style transforms for fine-tuning."""
    image_size = int(data_config.get("image_size", 224))
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    if train:
        return _build_train_transforms(data_config, normalize)
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def build_loaders(data_config: dict, device: torch.device) -> dict[str, DataLoader]:
    """Build train/val/test dataloaders from a config dictionary."""
    root = data_config["root"]
    batch_size = int(data_config.get("batch_size", 64))
    num_workers = int(data_config.get("num_workers", 4))
    pin_memory = bool(data_config.get("pin_memory", device.type == "cuda"))

    datasets = {
        "train": Flowers102Dataset(root, "train", build_transforms(data_config, train=True)),
        "val": Flowers102Dataset(root, "val", build_transforms(data_config, train=False)),
        "test": Flowers102Dataset(root, "test", build_transforms(data_config, train=False)),
    }
    return {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for split, dataset in datasets.items()
    }


def _build_train_transforms(data_config: dict, normalize: transforms.Normalize) -> transforms.Compose:
    image_size = int(data_config.get("image_size", 224))
    augment = str(data_config.get("augment", "basic")).lower()
    if augment == "strong":
        steps = _strong_augmentation_steps(data_config, image_size)
    elif augment == "mild":
        steps = _mild_augmentation_steps(image_size)
    elif augment == "none":
        steps = [transforms.Resize(int(image_size * 256 / 224)), transforms.CenterCrop(image_size)]
    else:
        steps = _basic_augmentation_steps(image_size)
    steps.extend([transforms.ToTensor(), normalize])
    erasing = float(data_config.get("random_erasing", 0.0))
    if erasing > 0:
        steps.append(transforms.RandomErasing(p=erasing, value="random"))
    return transforms.Compose(steps)


def _basic_augmentation_steps(image_size: int) -> list:
    return [
        transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ]


def _mild_augmentation_steps(image_size: int) -> list:
    return [
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    ]


def _strong_augmentation_steps(data_config: dict, image_size: int) -> list:
    ops = int(data_config.get("randaugment_ops", 2))
    magnitude = int(data_config.get("randaugment_magnitude", 9))
    return [
        transforms.RandomResizedCrop(image_size, scale=(0.55, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=ops, magnitude=magnitude),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    ]
