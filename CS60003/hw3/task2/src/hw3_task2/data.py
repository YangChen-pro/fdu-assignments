from __future__ import annotations

import io
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from tqdm import tqdm


@dataclass(frozen=True)
class EpisodeRef:
    parquet_path: str
    split: str
    rows: int


@dataclass(frozen=True)
class FrameRef:
    episode_index: int
    row_index: int


class _TableCache:
    def __init__(self, max_items: int = 8) -> None:
        self.max_items = max_items
        self.items: OrderedDict[str, dict] = OrderedDict()

    def get(self, path: str) -> dict:
        if path in self.items:
            self.items.move_to_end(path)
            return self.items[path]
        table = pq.read_table(
            path,
            columns=["image", "wrist_image", "state", "actions", "task_index"],
        ).to_pydict()
        self.items[path] = table
        self.items.move_to_end(path)
        while len(self.items) > self.max_items:
            self.items.popitem(last=False)
        return table


class CalvinActDataset(Dataset):
    """LeRobot-format CALVIN parquet dataset for action-chunking training."""

    def __init__(
        self,
        root: str | Path,
        splits: list[str],
        image_size: int,
        chunk_size: int,
        max_episodes: int | None = None,
        use_wrist_image: bool = True,
        seed: int = 42,
        show_progress: bool = True,
    ) -> None:
        self.root = Path(root)
        self.splits = splits
        self.image_size = image_size
        self.chunk_size = chunk_size
        self.use_wrist_image = use_wrist_image
        self.episodes = self._discover_episodes(max_episodes, seed, show_progress)
        self.frames = self._build_frame_index()
        self.cache = _TableCache()

    def _discover_episodes(self, max_episodes: int | None, seed: int, show_progress: bool) -> list[EpisodeRef]:
        episodes: list[EpisodeRef] = []
        for split in self.splits:
            files = sorted((self.root / split / "data").rglob("*.parquet"))
            if max_episodes is not None:
                rng = random.Random(seed + sum(ord(c) for c in split))
                files = sorted(rng.sample(files, min(max_episodes, len(files))))
            iterator = tqdm(files, desc=f"index {split}", disable=not show_progress)
            for path in iterator:
                rows = pq.ParquetFile(path).metadata.num_rows
                if rows > 0:
                    episodes.append(EpisodeRef(str(path), split, rows))
        if not episodes:
            raise FileNotFoundError(f"no parquet episodes found under {self.root} for {self.splits}")
        return episodes

    def _build_frame_index(self) -> list[FrameRef]:
        frames: list[FrameRef] = []
        for episode_idx, episode in enumerate(self.episodes):
            for row_idx in range(episode.rows):
                frames.append(FrameRef(episode_idx, row_idx))
        return frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        frame = self.frames[index]
        episode = self.episodes[frame.episode_index]
        table = self.cache.get(episode.parquet_path)
        row = frame.row_index
        image = self._decode_image(table["image"][row])
        wrist = self._decode_image(table["wrist_image"][row]) if self.use_wrist_image else None
        state = torch.tensor(table["state"][row], dtype=torch.float32)
        task_index = torch.tensor(int(table["task_index"][row]), dtype=torch.long)
        actions, valid = self._action_chunk(table["actions"], row)
        return {
            "image": image,
            **({"wrist_image": wrist} if wrist is not None else {}),
            "state": state,
            "task_index": task_index,
            "episode_index": torch.tensor(frame.episode_index, dtype=torch.long),
            "row_index": torch.tensor(row, dtype=torch.long),
            "actions": actions,
            "valid": valid,
        }

    def _decode_image(self, value: dict) -> torch.Tensor:
        data = value.get("bytes")
        if data is None and value.get("path"):
            data = (self.root / value["path"]).read_bytes()
        if data is None:
            raise ValueError("image row has neither bytes nor path")
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img = F.resize(img, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        return F.to_tensor(img)

    def _action_chunk(self, actions_column: list, row: int) -> tuple[torch.Tensor, torch.Tensor]:
        action_dim = len(actions_column[row])
        chunk = np.zeros((self.chunk_size, action_dim), dtype=np.float32)
        valid = np.zeros((self.chunk_size,), dtype=np.float32)
        end = min(row + self.chunk_size, len(actions_column))
        for out_idx, src_idx in enumerate(range(row, end)):
            chunk[out_idx] = np.asarray(actions_column[src_idx], dtype=np.float32)
            valid[out_idx] = 1.0
        return torch.from_numpy(chunk), torch.from_numpy(valid)


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


def summarize_dataset(root: str | Path, splits: list[str]) -> dict[str, dict]:
    root = Path(root)
    summary: dict[str, dict] = {}
    for split in splits:
        info = json.loads((root / split / "meta" / "info.json").read_text())
        files = sorted((root / split / "data").rglob("*.parquet"))
        rows = sum(pq.ParquetFile(path).metadata.num_rows for path in tqdm(files, desc=f"scan {split}"))
        summary[split] = {
            "episodes": len(files),
            "frames": rows,
            "fps": info.get("fps"),
            "features": list(info.get("features", {}).keys()),
        }
    return summary
