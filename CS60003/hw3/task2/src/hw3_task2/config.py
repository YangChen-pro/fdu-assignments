from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str
    train_splits: list[str]
    eval_split: str = "splitD"
    image_size: int = 96
    chunk_size: int = 16
    max_train_episodes: int | None = None
    max_eval_episodes: int | None = None
    use_wrist_image: bool = True


@dataclass
class ModelConfig:
    state_dim: int = 15
    action_dim: int = 7
    hidden_dim: int = 256
    nheads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    use_vae: bool = True
    kl_weight: float = 10.0


@dataclass
class TrainConfig:
    name: str
    output_dir: str
    epochs: int = 8
    batch_size: int = 128
    eval_batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 8
    seed: int = 42
    amp: bool = True
    log_every: int = 50
    eval_every_epochs: int = 1
    save_every_epochs: int = 1
    max_steps_per_epoch: int | None = None
    dry_run_steps: int | None = None


@dataclass
class TrackConfig:
    enable_swanlab: bool = True
    project: str = "CS60003_HW3_Task2"
    workspace: str | None = None
    mode: str = "cloud"
    secret_env: str = ".helloagents/secrets/hw3.env"


@dataclass
class UploadConfig:
    enable_modelscope: bool = True
    repo_id: str = "youngchen/CS60003"
    path_prefix: str = "hw3/task2"
    secret_env: str = ".helloagents/secrets/hw3.env"


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    track: TrackConfig = field(default_factory=TrackConfig)
    upload: UploadConfig = field(default_factory=UploadConfig)


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"missing required config key: {key}")
    return mapping[key]


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return ExperimentConfig(
        data=DataConfig(**_require(raw, "data")),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**_require(raw, "train")),
        track=TrackConfig(**raw.get("track", {})),
        upload=UploadConfig(**raw.get("upload", {})),
    )


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data": config.data.__dict__,
        "model": config.model.__dict__,
        "train": config.train.__dict__,
        "track": config.track.__dict__,
        "upload": config.upload.__dict__,
    }
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))
