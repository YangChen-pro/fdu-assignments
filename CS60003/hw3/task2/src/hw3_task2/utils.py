from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def write_json(path: str | Path, payload: dict | list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def append_csv(path: str | Path, row: dict, fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = fieldnames or list(row.keys())
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def available_gpu_ids(min_free_mib: int = 12_000) -> list[int]:
    if not torch.cuda.is_available():
        return []
    ids: list[int] = []
    for idx in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(idx)
        if free // (1024 * 1024) >= min_free_mib:
            ids.append(idx)
    return ids


def mean_dict(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    keys = rows[0].keys() if rows else []
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}
