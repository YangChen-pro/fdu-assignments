"""Shared utilities for HW3 Task1."""

from __future__ import annotations

import csv
import json
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def save_json(path: str | Path, data: Any) -> None:
    """Write JSON with stable UTF-8 formatting."""
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write a CSV from dictionaries."""
    rows = list(rows)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_run_dir(output_root: str | Path, name: str) -> Path:
    """Create a deterministic run directory."""
    run_dir = Path(output_root) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def copy_source_config(config_path: str | Path, run_dir: Path) -> None:
    """Copy the source YAML config into a run directory."""
    shutil.copy2(config_path, run_dir / "source_config.yaml")


def command_status(names: list[str]) -> dict[str, str]:
    """Return executable discovery status without failing on missing tools."""
    return {name: shutil.which(name) or "" for name in names}


def environment_summary() -> dict[str, Any]:
    """Return a compact runtime environment summary."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
