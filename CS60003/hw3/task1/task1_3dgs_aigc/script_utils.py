"""Shared script-generation helpers for Task1 shell orchestration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable


def resolve_swanlab_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract a normalized SwanLab config block from task config."""
    swanlab = config.get("logging", {}).get("swanlab", {})
    return {
        "enabled": bool(swanlab.get("enabled", False)),
        "project": str(swanlab.get("project", "cs60003-hw3-task1")),
        "group": str(swanlab.get("group", "real-high-quality")),
        "mode": str(swanlab.get("mode", "cloud")),
        "tags": list(swanlab.get("tags", [])),
        "env_file": str(swanlab.get("env_file", ".helloagents/secrets/hw3.env")),
    }


def swanlab_tag_args(swanlab: dict[str, Any], extra: Iterable[str]) -> str:
    """Return shell-ready `--tag` arguments for a script call."""
    tags = [*swanlab.get("tags", []), *extra]
    return " ".join(f'--tag "{tag}"' for tag in tags)


def to_cli_list(values: list[int | float | str]) -> str:
    """Format Python list values for a Threestudio CLI array syntax."""
    return "[" + ",".join(str(value) for value in values) + "]"


def load_env_file(path: str | Path) -> None:
    """Load KEY=VALUE pairs from an env file without printing secret values."""
    if not path:
        return
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def write_script(path: Path, text: str) -> Path:
    """Write executable shell text and return the path."""
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path
