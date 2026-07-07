from __future__ import annotations

import os
import re
from pathlib import Path

_ASSIGN_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_UNSET_RE = re.compile(r"^unset\s+([A-Za-z_][A-Za-z0-9_]*)$")


def load_env_file(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        unset_match = _UNSET_RE.match(stripped)
        if unset_match:
            os.environ.pop(unset_match.group(1), None)
            continue
        match = _ASSIGN_RE.match(stripped)
        if not match:
            continue
        key, value = match.groups()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
            values[key] = value
    return values


def has_any_key(names: list[str]) -> bool:
    return any(os.environ.get(name) for name in names)
