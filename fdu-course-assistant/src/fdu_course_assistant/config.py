"""Configuration loading and time parsing."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .models import AppConfig, CourseTarget, DEFAULT_TARGETS, RunWindow

try:  # PyYAML is optional; a small fallback parser handles the example schema.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - exercised when PyYAML is unavailable.
    yaml = None


def load_config(path: str | Path) -> AppConfig:
    """Load an AppConfig from a YAML or JSON file."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    text = config_path.read_text(encoding="utf-8")
    data = _parse_mapping(text, config_path.suffix.lower())
    courses = tuple(
        CourseTarget(category=str(item["category"]), ids=tuple(str(x) for x in item["ids"]))
        for item in data.get("courses", [])
    )
    return AppConfig(
        targets=_resolve_targets(data),
        cookie_env=str(data.get("cookie_env", "FDU_XK_COOKIE")),
        start_at=str(data.get("start_at", "00:00:00")),
        end_at=str(data.get("end_at", "23:59:59")),
        poll_interval_seconds=float(data.get("poll_interval_seconds", 0.3)),
        courses=courses,
    )


def resolve_cookie(config: AppConfig, override: str | None = None) -> str:
    """Resolve Cookie from explicit override or the configured environment variable."""

    cookie = override if override is not None else os.environ.get(config.cookie_env, "")
    if not cookie.strip():
        raise RuntimeError(f"未找到 Cookie。请设置环境变量 {config.cookie_env}，不要写入源码或提交配置。")
    return cookie


def resolve_run_window(start_text: str, end_text: str, now: datetime | None = None) -> RunWindow:
    """Resolve date/time strings into a concrete run window."""

    base = now or datetime.now()
    start_at = _parse_datetime_or_time(start_text, base)
    end_at = _parse_datetime_or_time(end_text, base)
    if start_at < base and len(start_text.strip()) == 8:
        start_at += timedelta(days=1)
    if end_at <= start_at and len(end_text.strip()) == 8:
        end_at += timedelta(days=1)
    return RunWindow(start_at=start_at, end_at=end_at)


def config_preview(config: AppConfig) -> dict[str, Any]:
    """Return a safe, serializable preview of the config without secrets."""

    return {
        "targets": list(config.targets),
        "cookie_env": config.cookie_env,
        "start_at": config.start_at,
        "end_at": config.end_at,
        "poll_interval_seconds": config.poll_interval_seconds,
        "courses": [{"category": c.category, "ids": list(c.ids)} for c in config.courses],
    }


def _resolve_targets(data: dict[str, Any]) -> tuple[str, ...]:
    targets = data.get("targets")
    if targets is None:
        single_target = data.get("target")
        if single_target:
            targets = [single_target]
        else:
            targets = list(DEFAULT_TARGETS)
    if isinstance(targets, str):
        targets = [targets]
    return tuple(str(target).strip() for target in targets if str(target).strip())


def _parse_datetime_or_time(value: str, base: datetime) -> datetime:
    value = value.strip().strip('"').strip("'")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%H:%M:%S"):
        try:
            parsed = datetime.strptime(value, fmt)
            if fmt == "%H:%M:%S":
                return base.replace(hour=parsed.hour, minute=parsed.minute, second=parsed.second, microsecond=0)
            return parsed
        except ValueError:
            pass
    raise ValueError(f"时间格式不支持：{value}")


def _parse_mapping(text: str, suffix: str) -> dict[str, Any]:
    if suffix == ".json":
        return json.loads(text)
    if yaml is not None:
        loaded = yaml.safe_load(text)
        return loaded or {}
    return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    courses: list[dict[str, Any]] = []
    current_course: dict[str, Any] | None = None
    in_ids = False
    in_targets = False
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped == "courses:":
            data["courses"] = courses
            in_targets = False
            continue
        if stripped == "targets:":
            data["targets"] = []
            in_targets = True
            continue
        if in_targets and stripped.startswith("-"):
            data["targets"].append(_clean_scalar(stripped[1:]))
            continue
        if stripped.startswith("- category:"):
            current_course = {"category": _clean_scalar(stripped.split(":", 1)[1]), "ids": []}
            courses.append(current_course)
            in_ids = False
            in_targets = False
            continue
        if stripped == "ids:":
            in_ids = True
            continue
        if in_ids and stripped.startswith("-") and current_course is not None:
            current_course["ids"].append(_clean_scalar(stripped[1:]))
            continue
        if ":" in stripped and not raw_line.startswith(" "):
            key, value = stripped.split(":", 1)
            data[key] = _clean_scalar(value)
            in_targets = False
    return data


def _clean_scalar(value: str) -> str:
    return value.strip().strip('"').strip("'")
