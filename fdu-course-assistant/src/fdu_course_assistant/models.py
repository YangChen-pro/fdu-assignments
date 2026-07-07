"""Shared data models for the course assistant."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Iterable

DEFAULT_TARGETS: tuple[str, ...] = ("yjsxk.fudan.edu.cn", "yjsxk.fudan.sh.cn")

COURSE_CLASSIFICATION: dict[str, int] = {
    "学位基础课": 8,
    "专业选修课": 8,
    "学位专业课": 8,
    "公共选修课": 9,
    "第一外国语": 7,
    "政治理论课": 7,
    "专业外语": 7,
}


class SubmitStatus(str, Enum):
    """Normalized result status for a course submission."""

    PENDING = "pending"
    SUCCESS = "success"
    REJECTED = "rejected"
    AUTH_EXPIRED = "auth_expired"
    NETWORK_ERROR = "network_error"
    INVALID_RESPONSE = "invalid_response"


@dataclass(frozen=True)
class CourseTarget:
    """A group of target course IDs under one course category."""

    category: str
    ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.category not in COURSE_CLASSIFICATION:
            supported = ", ".join(COURSE_CLASSIFICATION)
            raise ValueError(f"未知课程类别：{self.category}；支持：{supported}")
        if not self.ids:
            raise ValueError(f"课程类别 {self.category} 至少需要一个教学班代码")

    @property
    def classification_code(self) -> int:
        """Return the numeric `lx` value expected by the FDU endpoint."""

        return COURSE_CLASSIFICATION[self.category]


@dataclass(frozen=True)
class RunWindow:
    """Concrete start and end datetimes for one run."""

    start_at: datetime
    end_at: datetime

    def __post_init__(self) -> None:
        if self.end_at <= self.start_at:
            raise ValueError("end_at 必须晚于 start_at")


@dataclass(frozen=True)
class AppConfig:
    """Resolved application configuration."""

    targets: tuple[str, ...]
    cookie_env: str
    start_at: str
    end_at: str
    poll_interval_seconds: float
    courses: tuple[CourseTarget, ...]

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("targets 不能为空")
        for target in self.targets:
            if not target or "/" in target or target.startswith("http"):
                raise ValueError("targets 必须是域名，不包含 http:// 或路径")
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds 必须大于 0")
        if not self.courses:
            raise ValueError("courses 不能为空")

    @property
    def target(self) -> str:
        """Return the first candidate target for backward-compatible callers."""

        return self.targets[0]


@dataclass(frozen=True)
class SubmitResult:
    """Result returned by one course submission attempt."""

    course_id: str
    category: str
    status: SubmitStatus
    message: str
    raw_code: int | str | None = None
    submitted_at: datetime = field(default_factory=datetime.now)

    @property
    def ok(self) -> bool:
        """Whether this attempt selected the course successfully."""

        return self.status is SubmitStatus.SUCCESS


def flatten_course_ids(courses: Iterable[CourseTarget]) -> list[str]:
    """Return all course IDs from configured targets in deterministic order."""

    return [course_id for target in courses for course_id in target.ids]
