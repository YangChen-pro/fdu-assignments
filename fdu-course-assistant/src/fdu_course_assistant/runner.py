"""Course enrollment runner and logging."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Protocol

from .models import AppConfig, CourseTarget, RunWindow, SubmitResult, SubmitStatus

LogSink = Callable[[str], None]


class CourseClient(Protocol):
    """Minimal client interface used by the runner."""

    def fetch_csrf_token(self) -> str:
        """Return a current csrfToken."""

    def submit_course(self, target: CourseTarget, course_id: str, csrf_token: str) -> SubmitResult:
        """Submit one course ID and return a normalized result."""


@dataclass
class RunSummary:
    """Summary of one assistant run."""

    started_at: datetime
    ended_at: datetime | None = None
    results: list[SubmitResult] = field(default_factory=list)
    completed_ids: set[str] = field(default_factory=set)
    cycles: int = 0

    @property
    def success_count(self) -> int:
        return sum(1 for result in self.results if result.status is SubmitStatus.SUCCESS)


def run_enrollment(
    config: AppConfig,
    window: RunWindow,
    client: CourseClient,
    log: LogSink = print,
    wait: bool = True,
    once: bool = False,
) -> RunSummary:
    """Run enrollment until all targets complete, deadline arrives, or one cycle finishes."""

    summary = RunSummary(started_at=datetime.now())
    if wait:
        _wait_until(window.start_at, log)
    else:
        log(f"跳过等待，计划开始时间：{window.start_at:%Y-%m-%d %H:%M:%S}")
    log(f"开始运行，截止时间：{window.end_at:%Y-%m-%d %H:%M:%S}")
    if once:
        log("单轮验证模式：本次只提交一轮，不进入持续抢课循环")

    remaining = {course_id for target in config.courses for course_id in target.ids}
    while remaining and datetime.now() < window.end_at:
        summary.cycles += 1
        csrf_token = client.fetch_csrf_token()
        for target in config.courses:
            for course_id in target.ids:
                if course_id not in remaining:
                    continue
                result = client.submit_course(target, course_id, csrf_token)
                summary.results.append(result)
                _log_result(result, log)
                if result.ok:
                    remaining.remove(course_id)
                    summary.completed_ids.add(course_id)
                elif result.status is SubmitStatus.AUTH_EXPIRED:
                    raise RuntimeError(f"认证失效：{result.message}")
        if not remaining or once:
            break
        time.sleep(config.poll_interval_seconds)
    summary.ended_at = datetime.now()
    if remaining:
        log("运行结束，仍未完成：" + ", ".join(sorted(remaining)))
    else:
        log("全部课程目标已完成")
    return summary


def _wait_until(start_at: datetime, log: LogSink) -> None:
    seconds = (start_at - datetime.now()).total_seconds()
    if seconds <= 0:
        return
    log(f"当前时间：{datetime.now():%Y-%m-%d %H:%M:%S}，将在 {int(seconds)} 秒后开始")
    time.sleep(seconds)


def _log_result(result: SubmitResult, log: LogSink) -> None:
    timestamp = result.submitted_at.strftime("%H:%M:%S")
    if result.status is SubmitStatus.SUCCESS:
        log(f"[{timestamp}] {result.category} {result.course_id}: 成功 - {result.message}")
    elif result.status is SubmitStatus.PENDING:
        log(f"[{timestamp}] {result.category} {result.course_id}: 等待 - {result.message}")
    else:
        log(f"[{timestamp}] {result.category} {result.course_id}: {result.status.value} - {result.message}")
