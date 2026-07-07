from datetime import datetime
from pathlib import Path

import pytest

from fdu_course_assistant.cli import build_parser, main
from fdu_course_assistant.client import FduCourseClient, InspectResult
from fdu_course_assistant.config import load_config
from fdu_course_assistant.models import CourseTarget, RunWindow, SubmitResult, SubmitStatus
from fdu_course_assistant.runner import run_enrollment

ROOT = Path(__file__).parents[1]


class FakeClient:
    def __init__(self):
        self.submitted = []

    def fetch_csrf_token(self):
        return "0123456789abcdef0123456789abcdef"

    def submit_course(self, target: CourseTarget, course_id: str, csrf_token: str):
        self.submitted.append(course_id)
        return SubmitResult(course_id, target.category, SubmitStatus.PENDING, "继续尝试", raw_code=0)


def test_run_once_submits_one_cycle_only():
    config = load_config(ROOT / "configs" / "example.yaml")
    now = datetime.now()
    client = FakeClient()
    logs = []
    summary = run_enrollment(
        config,
        RunWindow(start_at=now, end_at=now.replace(hour=23, minute=59, second=59)),
        client,
        log=logs.append,
        wait=False,
        once=True,
    )
    assert summary.cycles == 1
    assert len(client.submitted) == 2
    assert "单轮验证模式" in "\n".join(logs)


def test_cli_help_has_only_real_commands(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    help_text = capsys.readouterr().out
    assert "mock" not in help_text
    assert "web" not in help_text
    assert "inspect" in help_text


def test_cli_check_cookie_requires_cookie(monkeypatch, capsys):
    monkeypatch.delenv("FDU_XK_COOKIE", raising=False)
    assert main(["check-cookie", "--config", str(ROOT / "configs" / "local.example.yaml")]) == 1
    assert "未找到 Cookie" in capsys.readouterr().err


def test_cli_inspect_uses_resolved_target(monkeypatch, capsys):
    monkeypatch.setenv("FDU_XK_COOKIE", "JSESSIONID=fake-token; XK_TOKEN=fake-xk-token")
    monkeypatch.setattr(
        FduCourseClient,
        "inspect",
        lambda self: InspectResult(
            self.settings.target,
            "0123456789abcdef0123456789abcdef",
            "研究生选课",
            "选课页片段",
            128,
        ),
    )
    assert main(["inspect", "--config", str(ROOT / "configs" / "local.example.yaml")]) == 0
    output = capsys.readouterr().out
    assert "命中域名：yjsxk.fudan.edu.cn" in output
    assert "fake-token" not in output
