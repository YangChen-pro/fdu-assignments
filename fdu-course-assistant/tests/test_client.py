import pytest

from fdu_course_assistant.client import FduCourseClient, InspectResult, parse_submit_response, resolve_working_client
from fdu_course_assistant.models import SubmitStatus


def test_parse_pending_response():
    result = parse_submit_response("C1", "政治理论课", '{"code": 0, "msg": "名额已满"}')
    assert result.status is SubmitStatus.PENDING


def test_parse_success_response():
    result = parse_submit_response("C1", "政治理论课", '{"code": 1, "msg": "成功"}')
    assert result.ok


def test_parse_rejected_response():
    result = parse_submit_response("C1", "政治理论课", '{"code": -1, "msg": "不在选课时间"}')
    assert result.status is SubmitStatus.REJECTED


def test_parse_auth_expired_response():
    result = parse_submit_response("C1", "政治理论课", '{"code": -1, "msg": "cookie 过期"}')
    assert result.status is SubmitStatus.AUTH_EXPIRED


def test_parse_invalid_response():
    result = parse_submit_response("C1", "政治理论课", "not json")
    assert result.status is SubmitStatus.INVALID_RESPONSE


def test_resolve_working_client_picks_first_with_token(monkeypatch):
    def fake_inspect(self):
        if self.settings.target == "bad.example":
            return InspectResult(self.settings.target, None, "登录", "no token", 20)
        return InspectResult(self.settings.target, "0123456789abcdef0123456789abcdef", "选课", "ok", 100)

    monkeypatch.setattr(FduCourseClient, "inspect", fake_inspect)
    client, result = resolve_working_client(("bad.example", "good.example"), "a=b")
    assert client.settings.target == "good.example"
    assert result.ok


def test_resolve_working_client_reports_all_failures(monkeypatch):
    monkeypatch.setattr(
        FduCourseClient,
        "inspect",
        lambda self: InspectResult(self.settings.target, None, "登录", "no token", 20),
    )
    with pytest.raises(RuntimeError, match="所有候选域名"):
        resolve_working_client(("bad.example",), "a=b")
