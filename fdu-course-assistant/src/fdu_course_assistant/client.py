"""HTTP client implementation for real course enrollment runs."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from .models import CourseTarget, SubmitResult, SubmitStatus
from .session import SessionSettings, extract_csrf_token

TITLE_PATTERN = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class InspectResult:
    """Read-only inspection result for a logged-in choose-course page."""

    target: str
    csrf_token: str | None
    title: str
    page_excerpt: str
    html_length: int

    @property
    def ok(self) -> bool:
        return self.csrf_token is not None


@dataclass
class FduCourseClient:
    """Real HTTP client for FDU graduate course enrollment endpoints."""

    settings: SessionSettings
    timeout_seconds: float = 10.0

    def fetch_choose_course_page(self) -> str:
        """Fetch the logged-in choose-course page without submitting anything."""

        url = f"{self.settings.base_url}/xsxkHome/gotoChooseCourse.do"
        last_error: Exception | None = None
        for attempt in range(3):
            request = urllib.request.Request(url, headers=self.settings.headers, method="GET")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return response.read().decode("utf-8", errors="replace")
            except (urllib.error.URLError, OSError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.6 * (attempt + 1))
        raise RuntimeError(f"访问选课页失败：{last_error}")

    def inspect(self) -> InspectResult:
        """Inspect the choose-course page and return safe, non-secret diagnostics."""

        html = self.fetch_choose_course_page()
        title = _extract_title(html)
        try:
            token = extract_csrf_token(html)
        except ValueError:
            token = None
        return InspectResult(
            target=self.settings.target,
            csrf_token=token,
            title=title,
            page_excerpt=_safe_excerpt(html),
            html_length=len(html),
        )

    def fetch_csrf_token(self) -> str:
        """Fetch csrfToken from the logged-in choose-course page."""

        return extract_csrf_token(self.fetch_choose_course_page())

    def submit_course(self, target: CourseTarget, course_id: str, csrf_token: str) -> SubmitResult:
        """Submit one course ID to the real choiceCourse endpoint."""

        timestamp = int(time.time() * 1000)
        url = f"{self.settings.base_url}/xsxkCourse/choiceCourse.do?_={timestamp}"
        form = urllib.parse.urlencode(
            {
                "bjdm": course_id,
                "lx": str(target.classification_code),
                "bqmc": target.category,
                "csrfToken": csrf_token,
            }
        ).encode("utf-8")
        request = urllib.request.Request(url, data=form, headers=self.settings.headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError) as exc:
            return SubmitResult(course_id, target.category, SubmitStatus.NETWORK_ERROR, str(exc))
        return parse_submit_response(course_id, target.category, body)


def resolve_working_client(targets: tuple[str, ...], cookie: str, timeout_seconds: float = 10.0) -> tuple[FduCourseClient, InspectResult]:
    """Try configured domains and return the first client that can extract csrfToken."""

    errors: list[str] = []
    for target in targets:
        client = FduCourseClient(SessionSettings(target=target, cookie=cookie), timeout_seconds=timeout_seconds)
        try:
            result = client.inspect()
        except Exception as exc:  # noqa: BLE001 - keep all target failures for diagnostics.
            errors.append(f"{target}: {exc}")
            continue
        if result.ok:
            return client, result
        errors.append(f"{target}: 未找到 csrfToken；title={result.title or '空'}；html_length={result.html_length}")
    detail = "；".join(errors) if errors else "未配置候选域名"
    raise RuntimeError(f"所有候选域名都无法提取 csrfToken：{detail}")


def parse_submit_response(course_id: str, category: str, body: str) -> SubmitResult:
    """Normalize the enrollment JSON response body into SubmitResult."""

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return SubmitResult(course_id, category, SubmitStatus.INVALID_RESPONSE, "接口返回非 JSON 内容")
    code = payload.get("code")
    message = str(payload.get("msg", ""))
    if code == 0:
        return SubmitResult(course_id, category, SubmitStatus.PENDING, message or "继续尝试", raw_code=code)
    if code == 1:
        return SubmitResult(course_id, category, SubmitStatus.SUCCESS, message or "提交选课成功", raw_code=code)
    if "过期" in message or "csrf" in message.lower() or "cookie" in message.lower():
        status = SubmitStatus.AUTH_EXPIRED
    else:
        status = SubmitStatus.REJECTED
    return SubmitResult(course_id, category, status, message or f"接口返回 code={code}", raw_code=code)


def _extract_title(html: str) -> str:
    match = TITLE_PATTERN.search(html)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()


def _safe_excerpt(html: str, limit: int = 360) -> str:
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]
