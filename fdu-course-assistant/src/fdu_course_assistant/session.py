"""HTTP session helpers, cookie handling, and csrfToken extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape

CSRF_PATTERN = re.compile(r"id=[\"']csrfToken[\"'][^>]*value=[\"']([a-fA-F0-9]{32})[\"']")
SENSITIVE_COOKIE_NAMES = {"JSESSIONID", "XK_TOKEN", "route", "_WEU"}


def normalize_cookie(cookie: str) -> str:
    """Trim a copied browser Cookie header into a single-line value."""

    return " ".join(part.strip() for part in cookie.replace("\r", "\n").splitlines() if part.strip())


def mask_cookie(cookie: str) -> str:
    """Mask sensitive Cookie values while preserving cookie names for debugging."""

    normalized = normalize_cookie(cookie)
    if not normalized:
        return ""
    masked_parts: list[str] = []
    for part in normalized.split(";"):
        item = part.strip()
        if "=" not in item:
            masked_parts.append("***")
            continue
        name, value = item.split("=", 1)
        if name in SENSITIVE_COOKIE_NAMES or len(value) > 12:
            shown = value[:3] + "..." + value[-3:] if len(value) >= 8 else "***"
            masked_parts.append(f"{name}={shown}")
        else:
            masked_parts.append(f"{name}=***")
    return "; ".join(masked_parts)


def extract_csrf_token(html: str) -> str:
    """Extract the 32-character csrfToken value from the choose-course HTML page."""

    match = CSRF_PATTERN.search(unescape(html))
    if not match:
        raise ValueError("未在页面中找到 csrfToken，Cookie 可能已过期或页面结构已变化")
    return match.group(1)


@dataclass(frozen=True)
class SessionSettings:
    """Resolved HTTP settings for FDU course requests."""

    target: str
    cookie: str
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
    )

    @property
    def base_url(self) -> str:
        """Return the HTTP base URL used by the enrollment app."""

        return f"http://{self.target}/yjsxkapp/sys/xsxkappfudan"

    @property
    def headers(self) -> dict[str, str]:
        """Return common headers for form and token requests."""

        return {
            "Cookie": normalize_cookie(self.cookie),
            "User-Agent": self.user_agent,
            "Content-Type": "application/x-www-form-urlencoded",
        }
