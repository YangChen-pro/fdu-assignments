import pytest

from fdu_course_assistant.session import extract_csrf_token, mask_cookie, normalize_cookie


def test_normalize_cookie_single_line():
    assert normalize_cookie(" a=1;\n b=2 ") == "a=1; b=2"


def test_mask_cookie_hides_values():
    masked = mask_cookie("JSESSIONID=abcdef1234567890; XK_TOKEN=token-value-123456")
    assert "abcdef1234567890" not in masked
    assert "token-value-123456" not in masked
    assert "JSESSIONID=" in masked


def test_extract_csrf_token():
    html = "<input id='csrfToken' value='0123456789abcdef0123456789abcdef'>"
    assert extract_csrf_token(html) == "0123456789abcdef0123456789abcdef"


def test_extract_csrf_token_missing():
    with pytest.raises(ValueError):
        extract_csrf_token("<html></html>")
