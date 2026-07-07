"""Command line interface for fdu-course-assistant."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .client import resolve_working_client
from .config import load_config, resolve_cookie, resolve_run_window
from .runner import run_enrollment
from .session import mask_cookie


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fdu-course", description="FDU course enrollment assistant")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "check-cookie", "inspect"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", required=True, help="Path to YAML/JSON config file")
        cmd.add_argument("--cookie", default=None, help="Cookie override; prefer environment variable")
    sub.choices["run"].add_argument("--once", action="store_true", help="Submit only one cycle for practical verification")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args, parser)
    except RuntimeError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    config = load_config(Path(args.config))
    cookie = resolve_cookie(config, getattr(args, "cookie", None))
    client, inspect_result = resolve_working_client(config.targets, cookie)

    if args.command == "check-cookie":
        token = inspect_result.csrf_token or client.fetch_csrf_token()
        print(f"Cookie 可用：{mask_cookie(cookie)}")
        print(f"命中域名：{inspect_result.target}")
        print(f"csrfToken：{token[:6]}...{token[-6:]}")
        return 0

    if args.command == "inspect":
        print(f"Cookie：{mask_cookie(cookie)}")
        print(f"命中域名：{inspect_result.target}")
        print(f"页面标题：{inspect_result.title or '未识别'}")
        print(f"HTML 长度：{inspect_result.html_length}")
        token = inspect_result.csrf_token
        print(f"csrfToken：{token[:6] + '...' + token[-6:] if token else '未提取到'}")
        print(f"页面片段：{inspect_result.page_excerpt or '空'}")
        return 0

    if args.command == "run":
        window = resolve_run_window(config.start_at, config.end_at)
        summary = run_enrollment(config, window, client, wait=not args.once, once=args.once)
        print(f"运行完成：成功 {summary.success_count} / 尝试 {len(summary.results)} / 轮次 {summary.cycles}")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
