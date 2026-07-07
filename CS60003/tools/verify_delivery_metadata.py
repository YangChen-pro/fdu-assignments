"""Validate HelloAGENTS delivery metadata for this repository."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLANS = ROOT / ".helloagents" / "plans"
VALID_VERIFY_MODES = {"test-first", "review-first"}


def _task_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("- [")
    ]


def _validate_contract(plan_dir: Path) -> list[str]:
    issues: list[str] = []
    contract_path = plan_dir / "contract.json"
    if not contract_path.is_file():
        return ["missing contract.json"]

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    verify_mode = str(contract.get("verifyMode", "")).strip()
    if verify_mode not in VALID_VERIFY_MODES:
        issues.append("missing valid verifyMode")
    if not contract.get("testerFocus"):
        issues.append("missing testerFocus")
    if verify_mode == "review-first" and not contract.get("reviewerFocus"):
        issues.append("missing reviewerFocus")
    return issues


def _validate_tasks(plan_dir: Path) -> list[str]:
    issues: list[str] = []
    task_path = plan_dir / "tasks.md"
    if not task_path.is_file():
        return ["missing tasks.md"]

    for line in _task_lines(task_path):
        missing = [
            label
            for label in ("涉及文件", "完成标准", "验证方式")
            if label not in line
        ]
        if missing:
            issues.append(f"{line}: missing {', '.join(missing)}")
    return issues


def main() -> None:
    issues: list[str] = []
    for plan_dir in sorted(p for p in PLANS.iterdir() if p.is_dir()):
        for issue in [*_validate_contract(plan_dir), *_validate_tasks(plan_dir)]:
            issues.append(f"{plan_dir.name}: {issue}")

    if issues:
        raise SystemExit("\n".join(issues))
    print("delivery metadata ok")


if __name__ == "__main__":
    main()
