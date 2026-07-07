"""Upload the best Task3 checkpoint and artifacts to ModelScope."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from modelscope.hub.api import HubApi

MODEL_ID = "youngchen/CS60003"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Task3 output run directory containing best.pt.")
    parser.add_argument("--remote-subdir", default=None, help="Remote subdir under hw2/task3/.")
    return parser.parse_args()


def main() -> None:
    """Upload selected Task3 artifacts to ModelScope."""
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not (run_dir / "best.pt").is_file():
        raise FileNotFoundError(f"Missing best.pt in {run_dir}")
    token = resolve_modelscope_token()
    if not token:
        raise RuntimeError("ModelScope token is required in MODELSCOPE_API_TOKEN or .helloagents/modules/hw2.md")
    api = HubApi()
    api.login(token)
    remote_subdir = args.remote_subdir or run_dir.name
    for name in [
        "best.pt",
        "source_config.yaml",
        "config.json",
        "dataset_stats.json",
        "env.json",
        "history.csv",
        "curves.png",
        "metrics.json",
        "val_samples.png",
        "palette_legend.png",
    ]:
        path = run_dir / name
        if path.is_file():
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"hw2/task3/{remote_subdir}/{name}",
                repo_id=MODEL_ID,
                repo_type="model",
                commit_message=f"Upload HW2 Task3 {remote_subdir} {name}",
            )
            print(f"uploaded hw2/task3/{remote_subdir}/{name}")


def resolve_modelscope_token() -> str:
    """Read the temporary homework ModelScope token from env or project notes."""
    token = os.environ.get("MODELSCOPE_API_TOKEN", "").strip()
    if token:
        return token
    notes_path = Path(__file__).resolve().parents[2] / ".helloagents/modules/hw2.md"
    if not notes_path.is_file():
        return ""
    text = notes_path.read_text(encoding="utf-8")
    match = re.search(r"ModelScope API token[^`]*`([^`]+)`", text)
    return match.group(1).strip() if match else ""


if __name__ == "__main__":
    main()
