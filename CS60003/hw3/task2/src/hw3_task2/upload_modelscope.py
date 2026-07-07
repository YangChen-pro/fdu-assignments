from __future__ import annotations

import argparse
import os
from pathlib import Path

from tqdm import tqdm

from .secrets import has_any_key, load_env_file
from .utils import write_json

UPLOAD_ALLOWLIST = {
    "config.yaml",
    "dataset_summary.json",
    "metrics.csv",
    "train_summary.json",
    "results_summary.json",
    "checkpoints/best.pt",
    "checkpoints/final.pt",
    "checkpoints/latest.pt",
}


def build_repo_path(prefix: str, run_name: str, rel_path: str) -> str:
    parts = [prefix.strip("/"), run_name.strip("/"), rel_path.strip("/")]
    return "/".join(part for part in parts if part)


def collect_upload_files(model_dir: Path) -> list[Path]:
    files = []
    for rel in sorted(UPLOAD_ALLOWLIST):
        path = model_dir / rel
        if path.is_file():
            files.append(path)
    if not files:
        raise FileNotFoundError(f"no uploadable files found in {model_dir}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--path-prefix", default="")
    parser.add_argument("--secret-env", default=".helloagents/secrets/hw3.env")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    load_env_file(args.secret_env)
    if not has_any_key(["MODELSCOPE_API_TOKEN", "MODELSCOPE_TOKEN", "MODELSCOPE_SDK_TOKEN"]):
        raise RuntimeError("ModelScope token not found in environment")
    token = (
        os.environ.get("MODELSCOPE_API_TOKEN")
        or os.environ.get("MODELSCOPE_TOKEN")
        or os.environ.get("MODELSCOPE_SDK_TOKEN")
    )

    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(access_token=token)
    try:
        api.create_model(args.repo_id, visibility=5, license="Apache License 2.0", token=token)
    except Exception as exc:  # existing repository is fine; upload_file will surface real auth/path errors.
        print(f"ModelScope repo create skipped: {type(exc).__name__}")

    model_dir = Path(args.model_dir)
    files = collect_upload_files(model_dir)
    uploaded = []
    run_name = model_dir.name
    for path in tqdm(files, desc=f"upload {args.repo_id}"):
        rel = path.relative_to(model_dir).as_posix()
        path_in_repo = build_repo_path(args.path_prefix, run_name, rel)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path_in_repo,
            repo_id=args.repo_id,
            repo_type="model",
            token=token,
            commit_message=f"upload {path_in_repo}",
        )
        uploaded.append(path_in_repo)
    result = {
        "repo_id": args.repo_id,
        "modelscope_url": f"https://modelscope.cn/models/{args.repo_id}",
        "path_prefix": args.path_prefix,
        "model_dir": str(model_dir.resolve()),
        "files": uploaded,
    }
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
