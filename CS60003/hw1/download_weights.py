"""下载并校验 HW1 已上传到 ModelScope 的权重。"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download


MODEL_ID = "youngchen/CS60003"
WEIGHTS = {
    "hw1/final_p/best_model.npz": "1d33521419a060a4670b86be58926522c5febffc5a64a1d63ec2d30793325d2a",
    "hw1/final_o/best_model.npz": "e1295599ece6be23a20016c110dbca63359ebc82cb715d7db4a154b30b2457f5",
}


def sha256_file(path: Path) -> str:
    """计算文件的 SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """下载正式模型和扩展实验权重，并校验摘要。"""
    local_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("hw1/model_weights")
    local_dir.mkdir(parents=True, exist_ok=True)

    # 某些本机环境会注入 SOCKS 代理变量，但 requests 未必带有对应依赖。
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)

    # 只下载本作业需要核验的两个权重文件，避免拉取无关内容。
    snapshot_download(MODEL_ID, local_dir=str(local_dir), allow_file_pattern=list(WEIGHTS))

    for relative_path, expected_hash in WEIGHTS.items():
        path = local_dir / relative_path
        actual_hash = sha256_file(path)
        status = "OK" if actual_hash == expected_hash else "FAILED"
        print(f"{status}  {relative_path}  sha256={actual_hash}")
        if actual_hash != expected_hash:
            raise SystemExit(f"SHA256 校验失败: {relative_path}")


if __name__ == "__main__":
    main()
