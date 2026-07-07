"""Runtime bootstrap helpers for HW3 Task1 Python launch wrappers."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

SYSTEM_CUDA_HOME = Path("/usr/local/cuda")


def ensure_swanlab_api_key() -> str:
    """Return SWANLAB_API_KEY from environment or raise a clear error."""
    api_key = os.environ.get("SWANLAB_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SWANLAB_API_KEY is required for training with SwanLab logging.")
    return api_key


def load_env_file(path: str | Path) -> None:
    """Load KEY=VALUE pairs from an env file."""
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def patch_trusted_torch_load() -> None:
    """Set torch.load(weights_only=False) default unless explicitly disabled."""
    enabled = os.environ.get("HW3_TRUSTED_TORCH_LOAD_WEIGHTS_ONLY_FALSE", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return

    import torch

    original_load = torch.load
    if getattr(original_load, "_hw3_trusted_patch", False):
        return

    def load_with_trusted_default(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    load_with_trusted_default._hw3_trusted_patch = True  # type: ignore[attr-defined]
    torch.load = load_with_trusted_default


def prepare_cuda_extension_env(logdir: str | Path | None = None) -> None:
    """Populate extension-related env vars used by PyTorch heavy wrappers."""
    from sysconfig import get_paths

    _prepend_path_env("PATH", [str(Path(sys.executable).resolve().parent)])
    purelib = Path(get_paths()["purelib"])
    nvidia_root = purelib / "nvidia"
    include_dirs = sorted(str(path) for path in nvidia_root.glob("*/include") if path.is_dir())
    lib_dirs = sorted(str(path) for path in nvidia_root.glob("*/lib") if path.is_dir())

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        include_dirs.insert(0, str(Path(conda_prefix) / "targets" / "x86_64-linux" / "include"))
        include_dirs.insert(0, str(Path(conda_prefix) / "include"))
        lib_dirs.insert(0, str(Path(conda_prefix) / "targets" / "x86_64-linux" / "lib"))
        lib_dirs.insert(0, str(Path(conda_prefix) / "lib"))

    cuda_home = _find_cuda_home(conda_prefix)
    if cuda_home is not None:
        os.environ["CUDA_HOME"] = str(cuda_home)
        _prepend_path_env("PATH", [str(cuda_home / "bin")])
        include_dirs.insert(0, str(cuda_home / "include"))
        lib_dirs.insert(0, str(cuda_home / "lib64"))

    for name, values in {
        "CPATH": include_dirs,
        "CPLUS_INCLUDE_PATH": include_dirs,
        "LIBRARY_PATH": lib_dirs,
        "LD_LIBRARY_PATH": lib_dirs,
    }.items():
        _prepend_path_env(name, values)

    if logdir is not None:
        extension_dir = _torch_extensions_dir(logdir)
        extension_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(extension_dir))

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _find_cuda_home(conda_prefix: str = "") -> Path | None:
    candidates = [
        Path(value)
        for value in (
            os.environ.get("CUDA_HOME", ""),
            str(SYSTEM_CUDA_HOME),
            conda_prefix,
        )
        if value
    ]
    for candidate in candidates:
        if (candidate / "bin" / "nvcc").is_file():
            return candidate.resolve()
    return None


def _torch_extensions_dir(logdir: str | Path) -> Path:
    log_path = Path(logdir).resolve()
    if len(log_path.parents) >= 2:
        return log_path.parents[1] / "torch_extensions"
    return Path.home() / ".cache" / "torch_extensions_hw3"


def _prepend_path_env(name: str, values: Iterable[str]) -> None:
    existing = [value for value in os.environ.get(name, "").split(":") if value]
    merged: list[str] = []
    for value in [*values, *existing]:
        if value and value not in merged:
            merged.append(value)
    if merged:
        os.environ[name] = ":".join(merged)


def add_repo_path(path: str | Path) -> None:
    """Prepend a path to PYTHONPATH if missing."""
    resolved = str(Path(path).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
