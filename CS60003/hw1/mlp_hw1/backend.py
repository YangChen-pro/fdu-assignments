"""CuPy 后端辅助函数。"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError as exc:  # pragma: no cover - depends on the local environment
    cp = None
    _CUPY_IMPORT_ERROR = exc
else:
    _CUPY_IMPORT_ERROR = None


def get_array_module() -> Any:
    """返回当前唯一支持的数组后端。"""
    if cp is None:
        raise RuntimeError("当前环境未安装 CuPy；本项目训练与评估默认需要 GPU + CuPy 环境。") from _CUPY_IMPORT_ERROR
    return cp


def seed_everything(seed: int) -> None:
    """为项目使用的 CPU/GPU 随机数生成器设置种子。"""
    np.random.seed(seed)
    if cp is None:
        raise RuntimeError("当前环境未安装 CuPy；无法为 GPU 随机数发生器设种子。") from _CUPY_IMPORT_ERROR
    cp.random.seed(seed)


def to_numpy(array: Any) -> np.ndarray:
    """将 CuPy 数组转换为 NumPy，便于报告或可视化。"""
    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)
