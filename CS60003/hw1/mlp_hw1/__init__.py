"""HW1：EuroSAT MLP 分类器。"""

from .config import TrainConfig, SearchConfig, build_search_config, build_train_config

__all__ = [
    "SearchConfig",
    "ThreeLayerMLP",
    "TrainConfig",
    "build_search_config",
    "build_train_config",
]


def __getattr__(name: str):
    """延迟加载模型，使配置与数据工具在无 CuPy 时仍可导入。"""
    if name == "ThreeLayerMLP":
        from .model import ThreeLayerMLP

        return ThreeLayerMLP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
