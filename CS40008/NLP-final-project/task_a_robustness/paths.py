"""Shared filesystem paths for Task A scripts."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
TASK_A_RESULTS_DIR = RESULTS_DIR / "task_a"
TASK_A_ROBERTA_RESULTS_DIR = TASK_A_RESULTS_DIR / "roberta"
TASK_A_VLLM_RESULTS_DIR = TASK_A_RESULTS_DIR / "vllm"
TASK_A_VLLM_PREDICTIONS_DIR = TASK_A_VLLM_RESULTS_DIR / "predictions"
TASK_A_VLLM_METRICS_DIR = TASK_A_VLLM_RESULTS_DIR / "metrics"
TASK_A_VLLM_RAW_OUTPUTS_DIR = TASK_A_VLLM_RESULTS_DIR / "raw_outputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "task_a"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "task_a_roberta"
HF_CACHE_DIR = PROJECT_ROOT / "checkpoints" / "hf_cache"


def processed_split_path(split: str) -> Path:
    """Return the processed ComVE Task A CSV path for a split."""

    return PROCESSED_DATA_DIR / f"comve_task_a_{split}.csv"


def ensure_runtime_dirs() -> None:
    """Create directories used by local runs."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TASK_A_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TASK_A_ROBERTA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TASK_A_VLLM_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    TASK_A_VLLM_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    TASK_A_VLLM_RAW_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
