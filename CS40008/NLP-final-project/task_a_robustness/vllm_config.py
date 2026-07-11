"""vLLM model registry and README-backed generation settings for Task A."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import PROJECT_ROOT


MODEL_ROOT = PROJECT_ROOT / "checkpoints" / "llm_models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "task_a" / "vllm" / "raw_outputs"
DEFAULT_MAX_TOKENS_BY_MODE = {"non_thinking": 10240, "thinking": 16384}


@dataclass(frozen=True)
class GenerationSettings:
    """README-backed generation settings for one thinking mode."""

    enable_thinking: bool
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float | None
    repetition_penalty: float | None
    source_note: str


@dataclass(frozen=True)
class ModelSpec:
    """Local model path and README-backed settings."""

    name: str
    path: Path
    family: str
    modes: dict[str, GenerationSettings]


QWEN3_NON_THINKING = GenerationSettings(
    enable_thinking=False,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
    presence_penalty=None,
    repetition_penalty=None,
    source_note="Qwen3 README non-thinking mode: temperature=0.7, top_p=0.8, top_k=20, min_p=0.",
)
QWEN3_THINKING = GenerationSettings(
    enable_thinking=True,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=None,
    repetition_penalty=None,
    source_note="Qwen3 README thinking mode: temperature=0.6, top_p=0.95, top_k=20, min_p=0; greedy decoding is discouraged.",
)
QWEN35_NON_THINKING = GenerationSettings(
    enable_thinking=False,
    temperature=1.0,
    top_p=1.0,
    top_k=20,
    min_p=0.0,
    presence_penalty=2.0,
    repetition_penalty=1.0,
    source_note="Qwen3.5 README non-thinking text tasks: temperature=1.0, top_p=1.0, top_k=20, min_p=0, presence_penalty=2.0, repetition_penalty=1.0.",
)
QWEN35_THINKING = GenerationSettings(
    enable_thinking=True,
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    repetition_penalty=1.0,
    source_note="Qwen3.5 README thinking text tasks: temperature=1.0, top_p=0.95, top_k=20, min_p=0, presence_penalty=1.5, repetition_penalty=1.0.",
)

MODEL_SPECS: dict[str, ModelSpec] = {
    "qwen3_0_6b": ModelSpec(
        name="Qwen3-0.6B",
        path=MODEL_ROOT / "Qwen3-0.6B",
        family="qwen3",
        modes={"non_thinking": QWEN3_NON_THINKING, "thinking": QWEN3_THINKING},
    ),
    "qwen3_1_7b": ModelSpec(
        name="Qwen3-1.7B",
        path=MODEL_ROOT / "Qwen3-1.7B",
        family="qwen3",
        modes={"non_thinking": QWEN3_NON_THINKING, "thinking": QWEN3_THINKING},
    ),
    "qwen3_5_0_8b": ModelSpec(
        name="Qwen3.5-0.8B",
        path=MODEL_ROOT / "Qwen3.5-0.8B",
        family="qwen3_5",
        modes={"non_thinking": QWEN35_NON_THINKING, "thinking": QWEN35_THINKING},
    ),
    "qwen3_5_2b": ModelSpec(
        name="Qwen3.5-2B",
        path=MODEL_ROOT / "Qwen3.5-2B",
        family="qwen3_5",
        modes={"non_thinking": QWEN35_NON_THINKING, "thinking": QWEN35_THINKING},
    ),
}
