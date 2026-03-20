"""Configuration dataclasses for experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BackendType(str, Enum):
    """Supported model serving backends."""

    VLLM = "vllm"
    SGLANG = "sglang"
    OPENAI = "openai"


class JudgeType(str, Enum):
    """Supported judge task modes."""

    PAIRWISE = "pairwise"
    SINGLE_ANSWER = "single_answer"
    REFERENCE_GUIDED = "reference_guided"


class PromptVariant(str, Enum):
    """Prompt-engineering variants used in ablation and causal analyses."""

    BASELINE = "baseline"
    POSITION_SWAP = "position_swap"
    FEW_SHOT = "few_shot"
    COT = "cot"
    REFERENCE_GUIDED = "reference_guided"
    MULTI_TURN = "multi_turn"


@dataclass
class ModelConfig:
    """Configuration for a judge LLM."""
    model_name: str
    base_url: str
    api_key: str = "EMPTY"
    backend: BackendType = BackendType.VLLM
    model_size_label: str = ""  # e.g., "1B", "7B", "70B"

    def __post_init__(self):
        """Infer a readable model label from model name when omitted."""
        if not self.model_size_label:
            self.model_size_label = self.model_name.split("/")[-1]


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    temperatures: list[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    top_p: float = 0.95
    max_tokens: int = 1024
    judge_types: list[JudgeType] = field(
        default_factory=lambda: [JudgeType.PAIRWISE, JudgeType.SINGLE_ANSWER, JudgeType.REFERENCE_GUIDED]
    )
    prompt_variants: list[PromptVariant] = field(
        default_factory=lambda: [
            PromptVariant.BASELINE,
            PromptVariant.POSITION_SWAP,
            PromptVariant.FEW_SHOT,
            PromptVariant.COT,
            PromptVariant.REFERENCE_GUIDED,
            PromptVariant.MULTI_TURN,
        ]
    )
    num_repeats: int = 10
    sample_size: int = 100  # number of pairs to sample
    random_seed: int = 42
    output_dir: str = "results"
    models: list[ModelConfig] = field(default_factory=list)
