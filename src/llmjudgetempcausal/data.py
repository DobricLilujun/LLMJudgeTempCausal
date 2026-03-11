"""Data loading and preprocessing from mt_bench_human_judgments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset


@dataclass
class JudgePair:
    """A single pairwise comparison from MT-Bench."""
    question_id: int
    model_a: str
    model_b: str
    human_winner: str  # "model_a", "model_b", "tie"
    conversation_a: list[dict]  # [{role, content}, ...]
    conversation_b: list[dict]
    turn: int

    @property
    def question_text(self) -> str:
        """Extract the user question from the first turn."""
        return self.conversation_a[0]["content"]

    @property
    def response_a(self) -> str:
        """Get model_a's response for the relevant turn."""
        # turn 1 → assistant at index 1; turn 2 → assistant at index 3
        idx = (self.turn * 2) - 1
        return self.conversation_a[idx]["content"]

    @property
    def response_b(self) -> str:
        idx = (self.turn * 2) - 1
        return self.conversation_b[idx]["content"]

    @property
    def follow_up_question(self) -> Optional[str]:
        """For turn 2, get the follow-up user question."""
        if self.turn == 2 and len(self.conversation_a) >= 3:
            return self.conversation_a[2]["content"]
        return None

    @property
    def response_a_turn1(self) -> str:
        """Get model_a's response for turn 1 (for multi-turn context)."""
        return self.conversation_a[1]["content"]

    @property
    def response_b_turn1(self) -> str:
        return self.conversation_b[1]["content"]


def load_mt_bench_human(split: str = "human") -> list[JudgePair]:
    """Load the MT-Bench human judgments dataset."""
    ds = load_dataset("lmsys/mt_bench_human_judgments", split=split)
    pairs = []
    for row in ds:
        pairs.append(JudgePair(
            question_id=row["question_id"],
            model_a=row["model_a"],
            model_b=row["model_b"],
            human_winner=row["winner"],
            conversation_a=row["conversation_a"],
            conversation_b=row["conversation_b"],
            turn=row["turn"],
        ))
    return pairs


def sample_pairs(
    pairs: list[JudgePair],
    n: int,
    seed: int = 42,
) -> list[JudgePair]:
    """Randomly sample n pairs (with seed for reproducibility)."""
    rng = random.Random(seed)
    n = min(n, len(pairs))
    return rng.sample(pairs, n)
