"""Judge execution: run LLM judges and parse outputs."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .client import LLMClient
from .config import JudgeType, PromptVariant, ExperimentConfig
from .data import JudgePair
from .prompts import build_messages

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    """Result of a single judge evaluation."""
    question_id: int
    model_a: str
    model_b: str
    human_winner: str
    judge_type: str
    prompt_variant: str
    temperature: float
    model_name: str
    model_size_label: str
    repeat_id: int
    raw_output: str
    # Parsed results
    pairwise_winner: Optional[str] = None   # "A", "B", "C" (tie), or None (error)
    score_a: Optional[float] = None
    score_b: Optional[float] = None
    is_swapped: bool = False
    parse_error: bool = False


def parse_pairwise(raw: str) -> Optional[str]:
    """Parse pairwise verdict from raw LLM output.

    Looks for [[A]], [[B]], or [[C]] patterns.
    Returns "A", "B", "C", or None if parse error.
    """
    # Look for [[X]] pattern
    match = re.search(r'\[\[([ABCabc])\]\]', raw)
    if match:
        return match.group(1).upper()
    # Fallback: look for "Assistant A" / "Assistant B" / "tie"
    raw_lower = raw.lower().strip()
    if "assistant a" in raw_lower and "better" in raw_lower:
        return "A"
    if "assistant b" in raw_lower and "better" in raw_lower:
        return "B"
    if "tie" in raw_lower:
        return "C"
    return None


def parse_score(raw: str) -> Optional[float]:
    """Parse a numeric score from raw LLM output.

    Looks for [[N]] where N is 1-10.
    """
    match = re.search(r'\[\[(\d+(?:\.\d+)?)\]\]', raw)
    if match:
        score = float(match.group(1))
        if 1 <= score <= 10:
            return score
    # Fallback: look for standalone number
    match = re.search(r'\b(\d+(?:\.\d+)?)\b', raw)
    if match:
        score = float(match.group(1))
        if 1 <= score <= 10:
            return score
    return None


def run_judge_single(
    client: LLMClient,
    pair: JudgePair,
    judge_type: JudgeType,
    prompt_variant: PromptVariant,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repeat_id: int,
    model_size_label: str,
) -> list[JudgeResult]:
    """Run a single judge evaluation for one pair.

    For pairwise: returns 1 result.
    For single_answer/reference_guided: returns 2 results (one per response).
    """
    results = []

    if judge_type == JudgeType.PAIRWISE:
        is_swapped = (prompt_variant == PromptVariant.POSITION_SWAP)
        messages = build_messages(pair, judge_type, prompt_variant, swapped=is_swapped)
        raw = client.generate(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        winner = parse_pairwise(raw)
        # If swapped, flip the winner back
        if is_swapped and winner in ("A", "B"):
            winner = "B" if winner == "A" else "A"

        results.append(JudgeResult(
            question_id=pair.question_id,
            model_a=pair.model_a,
            model_b=pair.model_b,
            human_winner=pair.human_winner,
            judge_type=judge_type.value,
            prompt_variant=prompt_variant.value,
            temperature=temperature,
            model_name=client.model_name,
            model_size_label=model_size_label,
            repeat_id=repeat_id,
            raw_output=raw,
            pairwise_winner=winner,
            is_swapped=is_swapped,
            parse_error=(winner is None),
        ))

    elif judge_type in (JudgeType.SINGLE_ANSWER, JudgeType.REFERENCE_GUIDED):
        for which in ("a", "b"):
            messages = build_messages(pair, judge_type, prompt_variant, which_response=which)
            raw = client.generate(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            score = parse_score(raw)
            result = JudgeResult(
                question_id=pair.question_id,
                model_a=pair.model_a,
                model_b=pair.model_b,
                human_winner=pair.human_winner,
                judge_type=judge_type.value,
                prompt_variant=prompt_variant.value,
                temperature=temperature,
                model_name=client.model_name,
                model_size_label=model_size_label,
                repeat_id=repeat_id,
                raw_output=raw,
                parse_error=(score is None),
            )
            if which == "a":
                result.score_a = score
            else:
                result.score_b = score
            results.append(result)

    return results


def run_judge_pair_consistency(
    client: LLMClient,
    pair: JudgePair,
    prompt_variant: PromptVariant,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repeat_id: int,
    model_size_label: str,
) -> tuple[JudgeResult, JudgeResult]:
    """Run pairwise judge in both original and swapped order for consistency check."""
    # Original order
    messages_orig = build_messages(pair, JudgeType.PAIRWISE, PromptVariant.BASELINE, swapped=False)
    raw_orig = client.generate(messages_orig, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    winner_orig = parse_pairwise(raw_orig)

    # Swapped order
    messages_swap = build_messages(pair, JudgeType.PAIRWISE, PromptVariant.BASELINE, swapped=True)
    raw_swap = client.generate(messages_swap, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    winner_swap = parse_pairwise(raw_swap)
    # Flip back
    if winner_swap in ("A", "B"):
        winner_swap = "B" if winner_swap == "A" else "A"

    result_orig = JudgeResult(
        question_id=pair.question_id,
        model_a=pair.model_a, model_b=pair.model_b,
        human_winner=pair.human_winner,
        judge_type="pairwise", prompt_variant=prompt_variant.value,
        temperature=temperature, model_name=client.model_name,
        model_size_label=model_size_label, repeat_id=repeat_id,
        raw_output=raw_orig, pairwise_winner=winner_orig,
        is_swapped=False, parse_error=(winner_orig is None),
    )
    result_swap = JudgeResult(
        question_id=pair.question_id,
        model_a=pair.model_a, model_b=pair.model_b,
        human_winner=pair.human_winner,
        judge_type="pairwise", prompt_variant=prompt_variant.value,
        temperature=temperature, model_name=client.model_name,
        model_size_label=model_size_label, repeat_id=repeat_id,
        raw_output=raw_swap, pairwise_winner=winner_swap,
        is_swapped=True, parse_error=(winner_swap is None),
    )
    return result_orig, result_swap
