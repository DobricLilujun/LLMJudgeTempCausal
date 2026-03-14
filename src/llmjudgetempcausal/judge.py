"""Judge execution: run LLM judges and parse outputs."""

from __future__ import annotations

import json
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
    judge_reason: Optional[str] = None
    # Parsed results
    pairwise_winner: Optional[str] = None   # "A", "B", "C" (tie), or None (error)
    score_a: Optional[float] = None
    score_b: Optional[float] = None
    is_swapped: bool = False
    parse_error: bool = False


def _parse_json_object(raw: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from raw model output."""
    text = raw.strip()
    if not text:
        return None

    candidates = [text]
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    if fenced != text:
        candidates.append(fenced)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj

    return None


def parse_judge_reason(raw: str) -> Optional[str]:
    """Parse judge reason from JSON output, if present."""
    obj = _parse_json_object(raw)
    if not obj:
        return None

    reason = obj.get("judge_reason", obj.get("reason"))
    if reason is None:
        return None
    reason = str(reason).strip()
    return reason or None


def parse_pairwise(raw: str) -> Optional[str]:
    """Parse pairwise verdict from raw LLM output.

    Looks for JSON {"judge_result": "A"} first, then legacy [[A]] fallback.
    Returns "A", "B", "C", or None if parse error.
    """
    obj = _parse_json_object(raw)
    if obj:
        value = obj.get("judge_result", obj.get("winner", obj.get("verdict")))
        if value is not None:
            value = str(value).strip().upper()
            if value in ("A", "B", "C"):
                return value

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

    Looks for JSON {"judge_result": N} first, then legacy [[N]] fallback.
    """
    obj = _parse_json_object(raw)
    if obj:
        value = obj.get("judge_result", obj.get("score", obj.get("rating")))
        if value is not None:
            try:
                score = float(value)
            except (TypeError, ValueError):
                score = None
            if score is not None and 1 <= score <= 10:
                return score

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
    seed: Optional[int] = None,
) -> list[JudgeResult]:
    """Run a single judge evaluation for one pair.

    For pairwise: returns 1 result.
    For reference_guided: returns 1 pairwise result.
    For single_answer: returns 2 results (one per response).
    """
    results = []

    if judge_type in (JudgeType.PAIRWISE, JudgeType.REFERENCE_GUIDED):
        is_swapped = (prompt_variant == PromptVariant.POSITION_SWAP)
        messages = build_messages(
            pair,
            judge_type,
            prompt_variant,
            swapped=is_swapped,
            model_name=client.model_name,
        )
        raw = client.generate(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)
        winner = parse_pairwise(raw)
        judge_reason = parse_judge_reason(raw)
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
            judge_reason=judge_reason,
            pairwise_winner=winner,
            is_swapped=is_swapped,
            parse_error=((winner is None) or (prompt_variant == PromptVariant.COT and judge_reason is None)),
        ))

    elif judge_type == JudgeType.SINGLE_ANSWER:
        for which in ("a", "b"):
            messages = build_messages(
                pair,
                judge_type,
                prompt_variant,
                which_response=which,
                model_name=client.model_name,
            )
            raw = client.generate(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)
            score = parse_score(raw)
            judge_reason = parse_judge_reason(raw)
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
                judge_reason=judge_reason,
                parse_error=((score is None) or (prompt_variant == PromptVariant.COT and judge_reason is None)),
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
    seed: Optional[int] = None,
) -> tuple[JudgeResult, JudgeResult]:
    """Run pairwise judge in both original and swapped order for consistency check."""
    # Original order
    messages_orig = build_messages(
        pair,
        JudgeType.PAIRWISE,
        PromptVariant.BASELINE,
        swapped=False,
        model_name=client.model_name,
    )
    raw_orig = client.generate(messages_orig, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)
    winner_orig = parse_pairwise(raw_orig)

    # Swapped order
    messages_swap = build_messages(
        pair,
        JudgeType.PAIRWISE,
        PromptVariant.BASELINE,
        swapped=True,
        model_name=client.model_name,
    )
    raw_swap = client.generate(messages_swap, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)
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
