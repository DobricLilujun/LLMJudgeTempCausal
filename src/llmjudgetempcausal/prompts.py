"""Prompt templates for different judge types and prompt variants.

All prompt text lives in Jinja2 ``.jinja`` files under the ``assets/`` directory
next to this module.  This keeps the templates easy to read, version, and modify
without touching Python code.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import JudgeType, PromptVariant
from .data import JudgePair

# ---------------------------------------------------------------------------
# Jinja2 environment – loads from assets/ once at import time
# ---------------------------------------------------------------------------

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"

_env = Environment(
    loader=FileSystemLoader(_ASSETS_DIR),
    autoescape=select_autoescape([]),  # plain-text templates, no HTML escaping
    keep_trailing_newline=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _render(template_name: str, **kwargs) -> str:
    """Render an asset template by name."""
    return _env.get_template(template_name).render(**kwargs)


def _is_gemma3_model(model_name: str | None) -> bool:
    if not model_name:
        return False
    return "gemma-3" in model_name.lower()


def adapt_messages_for_model(
    messages: list[dict[str, str]],
    model_name: str | None = None,
) -> list[dict[str, str]]:
    """Normalize messages for model-specific chat template constraints."""
    if not _is_gemma3_model(model_name):
        return messages

    normalized: list[dict[str, str]] = []
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        if not content:
            continue

        if role == "system":
            role = "user"
            content = f"[Instructions]\n{content}"

        if normalized and normalized[-1]["role"] == role:
            normalized[-1]["content"] = f'{normalized[-1]["content"]}\n\n{content}'
        else:
            normalized.append({"role": role, "content": content})

    return normalized


# ---------------------------------------------------------------------------
# Internal helpers – resolve swapped responses for template variables
# ---------------------------------------------------------------------------

def _pairwise_vars(pair: JudgePair, swapped: bool) -> dict:
    if swapped:
        return dict(response_a=pair.response_b, response_b=pair.response_a)
    return dict(response_a=pair.response_a, response_b=pair.response_b)


def _multi_turn_vars(pair: JudgePair, swapped: bool) -> dict:
    if swapped:
        return dict(
            response_a_turn1=pair.response_b_turn1,
            response_b_turn1=pair.response_a_turn1,
            response_a=pair.response_b,
            response_b=pair.response_a,
        )
    return dict(
        response_a_turn1=pair.response_a_turn1,
        response_b_turn1=pair.response_b_turn1,
        response_a=pair.response_a,
        response_b=pair.response_b,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_messages(
    pair: JudgePair,
    judge_type: JudgeType,
    prompt_variant: PromptVariant,
    swapped: bool = False,
    which_response: str = "a",
    model_name: str | None = None,
) -> list[dict[str, str]]:
    """Build the chat messages for a given judge type and prompt variant.

    Args:
        pair: The MT-Bench pair to judge.
        judge_type: Type of judgment (pairwise, single_answer, reference_guided).
        prompt_variant: Prompt engineering variant.
        swapped: Whether to swap A/B positions (for position bias).
        which_response: For single-answer scoring, which response to rate ("a" or "b").
        model_name: Optional model name used for model-specific message normalization.

    Returns:
        List of {role, content} dicts for the chat API.
    """
    messages: list[dict[str, str]] = []

    # --- Pairwise ---
    if judge_type == JudgeType.PAIRWISE:
        # System prompt
        if prompt_variant == PromptVariant.COT:
            messages.append({"role": "system", "content": _render("system_pairwise_cot.jinja")})
        else:
            messages.append({"role": "system", "content": _render("system_pairwise.jinja")})

        # Few-shot examples
        if prompt_variant == PromptVariant.FEW_SHOT:
            messages.append({"role": "user", "content": _render("few_shot_pairwise.jinja")})

        # User content
        if prompt_variant == PromptVariant.MULTI_TURN:
            mt = _multi_turn_vars(pair, swapped)
            user_content = _render(
                "user_multi_turn.jinja",
                question=pair.question_text,
                follow_up_question=pair.follow_up_question,
                **mt,
            )
        else:
            swap_flag = swapped or (prompt_variant == PromptVariant.POSITION_SWAP)
            pw = _pairwise_vars(pair, swap_flag)
            user_content = _render(
                "user_pairwise.jinja",
                question=pair.question_text,
                reference_hint=(prompt_variant == PromptVariant.REFERENCE_GUIDED),
                **pw,
            )

        messages.append({"role": "user", "content": user_content})

    # --- Single Answer ---
    elif judge_type == JudgeType.SINGLE_ANSWER:
        if prompt_variant == PromptVariant.COT:
            messages.append({"role": "system", "content": _render("system_single_cot.jinja")})
        else:
            messages.append({"role": "system", "content": _render("system_single.jinja")})

        if prompt_variant == PromptVariant.FEW_SHOT:
            messages.append({"role": "user", "content": _render("few_shot_single.jinja")})

        response = pair.response_a if which_response == "a" else pair.response_b
        user_content = _render("user_single.jinja", question=pair.question_text, response=response)
        messages.append({"role": "user", "content": user_content})

    # --- Reference Guided ---
    elif judge_type == JudgeType.REFERENCE_GUIDED:
        if prompt_variant == PromptVariant.COT:
            messages.append({"role": "system", "content": _render("system_reference_cot.jinja")})
        else:
            messages.append({"role": "system", "content": _render("system_reference.jinja")})

        if not pair.reference_answer:
            raise ValueError("reference_answer is required for reference_guided judging")

        swap_flag = swapped or (prompt_variant == PromptVariant.POSITION_SWAP)
        pw = _pairwise_vars(pair, swap_flag)
        user_content = _render(
            "user_reference.jinja",
            question=pair.question_text,
            answer_ref=pair.reference_answer,
            **pw,
        )
        messages.append({"role": "user", "content": user_content})

    return adapt_messages_for_model(messages, model_name=model_name)
