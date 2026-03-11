"""Prompt templates for different judge types and prompt variants."""

from __future__ import annotations

from .config import JudgeType, PromptVariant
from .data import JudgePair


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PAIRWISE = (
    "You are a fair and impartial judge. Your task is to evaluate two AI assistant "
    "responses to a user question and decide which one is better. You must output "
    "exactly one of: [[A]], [[B]], or [[C]] (for tie).\n"
    "Do NOT provide any other text before or after your verdict."
)

SYSTEM_PAIRWISE_COT = (
    "You are a fair and impartial judge. Your task is to evaluate two AI assistant "
    "responses to a user question. First, compare them on helpfulness, relevance, "
    "accuracy, depth, and creativity. Then compare on coherence, clarity, and "
    "language quality. Finally, give your verdict.\n"
    "End your response with exactly one of: [[A]], [[B]], or [[C]] (for tie)."
)

SYSTEM_SINGLE = (
    "You are a fair and impartial judge. Rate the quality of the AI assistant's "
    "response to the user question on a scale of 1 to 10. Output ONLY a single "
    "integer between 1 and 10, enclosed in double brackets like [[7]]."
)

SYSTEM_SINGLE_COT = (
    "You are a fair and impartial judge. Rate the quality of the AI assistant's "
    "response to the user question on a scale of 1 to 10. First analyze the "
    "response's helpfulness, relevance, accuracy, depth, creativity, and coherence. "
    "Then provide your rating.\n"
    "End your response with a single integer between 1 and 10 enclosed in double "
    "brackets like [[7]]."
)

SYSTEM_REFERENCE = (
    "You are a fair and impartial judge. You are given a reference answer and an "
    "AI assistant's response. Compare the assistant's response to the reference "
    "and rate it on a scale of 1 to 10.\n"
    "Output ONLY a single integer between 1 and 10, enclosed in double brackets "
    "like [[7]]."
)

# ---------------------------------------------------------------------------
# Few-shot examples (from MT-Bench style)
# ---------------------------------------------------------------------------

FEW_SHOT_PAIRWISE_EXAMPLES = """
Example 1:
[Question] What is the capital of France?
[Assistant A] The capital of France is Paris. It is known for the Eiffel Tower.
[Assistant B] Paris is the capital of France, a major European city renowned for art, fashion, gastronomy, and culture.
[Verdict] [[B]]

Example 2:
[Question] Explain gravity in simple terms.
[Assistant A] Gravity is the force that pulls objects toward each other. The more massive an object, the stronger its gravitational pull.
[Assistant B] Gravity makes things fall down.
[Verdict] [[A]]

Example 3:
[Question] What is 2+2?
[Assistant A] 4
[Assistant B] The answer is 4.
[Verdict] [[C]]
"""

FEW_SHOT_SINGLE_EXAMPLES = """
Example 1:
[Question] What is the capital of France?
[Response] Paris is the capital of France, a major European city renowned for art, fashion, gastronomy, and culture.
[Rating] [[9]]

Example 2:
[Question] Explain quantum computing.
[Response] Quantum computing uses qubits.
[Rating] [[3]]

Example 3:
[Question] Write a haiku about spring.
[Response] Cherry blossoms bloom, / Gentle rain on fresh green leaves, / New life awakens.
[Rating] [[8]]
"""


# ---------------------------------------------------------------------------
# Template builders
# ---------------------------------------------------------------------------

def _format_pairwise_user(pair: JudgePair, swapped: bool = False) -> str:
    """Format user message for pairwise comparison."""
    if swapped:
        resp_a, resp_b = pair.response_b, pair.response_a
    else:
        resp_a, resp_b = pair.response_a, pair.response_b

    text = (
        f"[User Question]\n{pair.question_text}\n\n"
        f"[The Start of Assistant A's Answer]\n{resp_a}\n"
        f"[The End of Assistant A's Answer]\n\n"
        f"[The Start of Assistant B's Answer]\n{resp_b}\n"
        f"[The End of Assistant B's Answer]\n\n"
        "Which assistant's answer is better? Output [[A]], [[B]], or [[C]] for tie."
    )
    return text


def _format_single_user(pair: JudgePair, which: str = "a") -> str:
    """Format user message for single-answer scoring."""
    response = pair.response_a if which == "a" else pair.response_b
    model = pair.model_a if which == "a" else pair.model_b
    text = (
        f"[User Question]\n{pair.question_text}\n\n"
        f"[The Start of Assistant's Answer]\n{response}\n"
        f"[The End of Assistant's Answer]\n\n"
        "Rate the quality of this response on a scale of 1 to 10. Output [[score]]."
    )
    return text


def _format_reference_user(pair: JudgePair, which: str = "a") -> str:
    """Format user message for reference-guided scoring."""
    response = pair.response_a if which == "a" else pair.response_b
    # Use the other response as a pseudo-reference
    reference = pair.response_b if which == "a" else pair.response_a
    text = (
        f"[User Question]\n{pair.question_text}\n\n"
        f"[Reference Answer]\n{reference}\n\n"
        f"[The Start of Assistant's Answer]\n{response}\n"
        f"[The End of Assistant's Answer]\n\n"
        "Compare the assistant's answer to the reference answer and rate it on a "
        "scale of 1 to 10. Output [[score]]."
    )
    return text


def _format_multi_turn_user(pair: JudgePair, swapped: bool = False) -> str:
    """Format multi-turn context (include turn 1 context for turn 2 questions)."""
    if swapped:
        resp_a_t1, resp_b_t1 = pair.response_b_turn1, pair.response_a_turn1
        resp_a, resp_b = pair.response_b, pair.response_a
    else:
        resp_a_t1, resp_b_t1 = pair.response_a_turn1, pair.response_b_turn1
        resp_a, resp_b = pair.response_a, pair.response_b

    parts = [f"[User Question (Turn 1)]\n{pair.question_text}\n"]
    parts.append(f"[Assistant A (Turn 1)]\n{resp_a_t1}\n")
    parts.append(f"[Assistant B (Turn 1)]\n{resp_b_t1}\n")

    if pair.follow_up_question:
        parts.append(f"\n[User Follow-up (Turn 2)]\n{pair.follow_up_question}\n")
        parts.append(f"[Assistant A (Turn 2)]\n{resp_a}\n")
        parts.append(f"[Assistant B (Turn 2)]\n{resp_b}\n")
    else:
        # For turn-1 items, fall back to standard pairwise
        parts.append(f"\n[Assistant A]\n{resp_a}\n")
        parts.append(f"[Assistant B]\n{resp_b}\n")

    parts.append(
        "\nConsidering the full conversation, which assistant is better? "
        "Output [[A]], [[B]], or [[C]] for tie."
    )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_messages(
    pair: JudgePair,
    judge_type: JudgeType,
    prompt_variant: PromptVariant,
    swapped: bool = False,
    which_response: str = "a",
) -> list[dict[str, str]]:
    """Build the chat messages for a given judge type and prompt variant.

    Args:
        pair: The MT-Bench pair to judge.
        judge_type: Type of judgment (pairwise, single_answer, reference_guided).
        prompt_variant: Prompt engineering variant.
        swapped: Whether to swap A/B positions (for position bias).
        which_response: For single/reference scoring, which response to rate ("a" or "b").

    Returns:
        List of {role, content} dicts for the chat API.
    """
    messages: list[dict[str, str]] = []

    # --- Pairwise ---
    if judge_type == JudgeType.PAIRWISE:
        if prompt_variant == PromptVariant.COT:
            messages.append({"role": "system", "content": SYSTEM_PAIRWISE_COT})
        else:
            messages.append({"role": "system", "content": SYSTEM_PAIRWISE})

        if prompt_variant == PromptVariant.FEW_SHOT:
            messages.append({"role": "user", "content": f"Here are some examples of how to judge:\n{FEW_SHOT_PAIRWISE_EXAMPLES}\nNow judge the following:"})

        if prompt_variant == PromptVariant.MULTI_TURN:
            user_content = _format_multi_turn_user(pair, swapped=swapped)
        elif prompt_variant == PromptVariant.REFERENCE_GUIDED:
            # Pairwise with reference hint
            user_content = _format_pairwise_user(pair, swapped=swapped)
            user_content += (
                "\n\nNote: Consider factual accuracy carefully. A response that "
                "is more factually accurate should be preferred."
            )
        else:
            swap_flag = swapped or (prompt_variant == PromptVariant.POSITION_SWAP)
            user_content = _format_pairwise_user(pair, swapped=swap_flag)

        messages.append({"role": "user", "content": user_content})

    # --- Single Answer ---
    elif judge_type == JudgeType.SINGLE_ANSWER:
        if prompt_variant == PromptVariant.COT:
            messages.append({"role": "system", "content": SYSTEM_SINGLE_COT})
        else:
            messages.append({"role": "system", "content": SYSTEM_SINGLE})

        if prompt_variant == PromptVariant.FEW_SHOT:
            messages.append({"role": "user", "content": f"Here are some examples:\n{FEW_SHOT_SINGLE_EXAMPLES}\nNow rate the following:"})

        user_content = _format_single_user(pair, which=which_response)
        messages.append({"role": "user", "content": user_content})

    # --- Reference Guided ---
    elif judge_type == JudgeType.REFERENCE_GUIDED:
        messages.append({"role": "system", "content": SYSTEM_REFERENCE})

        if prompt_variant == PromptVariant.FEW_SHOT:
            messages.append({"role": "user", "content": f"Here are some examples:\n{FEW_SHOT_SINGLE_EXAMPLES}\nNow rate the following:"})

        user_content = _format_reference_user(pair, which=which_response)
        messages.append({"role": "user", "content": user_content})

    return messages
