"""LLM client supporting vLLM, SGLang, and OpenAI backends via OpenAI-compatible API."""

from __future__ import annotations

import logging
from typing import Optional

from openai import OpenAI

from .config import ModelConfig, BackendType
from .prompts import adapt_messages_for_model

# logger = logging.getLogger(__name__)


def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
    """Convert chat messages to a single prompt string for text completions API."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"<start_of_turn>system\n{content}<end_of_turn>")
        elif role == "user":
            parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
        elif role == "assistant":
            parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
    parts.append("<start_of_turn>model\n")
    return "\n".join(parts)


class LLMClient:
    """Unified LLM client that talks to vLLM / SGLang / OpenAI via OpenAI-compatible API."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = OpenAI(
            base_url=self._get_base_url(),
            api_key=config.api_key,
        )
        self.model_name = config.model_name
        self._use_completions = False  # fallback flag

    def _get_base_url(self) -> str:
        url = self.config.base_url.rstrip("/")
        if self.config.backend in (BackendType.VLLM, BackendType.SGLANG):
            if not url.endswith("/v1"):
                url += "/v1"
        return url

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        seed: Optional[int] = None,
    ) -> str:
        """Generate a completion from the LLM.

        Tries chat completions first; falls back to text completions if needed.

        Args:
            seed: Random seed for reproducible sampling. Passed to the API when set.
                  Supported by vLLM, SGLang, and OpenAI (>=gpt-4-turbo).
        """
        request_messages = adapt_messages_for_model(messages, self.model_name)
        extra = {"seed": seed} if seed is not None else {}

        if not self._use_completions:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=request_messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    **extra,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                # logger.warning("Chat completions failed, falling back to text completions: %s", e)
                self._use_completions = True

        # Fallback: use text completions API
        try:
            prompt = _messages_to_prompt(request_messages)
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=["<end_of_turn>"],
                **extra,
            )
            return response.choices[0].text or ""
        except Exception as e:
            # logger.error("LLM generation failed: %s", e)
            return f"ERROR: {e}"

    def generate_batch(
        self,
        messages_list: list[list[dict[str, str]]],
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        seed: Optional[int] = None,
    ) -> list[str]:
        """Generate completions for a batch of message lists (sequentially)."""
        results = []
        for messages in messages_list:
            results.append(self.generate(messages, temperature, top_p, max_tokens, seed=seed))
        return results
