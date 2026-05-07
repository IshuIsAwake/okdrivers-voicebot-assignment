"""Groq cloud API LLM — Llama 3.1 8B instant (streaming)."""

import os
from typing import Iterator

from src.llm.base import BaseLLM, LLMChunk


class GroqLlama8B(BaseLLM):
    def __init__(self, model: str = "llama-3.1-8b-instant",
                 max_tokens: int = 150, temperature: float = 0.6):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to .env (see .env.example) "
                "or export it in your shell."
            )
        from groq import Groq  # type: ignore
        self._client = Groq(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[LLMChunk]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stream=True,
        )
        token_count = 0
        last: LLMChunk = LLMChunk(delta="", is_final=False, token_count=0)
        for event in stream:
            choice = event.choices[0] if event.choices else None
            if choice is None:
                continue
            delta = (choice.delta.content or "") if choice.delta else ""
            if delta:
                token_count += max(1, len(delta.split()))
            finish = choice.finish_reason
            chunk = LLMChunk(delta=delta, is_final=finish is not None, token_count=token_count)
            yield chunk
            last = chunk
        if not last.is_final:
            yield LLMChunk(delta="", is_final=True, token_count=token_count)

    def name(self) -> str:
        return "groq_8b"
