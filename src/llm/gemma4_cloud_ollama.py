"""Ollama Cloud Gemma 4 31B — cloud-hosted open-weights engine."""

import os

from src.llm._ollama_base import OllamaLLMBase


class Gemma4CloudOllama(OllamaLLMBase):
    def __init__(self, model: str = "gemma4:31b-cloud",
                 host: str = "https://ollama.com",
                 max_tokens: int = 150, temperature: float = 0.6):
        api_key = os.environ.get("OLLAMA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OLLAMA_API_KEY not set. Sign up at https://ollama.com, generate "
                "an API key, and add it to .env (see .env.example)."
            )
        super().__init__(
            model=model,
            host=host,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            engine_name="gemma4_cloud",
        )
