"""Local Ollama Gemma 4 e4b — true edge engine."""

from src.llm._ollama_base import OllamaLLMBase


class Gemma4LocalOllama(OllamaLLMBase):
    def __init__(self, model: str = "gemma4:e4b",
                 host: str = "http://localhost:11434",
                 max_tokens: int = 150, temperature: float = 0.6):
        super().__init__(
            model=model,
            host=host,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=None,
            engine_name="gemma4_local",
        )
        self._verify_local_available()
