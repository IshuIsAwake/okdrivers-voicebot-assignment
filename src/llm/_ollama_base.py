"""Shared base for Ollama-backed LLM engines (local + cloud)."""

from typing import Iterator, Optional

from src.llm.base import BaseLLM, LLMChunk


class OllamaLLMBase(BaseLLM):
    def __init__(self, model: str, host: str, max_tokens: int = 150,
                 temperature: float = 0.6, api_key: Optional[str] = None,
                 engine_name: str = "ollama"):
        from ollama import Client  # type: ignore
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = Client(host=host, headers=headers) if headers else Client(host=host)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._engine_name = engine_name
        self._host = host

    def _verify_local_available(self) -> None:
        """For the local engine, fail fast if Ollama isn't reachable or the model is missing."""
        try:
            available = self._client.list()
        except Exception as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self._host}. "
                "Is `ollama serve` running? "
                f"(underlying error: {e})"
            ) from e
        models = available.get("models", []) if isinstance(available, dict) else getattr(available, "models", [])
        names = []
        for m in models:
            n = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
            if n:
                names.append(n)
        if not any(self._model == n or n.startswith(self._model + ":") or n == self._model for n in names):
            raise RuntimeError(
                f"Ollama model '{self._model}' is not pulled. "
                f"Run:  ollama pull {self._model}"
            )

    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[LLMChunk]:
        # `think=False` disables the model's internal reasoning phase for "thinking"
        # models like Gemma 4 e4b — without this the model emits its chain-of-thought
        # into `message.thinking` and `message.content` stays empty until the budget
        # runs out. Some non-thinking models will silently ignore the flag.
        token_count = 0
        chat_kwargs = dict(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            options={
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        )
        try:
            gen = self._client.chat(think=False, **chat_kwargs)
        except TypeError:
            # Older ollama-python clients without the `think` kwarg.
            gen = self._client.chat(**chat_kwargs)
        last_done = False
        for chunk in gen:
            if isinstance(chunk, dict):
                msg = chunk.get("message") or {}
                delta = msg.get("content", "") if isinstance(msg, dict) else ""
                done = bool(chunk.get("done", False))
            else:
                msg = getattr(chunk, "message", None)
                delta = getattr(msg, "content", "") if msg is not None else ""
                done = bool(getattr(chunk, "done", False))
            if delta:
                token_count += max(1, len(delta.split()))
            yield LLMChunk(delta=delta, is_final=done, token_count=token_count)
            last_done = done
        if not last_done:
            yield LLMChunk(delta="", is_final=True, token_count=token_count)

    def name(self) -> str:
        return self._engine_name
