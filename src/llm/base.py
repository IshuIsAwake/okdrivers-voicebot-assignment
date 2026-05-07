from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass
class LLMChunk:
    delta: str
    is_final: bool
    token_count: int


class BaseLLM(ABC):
    @abstractmethod
    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[LLMChunk]:
        """Yields LLMChunks as the model generates."""
        ...

    @abstractmethod
    def name(self) -> str:
        ...
