from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class TTSResult:
    audio: np.ndarray
    sample_rate: int
    audio_duration_sec: float
    inference_duration_sec: float
    rtf: float
    peak_memory_mb: float
    model_name: str


class BaseTTS(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> TTSResult:
        """Synchronous synthesis of a single sentence/chunk."""
        ...

    @abstractmethod
    def name(self) -> str:
        ...
