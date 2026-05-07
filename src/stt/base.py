from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class STTResult:
    text: str
    audio_duration_sec: float
    inference_duration_sec: float
    rtf: float
    peak_memory_mb: float
    model_name: str


class BaseSTT(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> STTResult:
        """Synchronous transcription. Audio is float32 mono at sample_rate Hz."""
        ...

    @abstractmethod
    def name(self) -> str:
        ...
