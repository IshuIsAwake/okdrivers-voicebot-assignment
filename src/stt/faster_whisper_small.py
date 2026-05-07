"""faster-whisper STT engine using the `small` multilingual model."""

from src.stt.faster_whisper_base import FasterWhisperBase


class FasterWhisperSmall(FasterWhisperBase):
    MODEL_SIZE = "small"
