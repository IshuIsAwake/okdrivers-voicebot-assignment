"""faster-whisper STT engine using the `base` multilingual model."""

import time

import numpy as np

from src.metrics import measure_peak_memory
from src.stt.base import BaseSTT, STTResult


class FasterWhisperBase(BaseSTT):
    MODEL_SIZE = "base"

    def __init__(self, device: str = "auto", compute_type: str = "auto"):
        from faster_whisper import WhisperModel  # type: ignore
        self._model = WhisperModel(self.MODEL_SIZE, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> STTResult:
        if sample_rate != 16000:
            raise ValueError(f"Expected 16kHz audio, got {sample_rate}")
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        audio_dur = len(audio) / float(sample_rate)
        start = time.perf_counter()
        with measure_peak_memory() as mem:
            # Auto-detect language — Whisper's behavior on Hinglish is itself part
            # of the ablation finding (small models often pick Urdu/Devanagari script
            # for short utterances; larger models stay in Latin for code-switched
            # speech). WER is computed against a script-normalized reference, so the
            # cross-script penalty is real signal, not a measurement bug.
            segments, _info = self._model.transcribe(
                audio,
                language=None,
                task="transcribe",
                vad_filter=False,
                beam_size=1,
                without_timestamps=True,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
        inference = time.perf_counter() - start
        return STTResult(
            text=text,
            audio_duration_sec=audio_dur,
            inference_duration_sec=inference,
            rtf=(inference / audio_dur) if audio_dur > 0 else 0.0,
            peak_memory_mb=mem.peak_mb,
            model_name=self.name(),
        )

    def name(self) -> str:
        return f"fw_{self.MODEL_SIZE}"
