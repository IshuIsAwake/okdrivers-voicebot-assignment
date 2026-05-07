"""Silero VAD wrapper using the official `silero-vad` PyPI package (ONNX backend)."""

import numpy as np


class SileroVAD:
    SAMPLE_RATE = 16000
    FRAME_SAMPLES = 512  # Silero expects 512 samples @ 16kHz per frame

    def __init__(self, threshold: float = 0.5):
        from silero_vad import load_silero_vad  # type: ignore
        self._model = load_silero_vad(onnx=True)
        self.threshold = threshold
        self._buffer = np.zeros(0, dtype=np.float32)

    def is_speech(self, frame: np.ndarray) -> bool:
        """Append `frame` to internal buffer; consume in 512-sample windows.
        Returns True if the most recent window scored above threshold.
        """
        import torch  # type: ignore
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        self._buffer = np.concatenate([self._buffer, frame])
        last_score = 0.0
        while len(self._buffer) >= self.FRAME_SAMPLES:
            chunk = self._buffer[: self.FRAME_SAMPLES]
            self._buffer = self._buffer[self.FRAME_SAMPLES :]
            t = torch.from_numpy(chunk)
            last_score = float(self._model(t, self.SAMPLE_RATE).item())
        return last_score >= self.threshold

    def reset(self) -> None:
        try:
            self._model.reset_states()
        except Exception:
            pass
        self._buffer = np.zeros(0, dtype=np.float32)
