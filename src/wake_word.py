"""openWakeWord wrapper. Loads the built-in `hey_jarvis` preset by default."""

from typing import Optional

import numpy as np


class WakeWordDetector:
    def __init__(self, model_name: str = "hey_jarvis", threshold: float = 0.5,
                 sample_rate: int = 16000):
        from openwakeword.model import Model  # type: ignore
        # Trigger first-run download of pre-bundled models (best-effort).
        try:
            import openwakeword
            openwakeword.utils.download_models()
        except Exception:
            pass
        # Force ONNX backend — the bundled tflite_runtime is compiled against
        # NumPy 1.x and crashes under NumPy 2.x. ONNX models were already
        # downloaded by `download_models()` above.
        self._model = Model(wakeword_models=[model_name], inference_framework="onnx")
        self._key: Optional[str] = None
        for k in self._model.models.keys():
            if model_name in k:
                self._key = k
                break
        if self._key is None:
            self._key = next(iter(self._model.models.keys()))
        self.threshold = threshold
        self.sample_rate = sample_rate

    def process(self, frame: np.ndarray) -> bool:
        """Feed a chunk of float32 mono audio. Returns True if wake word detected."""
        if frame.dtype != np.int16:
            int16 = np.clip(frame * 32767.0, -32768, 32767).astype(np.int16)
        else:
            int16 = frame
        scores = self._model.predict(int16)
        score = scores.get(self._key, 0.0) if isinstance(scores, dict) else 0.0
        return score >= self.threshold

    def reset(self) -> None:
        try:
            self._model.reset()
        except Exception:
            pass
