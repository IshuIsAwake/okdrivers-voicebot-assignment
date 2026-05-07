"""Microphone capture and playback helpers."""

import queue
import threading
from typing import Optional

import numpy as np


def _import_sounddevice():
    try:
        import sounddevice as sd  # noqa: WPS433
        return sd
    except OSError as e:
        raise RuntimeError(
            "sounddevice failed to load PortAudio. "
            "Install system package 'libportaudio2' (Linux) or 'portaudio' (macOS)."
        ) from e


class MicCapture:
    """Continuous mic capture into a thread-safe queue of float32 frames."""

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30, channels: int = 1):
        self.sample_rate = sample_rate
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.channels = channels
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        # indata is float32 shape (frames, channels)
        mono = indata[:, 0] if indata.ndim == 2 else indata
        self._q.put(mono.copy())

    def start(self) -> None:
        sd = _import_sounddevice()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.frame_samples,
            callback=self._callback,
        )
        self._stream.start()

    def read_frame(self, timeout: Optional[float] = None) -> np.ndarray:
        return self._q.get(timeout=timeout)

    def drain(self) -> None:
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class Playback:
    """Synchronous PCM playback. play_blocking returns when buffer drains."""

    def play_blocking(self, audio: np.ndarray, sample_rate: int) -> None:
        sd = _import_sounddevice()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        sd.play(audio, samplerate=sample_rate, blocking=True)


class NullPlayback:
    """No-op playback used by benchmark.py — still allows latency measurement."""

    def play_blocking(self, audio: np.ndarray, sample_rate: int) -> None:  # noqa: ARG002
        return None
