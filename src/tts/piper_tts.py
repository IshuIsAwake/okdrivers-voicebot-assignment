"""Piper local TTS engine (ONNX). Uses the `piper-tts` Python package."""

import time
from pathlib import Path

import numpy as np

from src.metrics import measure_peak_memory
from src.tts.base import BaseTTS, TTSResult


class PiperTTS(BaseTTS):
    def __init__(self, voice: str = "hi_IN-pratham-medium",
                 voice_dir: str = "models/piper",
                 download_if_missing: bool = True):
        self._voice = voice
        self._voice_dir = Path(voice_dir)
        self._voice_dir.mkdir(parents=True, exist_ok=True)
        model_path = self._voice_dir / f"{voice}.onnx"
        config_path = self._voice_dir / f"{voice}.onnx.json"
        if not model_path.exists() or not config_path.exists():
            if download_if_missing:
                self._download_voice(voice, model_path, config_path)
            else:
                raise RuntimeError(
                    f"Piper voice '{voice}' not found in {voice_dir}. "
                    "Download from https://github.com/rhasspy/piper/blob/master/VOICES.md"
                )
        from piper.voice import PiperVoice  # type: ignore
        self._piper = PiperVoice.load(str(model_path), config_path=str(config_path))

    @staticmethod
    def _download_voice(voice: str, model_path: Path, config_path: Path) -> None:
        # Voice naming: <lang>_<COUNTRY>-<name>-<quality>
        import urllib.request
        try:
            lang = voice.split("_")[0]
            full_locale = "_".join(voice.split("-")[0].split("_"))
            voice_name = voice.split("-")[1]
            quality = voice.split("-")[2]
        except IndexError as e:
            raise RuntimeError(f"Could not parse voice id '{voice}'") from e
        base = (
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
            f"{lang}/{full_locale}/{voice_name}/{quality}/{voice}"
        )
        for suffix, dest in (("onnx", model_path), ("onnx.json", config_path)):
            url = f"{base}.{suffix}"
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download Piper voice from {url}. "
                    "Check internet, or download manually."
                ) from e

    def synthesize(self, text: str) -> TTSResult:
        start = time.perf_counter()
        with measure_peak_memory() as mem:
            chunks = list(self._piper.synthesize(text))
            if not chunks:
                audio = np.zeros(0, dtype=np.float32)
                sr = 22050
            else:
                # AudioChunk.audio_float_array is float32 mono in [-1,1].
                pieces = [c.audio_float_array.astype(np.float32) for c in chunks]
                audio = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
                sr = chunks[0].sample_rate
        inference = time.perf_counter() - start
        audio_dur = len(audio) / float(sr) if sr else 0.0
        return TTSResult(
            audio=audio,
            sample_rate=sr,
            audio_duration_sec=audio_dur,
            inference_duration_sec=inference,
            rtf=(inference / audio_dur) if audio_dur > 0 else 0.0,
            peak_memory_mb=mem.peak_mb,
            model_name=self.name(),
        )

    def name(self) -> str:
        return "piper"
