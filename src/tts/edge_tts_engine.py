"""Microsoft edge-tts cloud TTS engine."""

import asyncio
import io
import time

import numpy as np

from src.metrics import measure_peak_memory
from src.tts.base import BaseTTS, TTSResult


class EdgeTTSEngine(BaseTTS):
    def __init__(self, voice: str = "hi-IN-SwaraNeural"):
        # Lazy import — edge-tts pulls in aiohttp.
        import edge_tts  # type: ignore  # noqa: F401
        self._voice = voice

    async def _synth_async(self, text: str) -> bytes:
        import edge_tts  # type: ignore
        comm = edge_tts.Communicate(text, self._voice)
        chunks = []
        async for ev in comm.stream():
            if ev.get("type") == "audio":
                chunks.append(ev["data"])
        return b"".join(chunks)

    def _decode_mp3(self, mp3_bytes: bytes) -> tuple[np.ndarray, int]:
        # edge-tts returns mp3 by default. Decode via miniaudio if available, else pydub/ffmpeg.
        try:
            import miniaudio  # type: ignore
            decoded = miniaudio.decode(
                mp3_bytes,
                output_format=miniaudio.SampleFormat.FLOAT32,
                nchannels=1,
                sample_rate=24000,
            )
            audio = np.array(decoded.samples, dtype=np.float32)
            return audio, decoded.sample_rate
        except ImportError:
            pass
        # Fallback: use soundfile + an mp3-capable backend (libsndfile 1.1+).
        import soundfile as sf  # type: ignore
        data, sr = sf.read(io.BytesIO(mp3_bytes), dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        return data.astype(np.float32), sr

    def synthesize(self, text: str) -> TTSResult:
        start = time.perf_counter()
        with measure_peak_memory() as mem:
            mp3 = asyncio.run(self._synth_async(text))
            audio, sr = self._decode_mp3(mp3)
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
        return "edge_tts"
