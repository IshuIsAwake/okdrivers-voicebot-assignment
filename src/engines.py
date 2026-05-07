"""Engine registry: maps short config names to constructors. Lazy-imported."""

from typing import Dict


STT_NAMES = ["fw_base", "fw_small"]
LLM_NAMES = ["groq_8b", "gemma4_local", "gemma4_cloud"]
TTS_NAMES = ["piper", "edge_tts"]


def build_stt(name: str):
    if name == "fw_base":
        from src.stt.faster_whisper_base import FasterWhisperBase
        return FasterWhisperBase()
    if name == "fw_small":
        from src.stt.faster_whisper_small import FasterWhisperSmall
        return FasterWhisperSmall()
    raise ValueError(f"Unknown STT engine: {name}. Pick one of {STT_NAMES}.")


def build_llm(name: str, cfg: Dict):
    common = dict(
        max_tokens=cfg.get("max_tokens", 150),
        temperature=cfg.get("temperature", 0.6),
    )
    if name == "groq_8b":
        from src.llm.groq_llama8b import GroqLlama8B
        return GroqLlama8B(model=cfg.get("groq_model", "llama-3.1-8b-instant"), **common)
    if name == "gemma4_local":
        from src.llm.gemma4_local_ollama import Gemma4LocalOllama
        return Gemma4LocalOllama(
            model=cfg.get("gemma4_local_model", "gemma4:e4b"),
            host=cfg.get("ollama_local_host", "http://localhost:11434"),
            **common,
        )
    if name == "gemma4_cloud":
        from src.llm.gemma4_cloud_ollama import Gemma4CloudOllama
        return Gemma4CloudOllama(
            model=cfg.get("gemma4_cloud_model", "gemma4:31b-cloud"),
            host=cfg.get("ollama_cloud_host", "https://ollama.com"),
            **common,
        )
    raise ValueError(f"Unknown LLM engine: {name}. Pick one of {LLM_NAMES}.")


def build_tts(name: str, cfg: Dict):
    if name == "piper":
        from src.tts.piper_tts import PiperTTS
        return PiperTTS(
            voice=cfg.get("piper_voice", "hi_IN-pratham-medium"),
            voice_dir=cfg.get("piper_voice_dir", "models/piper"),
        )
    if name == "edge_tts":
        from src.tts.edge_tts_engine import EdgeTTSEngine
        return EdgeTTSEngine(voice=cfg.get("edge_voice", "hi-IN-SwaraNeural"))
    raise ValueError(f"Unknown TTS engine: {name}. Pick one of {TTS_NAMES}.")
