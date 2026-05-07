"""Interactive Rich-based CLI demo for okDriver voice assistant.

Wake word ("hey jarvis") -> VAD-gated mic recording -> STT -> streaming LLM ->
sentence-chunked TTS -> playback. One turn, then back to listening.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import IntPrompt
from rich.table import Table

from src.audio_io import MicCapture, Playback
from src.engines import LLM_NAMES, STT_NAMES, TTS_NAMES, build_llm, build_stt, build_tts
from src.pipeline import run_turn
from src.prompts import SYSTEM_PROMPT


def load_config(path: str = "config.yaml") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick(console: Console, label: str, options: list, default_idx: int = 0) -> str:
    table = Table(title=f"Choose {label}", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("Engine")
    for i, name in enumerate(options, start=1):
        table.add_row(str(i), name)
    console.print(table)
    idx = IntPrompt.ask("Selection", default=default_idx + 1, console=console)
    if idx < 1 or idx > len(options):
        idx = default_idx + 1
    return options[idx - 1]


def record_until_silence(mic: MicCapture, vad, max_sec: float, silence_ms: int,
                         sample_rate: int) -> tuple[np.ndarray, float]:
    """Returns (audio_buffer, t_end_of_speech)."""
    chunks = []
    silent_ms = 0
    speech_started = False
    started = time.perf_counter()
    frame_ms = 30
    silence_threshold_ms = silence_ms
    while True:
        frame = mic.read_frame(timeout=1.0)
        chunks.append(frame)
        is_speech = vad.is_speech(frame)
        if is_speech:
            speech_started = True
            silent_ms = 0
        else:
            if speech_started:
                silent_ms += frame_ms
        elapsed = time.perf_counter() - started
        if speech_started and silent_ms >= silence_threshold_ms:
            break
        if elapsed >= max_sec:
            break
    t_end_of_speech = time.perf_counter()
    audio = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)
    return audio, t_end_of_speech


def render_state(state: str, hint: str = "") -> Panel:
    title = f"okDriver  ·  state: [bold cyan]{state}[/bold cyan]"
    body = hint or {
        "IDLE": "Loading…",
        "LISTENING": 'Say "Hey Jarvis" to start.',
        "RECORDING": "Listening to your question…",
        "TRANSCRIBING": "Transcribing audio…",
        "THINKING": "Generating response…",
        "SPEAKING": "Speaking response…",
    }.get(state, "")
    return Panel(body, title=title, border_style="cyan")


def render_metrics_table(log: Dict) -> Table:
    t = Table(title="Turn metrics", show_lines=False)
    t.add_column("Metric")
    t.add_column("Value", justify="right")

    def f(key, fmt="{:.3f}"):
        v = log.get(key)
        return fmt.format(v) if isinstance(v, (int, float)) and v == v else "—"

    t.add_row("E2E latency (s)", f("e2e_latency_sec"))
    t.add_row("STT RTF", f("stt_rtf"))
    t.add_row("STT peak mem (MB)", f("stt_peak_mem_mb", "{:.1f}"))
    t.add_row("LLM TTFT (s)", f("llm_ttft_sec"))
    t.add_row("LLM time-to-1st-sentence (s)", f("llm_time_to_first_sentence_sec"))
    t.add_row("LLM TPS", f("llm_tps", "{:.1f}"))
    t.add_row("LLM word count", str(log.get("llm_word_count", "—")))
    t.add_row("TTS RTF", f("tts_rtf"))
    t.add_row("TTS peak mem (MB)", f("tts_peak_mem_mb", "{:.1f}"))
    t.add_row("Playback total (s)", f("playback_total_sec"))
    return t


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="okDriver interactive CLI demo")
    parser.add_argument("--stt", choices=STT_NAMES, help="STT engine")
    parser.add_argument("--llm", choices=LLM_NAMES, help="LLM engine")
    parser.add_argument("--tts", choices=TTS_NAMES, help="TTS engine")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log-jsonl", default="results/cli_log.jsonl")
    args = parser.parse_args()

    cfg = load_config(args.config)
    console = Console()
    console.print(Panel.fit(
        "[bold]okDriver Voice Assistant[/bold] — Hinglish, streaming pipeline",
        border_style="green",
    ))

    stt_name = args.stt or pick(console, "STT", STT_NAMES,
                                STT_NAMES.index(cfg["stt"]["default"]))
    llm_name = args.llm or pick(console, "LLM", LLM_NAMES,
                                LLM_NAMES.index(cfg["llm"]["default"]))
    tts_name = args.tts or pick(console, "TTS", TTS_NAMES,
                                TTS_NAMES.index(cfg["tts"]["default"]))

    with console.status(f"Loading STT ({stt_name})…"):
        stt = build_stt(stt_name)
    with console.status(f"Loading LLM ({llm_name})…"):
        llm = build_llm(llm_name, cfg["llm"])
    with console.status(f"Loading TTS ({tts_name})…"):
        tts = build_tts(tts_name, cfg["tts"])

    sample_rate = cfg["audio"]["sample_rate"]
    silence_ms = cfg["vad"]["silence_ms"]
    max_rec = cfg["vad"]["max_recording_sec"]
    threshold = cfg["wake_word"]["threshold"]
    ww_model = cfg["wake_word"]["model"]

    with console.status("Loading wake word + VAD…"):
        from src.wake_word import WakeWordDetector
        from src.vad import SileroVAD
        ww = WakeWordDetector(model_name=ww_model, threshold=threshold,
                              sample_rate=sample_rate)
        vad = SileroVAD()
        playback = Playback()

    mic = MicCapture(sample_rate=sample_rate, frame_ms=30)
    mic.start()

    log_path = Path(args.log_jsonl)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("a", encoding="utf-8")

    console.print(Panel.fit('Say "Hey Jarvis" to begin. Ctrl+C to quit.',
                            border_style="green"))
    try:
        while True:
            with Live(render_state("LISTENING"), console=console, refresh_per_second=8) as live:
                ww.reset()
                while True:
                    frame = mic.read_frame(timeout=1.0)
                    if ww.process(frame):
                        break
                live.update(render_state("RECORDING"))
                vad.reset()
                audio, t_end_of_speech = record_until_silence(
                    mic, vad, max_rec, silence_ms, sample_rate
                )
                if len(audio) < sample_rate // 2:  # <0.5s of audio
                    live.update(render_state("LISTENING",
                                             "(too short — say something then pause)"))
                    continue
                live.update(render_state("TRANSCRIBING"))
                turn_log: Dict = {
                    "run_id": str(uuid.uuid4()),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "stt_engine": stt.name(),
                    "llm_engine": llm.name(),
                    "tts_engine": tts.name(),
                    "audio_input_path": None,
                    "reference_transcript": None,
                    "wer": None,
                    "t_end_of_speech": t_end_of_speech,
                }
                live.update(render_state("THINKING"))
                run_turn(audio, sample_rate, stt, llm, tts, playback,
                         SYSTEM_PROMPT, turn_log)
                live.update(render_state("SPEAKING", "(playback complete)"))

            console.print(render_metrics_table(turn_log))
            console.print(Panel(turn_log.get("transcript", ""),
                                title="Transcript", border_style="yellow"))
            console.print(Panel(turn_log.get("llm_response", ""),
                                title="Response", border_style="magenta"))
            log_fh.write(json.dumps({k: v for k, v in turn_log.items()
                                     if not k.startswith("t_")}, default=str) + "\n")
            log_fh.flush()
    except KeyboardInterrupt:
        console.print("\n[bold]Shutting down…[/bold]")
    finally:
        mic.stop()
        log_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
