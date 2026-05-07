"""Streaming pipeline orchestrator. The `run_turn` function is the system's heart.

It is shared by both `cli.py` (live demo) and `benchmark.py` (ablation harness).
Only the `playback` argument differs: live uses `audio_io.Playback`, benchmark uses
`audio_io.NullPlayback` so latency is measured independent of speaker hardware.
"""

import re
import threading
import time
from queue import Queue
from typing import Any, Dict

import numpy as np

SENTENCE_END = re.compile(r'[.!?।]+[\s"\'\)]*')


def run_turn(
    audio_buf: np.ndarray,
    sample_rate: int,
    stt,
    llm,
    tts,
    playback,
    system_prompt: str,
    log: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute one full turn. `log` must already contain `t_end_of_speech`."""

    # --- STT (synchronous) ---
    t_end_of_speech = log["t_end_of_speech"]
    stt_result = stt.transcribe(audio_buf, sample_rate)
    log["t_stt_done"] = time.perf_counter()
    log["stt_rtf"] = stt_result.rtf
    log["stt_peak_mem_mb"] = stt_result.peak_memory_mb
    log["stt_inference_sec"] = stt_result.inference_duration_sec
    log["transcript"] = stt_result.text

    # --- TTS worker ---
    tts_in: "Queue[Any]" = Queue()
    audio_out: "Queue[Any]" = Queue()
    first_audio_ready = threading.Event()
    tts_stats: Dict[str, float] = {"peak_mem_mb": 0.0, "audio_dur": 0.0, "inf": 0.0}

    def tts_worker():
        while True:
            sentence = tts_in.get()
            if sentence is None:
                audio_out.put(None)
                break
            try:
                r = tts.synthesize(sentence)
            except Exception as e:
                import sys as _sys
                import traceback as _tb
                log.setdefault("tts_errors", []).append(repr(e))
                _tb.print_exc(file=_sys.stderr)
                continue
            tts_stats["peak_mem_mb"] = max(tts_stats["peak_mem_mb"], r.peak_memory_mb)
            tts_stats["audio_dur"] += r.audio_duration_sec
            tts_stats["inf"] += r.inference_duration_sec
            audio_out.put((r.audio, r.sample_rate))
            if not first_audio_ready.is_set():
                log["t_first_audio_ready"] = time.perf_counter()
                first_audio_ready.set()

    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    # --- Playback worker ---
    first_audio_played = threading.Event()
    playback_done = threading.Event()

    def playback_worker():
        while True:
            item = audio_out.get()
            if item is None:
                playback_done.set()
                break
            audio, sr = item
            if not first_audio_played.is_set():
                log["t_first_audio_played"] = time.perf_counter()
                first_audio_played.set()
            playback.play_blocking(audio, sr)

    playback_thread = threading.Thread(target=playback_worker, daemon=True)
    playback_thread.start()

    # --- LLM streaming + sentence chunking ---
    buffer = ""
    word_count = 0
    token_count = 0
    full_response = []
    first_token_logged = False
    first_sentence_pushed = False
    user_prompt = stt_result.text

    llm_start = time.perf_counter()
    llm_finished_early = False
    for chunk in llm.stream(system_prompt, user_prompt):
        if chunk.delta and not first_token_logged:
            log["t_llm_first_token"] = time.perf_counter()
            first_token_logged = True
        if chunk.delta:
            buffer += chunk.delta
            full_response.append(chunk.delta)
        token_count = chunk.token_count
        while True:
            m = SENTENCE_END.search(buffer)
            if not m:
                break
            end = m.end()
            sentence = buffer[:end].strip()
            buffer = buffer[end:]
            if sentence:
                tts_in.put(sentence)
                word_count += len(sentence.split())
                if not first_sentence_pushed:
                    log["t_first_sentence_to_tts"] = time.perf_counter()
                    first_sentence_pushed = True
        if chunk.is_final:
            llm_finished_early = True
            break

    if buffer.strip():
        tail = buffer.strip()
        tts_in.put(tail)
        word_count += len(tail.split())
        if not first_sentence_pushed:
            log["t_first_sentence_to_tts"] = time.perf_counter()
            first_sentence_pushed = True
    tts_in.put(None)  # sentinel: drain & exit

    llm_total = time.perf_counter() - llm_start
    log["llm_total_sec"] = llm_total
    log["llm_token_count"] = token_count
    log["llm_tps"] = (token_count / llm_total) if llm_total > 0 else 0.0
    log["llm_word_count"] = word_count
    log["llm_response"] = "".join(full_response).strip()
    log["llm_finished_streaming"] = llm_finished_early

    # Wait for TTS + playback to drain.
    playback_done.wait()
    log["t_playback_done"] = time.perf_counter()

    # Aggregate TTS metrics.
    log["tts_peak_mem_mb"] = tts_stats["peak_mem_mb"]
    log["tts_rtf"] = (tts_stats["inf"] / tts_stats["audio_dur"]) if tts_stats["audio_dur"] else 0.0
    log["tts_audio_duration_sec"] = tts_stats["audio_dur"]

    # Derived headline metrics.
    if "t_first_audio_played" in log:
        log["e2e_latency_sec"] = log["t_first_audio_played"] - t_end_of_speech
        log["playback_total_sec"] = log["t_playback_done"] - log["t_first_audio_played"]
    else:
        log["e2e_latency_sec"] = float("nan")
        log["playback_total_sec"] = float("nan")
    if "t_llm_first_token" in log:
        log["llm_ttft_sec"] = log["t_llm_first_token"] - log["t_stt_done"]
    else:
        log["llm_ttft_sec"] = float("nan")
    if "t_first_sentence_to_tts" in log:
        log["llm_time_to_first_sentence_sec"] = log["t_first_sentence_to_tts"] - log["t_stt_done"]
    else:
        log["llm_time_to_first_sentence_sec"] = float("nan")

    return log
