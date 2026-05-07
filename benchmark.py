"""Ablation harness: sweep every (STT × LLM × TTS) combo over a fixed audio set.

Outputs:
  - results/ablation.csv          per-turn metrics
  - results/ablation_summary.md   markdown table grouped by combo (median values)

Non-interactive: no prompts, no menus. Exits 0 on success, non-zero on errors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
import wave
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from src.audio_io import NullPlayback
from src.engines import LLM_NAMES, STT_NAMES, TTS_NAMES, build_llm, build_stt, build_tts
from src.pipeline import run_turn
from src.prompts import SYSTEM_PROMPT


CSV_COLUMNS = [
    "run_id", "timestamp",
    "stt_engine", "llm_engine", "tts_engine",
    "audio_input_path", "transcript", "reference_transcript", "wer",
    "llm_response", "llm_word_count",
    "stt_rtf", "stt_peak_mem_mb",
    "llm_ttft_sec", "llm_time_to_first_sentence_sec",
    "llm_total_sec", "llm_token_count", "llm_tps",
    "tts_rtf", "tts_peak_mem_mb",
    "e2e_latency_sec", "playback_total_sec",
]


def load_wav_mono_16k(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sw = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if sw == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sw == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width {sw} in {path}")
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    if sr != 16000:
        raise ValueError(
            f"{path}: expected 16000 Hz, got {sr}. Re-encode with `ffmpeg -ar 16000 -ac 1`."
        )
    return audio, sr


def parse_combos(arg_combos: str, arg_stt, arg_llm, arg_tts) -> List[tuple]:
    if arg_combos == "all" or arg_combos is None:
        sttx = STT_NAMES
        llmx = LLM_NAMES
        ttsx = TTS_NAMES
    else:
        # comma-separated list of either combo names or per-axis filters via flags
        items = [c.strip() for c in arg_combos.split(",") if c.strip()]
        sttx = [c for c in items if c in STT_NAMES] or STT_NAMES
        llmx = [c for c in items if c in LLM_NAMES] or LLM_NAMES
        ttsx = [c for c in items if c in TTS_NAMES] or TTS_NAMES
    if arg_stt:
        sttx = [s for s in sttx if s in arg_stt.split(",")]
    if arg_llm:
        llmx = [l for l in llmx if l in arg_llm.split(",")]
    if arg_tts:
        ttsx = [t for t in ttsx if t in arg_tts.split(",")]
    return list(product(sttx, llmx, ttsx))


def _normalize_for_wer(s: str) -> str:
    import re
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_wer(reference: str, hypothesis: str) -> float:
    if not reference.strip() or not hypothesis.strip():
        return float("nan")
    try:
        import jiwer  # type: ignore
        return float(jiwer.wer(_normalize_for_wer(reference),
                                _normalize_for_wer(hypothesis)))
    except Exception:
        return float("nan")


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description="okDriver ablation benchmark harness")
    p.add_argument("--test-dir", default="test_audio")
    p.add_argument("--output", default="results/ablation.csv")
    p.add_argument("--combos", default="all",
                   help="'all' or comma-separated engine names to filter axes")
    p.add_argument("--stt", default=None, help="comma-separated STT filter")
    p.add_argument("--llm", default=None, help="comma-separated LLM filter")
    p.add_argument("--tts", default=None, help="comma-separated TTS filter")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of audio files per combo (debug aid)")
    p.add_argument("--dry-run", action="store_true",
                   help="Build engines and report combo plan; do not run inference")
    args = p.parse_args()

    test_dir = Path(args.test_dir)
    manifest_path = test_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} missing. See test_audio/README.md.", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        print("ERROR: manifest.json must be a non-empty list.", file=sys.stderr)
        return 2

    audio_files = []
    for entry in manifest:
        fname = entry.get("filename")
        ref = entry.get("reference") or entry.get("reference_transcript") or ""
        path = test_dir / fname
        if not path.exists():
            print(f"WARN: skipping {fname} — file missing.", file=sys.stderr)
            continue
        audio_files.append((path, ref))
    if not audio_files:
        print(f"ERROR: No test audio found — see {test_dir}/README.md", file=sys.stderr)
        return 2
    if args.limit:
        audio_files = audio_files[: args.limit]

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    combos = parse_combos(args.combos, args.stt, args.llm, args.tts)
    print(f"Plan: {len(combos)} combos × {len(audio_files)} audio files = "
          f"{len(combos) * len(audio_files)} runs")

    if args.dry_run:
        for c in combos:
            print("  combo:", c)
        return 0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    null_playback = NullPlayback()

    for stt_name, llm_name, tts_name in combos:
        print(f"\n=== Combo: STT={stt_name}  LLM={llm_name}  TTS={tts_name} ===")
        try:
            stt = build_stt(stt_name)
            llm = build_llm(llm_name, cfg["llm"])
            tts = build_tts(tts_name, cfg["tts"])
        except Exception as e:
            print(f"  SKIP combo (engine load failed): {e}", file=sys.stderr)
            continue

        for audio_path, reference in audio_files:
            try:
                audio, sr = load_wav_mono_16k(audio_path)
            except Exception as e:
                print(f"  SKIP {audio_path.name}: {e}", file=sys.stderr)
                continue
            log: Dict = {
                "run_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "stt_engine": stt_name,
                "llm_engine": llm_name,
                "tts_engine": tts_name,
                "audio_input_path": str(audio_path),
                "reference_transcript": reference,
                "t_end_of_speech": time.perf_counter(),
            }
            try:
                run_turn(audio, sr, stt, llm, tts, null_playback, SYSTEM_PROMPT, log)
            except Exception as e:
                print(f"  ERROR {audio_path.name}: {e}", file=sys.stderr)
                continue
            log["wer"] = compute_wer(reference, log.get("transcript", ""))
            rows.append({k: log.get(k) for k in CSV_COLUMNS})
            print(f"  {audio_path.name}: e2e={log.get('e2e_latency_sec', float('nan')):.3f}s  "
                  f"WER={log.get('wer', float('nan')):.3f}  "
                  f"TTFT={log.get('llm_ttft_sec', float('nan')):.3f}s")

    if not rows:
        print("ERROR: no rows produced.", file=sys.stderr)
        return 3

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")

    summary = (df.groupby(["stt_engine", "llm_engine", "tts_engine"])
                 .agg({
                     "e2e_latency_sec": "median",
                     "stt_rtf": "median",
                     "llm_ttft_sec": "median",
                     "llm_tps": "median",
                     "tts_rtf": "median",
                     "wer": "median",
                     "stt_peak_mem_mb": "median",
                     "tts_peak_mem_mb": "median",
                 })
                 .reset_index())
    summary_path = out_path.with_name("ablation_summary.md")
    md_lines = ["# Ablation Summary (median across test clips)", ""]
    md_lines.append(summary.to_markdown(index=False, floatfmt=".3f"))
    md_lines.append("")
    md_lines.append("## Best per metric")
    for metric, ascending in [
        ("e2e_latency_sec", True),
        ("wer", True),
        ("stt_peak_mem_mb", True),
        ("tts_peak_mem_mb", True),
        ("llm_tps", False),
    ]:
        if metric not in summary.columns:
            continue
        s = summary.dropna(subset=[metric]).sort_values(metric, ascending=ascending)
        if s.empty:
            continue
        top = s.iloc[0]
        md_lines.append(
            f"- **{metric}** ({'lower' if ascending else 'higher'} better): "
            f"{top['stt_engine']} / {top['llm_engine']} / {top['tts_engine']} "
            f"= {top[metric]:.3f}"
        )
    summary_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
