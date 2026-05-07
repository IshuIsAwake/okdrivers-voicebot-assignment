# Test Audio Set

`benchmark.py` reads `manifest.json` from this directory and feeds each clip
through every engine combination. You must populate the actual `.wav` files
before running the benchmark.

## File requirements

- Format: **mono, 16 kHz, 16-bit PCM WAV**
- VAD-trimmed: <200 ms of leading/trailing silence
- 10–20 utterances total, varied lengths (3–15 words), all Hinglish
- Recorded in a moderately quiet environment (some background noise is realistic)

## Adding a clip

1. Record (e.g., with `arecord -f S16_LE -r 16000 -c 1 my_clip.wav`).
2. Trim silence with `ffmpeg -i my_clip.wav -af silenceremove=start_periods=1:start_silence=0.2:start_threshold=-40dB clip_trimmed.wav`.
3. Add an entry to `manifest.json`:
   ```json
   {"filename": "04_my_clip.wav", "reference": "exact transcription in Hinglish"}
   ```
4. Drop the WAV into this directory (gitignored by default).

## Re-encoding existing audio

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 output.wav
```
