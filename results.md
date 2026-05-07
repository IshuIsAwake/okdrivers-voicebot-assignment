# Results — okDriver Voice Assistant - https://github.com/IshuIsAwake/okdrivers-voicebot-assignment

What this prototype actually does, and how well, with the numbers from a real
run. No glossing over the broken bits.


## TL;DR

- The streaming pipeline works. Audio playback starts ~0.6s after you stop
  speaking on the fastest combo (`fw_base + groq_8b + piper`), well under the
  1.5s success bar.
- The LLM layer is the surprise — `gemma4_local` on a 6 GB RTX 3050 hits the
  same TTFT (~0.2s) as Groq's cloud API, with no network. That's not nothing.
- Whisper-base on 3-word Hinglish utterances is bad. WER is high not because of
  pronunciation, but because Whisper auto-detects Urdu/Devanagari script and the
  reference is romanized. With longer clips the gap closes; with `fw_small` it's
  still rough but coherent enough.
- The whole thing is modular enough that swapping any of the 7 engines is a
  config change, not a refactor. That was the main architectural goal.
- One failure to flag up front: a real "Accident hogya" got transcribed as
  "Exit & done!" and the assistant said "bye bye!" — exactly wrong for a
  road-safety product. Whisper-base on short Hinglish is the weak link, and
  for a real okDriver deployment a safety-keyword fallback is non-optional.

## Ablation results (12 combos × 3 clips, median)

Generated with `python benchmark.py --combos all`. Full CSV at
[results/ablation.csv](results/ablation.csv).

| STT      | LLM           | TTS       | E2E (s) | STT RTF | LLM TTFT (s) | LLM TPS | TTS RTF | WER   | STT mem (MB) | TTS mem (MB) |
|----------|---------------|-----------|--------:|--------:|-------------:|--------:|--------:|------:|-------------:|-------------:|
| fw_base  | groq_8b       | **piper** |   0.583 |   0.033 |        0.292 |    69.9 |   0.026 | 1.000 |        325.5 |        459.0 |
| fw_base  | gemma4_local  | piper     |   0.989 |   0.036 |        0.225 |    19.9 |   0.029 | 1.000 |        379.7 |        430.7 |
| fw_base  | gemma4_cloud  | piper     |   1.283 |   0.034 |        0.758 |    20.9 |   0.023 | 1.000 |        431.2 |        525.0 |
| fw_small | groq_8b       | piper     |   0.588 |   0.092 |        0.149 |   131.9 |   0.024 | 1.571 |        697.2 |        800.4 |
| fw_small | gemma4_local  | piper     |   1.129 |   0.092 |        0.201 |    20.4 |   0.035 | 1.571 |        725.0 |        755.7 |
| fw_small | gemma4_cloud  | piper     |   1.577 |   0.092 |        0.756 |    16.1 |   0.024 | 1.571 |        715.2 |        816.6 |
| fw_base  | groq_8b       | edge_tts  |   1.986 |   0.023 |        0.178 |   110.7 |   0.341 | 1.000 |        364.2 |        364.8 |
| fw_base  | gemma4_local  | edge_tts  |   2.281 |   0.033 |        0.199 |    23.1 |   0.374 | 1.000 |        385.8 |        386.0 |
| fw_base  | gemma4_cloud  | edge_tts  |   2.459 |   0.023 |        0.718 |    25.7 |   0.387 | 1.000 |        426.4 |        426.5 |
| fw_small | groq_8b       | edge_tts  |   2.484 |   0.086 |        0.346 |    88.4 |   0.284 | 1.571 |        803.5 |        805.4 |
| fw_small | gemma4_local  | edge_tts  |   2.350 |   0.086 |        0.194 |    22.4 |   0.353 | 1.571 |        812.9 |        812.9 |
| fw_small | gemma4_cloud  | edge_tts  |   2.014 |   0.086 |        0.728 |    19.4 |   0.460 | 1.571 |        841.4 |        841.7 |

### Best combo per metric

- **E2E latency (lower better):** `fw_base / groq_8b / piper` at **0.583s** — beats the 1.5s success target by ~3×.
- **WER (lower better):** `fw_base / gemma4_cloud / edge_tts` at 1.000 — but see "Whisper script issue" below; raw WER is misleading on these clips.
- **STT peak memory:** `fw_base / groq_8b / piper` at 325.5 MB.
- **TTS peak memory:** `fw_base / groq_8b / edge_tts` at 364.8 MB.
- **LLM TPS (higher better):** `fw_small / groq_8b / piper` at **131.9 tps** — Groq is ~5× faster than the Ollama engines on token throughput.

### What the numbers actually mean

**Piper crushes edge-tts on E2E** (~600ms vs ~2s) even though both have similar
RTF. Reason: edge-tts goes over HTTPS per sentence — that's a ~300ms round-trip
on top of synthesis, paid for every sentence. Piper is local ONNX, no network.
For a dashcam, this is the right choice anyway; you can't depend on cellular.

**`gemma4_local` matches Groq on TTFT.** Both ~0.2s. Groq wins on throughput
(110 tps vs 20 tps) but for a 1–3 sentence response the user doesn't notice —
the bottleneck is TTS playback length, not LLM tokens-per-second. The local
8B-Q4 model on a mid-range GPU is genuinely competitive with a hosted endpoint
*if* you only need short answers.

**`gemma4_cloud` is the slowest LLM.** TTFT ~0.7s. The 31B-cloud variant has
larger context and more capability but pays a network round-trip *and* is
serving from a busier free tier. Worth keeping for harder queries; not worth
making the default.

**WER ≥ 1.0 is structurally weird.** WER can exceed 1.0 when the hypothesis is
longer than the reference (every reference word is a substitution + extra
insertions). `fw_small` got 2.250 on the route clip because it added speculative
words. This is faithfully reported, not a bug.

## Live CLI conversation — `gemma4_cloud + edge_tts`

This is what the demo actually sounds like with the cloud-hosted LLM. STT
mistakes are real; the LLM compensates because Hinglish is robust to noisy
transcripts — until it isn't.

| You said (intended) | STT transcribed | LLM responded | E2E |
|---|---|---|--:|
| "Delhi mein mausam kya sa hai" | "Delhi me mason kya sa hai" | "Delhi mein abhi thodi thand hai aur mausam kaafi pleasant hai. Bahar nikal rahe ho toh ek light jacket saath rakh lena." | 3.20s |
| (asking about car brands) | "i.e. konse brine ki jagit penu" | "Bhai, aap shayad brand ke baare mein pooch rahe ho. Aapki car ke liye best brand kaunsa rahega, wo mujhe batao ya phir model share karo." | 2.76s |
| "What do you think of Lamborghini?" | "What do you think of Lamborghini?" | "Bhai, Lamborghini toh ekdum beast hai! Iska look aur speed dono mind blowing hote hain. Dream car hai boss!" | 2.59s |
| "I like Porsche 911 GT3 RS more." | "I like Porsche 911 GT3 RS more." | "Bhai, choice toh ekdum top class hai! Porsche 911 GT3 RS ek beast hai, uska track performance next level hai. Sahi taste hai aapka!" | 2.30s |
| "Tehri ke liye kaunsi best gaadi hai" | "Tehri kohansi favint gadi hai." | "Bhai, Tehri ke pahadon ke liye 4x4 SUVs best hoti hain. Mahindra Thar ya Toyota Fortuner jaisi gaadiyan wahan ke rasteon par ekdum mast chalti hain." | 3.49s |
| "How are you?" | "How are you?" | "Main ekdum mast hoon bhai! Aap batao, drive kaisi chal rahi hai?" | 1.89s |
| **"Accident hogya"** | **"Exit & done !"** | **"Theek hai bhai, done! Safe driving and apna khayal rakhna. Bye bye!"** | 2.30s |

Three observations from this run:

1. **English-only inputs work fine.** Whisper-base nails clean English; the LLM
   responds in Hinglish per the system prompt. The Lamborghini and Porsche
   turns are clean end-to-end.
2. **Heavy Hindi short utterances are where Whisper-base falls apart.** "Tehri
   kohansi favint gadi hai" should have been "Tehri ke liye kaunsi best gaadi
   hai." But the LLM still recovered — it heard the city name and inferred the
   intent. That's the streaming pipeline doing its job: the LLM sees garbage
   text, makes a reasonable guess, and the response is coherent in 3.49s.
3. **The accident turn is the important failure.** "Accident hogya" got
   transcribed as "Exit & done !" and the assistant cheerfully said *"Theek hai
   bhai, done! Bye bye!"* — exactly the wrong response for a road-safety
   product. This isn't a quirk; it's a category of failure okDriver has to
   solve before this prototype becomes a product. Possible mitigations: a
   bigger/multilingual STT, an intent classifier on a hot list of safety
   keywords ("accident", "police", "ambulance", "help"), or a tiered fallback
   where low STT confidence triggers a clarifying re-prompt instead of a
   confident reply. None are implemented here.
4. **System prompt is doing real work.** The Hinglish tone, the 1–3 sentence
   limit, the "bhai" register — all from `src/prompts.py`. Without it the LLM
   tends to over-explain in English.

E2E latencies are 1.9–3.5s on this combo. The success bar of <1.5s is met by
`fw_base + groq_8b + piper` (0.6s) — see ablation table — not by this slower
combo, which is fine because it was picked for the higher-quality cloud LLM.

## Live CLI conversation — `gemma4_local + edge_tts` (recorded video)

Same pipeline, but the LLM is running entirely on a 6 GB RTX 3050 — no network
for the language model. This is the "true edge" demonstration. There's a
recorded video of this run.

| You said (intended) | STT transcribed | LLM responded | E2E |
|---|---|---|--:|
| "Kya ho raha hai" | "What is going on?" | "Sab theek hai bhai, tension nahi lene ka. Aapko koi specific info chahiye ya kuch poochhna hai?" | 2.20s |
| "Mereko drifting karni hai" | "میرے کو Drifting kanyi hai" *(mixed Urdu + Latin)* | "Bhai, drifting ke liye aapko kisi safe, empty track ya closed area mein jaana chahiye. Pehle basic control aur car handling seekhna zaroori hai!" | 3.28s |
| "Lamborghini aur Porsche mein se kaunsi gaadi achhi hai" | "Lamagini & Purusha ke bhi se koon se gaadi akchi hai" | "Bhai, dono hi achhe hain, but aapka budget aur use ka purpose batao toh main better suggest kar paunga! Basically, Lamagini thoda premium feel deta hai aur Purusha might be more mileage-friendly." | 3.65s |

Why this run matters more than the cloud one:

1. **TTFT is ~0.2s on a 6 GB consumer GPU.** That's effectively the same as
   Groq's hosted API. The intuition that local LLMs are "slow" is wrong for
   8B-Q4 models on modern hardware — the bottleneck is throughput (~22 tps),
   not first token. For 1–3 sentence responses the user can't tell the
   difference.
2. **The drifting transcript is half Urdu, half Latin.** "میرے کو Drifting
   kanyi hai" — Whisper-base genuinely emitted both scripts in one transcript.
   The LLM didn't care; it parsed "Drifting", inferred the intent, and gave
   correct safety advice. This is the streaming pipeline's robustness story
   in a single example.
3. **"Lamagini" and "Purusha" are the funniest part of this whole project.**
   Whisper mangled "Lamborghini" → "Lamagini" and "Porsche" → "Purusha", and
   the LLM **confidently kept using those fake brand names** while
   constructing plausible advice ("Lamagini thoda premium feel deta hai aur
   Purusha might be more mileage-friendly"). It's a perfect illustration of
   LLM hallucination when grounded on bad input — and a perfect illustration
   of why an okDriver-grade system needs better STT than Whisper-base on
   short Hinglish.
4. **No network for the LLM.** Only edge-tts hits the network on this run.
   Switch the TTS to Piper and you have a completely offline-capable demo,
   modulo the wake-word weights cached at install time.

## What's broken or limited (honest)

- **Whisper-base on short Hinglish utterances:** auto-detects Urdu/Devanagari
  script and the response can drift into pure English translation. Forcing
  `language="hi"` or `"en"` did not help (we tried). Real fix is `fw_small` or
  Whisper-large, or longer/cleaner audio.
- **No safety-keyword fallback.** A real road-safety product cannot respond
  "bye bye!" to "Accident hogya" just because STT mistranscribed it. The fix
  is a pre-LLM intent layer: a small classifier or keyword hot-list checked
  against the *audio embedding* (not just the transcript), with a clarifying
  re-prompt on low confidence. Out of scope for this prototype but the most
  important production gap.
- **LLMs hallucinate confidently on mangled input.** "Lamborghini" → "Lamagini"
  and the LLM kept using the fake brand. There's no transcript-confidence
  signal flowing into the LLM prompt; it should be. A simple version: if
  Whisper's avg log-prob is below a threshold, prefix the user prompt with
  "(low STT confidence — ask for clarification before answering)".
- **WER as a metric for code-switched speech:** unreliable. Reference is
  romanized Hinglish; STT may emit Devanagari (correct phonetically, WER 1.0).
  A normalized phonetic-edit-distance metric would be fairer. Out of scope.
- **`gemma4:e4b` is a thinking model.** Without `think=False` in the Ollama
  client call, it streams reasoning tokens into `message.thinking` and content
  stays empty until the budget runs out. We pass `think=False`. Some future
  Ollama client version may rename or remove this parameter — the code falls
  back gracefully.
- **Wake word is the off-the-shelf `hey_jarvis`.** A real okDriver deployment
  would train a custom wake word; that's out of the prototype scope.
- **edge-tts requires network.** Piper is the offline-capable option.
- **Single turn.** No conversation memory by spec.

## Reproducing these numbers

```bash
# Ablation across all 12 combos (after populating test_audio/)
python benchmark.py --combos all
# → results/ablation.csv + results/ablation_summary.md

# Live conversation (the fun one)
python cli.py --stt fw_base --llm groq_8b --tts edge_tts        # fastest combo
python cli.py --stt fw_base --llm gemma4_cloud --tts edge_tts   # the run above
python cli.py --stt fw_base --llm gemma4_local --tts piper      # all-local
```

Each CLI turn appends a JSON line to `results/cli_log.jsonl` so you can
post-hoc compare live numbers against the benchmark.
