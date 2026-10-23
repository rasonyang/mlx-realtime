# S3 / S4 / S9 — TTS, ASR and normalization probe (M0 B2, 2026-10-23)

**Code and data.** Scripts in `spikes/audio/` (mlx-audio 0.6.1, mlx 0.32.4, Python 3.12.12); raw data
in `spikes/results/s3_*`, `s4_*` and `s9_*.json`.

**Servers.** Stock `mlx_audio.server` (`uv run --locked --project spikes/audio python -m
mlx_audio.server`), with `HF_HUB_OFFLINE=1`: TTS (Qwen3-TTS + Kokoro) on 127.0.0.1:8103, ASR on :8102.
Cancellation and pinning used `s3_traced_server.py`, which only adds trace lines.

**Client.** A raw-socket client; TTFC = first audio byte, timed from before `connect()`. Requests are
serial, and the 1-min load average (2–7) is recorded per request.

**TTS request.** `{lang_code: chinese|english (mixed → chinese), response_format: pcm, stream: true,
streaming_interval}`. Sampling comes from the repo config (temperature 0.9); `max_tokens` is 1200.

**Models.** TTS 0.6B-8bit `049ef77f`, ASR 1.7B-8bit `a8379a2e`, Kokoro `a71e4d38`. For comparison:
TTS 0.6B-4bit `08c72cad`, TTS 1.7B-8bit `41d3337e`, ASR 0.6B-8bit `89e96d92`.

About 3,000 TTS and 2,700 ASR requests ran, with 0 HTTP errors.

## The Mac is in Low Power Mode (affects every M0 timing)

`pmset -g` shows `powermode 1`, and the PowerManagement plist has `LowPowerMode = 1` on both AC and
battery.
- Under continuous synthesis the GPU runs at full speed for ~20–40 s, then steps down to ~0.6× and
  stays there while the load lasts. A few seconds of idle restores full speed.
- The step was seen twice: at 01:13:57, and at 05:33 after 3 h idle.

| interval | TTFC full / capped | steady RTF full / capped |
|---|---|---|
| 0.16 s | 76 / 119 ms | 0.33 / 0.56 |
| 0.32 s | 118 / 189 ms | 0.30 / 0.50 |
| 0.8 s | 238 / 401 ms | 0.28 / 0.47 |

**Recommendation:** set Energy Mode to High Power (or Automatic) on the power adapter, for S2, S5, S10
and in production. That is the owner's call; I changed nothing. Comparisons below are like-for-like.

## S3 — TTS

**Streaming** (n = 3).
- With `stream: true`, the body is chunked `audio/pcm` (s16le, 24 kHz). Each chunk is
  `int(streaming_interval × 12.5)` frames of 80 ms.
- The default interval of 2.0 s gives a TTFC of 526–581 ms.
- Without `stream`, everything arrives at the end (1.8–5.3 s). That path is a batch session and ~25%
  slower.

**First-chunk latency, warm** (capped state). 1,080 requests, n = 40 per cell (vivian and ryan pooled).
Cells give TTFC p50 in ms for zh / en / mixed text; p95 is within 15 ms of p50, and there were 0 stalls.

| interval | short | medium | long | RTF p50 | chunk gap p95 |
|---|---|---|---|---|---|
| 0.16 | 117 / 117 / 117 | 119 / 119 / 119 | 131 / 132 / 132 | 0.57 | 96 ms |
| 0.32 | 186 / 184 / 187 | 187 / 188 / 187 | 200 / 200 / 202 | 0.51 | 165 ms |
| 0.8 | 394 / 392 / 394 | 395 / 396 / 396 | 407 / 410 / 409 | 0.47 | 380 ms |

- **Language, speaker and text length:** language and speaker don't matter, and long text adds ~13 ms.
- **Cold start:** the server is ready after 12.3 s. The first request's TTFC is 1.90 s (1.77 s of it is
  the model load); the second's is 113 ms.
- **Comparison models** (0.32 s, capped): 4-bit has TTFC 178 ms and RTF 0.48; 1.7B has 250 ms and 0.65.
- **~01:53 perturbation:** request wall times were recovered from the server's timestamped access log.
  The 44 requests in 01:52:00–01:54:45 had TTFC 0.95–1.04× their cell median (the normal spread); they
  were still excluded and re-run (`s3_sweep_perturbation.json`, `s3_sweep_topup.json`).

**HTTP overhead.** The in-process run used the server's own kwargs, on 21 cells in the same state.
Server minus in-process TTFC p50 was **+2.0 ms** median (range −6 to +5); the p95 difference was 3.5 ms;
RTF 0.51 vs 0.52. For ASR the gap was noise (−21 to +11 ms).

**Cancellation** (close after the first chunk; n = 20 per interval; mostly full speed):

| interval | old generation stops after close (p50 / max) | chunks after close | new TTFC vs control |
|---|---|---|---|
| 0.16 | 41 / 44 ms | 0 | 96 vs 65 ms |
| 0.32 | 75 / 81 ms | 0 | 173 vs 110 ms |
| 0.8 | 241 / 316 ms | 0 | 519 vs 226 ms |

The disconnect is seen within 0.3 ms. The chunk in progress finishes, nothing more is sent, and the
next request starts within 1 ms.

**Concurrency.** Streams are never batched. With 2 simultaneous requests, TTFC is 190 and 2,871 ms;
with 4 it is 191, 2,870, 5,698 and 8,437 ms.

**Quality guardrail** (100 runs: 5 zh sentences with vivian, 5 en with ryan, 10 runs each).
- **zh:** every run within 0.8× of its sentence median, and CER 0%.
- **en:** 3 runs fell below 0.6×, one of them 30 s of babble; WER 6.9%.
- **The comparison models show the same split:** 4-bit has 6/50 en outliers (up to 78 s of audio),
  1.7B has 1/50, and zh has 0/50 on all three.
- **It is ryan's voice.** In the sweep, 33 of 588 ryan outputs (5.6%) were over 1.5× the usual length,
  up to 20× (43 s for a 2.2 s sentence). For comparison: vivian 1/590, aiden 2/66, serena 0/66.
- **Edge silence:** every output starts with ~0.45 s of silence (p95 0.63 s) and ends with ~0.5 s.

**Speakers** (10 sentences × 2). zh CER is 0% for vivian, serena, uncle_fu and aiden, and 1.2% for
ryan. en WER is 0% for aiden and vivian, 0.4% for ryan and 0.9% for serena.

**Kokoro (D13).** Voices zf_xiaobei / af_heart; 20 sentences per set.

| set | Kokoro error | Qwen error | English words heard (Kokoro / Qwen) | latency, Kokoro / Qwen |
|---|---|---|---|---|
| zh | 0% | 0% | – | 220 ms total / 133 ms TTFC |
| en | 0% | 0% | – | 181 / 139 ms |
| mixed | 15.9% (18 of 20 sentences wrong) | 2.2% | 5/25 / 25/25 | 199 / 142 ms |

- Kokoro returns a whole sentence in one chunk (RTF 0.04). Its first request costs 2.2 s, and each new
  language costs more (0.75 s for zh).
- English inside Chinese comes out garbled, e.g. account → 宁达州. The cause: mlx-audio's legacy
  `ZHG2P` copies Latin text unchanged into the phoneme string.

**Warmup and pinning.**
- The model loads once, at the warmup.
- It never reloads after that: not in 49 further requests, and not after idle gaps of 5–120 s (first
  TTFC 115–142 ms). Swap stays at 0 and memory is flat (2.77 GB).
- TTFC drifting from 118 to 190 ms was the power cap, not a reload.
- There is no preload option and no way to wire memory (nothing calls `mx.set_wired_limit`).

## S4 — ASR

Latency: warm, p50 ms, with the language omitted (auto) / given; n = 20 per cell.

| clip | 1.7B zh | 1.7B en | 1.7B mixed | 0.6B zh | 0.6B en | 0.6B mixed |
|---|---|---|---|---|---|---|
| 2 s | 184 / 158 | 212 / 189 | 221 / 178 | 95 / 85 | 102 / 100 | 103 / 88 |
| 5 s | 323 / 275 | 350 / 304 | 311 / 287 | 153 / 138 | 153 / 163 | 157 / 143 |
| 10 s | 446 / 403 | 521 / 499 | 511 / 485 | 211 / 204 | 257 / 231 | 241 / 248 |

p95 is at most 140 ms above p50. The cold first request takes ~1.3 s.

Accuracy, in %: utterances without numbers / with numbers, language auto-detected.

| set | 1.7B wideband | 1.7B telephone | 0.6B wideband | 0.6B telephone |
|---|---|---|---|---|
| zh CER | 5.9 / 19.4 | 5.6 / 23.8 | 6.2 / 21.1 | 6.0 / 22.7 |
| en WER | 2.4 / 2.5 | 3.0 / 4.2 | 2.8 / 5.5 | 3.6 / 10.5 |
| mixed MER | 10.3 / – | 11.4 / – | 12.7 / – | 11.9 / – |

- The telephone copy costs at most 1.4 points; giving `language` changes the rate by ≤ 0.4 points.
- Utterances with numbers score worse because the references use digits while the model writes what it
  hears (一九四零年).

**Output shape.**
- **verbose_json:** `text`, `segments` (one per ≤ 30 s chunk, no word timestamps), `language` (a list),
  plus token counts. `json` gives only the text.
- **Formatting:** full-width punctuation; numbers written as heard. The language hint changes digit
  style: with "English", the same zh audio gives 1940年8月15日.
- **Pitfalls:**
  - Auto-detection labels a clip "English" whenever the model omits its tag: 7 of 42 zh clips, all
    transcribed correctly.
  - Any `language` value is accepted and echoed back.
  - A wrong hint can make it translate.
  - Clips longer than 30 s come back as segments joined with a space.

## S9 — normalization probe (G5)

- **Sentences:** 43 call-center sentences plus 6 English ones, each synthesized raw 3 times.
- **Scoring:** each was transcribed back and scored against the intended reading of every number, ID
  and amount.
- **Calibration:** the intended readings, written out, come back exactly. Plausible wrong readings (e.g.
  一百三十八 vs 幺三八) come back differently, so the ASR can tell a correct reading from a wrong one.
- **Audio:** WAVs are in `spikes/data/s9/`.

| category | runs correct / partial / wrong | failures |
|---|---|---|
| numbers (cardinal, decimal, %, ordinal) | 15 / 0 / 3 | 15600 always read 一千五百六十 |
| money (¥, 元, $) | 18 / 0 / 0 | – |
| dates, times | 17 / 1 / 0 | one spurious 零 |
| phone numbers | 8 / 3 / 10 | mobiles, 尾号5678 read as a cardinal, dropped or repeated digits |
| order and tracking IDs | 12 / 0 / 6 | TK-2024-0789, RT…A3, a dropped digit |
| units | 17 / 0 / 1 | one garbled sentence |
| mixed zh/en | 15 / 2 / 1 | the "@" of an e-mail never read |
| English extras | 2 of 6 sentences always right | IDs, 03/15/2024, $1,280.50 |

## Gate recommendations

**G1: keep `mlx_audio.server` for ASR and TTS.** It passes the rule:
- HTTP overhead is +2 ms.
- Warmup is reliable: one load, no reload or swap. "Pinned" here means it stays loaded once warmed;
  wired memory is not available.
- Cancellation stops within one chunk (41 / 75 ms at 0.16 / 0.32 s).
- No errors so far; S6 still has to confirm stability.

Caveat: streams run one at a time. With `max_sessions: 8`, overlapping replies would miss AICC's 3 s
first-audio deadline. If several sessions must speak at once, that, not latency, would justify a thin
worker built on mlx-audio's streaming `batch_generate`. S5/S6 should measure it.

**G5: normalize only:**
- phone numbers (including 4-digit 尾号 and extensions; digit by digit, 1 → 幺);
- alphanumeric IDs, including dashed ones;
- e-mail addresses;
- cardinals of 5 or more digits;
- in English segments: IDs, numeric dates and amounts ≥ 1,000.

Leave money, decimals, percent, ordinals, dates and times, units, URLs and brand names alone.

**D13 / G6: Kokoro does not pass as is.**
- Pure zh and en: intelligible and fast.
- Mixed text: English inside Chinese is lost (5 of 25 words heard).
- So: use it only for segments without Latin letters, or drop it. No substitute was tested.

**TTS defaults.**
- **`streaming_interval` 0.16 s.** TTFC is 119 ms capped / 76 ms at full speed, against 189 / 118 ms at
  0.32 s. Cancellation stops in 41 ms. RTF is 0.57 with no stalls. The cost is +12% GPU time. If S5
  shows RTF near 1 under LLM decode, use 0.32 s instead.
- **Speakers:** vivian for zh and mixed text (it reads English at 0% WER), aiden for en. Avoid ryan.
- **Two more gateway guards:**
  - Cap `max_tokens` in proportion to the text, because a runaway reaches the full 1,200-frame limit
    (96 s).
  - Trim edge silence from each segment.

## Surprises and incidents

- **Low Power Mode**, described above.
- **The slow-speech defect (mlx-audio#1002) also hits 0.6B**, and it follows the ryan voice.
- **Accidental deletion:** while removing my smoke results, a glob (`*_smoke.json`) also deleted A1's
  `spikes/results/a1_audio_smoke.json`. The directory is untracked, so no copy exists.
  `spikes/audio/smoke_audio.py` regenerates the file; it uses port 8103 and rewrites the `a1_*.pcm`
  files.
- **`HF_HUB_OFFLINE` on the sweep's client:** the client process lacked it (its server had it). All
  later scripts force it.

## Reproduce

```
export B2_SCRATCH=<dir>; R="uv run --locked --project spikes/audio python"
$R spikes/audio/s3_tts.py sweep --n 20 && $R spikes/audio/s3_sweep_check.py
$R spikes/audio/s3_tts.py sweep --topup spikes/results/s3_sweep_perturbation.json --tag _topup
$R spikes/audio/s3_inproc.py tts --n 20 --lengths short,medium
$R spikes/audio/s3_inproc.py tts --n 20 --intervals 0.32 --lengths long --tag _long; $R spikes/audio/s3_inproc.py asr
for s in cancel concurrency cold stream speakers guardrail kokoro pinning; do $R spikes/audio/s3_tts.py $s; done
$R spikes/audio/s3_tts.py sweep --speakers aiden,serena --intervals 0.32 --lengths short,medium --n 10 --tag _spk
$R spikes/audio/s4_asr.py clips; for s in latency accuracy shape; do $R spikes/audio/s4_asr.py $s; done
$R spikes/audio/s9_norm_probe.py synth --runs 3 && $R spikes/audio/s9_norm_probe.py score
# comparisons: s3_tts.py sweep|cold|guardrail --model <repo> --tag _4bit|_1.7b;
#              s4_asr.py latency|accuracy --model mlx-community/Qwen3-ASR-0.6B-8bit --tag _0.6b
$R spikes/audio/b2_summary.py sweep overhead cancel cold stream concurrency pinning speakers guardrail kokoro s4_latency s4_accuracy s4_shape
```
