# S7 — CPU audio models and gateway audio libraries

M0 work package A4. Parts 1, 6, 7 and the extras ran on 2026-10-22, parts 2–5 on 2026-10-23
(after a pause). Throwaway spike: scripts in `spikes/cpu/` (a standalone uv project), raw numbers
in `spikes/results/s7_*.json`. Nothing here is production code.

## 0. Setup

**Machine.** Apple M3 Max (12 P + 4 E cores), 128 GB, macOS 26.6.2, on mains power. Other agents
were running at the same time (model downloads over a slow link, an implementation agent running
tests). Every timing run records the 1-minute load average before and after it (§9).

**Environment** (`spikes/cpu/pyproject.toml`, `uv.lock`). Python 3.12.12, onnxruntime 1.31.0
(CPUExecutionProvider, set explicitly), numpy 2.5.4, soxr 1.1.0, soundfile 0.14.0,
huggingface_hub 2.2.0. Dev group, test-only: transformers 5.20.2 (parity check only; torch is not
installed, so it takes its numpy path) and pyarrow 26.0.0 (test-data fetch only). mlx, mlx-lm,
mlx-audio and openai are absent on purpose: this environment mirrors the MLX-free gateway.
`iogpu.wired_limit_mb` = 0; `mx.device_info()` is not applicable here.

**Models** (`spikes/cpu/models/MANIFEST.json`, files git-ignored):

| File | Source and revision | License | sha256 |
|---|---|---|---|
| `silero_vad.onnx` | snakers4/silero-vad v6.2.3 = 5cd7945 (git blob verified) | MIT | `1a153a22…88e3` |
| `smart-turn-v3.2-cpu.onnx` (int8) | HF pipecat-ai/smart-turn-v3 @ f766f81d | BSD-2-Clause | `2bb02631…7e4f` |
| `smart-turn-v3.2-gpu.onnx` (fp32) | same | BSD-2-Clause | `ab8dc64b…39be` |

The desk-verified I/O was re-checked on the files: Silero `input` [B, N], `state` [2, B, 128],
`sr` int64 scalar → `output` [B, 1], `stateN`; Smart Turn `input_features` [B, 80, 800] →
`logits` [B, 1], already in [0, 1]. The int8 file was produced by `onnx.quantize`, the fp32 file
by PyTorch.

**Ported code.** `whisper_features.py` is pipecat v1.13.0's numpy Whisper log-mel, byte-identical
below a header that reproduces pipecat's BSD-2 license and keeps the Apache-2.0 notice for the
transformers-derived parts. `silero_vad.py` ports pipecat's `SileroOnnxModel` (BSD-2 header kept),
plus a spike-only batched-state helper.

**Rerun** (from `spikes/cpu/`, each writes its JSON):

| Part | Command | Output |
|---|---|---|
| models | `uv run python fetch_models.py` | `models/`, `MANIFEST.json` |
| test data | `uv run python fetch_testdata.py --verify-rowgroup` | `data/…/zho/`, `s7_testdata.json` |
| 1 | `uv run python s7_1_smart_turn_timing.py` | `s7_1_smart_turn_timing.json` |
| 1b (extra) | `uv run python s7_1b_smart_turn_batch.py` | `s7_1b_smart_turn_batch.json` |
| 2 | `uv run python s7_2_smart_turn_accuracy.py` | `s7_2_smart_turn_accuracy.json` |
| 3 | `uv run python s7_3_feature_parity.py` | `s7_3_feature_parity.json` |
| 4 | `uv run python s7_4_silero.py` | `s7_4_silero.json` |
| 5 | `uv run python s7_5_concurrency.py` (+ variants, §5) | `s7_5_concurrency*.json` |
| 5b (extra) | `uv run python s7_5b_gil_probe.py` | `s7_5b_gil_probe.json` |
| 6 | `uv run python s7_6_resample.py` | `s7_6_resample.json` |
| 7 | `uv run python s7_7_g711.py` | `s7_7_g711.json` |

## Recommendations (for approval)

**G3 — Smart Turn file: fp32 (`smart-turn-v3.2-gpu.onnx`).**

- **Speed.** Its decision time (features + model, p95) is 61 ms at 2 threads, 33–35 ms at 4. G4
  has not set the turn budget yet, but 61 ms after the 200 ms silence that triggers the
  evaluation is unlikely to break any plausible budget.
- **Accuracy.**
  - Clean Chinese: 90.5% vs 86.3% for int8.
  - Telephone copies: 89.9% vs 87.3%.
  - It is more robust: false "complete" verdicts rise by +3.1 points on telephone audio, vs +6.1.
  - It is numerically stable (§3).
- **Fallback.** int8 is the fallback only if CPU becomes the constraint: half the CPU per
  evaluation.

**G3 — Smart Turn executor sizing.**

- **Session options.** `intra_op_num_threads = 2`, `session.intra_op.allow_spinning = "0"`,
  inter-op 1, sequential, ORT_ENABLE_ALL, CPUExecutionProvider, one session per worker. Spinning
  buys ≤ 3% at 2 threads and keeps cores busy.
- **Default: `smart_turn.workers` (K) = 2** for up to 64 sessions.
  - Queue p50 0.04 ms; bursts of 8 give queue p99 ≈ 200 ms and total p99 260 ms.
  - Stable at 16 evaluations/s (total p99 301 ms).
  - Uses 4 cores at peak and ≈ 1.1 cores on average at ~10 evaluations/s.
- **K = 4**, or 4 threads per worker, halves the burst tail (total p99 141–149 ms) for 8 cores at
  peak. K = 1 is too small for 64 sessions under load. At the sketched `max_sessions: 8`, K = 1
  is enough.
- **No batching.** Do not batch evaluations into one call (2–13% gain, and every verdict waits
  for the whole batch).

**Silero: per-stream calls on one single-thread executor, as planned.**

- 64 streams cost ≈ 0.55 core including the ingest path; loop lag p99 ≤ 0.54 ms; verdicts in
  0.3 ms (p50).
- Batching saves ≈ 0.2 core but delays every VAD verdict by up to one 32 ms tick (p50 16 ms).
  Keep it as the M6 option.
- GIL: nothing to do at 64 streams. A 1 ms switch interval trims the tails slightly. If the loop
  ever gets busy, executor work slows by GIL handoffs (§5b), so keep CPU work off the loop.

**Silero at 8 kHz (M7).**

- Keep the canonical path: upsample once, run Silero at 16 kHz. Native 8 kHz is as good on
  separate utterances, but it drifts in a long session (onsets up to +0.67 s at p95 after line
  noise), and pipecat's reset heuristic did not cure it.
- Upsampling, however, costs latency: soxr HQ 8k→16k delays every VAD verdict by 64 ms on average
  (115 ms max). For 8 kHz input use LQ (32 / 61 ms) or a short polyphase FIR, and re-check on real
  telephone recordings.

**soxr.**

- Quality HQ for 24k→16k: 21 ms mean and 37 ms max lag. MQ is no faster; VHQ doubles the lag.
- Run the streams in **float32** and convert int16 ↔ float32 in our code. float32 streaming is
  bit-exact with one-shot and free of dither, so tests can compare exactly; int16 I/O is dithered
  (±2 LSB), and python-soxr cannot disable that.
- Count the lag in the S10 latency budget and in the AEC reference alignment.

**G.711.** The tables are ready: zero mismatches against `audioop` on every input and code. Port
`audio/g711.py` from `audioop.c`'s 14-/13-bit variant, not from Sun's later 16-bit `g711.c`.

**Proposed changes**, for approval:

- `config.yaml`:
  - `turn.smart_turn.model: models/smart-turn-v3.2-gpu.onnx`; `workers: 2` (already in the
    sketch);
  - new keys `intra_op_threads: 2` and `allow_spinning: false`.
- ARCHITECTURE §4.3: `audio/resample.py` uses float32 I/O. The 8 kHz input path gets a
  lower-latency resampler (M7).
- ARCHITECTURE §4.2: the Smart Turn executor uses one session per worker, and K is sized by burst
  size.

## 1. Smart Turn timing (`s7_1_smart_turn_timing.json`)

Fresh session per configuration (sequential, inter-op 1, ORT_ENABLE_ALL, CPUExecutionProvider),
20 warm-up runs, then 200 runs on distinct random 8 s inputs. Features = the numpy port; model =
`session.run`. Milliseconds.

| File | Threads | Spinning | Features p50 / p95 / max | Model p50 / p95 / max | Cores busy | Load before → after |
|---|---|---|---|---|---|---|
| int8 (cpu) | 1 | on | 1.92 / 2.09 / 2.25 | 51.8 / 53.1 / 57.6 | 1.00 | 2.91 → 2.77 |
| int8 (cpu) | 1 | off | 1.92 / 2.07 / 2.27 | 51.8 / 53.6 / 56.5 | 1.00 | 2.77 → 3.11 |
| int8 (cpu) | 2 | on | 1.90 / 2.09 / 2.31 | 27.4 / 28.5 / 31.8 | 2.00 | 3.11 → 3.26 |
| int8 (cpu) | 2 | off | 1.89 / 2.07 / 2.28 | 28.1 / 29.6 / 32.4 | 1.88 | 3.26 → 3.24 |
| int8 (cpu) | 4 | on | 1.89 / 2.01 / 2.21 | 15.0 / 15.5 / 17.0 | 4.00 | 3.24 → 3.46 |
| int8 (cpu) | 4 | off | 1.91 / 2.16 / 2.25 | 17.0 / 18.0 / 21.5 | 3.20 | 3.46 → 3.58 |
| fp32 (gpu) | 1 | on | 1.94 / 2.11 / 2.57 | 107.9 / 110.4 / 114.3 | 1.00 | 3.58 → 3.47 |
| fp32 (gpu) | 1 | off | 1.94 / 2.10 / 2.79 | 107.9 / 110.6 / 114.2 | 1.00 | 3.47 → 5.61 ⚑ |
| fp32 (gpu) | 2 | on | 1.93 / 2.06 / 2.26 | 56.6 / 58.4 / 63.3 | 2.00 | 3.63 → 3.68 |
| fp32 (gpu) | 2 | off | 1.92 / 2.08 / 2.30 | 57.1 / 59.2 / 63.2 | 1.95 | 3.68 → 3.53 |
| fp32 (gpu) | 4 | on | 1.93 / 2.05 / 2.93 | 30.0 / 30.5 / 32.7 | 4.00 | 3.53 → 3.65 |
| fp32 (gpu) | 4 | off | 1.96 / 2.15 / 2.28 | 31.8 / 33.2 / 35.1 | 3.60 | 3.65 → 4.23 ⚑ |

- Features + model, p95: fp32 112 / 61 / 33–35 ms at 1 / 2 / 4 threads; int8 55 / 31 / 18–20 ms.
- int8 is only ~2× faster than fp32 at every thread count, and threads scale almost linearly:
  CPU cost per evaluation is ~110–122 core-ms for fp32 and ~54–62 for int8.
- Spinning matters only from 4 threads: ~6–12% lower latency, paid for with fully busy cores
  (4.0 vs 3.2–3.6). With 1 thread there is no intra-op pool, so the flag does nothing.
- Cold start: session creation 25–60 ms; the first call costs the same as a warm one.
- ⚑ The 5.61 after-reading was a spike in outside load; the run's p95/max match its spinning-on twin.

**1b (extra): batching a burst into one call does not pay** (`s7_1b_smart_turn_batch.json`).
B = 4 in one call costs 27.0 vs 28.1 ms per item (int8, 2 threads), 15.2 vs 17.4 (int8, 4),
56.0 vs 57.1 (fp32, 2) and 28.5 vs 31.4 (fp32, 4): 2–13% less CPU, but every verdict waits for the
whole batch (fp32, 2 threads: 224 ms). Parallel workers are the better way to absorb bursts.

## 2. Smart Turn accuracy, Chinese subset (`s7_2_smart_turn_accuracy.json`)

**Data.** HF `pipecat-ai/smart-turn-data-v3.2-test` at dfc17924, licensed **CC BY 4.0**: the dataset
card and its LICENSE file; the mixed-in background noise is CC0 from freesound. Language is the
`language` column with ISO 639-3 codes; Chinese is `zho`, 929 of 31 527 rows. The label is
`endpoint_bool` (True = complete), with 473 complete and 456 incomplete. **All 929 Chinese clips are
synthetic** (`synthetic` = True, source `chirp3_1`, Google Chirp 3 TTS): 480 have mid-utterance
fillers, 338 end fillers. They are 16 kHz mono PCM16, 4.0–15.9 s long (median 8.9 s).

**Download** (`fetch_testdata.py`; `s7_testdata.json` lists every row with its sha256). Only the
Chinese rows were fetched: 173 MB of audio, ≈206 MB transferred in all. Fetching only Chinese rows
from the parquet was not possible: rows are ordered by id, so `zho` rows sit in every 100-row
group, and each group stores its audio as one ~15 MB dictionary page. Instead the script
range-reads only the small metadata columns of each group (340 requests, 2 MB). It then downloads
each Chinese row's FLAC from the dataset-viewer API, authenticated because anonymous calls hit
HTTP 429. For one row group, the downloaded assets were checked against the original parquet
audio: byte-identical. The asset URLs carry the pinned revision.

**Method.** As the reference `inference.py` and pipecat do: the last 8 s of 16 kHz audio,
zero-padded at the front, then the numpy features, then the model; threshold 0.5. Telephone copy:
the whole clip goes 16 kHz → 8 kHz (soxr HQ) → G.711 μ-law round trip (`audioop`) → 16 kHz
(soxr HQ). That copy keeps 37.6 dB SNR against a band-limited clean copy, and 22.8 dB against the
wideband original.

| File | Audio | Accuracy % | Precision | Recall | FNR % (FN / complete) | FPR % (FP / incomplete) | FN / all % |
|---|---|---|---|---|---|---|---|
| int8 (cpu) | clean | 86.33 | 0.897 | 0.827 | 17.34 | 9.87 | 8.83 |
| int8 (cpu) | telephone | 87.30 | 0.854 | 0.905 | 9.51 | 16.01 | 4.84 |
| | **Δ** | +0.97 | −0.043 | +0.078 | −7.83 | +6.14 | −3.99 |
| fp32 (gpu) | clean | **90.53** | 0.899 | 0.917 | 8.25 | 10.75 | 4.20 |
| fp32 (gpu) | telephone | **89.88** | 0.875 | 0.934 | 6.55 | 13.82 | 3.34 |
| | **Δ** | −0.65 | −0.023 | +0.017 | −1.70 | +3.07 | −0.86 |

- fp32 reproduces the model card's Chinese figures exactly: 90.53%, P 0.899, R 0.918, and FP/all
  5.27%. int8 lands 0.54 points above the card's 85.79%, consistent with its sensitivity to tiny
  feature differences (§3).
- **Telephone audio pushes both models towards "complete".** Mean p rises by +0.029 for fp32 and
  +0.075 for int8.
  - fp32: mean |Δp| 0.054 (p95 0.33, max 0.92); 44 of 929 verdicts flip, 33 of them incomplete →
    complete.
  - int8: mean |Δp| 0.092 (p95 0.50, max 0.87); 93 flips, 79 incomplete → complete.
  - The cost is false "complete" verdicts, where the bot would answer a caller who has not
    finished: FPR rises from 10.8% to 13.8% (fp32) and from 9.9% to 16.0% (int8). int8's accuracy
    rises only because its clean recall was low.
- The two files disagree on 73 of 929 clean clips (mean |Δp| 0.080).
- Caveat: the Chinese test set is TTS speech. Real callers' Chinese, with real line noise, may do
  worse; M2/M4 should re-check on recorded calls.

## 3. Feature parity (`s7_3_feature_parity.json`)

50 Chinese clips (seeded random), plus their telephone copies. pipecat's numpy port vs
`WhisperFeatureExtractor(chunk_length=8)` called exactly as the reference `inference.py` does it.
transformers 5.20.2 ran its **numpy** path because torch is not installed; the torch STFT path,
which the model was likely trained with, was not compared.

| Clips | Features max / mean abs Δ | fp32: max / mean abs Δp | int8: max / mean abs Δp | Flips at 0.5 |
|---|---|---|---|---|
| clean | 1.19e-7 / 1.7e-8 | 2.5e-6 / 1.5e-7 | **0.036** / 0.0011 | 0 / 0 |
| telephone | 1.19e-7 / 2.0e-8 | 7.7e-7 / 1.0e-7 | **0.096** / 0.0019 | 0 / 0 |

- The features agree to float32 rounding (one ulp of values in [−0.23, 1.94]), but no clip is
  bit-identical. The port takes 1.85 ms per clip vs 5.9 ms for transformers.
- fp32 is insensitive to that rounding. **int8 is not**: one-ulp feature noise moves its output by
  up to 0.1, probably because its activations are quantized dynamically. Any int8 parity test needs
  a loose tolerance, and int8 verdicts near 0.5 are not stable to implementation details.

## 4. Silero VAD v6.2.3 (`s7_4_silero.json`)

**Cost** (intra-op 1; 2000 calls after 200 warm-up; load 3.63 → 3.92):

| Call | p50 ms | p95 ms | Per stream-window |
|---|---|---|---|
| B = 1, 16 kHz (512 + 64), through the wrapper | 0.129 | 0.152 | 129 µs |
| B = 1, 8 kHz (256 + 32), through the wrapper | 0.098 | 0.102 | 98 µs |
| B = 4, 16 kHz, batched | 0.366 | 0.400 | 92 µs |
| B = 16 | 1.314 | 1.342 | 82 µs |
| B = 64 | 5.10 | 5.27 | 80 µs |
| B = 64, intra-op 2 / 4 | 4.49 / 4.92 | 4.75 / 5.18 | 70 / 77 µs |

At 64 streams × 31.25 windows/s: per-stream calls cost ≈ 26% of one core in ONNX alone; one B = 64
call per tick costs ≈ 16%. The cost does not depend on content (zeros, noise and speech give the
same).

**8 kHz native vs upsampled 16 kHz on telephone audio.** All 929 clips: telephone 8 kHz (after
μ-law) run natively (`tel8`), the same audio upsampled to 16 kHz with soxr HQ (`tel16`), and the
original 16 kHz clip as reference. Windows are aligned (32 ms each). Speech = hysteresis 0.5 /
0.35; "confirmed" = 5 consecutive speech windows (150 ms, the planned `min_speech_ms`).
Δ = path − reference, in ms.

| Test | Pair | Agreement p ≥ 0.5 / hysteresis | First-speech Δ: mean / p95 / max; share later | Confirmed-start Δ p95 | End-of-speech Δ mean / p95 |
|---|---|---|---|---|---|
| Per clip, fresh state (929) | clean vs tel16 | 98.50 / 98.45% | +20 / +96 / +640; 28% | +96 | −5 / +32 |
| | clean vs tel8 | **98.68 / 98.59%** | **−5 / +64** / +576; 14% | **+64** | +8 / +32 |
| | tel16 vs tel8 | 98.30 / 98.22% | −25 / 0 / +224 | 0 | +13 / +64 |
| One stream, 2 s noise gaps (300 clips) | clean vs tel16 | 98.39 / 98.28% | **−41 / +64** / +608; 17% | **+96** | −5 / +32 |
| | clean vs tel8 | 98.45 / 98.25% | **+96 / +672 / +1024; 40%** | **+608** | +8 / +32 |
| Same, with pipecat's resets | clean vs tel16 | 97.80 / 97.72% | +88 / +578 / +1312; 49% | +384 | −5 / +32 |
| | clean vs tel8 | 97.39 / 97.18% | −9 / +515 / +960; 34% | +482 | +6 / +32 |

- On short, separate utterances, native 8 kHz is as good as upsampling, or slightly better. Its
  speech-window count is +0.4% vs the original, against −1.4% for tel16. No onsets were missed on
  any path.
- **In a continuous stream with one state (a session), native 8 kHz drifts.** After 2 s of
  −50 dBFS line noise, 40% of utterances are detected later, with p95 +0.67 s and max 1 s. The
  upsampled path stays tight (p95 +64 ms). This is the effect pipecat's comment describes.
- pipecat's periodic reset policy (reset after ≥ 2 s once 5 windows score < 0.1 and the VAD is
  quiet) does not fix it here. The 8 kHz tail stays at p95 ≈ +0.5 s, and resets add a similar
  tail to the 16 kHz path and to the clean reference (gap false alarms 40 → 174 windows).
- False alarms in the noise gaps are rare on the telephone paths (4–6 of 18 900 windows).
- Upsampling has a wall-clock cost not visible here: soxr HQ 8k→16k lags 64 ms on average and
  115 ms at most when streamed (§6).

## 5. Concurrency: 64 streams in one asyncio loop (`s7_5_concurrency*.json`)

**Simulation.**

- **Streams.** 64 streams, each sending a 20 ms frame every 20 ms (random phase) as a JSON
  `input_audio_buffer.append` with base64 PCM16 at 24 kHz, cut from real speech.
- **On the loop, per frame:** `json.loads`, base64 decode, `soxr.ResampleStream` 24k→16k (HQ,
  float32), then 512-sample windows.
- **Silero per-stream:** every window goes at once to a single-thread executor.
- **Silero batched:** one call per 32 ms tick for all ready streams (mean B = 22, max 64).
- **Smart Turn:** `ThreadPoolExecutor(K)`, one session per worker (fp32, 2 threads, spinning off).
  A job is features + model on a snapshot of the stream's last 8 s.
- **Load.** Poisson end-of-turn evaluations at 8/s, plus a burst of 8 simultaneous evaluations
  every 5 s (≈ 9.6/s, i.e. 0.15 per stream per second). "Heavy" doubles the Poisson rate to 16/s.
- **Measurement.** 30 s after a 3 s warm-up (variants: 20 s). Loop lag = oversleep of a 5 ms
  sleeper. CPU = process CPU / wall, where 100% is one core. ST = Smart Turn. Times in ms.

The runs reported here, from `spikes/cpu/`:

```sh
uv run python s7_5_concurrency.py --max-load-wait 60
uv run python s7_5_concurrency.py --modes per_stream --workers 0 --max-load-wait 180 --out s7_5_concurrency_baseline.json
uv run python s7_5_concurrency.py --st-rate 16 --modes per_stream --workers 1,2,4 --no-baseline --duration 20 --max-load-wait 60 --out s7_5_concurrency_heavy.json
uv run python s7_5_concurrency.py --st-intra 4 --modes per_stream --workers 1,2 --no-baseline --duration 20 --max-load-wait 60 --out s7_5_concurrency_t4.json
uv run python s7_5_concurrency.py --switch-interval-ms 1 --modes per_stream,batched --workers 2 --no-baseline --duration 20 --max-load-wait 60 --out s7_5_concurrency_switch1ms.json
uv run python s7_5_concurrency.py --st-model v3.2-cpu --st-intra 2 --modes per_stream --workers 1,2 --no-baseline --duration 20 --max-load-wait 60 --out s7_5_concurrency_int8.json
```

| Silero | K | CPU % | Loop lag p50 / p99 | Silero verdict p50 / p99 | ST queue p50 / p99 / max | ST total p50 / p99 |
|---|---|---|---|---|---|---|
| none (ingest only) | – | 23–29 | 0.14–0.17 / 0.82 | – | – | – |
| per-stream | 0 | 53–58 | 0.08 / 0.42–0.54 | 0.30–0.34 / 2.6–5.5 | – | – |
| per-stream | 1 | 168 | 0.07 / 0.35 | 0.30 / 8.0 | 48 / 625 / 657 | 110 / 685 |
| per-stream | **2** | **161** | 0.07 / 0.33 | 0.29 / 1.2 | **0.04 / 199 / 237** | **61 / 260** |
| per-stream | 4 | 161 | 0.07 / 0.32 | 0.29 / 1.0 | 0.03 / 80 / 117 | 61 / 141 |
| batched | 0 | 34 | 0.10 / 0.54 | 15.8 / 36.0 | – | – |
| batched | 1 | 140 | 0.08 / 0.32 | 15.8 / 33.4 | 42 / 470 / 514 | 101 / 529 |
| batched | 2 | 141 | 0.09 / 0.33 | 15.8 / 33.4 | 0.04 / 200 / 239 | 61 / 260 |
| batched | 4 | 141 | 0.09 / 0.33 | 15.8 / 33.4 | 0.04 / 82 / 115 | 60 / 142 |

Variants, all with per-stream Silero:

| Variant (command flags) | K | CPU % | ST queue p99 / max | ST total p50 / p99 |
|---|---|---|---|---|
| heavy: 16/s + bursts (`--st-rate 16`) | 1 | 241 | 1696 / 1745, **unstable** | 902 / 1756 |
| | 2 | 255 | 239 / 287 | 62 / 301 |
| | 4 | 254 | 110 / 122 | 61 / 171 |
| fp32, 4 threads per worker (`--st-intra 4`) | 1 | 169 | 242 / 263 | 35 / 278 |
| | 2 | 171 ⚑ | 114 / 135 | 35 / 149 |
| int8, 2 threads (`--st-model v3.2-cpu`) | 1 | 103 | 223 / 233 | 31 / 253 |
| | 2 | 104 | 94 / 113 | 31 / 125 |
| GIL switch interval 1 ms (`--switch-interval-ms 1`) | 2 | 162 ⚑ | 183 / 230 | 61 / 244 |

- **The loop never fell behind.** Loop lag p99 was ≤ 0.54 ms in every scenario. Isolated maxima of
  8–123 ms appeared with and without inference: OS hiccups on the shared machine.
- **Smart Turn queueing comes from the bursts.** The last job of a burst of B waits about
  (⌈B/K⌉ − 1) × service time. Outside bursts the queue p50 is ≈ 0.04 ms.
- **Capacity.** K = 1 cannot carry 16 evaluations/s (16 × 61 ms > 1 s); K = 2 can.
- Features computed inside the workers took 2.0–2.1 ms p50 (p99 ≤ 5.3 ms). At this loop load the
  GIL penalty of §5b does not show up.
- **Per-stream vs batched Silero.** Per-stream costs ≈ 20–24 points of one core more, and
  returns verdicts ≈ 15 ms sooner at p50 and ≈ 30 ms sooner at p99. The per-stream executor's
  backlog peaks at 21–256 windows, drained within milliseconds.
- **Ingest-only baseline.** ≈ 25% of one core. The audio work is only 3.6% (11 µs per frame on a
  busy core, measured separately); the rest is asyncio scheduling for 3200 wake-ups/s. Its higher
  loop-lag p50 (0.15 vs 0.07 ms with Silero running) suggests that a lightly loaded loop thread
  is clocked down or placed on an E-core.
- 1 ms switch interval: Smart Turn total p99 260 → 244 ms and Silero p99 1.2 → 0.8 ms. A small
  gain.

**5b (extra): GIL probe** (`s7_5b_gil_probe.json`). The main thread repeats one piece of work while
a second thread spins in pure Python, the worst case for a busy loop. Per call, p50 ms:

| Work | Alone | With busy thread, 5 ms switch (default) | With busy thread, 1 ms switch |
|---|---|---|---|
| Smart Turn fp32 model call (1 thread) | 108.4 | 115.5 | 109.6 |
| numpy Whisper features | 1.83 | **158.5** | 33.8 |
| Silero window (wrapper + ORT) | 0.128 | **7.66** | 1.66 |

ORT releases the GIL: the spinning thread kept its full rate. The numpy features and the Silero
wrapper each re-acquire the GIL many times per call, so they inherit the switch interval whenever
another thread holds it.

## 6. Resampling (`s7_6_resample.json`)

20 s of noise plus a chirp, fed to `soxr.ResampleStream` in random 10–40 ms chunks (5 chunkings per
case), flushed with `resample_chunk(empty, last=True)`, and compared with `soxr.resample` on the
whole signal.

| Pair | Quality | float32: stream vs one-shot | int16: max abs diff | `delay()` max, samples / ms | Lag a caller sees, mean / max ms |
|---|---|---|---|---|---|
| 24k→16k | MQ | identical | 2 LSB | 603 / 37.7 | 21.3 / 37.6 |
| 24k→16k | **HQ** | identical | 2 LSB | 585 / 36.6 | 21.2 / 36.6 |
| 24k→16k | VHQ | identical | 2 LSB | 1232 / 77.0 | 42.5 / 76.3 |
| 8k→16k | MQ | identical | 2 LSB | 1888 / 118.0 | 63.9 / 117.5 |
| 8k→16k | **HQ** | identical | 2 LSB | 1852 / 115.8 | 63.8 / 115.0 |
| 8k→16k | VHQ | identical | 2 LSB | 1780 / 111.2 | 64.0 / 111.3 |
| 24k→8k | MQ | identical | 2 LSB | 603 / 75.4 | 42.7 / 75.3 |
| 24k→8k | **HQ** | identical | 2 LSB | 586 / 73.2 | 42.7 / 73.1 |
| 24k→8k | VHQ | identical | 2 LSB | 1232 / 154.0 | 85.2 / 152.5 |

- **float32 I/O is bit-exact**: the concatenated stream equals the one-shot output, same length, in
  every case and chunking.
- **int16 I/O is dithered.** libsoxr adds TPDF dither to int16 output, and python-soxr offers no
  way to turn it off: its `quality` maps to a fixed enum (QQ/LQ/MQ/HQ/VHQ), with no flags. Digital
  silence comes back as ±1 LSB noise (4245 of 16000 samples non-zero). Stream and one-shot then
  differ by up to 2 LSB in ~45% of samples. One-shot int16 is repeatable, so the dither is seeded.
- **`delay()` is the pending output**, and it equals the lag a chunked caller sees (expected output
  minus output received). The lag comes from soxr's block processing, not the filter. It
  saw-tooths between a floor (6 ms at 24k→16k HQ) and the maximum, whatever the chunk size
  (10, 20, 40 or 100 ms all give the same range). Only a lower quality shortens it: LQ halves it
  (24k→16k mean 11 / max 20 ms; 8k→16k 32 / 61 ms). QQ (cubic) adds none but barely filters.
- Cost is negligible: ≈7 µs per 20 ms chunk at 24k→16k HQ (p50), ≈18 µs to construct a stream.

## 7. G.711 lookup tables (`s7_7_g711.json`)

Tables built from the ITU-T G.711 segment algorithm as CPython's `audioop.c` writes it. μ-law
encode is `st_14linear2ulaw(x >> 2)` and A-law encode is `st_linear2alaw(x >> 3)`. Decode uses
Sun's `g711.c` `ulaw2linear` / `alaw2linear`; CPython's `_st_*2linear16` tables hold the same
values. Checked exhaustively against `audioop` (width 2) on Python 3.12.12:

| Check | Inputs | Mismatches |
|---|---|---|
| μ-law encode vs `lin2ulaw` | 65 536 | **0** |
| A-law encode vs `lin2alaw` | 65 536 | **0** |
| μ-law decode vs `ulaw2lin` | 256 | **0** |
| A-law decode vs `alaw2lin` | 256 | **0** |
| μ-law encode through the LUT function, reversed input order | 65 536 | **0** |
| *For information:* Sun's later 16-bit-input `linear2ulaw` vs `lin2ulaw` | 65 536 | 381 |
| *For information:* Sun's later 16-bit-input `linear2alaw` vs `lin2alaw` | 65 536 | 896 |

- The tables are correct against `audioop`. Building them takes 0.11 s in pure Python.
- Trap: Sun's newer `g711.c` takes 16-bit input and computes the magnitude before shifting.
  `audioop` shifts first, with an arithmetic (floor) shift. The two disagree by one code step
  for some negative inputs (all mismatches are in −31 751…−1). Port `audioop.c`'s variant.
- Round-trip error is at most 644 (μ-law) and 512 (A-law) in int16 units. Every code is a fixed
  point except μ-law 0x7F (−0), which re-encodes as 0xFF.
- Speed is not an issue: LUT encode 2.3 µs and decode 1.3 µs per 20 ms frame (`audioop`: 0.8 /
  0.3 µs). For 1 s, the LUT encodes faster than `audioop` (15 vs 42 µs).

## 8. Surprises

1. **int8 buys only 2× speed** on the M3 Max, for 4.2 points less Chinese accuracy (86.3 vs 90.5%;
   4.7 on the model card) and twice the sensitivity to telephone audio. fp32 at 2 threads already
   decides in 61 ms (p95).
2. **int8 output is unstable to one-ulp feature noise** (Δp up to 0.1 between two float32-identical
   feature implementations). fp32 moves by 1e-6.
3. **Telephone audio biases Smart Turn towards "complete"**: false "complete" verdicts on unfinished
   turns rise from 10.8% to 13.8% (fp32) and from 9.9% to 16.0% (int8).
4. **The Chinese test set is entirely TTS** (`chirp3_1`): no human Chinese speech in it.
5. **numpy feature extraction is GIL-bound in small steps** (§5b). In a worker thread it slows from
   1.8 ms to 158 ms when another thread runs Python flat out (5 ms switch interval), or to 34 ms
   at a 1 ms interval. A Silero call slows from 0.13 to 7.7 ms the same way. ORT's `run()` releases
   the GIL as documented (verified).
6. **soxr**: int16 output is dithered, and python-soxr cannot turn that off. The streaming lag
   comes from block processing, not the filter: 24k→16k HQ adds 21 ms on average (37 ms max),
   8k→16k adds 64 ms (115 ms max), whatever the chunk size.
7. **Silero native 8 kHz drifts in long streams**: onsets come late, p95 +0.67 s after 2 s of line
   noise. pipecat's reset heuristic did not fix it here.
8. **G.711 references differ**: Sun's later 16-bit `g711.c` disagrees with `audioop` on 381 (μ-law)
   and 896 (A-law) negative inputs. Port `audioop.c`'s 14/13-bit variant.
9. **Batching Smart Turn calls barely helps** on CPU (2–13% per item).
10. **Data access**: the test set's parquet layout made a language-only download impossible, and
    the dataset-viewer API rate-limits anonymous use (HTTP 429). Both are worked around in
    `fetch_testdata.py`.

## 9. Load-average notes

The 1-minute load average was recorded before and after every timing run. Runs wait up to 5 min for
load ≤ 4 before starting, then run anyway and are flagged. "After" includes this spike's own busy
threads (a 4-thread run adds ≈ 1.5–4).

- S7.1 timing: before-load 2.77–3.68 for all 12 configurations. One configuration waited 50 s
  first. After-load exceeded 4 twice: 5.61 (fp32, 1 thread, spinning off) from an outside spike,
  and 4.23 (fp32, 4 threads, spinning off, own threads). Neither run's p95/max differs from its
  twin.
- S7.1b batched calls: 3.41–3.91 before. After-load reached 4.11–4.89 on the B = 4 runs (own
  threads); the fp32 4-thread B = 4 p95 (167 ms vs p50 114) is probably disturbed.
- S7.4 Silero cost: 3.63 → 3.92. S7.5b GIL probe: 2.98 → 2.67. S7.6 resampler cost: 3.24 → 3.24.
- S7.5 concurrency (waits of up to 60 s, then flag):
  - Main matrix: before-load 3.63–3.99 for every scenario except the first, `ingest_only`, which
    started at 4.58 after waiting 60 s ⚑. A rerun at 3.82 confirmed it (≈ 25–29% CPU). After-loads
    were 3.53–4.54, mostly this spike's own threads.
  - Variants: before-load 3.22–3.88, except `--st-intra 4` K = 2 at 5.78 ⚑ and `--switch-interval-ms 1`
    per-stream at 4.53 ⚑. Their numbers match the pattern of the unflagged runs (the 4-thread K = 2
    tail is what the queueing formula predicts).
  - Outside load during S7.5 came largely from `syspolicyd` at ~100% of one core and a VM.
- Two bugs were found in the S7.5 simulator.
  - A scenario that waited in the load guard started behind schedule and replayed a catch-up
    burst. Fixed before every reported run; the affected runs were discarded.
  - Ingest-only mode queued windows that nothing consumed. This touched only the `ingest_only` row
    of `s7_5_concurrency.json` (23% CPU). The corrected rerun, `s7_5_concurrency_baseline.json`,
    gives 29%, so the queued windows did not explain the cost.
- S7.2 / S7.3 / S7.4 accuracy and agreement figures do not depend on load.
