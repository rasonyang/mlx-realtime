# S11 part 2 — echo cancellation on simulated telephone echo (G7)

B4, 2026-10-23. Scripts: `spikes/aec/s11_*.py`. Raw results: `spikes/results/s11_{core,timeline,slices,dt,cpu}.json`
and `s11_cer.json`. Cached calls and ASR clips: `spikes/data/s11/` (git-ignored, 1.9 GB). G.168 coefficients are
read from the git-ignored `spikes/data/echo/g168_paths.json`; nothing ITU-derived is written anywhere tracked.
3020 engine runs, each call in a child process: 0 crashes, 0 errors.

## What ran

A3's recipe (§5), seeded:
- **Far end:** manifest zh/en speech at 24 kHz, sent with S3's TTS cadence.
- **AICC model:** AICC's `Run` playout state machine (20 ms ticks, 3-frame prebuffer, 2-tick grace), 33-tap ↓3 and framer, then G.711.
- **Echo:** G.168 m1–m8 (K_css), bulk delay D; a path change m_i → m_(i+4) at 45 s.
- **Near end:** near-end speech and 300–3400 Hz noise.
- **Uplink:** G.711, AICC's per-frame linear ↑3, soxr to 16 kHz.
- **VAD:** the gateway's own Silero gate (0.5/0.35, 2 s state resets).
- **Calls (60 s):** far end only, then alternating talk, then double talk, then the path change.
- **Grid:**
  - core: 96 calls (m1–m8 × ERL 6/10/15/20 × D 20/120/300 ms);
  - timeline: 38 calls (m4/m5, ERL 10, ε ±50/±200 ms, and four mid-call ε steps);
  - double talk: 32 calls × 10 utterances (40 zh + 40 en, ERL 6/15, near end 0/−15 dB);
  - slices: 10 calls (noise −40 dBm0, A-law, open path, RTF 1.1).

**Deviations, all forced by the data:**
1. **Manifest speech stands in for TTS:** no S3 WAVs existed.
2. **Tick-phase ambiguity.** The gateway cannot know AICC's tick phase, so the "blind" model (A3 as written) puts **24 % of
   playback starts one 20 ms tick off**. Added variants:
   - **locked**: the phase is known;
   - **learned**: blind for two responses, then locked; this emulates learning the phase from the delay estimator.
3. **GCC-PHAT keep range.** Re-shifting on every confirmed estimate moved the reference by ±1 ms every second and kept
   linear filters from converging. A shift now happens only when the residual leaves a keep range:
   - [0, 44] ms for Speex and the baseline;
   - [5, 400] ms for AEC3, as a guard.
4. **Speex "preprocessor on" was a no-op.** With denoise off, speexdsp applies no spectral gain, so its residual-echo
   suppression never acts. Added `ppd` (denoise on).
5. **ERLE on speech needs a measurability rule.** Quiet stretches put the echo at the noise floor (c/e ≈ 0 dB whatever the
   engine does). Convergence and the post-double-talk check use windows whose echo stands ≥ 25 dB above the noise
   ("meas"); all-window values are kept in the JSON.
6. **ΔCER ran last,** after B2 had released the GPU.

## Results

**Core grid, 96 calls per row.** C1 counts calls with Silero ≤ 1 % on echo-only audio after the first 1 s of far end
**and** no false `speech_started`, at `min_speech_ms` 150 / 400. Engines:
- `pw` = AEC3 via pywebrtc-audio 0.2.0; `lk` = AEC3 via livekit 1.1.21;
- `sx…ppd` = SpeexDSP 1024 taps + preprocessor + 3.8 kHz pre-filter; `bl` = numpy PBFDAF baseline.

| Engine : alignment | ERLE_ss meas p50 | ≥ 20 dB within 1 s (start / after change) | false starts after 1 s, 150 / 400 ms (96 calls) | C1 calls 150 / 400 | CPU ms per audio s | 8 threads vs 1 |
|---|---|---|---|---|---|---|
| pw : blind | 31.1 dB | 72 % / 49 % | 51 / 13 | 58 / 75 | 7.8 | 6.6× |
| pw : learned | 31.0 | 72 % / 51 % | 36 / 12 | 65 / 82 | 7.8 | 6.6× |
| **pw : locked** | 31.7 | 69 % / 50 % | 11 / 5 | **84 / 89** | 7.8 | 6.6× |
| pw : oracle | 31.8 | 77 % / 52 % | 5 / 1 | 91 / 95 | 7.8 | 6.6× |
| lk : blind | 31.7 | 68 % / 38 % | 140 / 73 | 29 / 40 | 9.0 | 1.8× |
| lk : locked | 32.2 | 72 % / 41 % | 34 / 16 | 70 / 79 | 9.0 | 1.8× |
| sx1024ppd : GCC + locked | 44.8 | 0 % / 17 % | 131 / 117 | 0 / 0 | 5.6 | 1.2× |
| sx1024 (linear) : oracle | 26.4 | 3 % / 0 % | 198 / 184 | 0 / 0 | 4.4 | 1.1× |
| bl μ0.5 : oracle | 13.2 | 0 % / 0 % | 596 / 397 | 0 / 0 | 9.2 | 0.9× |

Without a canceller, Silero flags 75 % of echo-only windows, with 1005 false starts in 96 calls. Every AEC3 row has
ERL + ERLE_ss ≥ 27 dB in 100 % of calls.
The livekit pre-filter and "no delay hint" variants did not differ from plain livekit.

**Why blind fails.** A 20 ms jump makes AEC3 re-acquire the delay and reset its filters; echo leaks 1–3 s after the
playback onset. Livekit, blind: 121 false starts in 235 responses that start with a jump, 40 in 491 without.

**Timeline, pw** (12 calls per ε; |ε| = 200 passes C1 after 2 s):

| Alignment | ε ±50 (pass after 1 s) | ε ±200 (pass after 2 s) | ε steps mid-call (Silero after 2 s, false starts) |
|---|---|---|---|
| engine only (blind / locked) | 14 / 16 of 18 | 6 / 6 of 12 | +200 step: 42–44 %, 14–16 |
| GCC-PHAT guard (blind / locked) | 12 / 14 of 18 | 10 / 10 of 12 | worst +200: 8 %, 9 |

Engine-only AEC3 fails exactly where the residual delay leaves its range. With ε = +200 and D ≤ 120 ms the residual is
negative; with ε = −200 and D = 300 ms it is ≈ 565 ms. Silero then fires on 54–80 % of echo-only windows and ERLE is
≈ 0 dB. The guard fixes both.

The guard's 400 ms bound fired needlessly at a residual of 415 ms, which AEC3 handled alone; the cost was 1–2 false
starts in two calls. AEC3 handled 415 ms and failed at 565 ms, so the bound belongs between them.

**Double talk** (320 utterances). The spectral SDR below compares the canceller output with the clean near end during
double talk (no canceller: 4.2 dB). The post-DT column counts bursts after which ERLE stays within 10 dB (G.168 3B).

| Engine | spectral SDR in DT | post-DT within 10 dB | Silero after 1 s, p50 |
|---|---|---|---|
| pw : locked | 6.1 dB | 238 / 244 | 0.0 % |
| lk : locked | 6.8 dB | 239 / 244 | 0.1 % |
| sx1024 linear : GCC + locked | 20.0 dB | 230 / 244 | 16.5 % |
| sx1024ppd : GCC + locked | 18.0 dB | 216 / 244 | 7.0 % |

AEC3's suppressor removes the echo but also much of the near end during double talk. The linear filters keep the
near end but leave echo that Silero hears.

**ΔCER** (`s11_cer.json`, run after B2 finished):
- Qwen3-ASR-1.7B-8bit @a8379a2, own server on :8112, stopped afterwards; 2880 requests, 0 errors.
- Each near-end utterance is cut ±300 ms from double talk; `language` is given.
- Clean chain (near end + noise, same telephone chain, no echo): zh CER 10.0 / 10.5 %, en WER 3.3 / 5.1 % (near end 0 / −15 dB).
- Points vs clean, paired, 95 % bootstrap CI; 80 utterances per cell, ERL 6 and 15 pooled:

| Variant | zh, near 0 dB | zh, near −15 dB | en, near 0 dB | en, near −15 dB |
|---|---|---|---|---|
| no canceller | +13.0 [8.3, 18.6] | +68.1 | +5.7 | +61.4 |
| pw : locked | +2.0 [0.8, 3.4] | **+10.9** [8.2, 13.9] | +4.9 [2.9, 7.2] | **+28.4** [22.7, 34.6] |
| pw : blind | +3.3 | +15.6 | +9.2 | +33.9 |
| lk : locked | +1.2 | +8.4 | +2.6 | +20.4 |
| sx1024 linear : GCC + locked | +0.9 | +0.6 | +0.7 | −0.8 |
| **sx1024ppd : GCC + locked** | **+0.1** [−0.8, 0.9] | **+0.2** | **+0.6** | **−0.6** |
| bl μ0.5 : GCC + locked | +0.7 | +6.4 | +11.3 | +50.5 |

AEC3 is worst at ERL 6 with the near end 15 dB down: pw:locked +16.2 zh and +40.7 en. Its suppressor removes quiet near-end
speech along with the echo, so ASR loses words. Both Speex variants stay within A3's limits (≤ 2 points at equal level,
≤ 5 at −15 dB) in every cell.

**Barge-in onset guard** (pw, `min_speech` 400 ms): false starts left in all 96 calls, counted from the modelled playback
start.

| Guard | 0 ms | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---|---|---|---|---|---|
| learned | 15 | 15 | 12 | 9 | 6 | 3 |
| locked | 9 | 9 | 8 | 5 | 3 | 1 |

With the locked model, the false starts left at 150 ms fall:
- 12 within 3 s of a path change, which no onset guard covers;
- 5 in the call's first 2 s of far-end speech;
- 3 shortly after other playback onsets.

Raising `min_speech` from 150 to 400 ms cuts false starts 2–4×.

**Slices.**
- **A-law** and **open path:** clean.
- **RTF 1.1 (underruns):** locked 0–1 false starts; blind 2–3.
- **Noise −40 dBm0:** one call in two left pw:locked with 8 % Silero hits and 6 false starts.

## G7 recommendation

No candidate passes all four criteria, so the trade-off goes to the owner.
- **AEC3 passes the VAD side** (C1, mostly) and fails ASR in double talk (C2).
- **SpeexDSP with its preprocessor** passes ASR (C2) but leaves residual that Silero hears (C1 fails).

**Suggested resolution for M4 (not tested as a combined system):** run both on the same capture.
- AEC3's output feeds Silero and Smart Turn; the Speex preprocessor's output (GCC-PHAT aligned) feeds the InputBuffer/ASR.
- Each half's numbers above come from the same calls.
- Combined cost: about 7.8 + 5.6 + 1.4 ms CPU per audio second.
- Before GCC-PHAT locks (about the first 2.5 s of the call) the ASR path has no cancellation.

If one engine must be chosen, the owner decides between false barge-ins (Speex) and lost words when the caller talks over
the bot (AEC3). The AEC3-only configuration:

- **Engine:** WebRTC AEC3 via **pywebrtc-audio 0.2.0**:
  - `EchoCanceller(sample_rate=16000, num_channels=1, stream_delay_ms=0)`, one per session, AEC only;
  - no 3.8 kHz pre-filter (no gain for AEC3);
  - same ERLE as livekit with a third of the leakage, and it releases the GIL; livekit's FFI does not.
  - Fallback: livekit 1.1.21.
  - Reject SpeexDSP and the baseline: their residual triggers Silero, they converge in 3–5 s, and they scale poorly.
- **Alignment:**
  - AICC playout model + 60 ms lead;
  - **learn AICC's tick phase:** fit the 20 ms grid from GCC-PHAT echo-onset estimates of the first playout runs, then
    place each run on it;
  - a **GCC-PHAT guard:** 2 s windows, 0.5 s hop, lags −300…+1000 ms, two windows within 2 ms; re-align to 60 ms only
    outside the keep range; 1.4 ms CPU per audio second.

  A gateway-side delay estimator **is needed**: as the guard, and as the source for the tick phase.
- **Defaults:**
  ```yaml
  aec:      {enabled: true, engine: webrtc_aec3, client_prebuffer_ms: 60, reference_lead_ms: 60,
             learn_tick_phase: true, max_residual_delay_ms: 450}
  barge_in: {echo_guard_ms: 2000, min_speech_ms_while_playing: 400, echo_turn_suppression: true}
  ```
  - `reference_lead_ms` and `learn_tick_phase` are new keys.
  - `max_residual_delay_ms` is 450 rather than the tested 400: it lies between the 415 ms that AEC3 handled and the
    565 ms where it failed.
  - `echo_guard_ms`: a guard of ≤ 800 ms adds nothing over AICC's own guard. 2000 ms covers AEC3's start-up
    (p90 1.7 s); the price is that early interruptions wait until the guard ends.

**Criteria, for pw with the tick phase locked:**

| Criterion | Verdict |
|---|---|
| **C1** echo-only | Fails A3's every-call rule. 84 / 89 of 96 calls pass (150 / 400 ms). At \|ε\| = 200 with the guard, 10 of 12 pass. |
| **C2** double talk | **Fails.** ΔCER: zh +2.0 / +10.9 and en +4.9 / +28.4 points at near end 0 / −15 dB (limits 2 / 5). Residual after double talk within 10 dB in 98 % of bursts. Speex with its preprocessor passes C2 (≤ +0.6 points) but fails C1 (0 / 96 calls) and C3. |
| **C3** convergence | Fails. ≥ 20 dB within 1 s in 69 % of calls (p50 0.77 s, p90 1.7 s); after the path change 50 % (p90 2.5 s). ERL + ERLE ≥ 27 dB passes. |
| **C4** CPU | Passes. 7.8 ms per audio second; p99 0.12 ms per frame; 8 sessions give 6.6× the throughput of one. |

## Notes for M4

- **livekit aborts (SIGABRT)** on any frame that is not a whole 10 ms: assert the frame size first.
- **pywebrtc zero-pads partial frames silently.** 15 ms chunks changed its output by up to 9189 LSB. Feed exact 10 ms
  frames.
- **AEC3 alters the audio:**
  - it delays the capture by a fixed 8 ms (127–128 samples);
  - its forced high-pass cuts about 7 dB below 150 Hz and shifts the phase, so waveform SDR is meaningless for it;
  - its suppressor removes much of the near end during double talk, and touches it even with no echo while the
    reference plays (spectral SDR 13 dB);
  - it adds comfort noise.
- **Path changes:** AEC3 needs p50 ~1 s (p90 2.5 s) to recover, and these are the main source of remaining false
  starts. `echo_turn_suppression` is the backstop.
- **Not simulated: clock drift** between AICC's ticker and the uplink sample clock (20–100 ppm is 1–6 ms per minute).
  The tick-phase learner and the guard must keep re-anchoring.
- **SpeexDSP** needs Homebrew's libspeexdsp at run time (installed here). Its preprocessor adds 10 ms of delay.

## Reproduce

```sh
cd spikes/aec && uv sync --locked
for p in core timeline slices dt; do uv run --locked python s11_run.py gen --part $p; done
for p in core timeline slices dt; do uv run --locked python s11_run.py run --part $p; done
uv run --locked python s11_run.py run --part core --variants lk:learned,pw:learned,sx1024ppd_pf:gcc,sx1024ppd_pf:gcc_locked,sx1024ppd_pf:oracle,sx2048ppd_pf:gcc_locked
for p in core timeline slices dt; do uv run --locked python s11_run.py summarize --part $p; done
uv run --locked python s11_cpu.py
uv run --locked python s11_cer.py        # only after B2 (spikes/results/s3s4s9_report.md) — GPU
uv run --locked python s11_tables.py     # markdown tables from the JSON
```
