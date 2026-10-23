# A3: echo-canceller engines and the S11 part 2 recipe

M0, S11 part 1, for G7: desk research and install checks only (2026-10-23). [n] marks a source;
**UNVERIFIED** means no primary source was found. Artifacts are in `spikes/aec/` (the uv project,
`install_check.py`, `baseline.py`, `speex_ctypes.py` and `g168_echo_paths.json`) and in
`spikes/results/a3_aec_install_check.json`.

## 1. Candidates

| | AEC3 via `livekit` | AEC3 via `pywebrtc-audio` | SpeexDSP via `speex_ctypes.py` | numpy baseline |
|---|---|---|---|---|
| License | Apache-2.0; WebRTC BSD-3 [1] | Apache-2.0 [2] | BSD-3 [3] | ours |
| Version | 1.1.21, 2026-10-16; WebRTC M150 [1][4] | 0.2.0, 2026-09-03; AEC3 from 2020, milestone UNVERIFIED [2] | 1.2.1, 2022-06-13 [3] | numpy 2.5.4 |
| macOS arm64 cp312 | wheel | wheel | Homebrew library, loaded via ctypes | wheel |
| API | `process_reverse_stream(f)`, then `process_stream(f)`, in place; `set_stream_delay_ms` | `EchoCanceller(rate).process(near, far)` | `SpeexAec(...).process(near, far)` | `PbfdafGeigel().process(near, far)` |
| Rates, frame | 8–384 kHz; whole 10 ms frames only [4] | 16, 32 or 48 kHz; any length | any rate; 10–20 ms frames [3] | any rate; 10 ms |
| Delay handling | own estimator, ≈0–509 ms [4] | same estimator; an issue reports a 600 ms echo left uncancelled [2] | none [3] | none |
| Echo tail | 52 ms after alignment [4] | same (UNVERIFIED) | `filter_length` (128 ms) | 130 ms |
| Double talk | step scaled by error; two filters; near-end detector [4] | same | variable learning rate; two filters [3][5] | Geigel detector freezes adaptation [6] |
| Alters audio | suppressor and comfort noise always on; HPF forced on [4] | same, with no settings [2] | only if the optional preprocessor is enabled [3] | no |
| GIL | released in the native call, but the protobuf is built in Python [1][7] | released; not thread-safe per instance [2] | released [7] | held |
| Maintenance | active | new | last release 2022 | n/a |
| Install check | OK; a 15 ms frame aborts the process (SIGABRT) | OK | OK | OK |

State is per instance: one instance per session.

**Excluded** (reasons in the appendix): `pyaec`, `speexdsp`, `speexaec`, `webrtc-audio-processing`, `aec-audio-processing`, `webrtc-apm`, `aec3-py`, packages without AEC, per-sample toolkits, and neural or Apple AEC.

## 2. Line-echo facts

From G.168 (04/2015) with Cor. 1 [9], unless noted:
- **Echo paths.** Annex D models m1–m8 are 64–128 integer taps at 8 kHz, applied as g(k) = 10^(−ERL/20)·Ki·mi(k−δ). The minimum ERL is 6 dB, except m2 (6.55), m7 (11.06) and m8 (9.27). They are transcribed and checked in `g168_echo_paths.json`.
- **Dispersion.** Measured echo paths average 6.02 ms (SD 2.26) and never exceed 12 ms.
- **ERL.** G.168 calls 6 dB "a typical worst-case value"; G.131 gives a mean of 14 dB (SD 3) [10]; G.122 requires a mean echo loss ≥ 15 dB [11].
- **Tail.** G.168 gives no typical tail capacity, so the usual 64/128 ms is UNVERIFIED.
- **Convergence (2A/2B).** Loss ≥ 20 dB from 1 s + t_d; with the NLP off, 25–30 dB from 10 s + t_d (≈ 26.7 dB at −20 dBm0); the same after an echo-path change.
- **Double talk (3A/3B).** With the near end 15 dB below, converge within 5 s. Afterwards the residual stays within 10 dB of the steady-state limit, or 3 dB when the near end is 6–30 dB lower.
- **Acoustic echo** is far weaker: TCLw is > 45 dB for handsets and > 40 dB hands-free [12].
- **Carriers** usually cancel line echo, but not always [13].

## 3. Reference alignment

**Prior art.**
- **Local stacks** measure their device delay, which AEC3 uses only as a seed [4][14].
- **Azure Voice Live**, the only remote-playout case found, "assumes the client plays response audio as soon as it receives it" [15].
- **LiveKit agents and pipecat** run no server AEC; LiveKit ignores interruptions for 3 s [14][16].
- **FreeSWITCH** has an opt-in Speex canceller [17].

**AICC** (8fabadc):
- Playout runs on a 20 ms tick. It starts at 3 frames or 60 ms, and re-buffers after more than 2 empty ticks (`voice/rtp.go` L22-31, L391-437).
- The uplink is upsampled by linear interpolation, so spectral images at 4–8 kHz reach us (`media/resample.go` L30).
- Gaps under 500 ms are silence-filled (`voice/jitter.go` L15-19).
- Truncate played-ms counts the frames sent (`aicall/session.go` L961-975).

**Design.**
1. **Model playout.** Replay AICC's playout state machine on our send log, with one unknown offset θ.
2. **Clock.** Index the reference and the capture by uplink sample count.
3. **Lock-step feeding.** One 10 ms reference frame per capture frame, with zeros when idle. Carry the delay in the content; AEC3 resets if render leads by more than 32 ms [4].
4. **Causality lead.** Emit the reference 60 ms early, because echo that precedes its reference is uncancellable [3].
5. **Re-anchor.** Correct θ from each truncate, and drop reference audio that was never played.
6. **Residual delay** (20–500 ms). AEC3 absorbs it. SpeexDSP and the baseline need a gateway estimator, which also guards AEC3's 509 ms limit.
7. **Estimator.** GCC-PHAT [18] on 8 kHz copies: 2 s windows, 0.5 s hop, lags −300…+1000 ms, only while the reference is active. Shift only when two windows agree within 2 ms. Weak in noise [19]; pitch side-peaks possible (UNVERIFIED).
8. **Telephone band.** Low-pass both signals at 3.8 kHz, because the images are not linear in the reference.
9. **Hazards.**
   - livekit aborts on any frame that is not a whole number of 10 ms, so assert the size first.
   - `set_stream_delay_ms` clamps to 0–500 ms and covers one frame only, so shift the content instead.

## 4. Shortlist

1. **AEC3 via `livekit`.** The only engine that covers our residual delay by itself; mature and maintained. Risks:
   - the always-on suppressor and comfort noise may hurt ASR;
   - the forced HPF;
   - a protobuf per call;
   - the abort hazard.

   Run `pywebrtc-audio` alongside it to choose the vehicle.
2. **SpeexDSP + GCC-PHAT.** Its output is linear only, the best case for ASR, and it is cheap. It needs the pre-aligner, and M4 would vendor its sources.
3. **numpy PBFDAF + Geigel + GCC-PHAT.** A transparent, G.168-style reference point.

## 5. S11 part 2 recipe

**Material.**
- Far end: Qwen3-TTS at 24 kHz (S3).
- Near end: the S0 telephone copies, with transcripts.
- Echo paths: the JSON models.
- Noise: band-limited white noise (G.168 test 9).
- Levels per G.168 §6.4.1.2, with the far end at −20 dBm0.

**Chain** (offline, seeded):
1. Send schedule: TTS speed, 10 s lead cap; one slice at RTF 1.1 forces underruns.
2. True playout: AICC's state machine plus transit, tick phase and ε.
3. Downlink: AICC's 33-tap filter, ↓3, G.711 μ-law.
4. Echo: g(k), then the bulk delay D.
5. Add the near end and noise.
6. Uplink: G.711, AICC's linear ↑3, soxr to 16 kHz. This is the capture.
7. Reference: TTS → soxr → model timeline + lead → estimator → engine.
8. Engine output → Silero and ASR.
9. Keep the echo and near-end components as ground truth.

**Call (60 s):** far end only for 0–15 s; alternating single talk for 15–30 s; double talk for 30–45 s; an echo-path change at 45 s, then far end only.

**Grid.**
- **Core:** m1–m8 × ERL {6, 10, 15, 20} × D {20, 120, 300} ms, with noise at −65 dBm0.
- **Timeline:** m4/m5, ERL 10, D {20, 120, 300} × ε {−200, −50, 0, +50, +200} ms, plus a mid-call ε step; aligned by the engine, GCC-PHAT or an oracle.
- **Double talk:** ERL {6, 15} × near end 0 or 15 dB below the far end; 40 zh + 40 en utterances.
- **Slices:** noise at −40 dBm0; A-law; an open path.
- **Variants:**
  - AEC3 with and without the pre-filter;
  - Speex at 1024 and 2048 taps, preprocessor off and on;
  - the baseline at μ 0.3 and 0.5.

**Metrics.** EO (echo-only) means far end active and near end absent, by ground truth.
- **ERLE(W)** = 10·log10(Σc²/Σe²) over 200 ms EO windows with a 100 ms hop, where c is the engine input and e its output.
  - ERLE_ss is the median from 5 s after onset until the change.
  - For AEC3 this includes suppression.
- **Convergence time.** Far-end-active time from onset, or from the change, until ERLE stays ≥ 20 dB for 1 s of EO.
- **Silero rate.** The gateway's Silero v6.2.3 (enter at 0.5, exit at 0.35) on EO spans, before and after the engine.
  - Rate = frames with p ≥ 0.5 / EO frames.
  - Also count `speech_started` events at `min_speech_ms` 150 and 400.
  - Report the first 1 s of each playback separately, for `echo_guard_ms`.
- **ΔCER.** Transcribe each double-talk near-end utterance ±300 ms with Qwen3-ASR (S4 setup).
  - CER for zh, WER for en, micro-averaged with bootstrap CIs.
  - ΔCER = error with the engine − error through the same chain without echo.
- **CPU per session-second.** Σ `thread_time_ns` over engine calls ÷ audio seconds, cross-checked with `process_time`.
  - Per-frame p50/p95/p99 over ≥ 6000 frames.
  - 8 sessions on a `ThreadPoolExecutor(8)` against 1, for the GIL.
- **Timeline sensitivity.** ΔERLE_ss, the change in Silero rate, and the re-convergence time at ε ±50 and ±200 ms against 0, per alignment method.

**Proposed G7 pass criteria** (core grid, |ε| ≤ 50 ms):
1. **Echo-only.** After the first 1 s of far-end speech: Silero rate ≤ 1% and no `speech_started`. At |ε| = 200 ms, the same within 2 s.
2. **Double talk.** ΔCER ≤ 2 points at equal levels and ≤ 5 at −15 dB. The residual stays within 10 dB or 3 dB of the Fig. 11 limit (G.168 3B).
3. **Convergence.** ≥ 20 dB within 1 s, both initially and after the change. ERL + ERLE_ss ≥ 27 dB in quiet.
4. **CPU.** ≤ 20 ms of CPU per audio second per session; p99 ≤ 2 ms per frame; 8 sessions ≥ 6× the throughput of one.

If no engine wins on all four, the trade-off goes to the owner.

## Sources

1. livekit 1.1.21: PyPI JSON; the installed `livekit/rtc/apm.py`, `_ffi_client.py` and `media_devices.py`; github.com/livekit/python-sdks tag rtc-v1.1.21 (cf9aa12).
2. pywebrtc-audio 0.2.0: PyPI JSON; github.com/strands-labs/pywebrtc-audio @1bc860a, `bindings/webrtc_audio_bindings.cpp` L355-528 (gil_scoped_release L473, L498), issues #5 and #6; the installed `_webrtc_audio.pyi`.
3. SpeexDSP 1.2.1, github.com/xiph/speexdsp/blob/SpeexDSP-1.2.1/:
   - `include/speex/speex_echo.h` L73-74, L93-112;
   - `libspeexdsp/mdf.c` L33-50, L426-427, L1231-1247;
   - `libspeexdsp/preprocess.c` L79-81, L426-438;
   - `doc/manual.lyx` L4584, L4812;
   - the Homebrew headers `/opt/homebrew/include/speex/*.h`.
4. AEC3 at webrtc-sdk/webrtc 89d790b = webrtc.googlesource.com branch-heads/7871:
   - `api/audio/echo_canceller3_config.h` L35-55, L92-119, and `api/audio/audio_processing.h` L66-72, L147-148, L199-204 (both re-checked by me);
   - `modules/audio_processing/aec3/aec3_common.h` L44-83, `matched_filter.cc` L606-609 and L724-729, `render_delay_buffer.cc` L170-188 and L269-275, `suppression_filter.cc` L98-121;
   - `audio_processing_impl.cc` L162-175 and L1634-1651;
   - livekit/rust-sdks @5a656c4: `webrtc-sys/include/livekit/apm.h` L35-43, `libwebrtc/src/native/apm.rs` L47-104, `livekit-ffi/src/server/requests.rs` L990-1071.
5. Valin, IEEE TASLP 15(3):1030–1034, 2007, doi:10.1109/TASL.2006.885935.
6. Duttweiler, IEEE Trans. Commun. 26(5):647–653, 1978, doi:10.1109/TCOM.1978.1094133; the Geigel statistic as in Schneider & Habets, EUSIPCO 2015, eq. (11).
7. docs.python.org/3.12/library/ctypes.html (PyDLL: "the Python GIL is not released", unlike CDLL); numpy.org/doc/stable/reference/thread_safety.html.
8. PyPI JSON for every package named. github.com/thewh1teagle/aec @0467306, `src/lib.rs` (no SET_SAMPLING_RATE). The speexdsp 0.1.1 sdist `setup.py` and `echo_canceller.cpp`. freedesktop `webrtc-audio-processing` v2.1 NEWS. Local build logs.
9. ITU-T G.168 (04/2015) and Cor. 1 (12/2022), itu.int/rec/T-REC-G.168:
   - definitions §3.3, §3.12–3.15;
   - §6.4.1.2;
   - the "6 dB is a typical worst-case value" note (p. 15);
   - tests 2A/2B (Figs 10a, 11, 12a), 3A/3B (§6.4.2.4);
   - Annex D, Tables D.1a–D.9;
   - Appendices II.4–II.5 and I.2.3.
10. ITU-T G.131 (11/2003), Appendix I.
11. ITU-T G.122 (03/1993).
12. ITU-T P.310 (06/2009) §5.4.1; ITU-T P.342 (06/2009) §5.3.1.
13. RFC 3611 §4.7.4.
14. Delay handling elsewhere:
    - webrtc.googlesource.com/src `aec3/render_delay_buffer.cc` L170-195, L304-313;
    - PulseAudio v17.0 `src/modules/echo-cancel/module-echo-cancel.c` L152-179 and `webrtc.cc` L345;
    - PipeWire 1.6.9 `src/modules/module-echo-cancel.c`;
    - docs.livekit.io (noise cancellation);
    - docs.pipecat.ai (choosing a transport).
15. learn.microsoft.com/azure/ai-services/speech-service/voice-live-how-to, updated 2026-10-08.
16. github.com/livekit/agents @31c6026, `livekit-agents/livekit/agents/voice/agent_session.py` L531-534 (`aec_warmup_duration`, default 3.0 s, None for outbound SIP).
17. github.com/signalwire/freeswitch @ef32e20, `src/switch_ivr_async.c` L3417 and L3555-3581 (`echo_cancel`, 1024-sample tail, the last written frame as reference).
18. Knapp & Carter, IEEE TASSP 24(4):320–327, 1976, doi:10.1109/TASSP.1976.1162830.
19. Chen, Benesty & Huang, EURASIP J. Applied Signal Processing, 2006, doi:10.1155/ASP/2006/26503.

The AICC references are to ai-native-callcenter @8fabadc, read-only.

## Appendix: notes for S11 part 2

These notes are not part of the 1500-word report.

**Excluded candidates** [8]:
- **pyaec 1.0.1** (thewh1teagle/aec-rs). It installs from a wheel and runs, but:
  - it never calls `SPEEX_ECHO_SET_SAMPLING_RATE`, so it uses 8 kHz constants at any rate;
  - it is fixed point;
  - the Python wrapper does not check buffer length, so a short buffer reads and writes out of bounds;
  - the MIT license is in Cargo.toml only.

  Removed.
- **speexdsp 0.1.1 and speexaec 0.1.3.**
  - Both build only against Homebrew's libspeexdsp: speexdsp needs `CFLAGS`/`LDFLAGS`, speexaec needs `SPEEXDSP_PREFIX`.
  - Both hold the GIL.
  - speexdsp's preprocessor is commented out; speexaec's GitHub repository returns 404.

  Removed.
- **webrtc-audio-processing 0.1.3.** AECM or AEC2 only, never AEC3. The build fails: `-mfloat-abi=hard -mfpu=neon` is unsupported on arm64-apple-darwin.
- **aec-audio-processing 1.0.1.** The full APM from M131, but wheels are Windows-only. The sdist build fails against Homebrew abseil 20250814: `no template named 'Nullable'`.
- **webrtc-apm 0.1.6.** Only a win_amd64 cp311 wheel.
- **aec3-py.** Not on PyPI, no license, and holds the GIL.
- **echoff 0.3.0.** Depends on livekit.
- **No AEC at all:** webrtc-noise-gain 1.3.0, speexdsp-ns and pyspeex-noise.
- **Per-sample toolkits:** pyroomacoustics, padasip and adaptfilt are per-sample Python loops.
- **spandsp 3.x:** per-sample 8 kHz C, a power-of-two tail, prints to stdout on every sample, and no binding.
- **OSLEC:** GPL, and removed from Linux 6.16.
- **pjsua2:** device path only.
- **Mask-based neural AEC:** DTLN-aec and the AEC Challenge baseline.
- **Apple voice-processing I/O:** local devices only.
- **M4 fallback, if AEC3's linear output or tuning is needed:** freedesktop `webrtc-audio-processing` 2.1 (M131, BSD-3, C++).

**Install-check record** (`spikes/results/a3_aec_install_check.json`). Each candidate imported, listed its API, and processed 100 frames of 10 ms at 16 kHz.
- **livekit.**
  - Accepted 10 ms frames at 8, 16, 24, 32, 44.1 and 48 kHz.
  - Split a 20 ms frame into two 10 ms calls.
  - A 15 ms frame panicked at `libwebrtc/src/native/apm.rs:48:9` ("slice must have a multiple of 10ms worth of samples"); the child process exited with −6.
- **pywebrtc-audio.** Imported cleanly on CPython 3.12.12 arm64, so its issue #6 did not reproduce. It took a 1 s array in one call, and returned 240 samples for a 15 ms frame.
- **speexaec.** Rejected a 100-sample frame with a `ValueError`.
- **Removed after the check, with `uv remove`:** speexdsp, speexaec and pyaec. The spike keeps livekit, numpy and pywebrtc-audio.

**AEC3 details** [4]:
- **LiveKit's settings.** Only four flags: AEC on or off, AGC2 adaptive digital, HPF, and NS (kModerate). The default `EchoCanceller3Config` is used.
- **Forced HPF.** `enforce_high_pass_filtering` defaults to true, so the HPF runs whenever AEC is on, even with `high_pass_filter=False`.
- **Threads.**
  - LiveKit's FFI locks a mutex per APM on every call; separate APMs are independent.
  - The APM header says stream setters "are never called concurrently with ProcessStream()".
  - The Python `FfiClient.request` takes no lock.
- **Suppression** stays conservative for the first 2.5 s of active render.
- **Delay change.** A newly detected delay resets both filters and the suppressor. The ≈ 0.5 s re-estimation time is an inference (UNVERIFIED).
- **Small changes.** Increases of one block (4 ms) are ignored.
- **Render queue.** Render frames beyond a queue of 100 are dropped silently.
- **pywebrtc-audio** wraps `EchoCanceller3` directly, without the APM.

**`speex_ctypes.py`:**
- Sets the sampling rate.
- Rejects any frame whose length is not `frame_size`.
- Leaves the preprocessor off by default. When it is on, it is linked to the echo state with denoise off.
- Needs `brew install speexdsp` at run time; override the library path with `SPEEXDSP_LIB`.
- For M4, vendor the BSD C sources into a built extension.

**`baseline.py`:**
- Overlap-save PBFDAF with a fully constrained gradient: 160-sample partitions, 13 partitions.
- Step μ = 0.5, normalized by the far-end power summed over the partitions.
- Geigel detector: threshold 0.5, i.e. 10^(−6/20) for the 6 dB minimum ERL, with a 30 ms hangover.
- No delay estimation, NLP or comfort noise.
- Smoke-tested only. Validate convergence on white noise through m1 before using its numbers.

**`g168_echo_paths.json`:**
- Transcribed from G.168 Annex D and cross-checked against both PDFs and the Cor. 1 Word file. The computed tone Ki match Table D.1b.
- The ITU text says "All rights reserved". The owner should decide whether it may be committed; if not, regenerate it from the PDF into git-ignored `spikes/data/`.

**AICC:**
- The downlink filter is a 33-tap Hamming windowed sinc (`media/resample.go` L61-123, `media/format.go` L116).
- AICC's played-ms excludes the 60 ms prebuffer and the RTP path to the phone; that residual is part of θ.
- Only gaps of 500 ms or more break the uplink sample clock.
