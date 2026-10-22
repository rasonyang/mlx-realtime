# M0 — Validation spikes

> **Status: PLAN — not started.** M0 waits for the owner's explicit go-ahead. Decisions D1–D19 were
> approved on 2026-10-22, and S11 (echo cancellation) was added the same day. M0 writes no production
> code: spike scripts live in `spikes/` and are thrown away. Findings, measured numbers and
> recommended plan changes will be written into this file.

## 0. Already settled by desk research (2026-10-22)

Sources are in `docs/ARCHITECTURE.md` §14. Nothing below has been run yet, except the SDK URL probe.

**Protocol.**

- Every client event in the brief and every server event except `conversation.item.created` has a GA
  name. The GA guides use `conversation.item.added` / `.done`.
- Audio formats are `audio/pcm` (24 kHz only), `audio/pcmu` and `audio/pcma`.
- openai-python 3.36.1 turns `base_url="http://…/v1"` into `ws://…/v1/realtime?model=…`. This was
  probed against a local capture server.

**Models.** Every repo exists on HF:

| Role | Repo |
|---|---|
| ASR | `mlx-community/Qwen3-ASR-1.7B-8bit` |
| TTS | `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit` |
| TTS fallback | `mlx-community/Kokoro-82M-bf16` |
| LLM | `mlx-community/Qwen3.6-35B-A3B-4bit` and `mlx-community/Qwen3.5-35B-A3B-4bit` (hybrid thinking, model_type `qwen3_5_moe`) |

**mlx-lm 0.32.2** supports `qwen3_5_moe`. Use 0.32.0 or later for these mlx-vlm conversions.

- Thinking is on by default. It can be turned off server-wide with `--chat-template-args`.
- `tool_calls` are parsed on the server with the qwen3_coder XML parser.
- `tool_choice` and `parallel_tool_calls` are ignored.
- The server keeps one MLX generation thread with continuous batching, and an LRU prompt cache. This
  hybrid architecture allows only exact-prefix or shorter-prefix cache reuse.

**mlx-audio 0.6.1.**

- `/v1/audio/speech` streams only with `stream: true`. `pcm` output is s16le at 24 kHz.
- Models load lazily on first use, and inference runs on a single broker thread.
- Qwen3-TTS has no `speed` control, and the whole text is prefilled before audio starts.
- Qwen3-ASR has no live-input streaming.

**Silero VAD v6.2.3.**

- ONNX I/O: `input` [B,N], `state` [2,B,128], `sr`.
- Frames are 512 samples at 16 kHz, plus a 64-sample context.

**Smart Turn v3.2.**

- Input `input_features` [B,80,800]; output a probability in [B,1].
- Audio is 16 kHz, at most 8 s, zero-padded at the front.
- Chinese accuracy: 85.8% (int8) and 90.5% (fp32).
- pipecat runs it after a 0.2 s VAD stop and falls back to ending the turn after 3 s.

**Gateway libraries.**

- onnxruntime 1.31.0 releases the GIL.
- `soxr.ResampleStream` handles chunked resampling.
- `audioop` is deprecated in 3.12, so G.711 is implemented with our own lookup tables.

**ai-native-callcenter (AICC).** It can attach unchanged through its `gateway` profile
(`docs/ARCHITECTURE.md` §2). Its client enforces four deadlines:

- 3 s from `response.create` to `response.created`;
- 3 s from `response.created` to the first audio;
- under 2 s between audio deltas;
- 3 s from `response.created` to `response.done` for a tool-only response.

It streams caller audio while the bot speaks, with no echo cancellation. Its client frames are in
`spikes/aicc/traces/`.

**What desk research cannot tell us:**

- latency on this machine, especially against AICC's deadlines;
- GPU contention between the three processes;
- multi-process stability;
- ASR and TTS quality on our kind of speech;
- how well an in-service echo canceller handles telephone echo (S11). FreeSWITCH provides none.

## 1. Ground rules

**Environment.** Apple M3 Max (12P + 4E cores), 128 GB, macOS 26.6.2, on mains power, with nothing
else heavy running.

**Record for every run:**

- the versions of Python, mlx, mlx-lm, mlx-audio, onnxruntime and openai;
- `mx.device_info()` and `sysctl iogpu.wired_limit_mb`;
- the commit SHA of every HF repo.

**Separate environments, uv only** (see `CLAUDE.md`).

- Each spike is a standalone uv project under `spikes/<name>/`, created with `uv init --no-workspace`,
  mirroring the planned `backends/llm` and `backends/audio` split. The gateway environment stays
  MLX-free.
- Dependencies are added with `uv add`, and scripts run with `uv run`.
- Every spike's `uv.lock` is committed with its results, so each number can be reproduced against the
  exact versions that produced it.
- No pip, no venv and no `requirements.txt`.

**Measurement.**

- Warm-up runs are discarded. Cold start is reported separately.
- Sample size: n ≥ 20 for GPU paths; n ≥ 200 for CPU micro-benchmarks.
- Report p50, p95 and max. Time with `time.perf_counter_ns`; first-chunk times are measured on the
  client side of the HTTP call, so they include the transport.
- Raw results are saved as JSON in `spikes/results/<spike>.json`, and summarized here.

**Downloads.** About 25 GB for the default models. Comparison builds — the other LLM, TTS 4-bit/1.7B
and ASR 0.6B — add about 30 GB. 377 GB is free.

## 2. Decision gates

| Gate | Question | Decided by | Rule |
|---|---|---|---|
| G1 | ASR and TTS: run `mlx_audio.server` as is, or thin workers on the mlx-audio Python API | S3, S4, S6 | Thin workers if the server path adds more than 30 ms to the first chunk, cannot be warmed and pinned reliably, cancels too coarsely, or is unstable in S6 |
| G2 | LLM repo and revision (Qwen3.6 vs Qwen3.5); sampling flags | S1, S2 | Thinking fully off, and tool calls correct in ≥ 19/20 zh and en cases. Then Qwen3.6, the brief's first choice, unless it shows a hard defect or a clearly worse TTFT |
| G3 | Smart Turn file (int8 vs fp32) and executor sizing | S7 | fp32 if its p95 on M3 Max fits the turn budget agreed at G4 |
| G4 | Latency budget per stage, and whether the targets are reachable | S2, S10 | A budget table; targets proposed to you for approval. Hard floor: AICC's deadlines at p95 with margin. ASR + LLM TTFT + first segment + TTS first chunk must stay under 3 s from `response.created`, including a full re-prefill after an instructions change. Otherwise the mitigations in S2 become plan changes |
| G5 | Text normalization scope | S9 | Normalize only the categories the TTS gets wrong |
| G6 | TTS fallback (D13) and 3 vs 2 model processes | S3, S6 | Kokoro stays if zh and mixed text are intelligible; process merge proposed only if S6 shows instability |
| G7 | Echo-canceller engine and reference alignment (D17) | S11 | On simulated telephone echo, pick the candidate that wins on all four: echo-only segments must not trigger VAD speech; near-end speech in double talk must stay recognisable (CER close to clean); it must converge after an echo-path change; CPU per session must fit the executor budget. If none wins on all four, the result and the trade-off come back to you |

## 3. Spikes

### S0 — Fixtures and environment

- Create the spike uv projects, which pins versions in their `uv.lock`.
- Download the models with the `hf` CLI, run through `uv run` from a spike project that declares
  `huggingface_hub`, and record each revision.
- **Text set** (≈ 60 sentences): Chinese, English and code-switched; short (≤ 8 chars), medium and
  long; numbers, money, dates, phone numbers and order IDs; and the call-center phrases
  ai-native-callcenter uses (greetings, wait lines).
- **Audio set: public datasets only** (decided 2026-10-22).
  - ≈ 40 utterances each of zh, en and code-switched zh/en speech, 1–10 s, with reference
    transcripts.
  - Candidates, verified for license and download before use: FLEURS (cmn_hans_cn, en_us), and ASCEND
    for code-switching.
  - Each utterance also gets a telephone copy: band-limited to 8 kHz, through a G.711 μ-law round trip,
    and back to 16 kHz.
  - A manifest records source, license and reference text. Audio stays out of git.
- **Echo material for S11.** Far-end and near-end speech from the same datasets, plus the
  standard line-echo path models (ITU-T G.168), verified before use. If a public AEC test set (e.g.
  the Microsoft AEC Challenge data) has a compatible license, add it for acoustic echo.
- **AICC prompt fixture.** `line_session_zh.json`, used read-only from that repository's
  `internal/provider/testdata/`. It is a real Chinese session with its instructions and 6 tools, and
  serves as the realistic prompt for S1 and S2. It is text, not audio.

### S1 — LLM behaviour (`mlx_lm.server`)

Start command: `--model <repo> --chat-template-args '{"enable_thinking":false}' --max-tokens 2048`
plus the card's non-thinking sampling.

1. **Thinking.** 20 zh/en prompts: no `delta.reasoning`, no `<think>`. Repeat with per-request
   `enable_thinking: true` to show the switch works.
2. **Request fields.** Check what happens with each of these:
   - `stream_options.include_usage` (does the usage chunk come before `[DONE]`?);
   - `max_completion_tokens`, `temperature`, `top_p`;
   - `presence_penalty`, which the card recommends: is it accepted?
   - the `model` field omitted vs set to the exact `--model` string;
   - a wrong `model` value (confirm it triggers a swap or lookup).
3. **Template edge cases.** For each, record the error:
   - no user message (the greeting case);
   - a system message that is not first;
   - an assistant text message between a `tool_calls` message and its `tool` result (AICC's wait
     line);
   - an empty conversation with only instructions.

   Then validate the gateway's planned fixes: a synthetic cue, and folding later system messages.

   Then the history shapes AICC produces:
   - a `function_call`, then an assistant wait line, then the `function_call_output`;
   - a deleted `function_call`;
   - several outputs followed by one request;
   - user-role stage notes: keypress, dead air, "(System note …)";
   - a `function_call_output` whose JSON carries a `hint` with a line to say.
4. **Tool calls.** Two tools; 20 prompts (zh/en; 10 that should call, 10 that should not). Record:
   - the raw output and the parsed `tool_calls` deltas;
   - typed arguments;
   - silent drops (`finish_reason: tool_calls` with no call);
   - a multi-turn round with the tool result;
   - whether `tool_choice` is ignored (expected), and the effect of sending no `tools` (our
     enforcement of `none`).
5. **Prompt cache.** Conversations A and B, 3 turns each, interleaved. Log `cached_tokens` and TTFT per
   turn, and check the system-prompt and tool-list reuse that the batched path claims.
6. **Cancellation.**
   - Close the stream after the first delta, then during a 16k-token prefill. Is generation stopped,
     judging by the server log and the next request's TTFT compared with idle?
   - Repeat with the disconnect inside a tool-call span.
7. **Concurrency.** Two and four simultaneous streams: are they batched? Report TTFT and tokens/s per
   stream.

### S2 — LLM performance

- TTFT and decode tokens/s for prompts of 300, 1k, 4k and 16k tokens (cache cold and warm), using the
  winning flags from S1.
- **AICC-realistic prompt** (`line_session_zh.json`, instructions and tools). Measure TTFT in four
  cases:
  - cold;
  - warm, appending one turn;
  - after an instructions-only change, which costs a full re-prefill of the hybrid model;
  - with a per-response `instructions` override.

  Then try the mitigations:
  - warming the cache with a 1-token request on `session.update`;
  - verbatim lines (D16), which remove the override case for AICC's lines.

  Compare all of these with AICC's 3 s first-audio deadline.
- Time to a complete tool call (`finish_reason: tool_calls`) on the realistic prompt. Compare it with
  AICC's 3 s deadline for tool-only responses.
- The "first segment" time: tokens until the first `，。！？` in typical zh/en answers.
- TTFT after 15 min idle, with and without `MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS`.
- Repeat the core numbers for Qwen3.5-35B-A3B-4bit (G2).

### S3 — TTS (`mlx_audio.server`, Qwen3-TTS and Kokoro)

1. **Streaming check.** With `stream: true`, `response_format: "pcm"`, record each chunk's arrival
   time, its size, and the total. Repeat without `stream` (expect the first byte only at the end).
2. **First-chunk latency.** Sweep `streaming_interval` over {0.16, 0.32, 0.8}, across:
   - zh, en and mixed;
   - short, medium and long text;
   - two speakers;
   - cold vs warm.

   Report p50/p95 time to first chunk, real-time factor (RTF) and chunk duration. Compare 0.6B 8-bit
   (default) with 0.6B 4-bit and 1.7B 8-bit, for information only.
3. **Same in-process.** Repeat 2 calling the Python API directly, to get the lower bound and so the
   HTTP overhead.
4. **Cancellation.** Close the request after its first chunk, then send a new request at once. When
   does the old generation stop, and what is the new request's time to first chunk?
5. **Quality guardrail.** Run 100 runs and measure the speaking-rate distribution. This checks the
   slow-speech defect reported on 1.7B (about 7–8% of runs at 0.45× rate, mlx-audio#1002) and the
   effect of sampling settings.
6. **Kokoro (D13).** zh (`lang_code: "z"` with `misaki[zh]`), en, and mixed text: first-chunk and
   total latency, and intelligibility, judged by back-transcription with ASR plus a short listening
   check.
7. **Warmup and pinning.** Does one warmup request leave the model resident? Does a lazy-load or swap
   ever happen mid-run?

### S4 — ASR (`mlx_audio.server`, Qwen3-ASR)

- Latency for 2, 5 and 10 s clips (warm, n = 20 each), zh / en / mixed, with and without `language`.
- 1.7B 8-bit (default) vs 0.6B, for information.
- CER/WER on the S0 audio set, wideband and telephone copies. Use CER for zh, WER for en, and a mixed
  error rate for code-switched speech.
- Output shape: `text`, `language`, segments, punctuation, how numbers are written.

### S5 — GPU contention

These measure the pipeline as it really runs, with the LLM still decoding while TTS synthesizes:

- TTS time to first chunk and RTF, while the LLM decodes a long answer, and while it prefills 16k
  tokens.
- ASR latency while the LLM decodes.
- LLM TTFT and tokens/s while TTS streams.
- If RTF exceeds 1 under contention, try `--prefill-step-size` and a 4-bit TTS.

### S6 — Stability and memory

- Run all three processes for 30 minutes:
  - 1000 TTS calls under concurrent LLM and ASR load;
  - count hangs (no chunk for 10 s), errors and any kernel panic;
  - record per-process memory (`footprint`), wired-limit settings, and `mx.device_info()`.
- Kill each backend in turn: confirm the health checks see it and that a supervisor-style restart
  recovers.
- If instability appears, also measure ASR and TTS merged into one process (G6).

### S7 — CPU audio models and gateway audio libraries

1. **Smart Turn timing.** Test {v3.2-cpu, v3.2-gpu (fp32)} × `intra_op_num_threads` {1, 2, 4} ×
   spinning {on, off}, with 200 runs on 8 s inputs. Report feature-extraction time and model time
   separately.
2. **Smart Turn accuracy.** Measure on the Chinese subset of `pipecat-ai/smart-turn-data-v3.2-test`,
   and again on telephone-degraded copies.
3. **Feature parity.** Compare pipecat's numpy Whisper features with `transformers`'
   `WhisperFeatureExtractor(chunk_length=8)` on 50 clips: max |Δ| on the features, and |Δp| on the model
   output.
4. **Silero.** Per-frame cost, and the native 8 kHz mode vs upsampled 16 kHz on telephone audio
   (frame agreement, onset delay).
5. **Concurrency.** Run 64 real-time streams through the planned executors: report event-loop lag p99
   and CPU, per-stream vs batched Silero, and 4 concurrent Smart Turn runs.
6. **Resampling.** `soxr.ResampleStream` fed random 10–40 ms chunks must equal one-shot
   `soxr.resample`. Record `delay()`.
7. **G.711.** Check the lookup tables against `audioop` (3.12) on all 65 536 inputs and all 256 codes.

### S8 — Protocol clients

1. **The official SDK.** Write a pinned script: openai 3.36.1 against a minimal stub server; record the
   URL, the headers, and GA parsing of the events we plan to emit.
2. **ai-native-callcenter: wire shape.** This is already captured: the audit ran AICC's real client
   with the `gateway` profile against AICC's fake server, and the frames are in `spikes/aicc/traces/`.
   M0 only turns the audit into the AICC section of `docs/PROTOCOL.md`. The live acceptance run
   comes later (M3/M4), with AICC's `probe` and `measureResponseCreated` harness pointed at
   mlx-realtime; a scratch wrapper already exists.
3. **Echo level on a real call.** Deferred by decision on 2026-10-22. S11 works on simulated echo
   instead.

### S9 — TTS text-normalization probe

- Feed the S0 numbers, money, dates, phone numbers, IDs and mixed-text sentences to Qwen3-TTS **raw**.
- Back-transcribe with Qwen3-ASR, and listen to a sample.
- Output: for each category, read correctly or not, which decides what the normalizer handles (G5).

### S10 — End-to-end floor and budget

- Combine the measured stages into a budget table:

  ```
  VAD silence (500 ms as AICC sends it; 200 ms our default) → Smart Turn → ASR (3 s utterance)
  → LLM TTFT (warm cache, typical prompt) → first segment → TTS first chunk → transport
  ```

- Check the sum with one scripted run that chains the real calls, without the gateway.
- Propose P50/P95 targets for speech end → first audio, for you to approve (G4).
- Check the AICC window separately: the stages after `response.created` (ASR onward) must stay under
  3 s at p95 in every case S2 covers.

### S11 — Echo cancellation (D17, G7)

M0 only selects the engine and its defaults; the canceller is implemented in M4.

1. **Desk check.** Look at echo-canceller engines that install with uv on macOS arm64 and Python
   3.12: WebRTC AEC3 bindings, SpeexDSP bindings, and a numpy NLMS / frequency-domain line-echo
   canceller as a baseline. For each, record:
   - license and API;
   - frame size and supported sample rates;
   - the delay range it can absorb;
   - how it handles double talk.

   Also fetch the ITU-T G.168 echo path models.
2. **Simulation.**
   - Build test calls: far-end speech at 24 kHz, as the gateway sends it, played out on the modelled
     AICC timeline:
     - downsampled to 8 kHz;
     - G.711-coded;
     - convolved with each G.168 echo path, at echo return losses of 6–20 dB and bulk delays of
       20–300 ms;
     - mixed with near-end speech for double talk, and with background noise;
     - then converted back the way AICC sends it (24 kHz) and the way the gateway sees it (16 kHz).
   - Include an echo-path change mid-call.
3. **Measure, per candidate:**
   - echo return loss enhancement (ERLE) over time, and convergence time;
   - Silero speech-frame rate on echo-only segments, before and after cancellation;
   - near-end ASR CER during double talk;
   - CPU per session per second of audio;
   - sensitivity to errors in the modelled playout timeline (±50 / ±200 ms).
4. **Decide G7**, and the defaults for `aec.*` and `barge_in.*`.

## 4. Inputs received (2026-10-22)

- D1–D19 approved.
- M0 test audio comes from public datasets.
- The echo measurement on a real call is deferred. FreeSWITCH provides no echo cancellation; the
  service must do it (S11).
- uv was upgraded by the owner to 0.13.5 (2026-10-21 build). Every uv project pins
  `[tool.uv] required-version = ">=0.13.5"`. The flags the plan relies on (`uv init --no-workspace`,
  `--locked`, `--frozen`, `uv add --dev`) were re-checked on 0.13.5.

## 5. Deliverable and exit criteria

**This file is updated with:**

- per-spike findings and numbers;
- gate decisions G1–G7, with evidence;
- pinned versions and HF revisions;
- proposed `config.yaml` values;
- recommended changes to the brief and `docs/ARCHITECTURE.md`, each listed for approval.

**M0 is done when:**

- every gate is decided;
- the LLM repo id and revision are pinned;
- nothing unknown blocks M1–M3.

## 6. Findings (filled in as spikes complete)

### S0 — Test audio (A2, 2026-10-22)

126 utterances (42 zh, 42 en, 42 code-switched), 781.5 s in total, all with a license, a telephone copy
and an 8 kHz G.711 stream (84 μ-law, 42 A-law). Rebuilding reproduces the files and the manifest byte
for byte. Scripts and the versioned manifest: `spikes/datasets/` (`MANIFEST.md`, `manifest.jsonl`).
Audio: `spikes/data/` (git-ignored, 132 MB).

| Set | Source (license, revision) | Count | Duration min / median / max | With numbers |
|---|---|---|---|---|
| zh | FLEURS `cmn_hans_cn` test (CC-BY-4.0, `70bb2e84`) | 42 | 4.74 / 7.53 / 9.72 s | 10 |
| en | FLEURS `en_us` test (CC-BY-4.0, `70bb2e84`) | 42 | 4.32 / 6.81 / 9.90 s | 13 |
| mix | ASCEND test, `language=mixed` (CC-BY-SA-4.0, `737e9800`) | 42 | 1.65 / 3.80 / 8.62 s | 2 |

**Processing.**

- Every utterance is levelled to −26 dBov (ITU-T P.56 measure, peak-limited at −1 dBFS) before either
  copy is made. Source levels differed by about 50 dB, and leaving them would have clipped ASCEND and
  buried quiet English material in G.711's noise floor. The gain is recorded per row, and the
  originals are kept.
- Telephone chain: 16 kHz → 8 kHz (soxr HQ) → G.711 round trip (stdlib `audioop` on 3.12) → 16 kHz
  (soxr HQ). No extra 300–3400 Hz filter.

**Caveats.**

- FLEURS has no clip under 3.6 s, so the short utterances all come from ASCEND.
- ASCEND's test split has only 2 speakers.
- `text_norm` does not expand numbers. FLEURS writes "1990 年" for spoken 一九九零年, so rows with
  `has_number` are scored separately in S4.
- ASCEND is share-alike: the derived copies may only be redistributed under CC-BY-SA-4.0.

**Echo material.**

- ITU-T G.168 (04/2015, Annex D, Tables D.2–D.9): 8 hybrid echo-path models at 8 kHz, 64–128 taps
  (8–16 ms). The extraction was checked against the published scaling factors and the 2022
  corrigendum. The coefficients are © ITU and stay in git-ignored `spikes/data/echo/`.
- A 20-example Microsoft AEC Challenge subset: synthetic, 10 linear and 10 nonlinear examples, 16 kHz.
  It is acoustic echo, for internal evaluation only.

### S11 part 1 — Echo-canceller desk research (A3, 2026-10-23)

Full report: `spikes/results/a3_aec_research.md`. Install checks: `spikes/results/a3_aec_install_check.json`.

| Candidate | Delay handling | Alters audio | Notes |
|---|---|---|---|
| WebRTC AEC3 via `livekit` 1.1.21 (Apache-2.0; WebRTC M150) | own estimator, ~0–509 ms | suppressor and comfort noise always on; high-pass filter forced on | **Primary.** Aborts the process (SIGABRT) on any frame that is not a whole multiple of 10 ms, so the size is checked before every call. One protobuf build per call |
| WebRTC AEC3 via `pywebrtc-audio` 0.2.0 | same engine | same, with no settings | Alternative wrapper for the same engine |
| SpeexDSP 1.2.1 (BSD-3) via ctypes | none: needs a gateway-side GCC-PHAT pre-aligner | only with the optional preprocessor | Linear-only output, which is best for ASR. Its Homebrew library is outside the lockfile, so it serves only as a spike comparison; M4 would vendor the C sources |
| numpy PBFDAF + Geigel double-talk detector (ours) | none: needs GCC-PHAT | no | G.168-style reference point |

**Alignment design, carried into S11 part 2.**

- Replay AICC's playout state machine on our own send log, with one unknown offset θ:
  - playback starts once 3 frames are queued or after 60 ms;
  - it runs on a 20 ms tick;
  - it re-buffers after more than 2 empty ticks.
- Feed the engine in lock step with the uplink sample count.
- Lead the reference by 60 ms, so it always arrives before its echo.
- Re-anchor θ on every truncate.
- Low-pass both signals at 3.8 kHz, because AICC's linear-interpolation upsampling folds images into
  4–8 kHz.

**G.168 facts used.** Hybrid dispersion is at most 12 ms. The worst-case ERL is 6 dB; the typical mean
is about 14–15 dB. Convergence requires at least 20 dB within 1 s, and 25–30 dB at steady state.
