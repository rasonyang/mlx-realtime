# M0 — Validation spikes

> **Status: done (findings through 2026-10-24).** Every gate is decided; G4's budget and targets
> await the owner's approval. S5, S6 and S10 ran through the real gateway in the real-stack
> validation (§6, last finding). Decisions D1–D19 were approved on 2026-10-22, and S11 (echo
> cancellation) was added the same day. M0 writes no production code: spike scripts live in
> `spikes/`. Findings, measured numbers and recommended plan changes are in §6.

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

G4 and G6 waited for S5, S6 and S10, which ran through the real gateway in the real-stack validation
(2026-10-23/24); their outcomes are the last finding in §6:

- **G4:** a budget and targets are proposed for approval (`docs/LATENCY.md` §4). One call meets the
  targets and AICC's floor with margin; three calls miss the P95 target (2.53–2.78 s against 2.5 s)
  and do not hold AICC's floor reliably, because of the one TTS stream.
- **G6:** keep three model processes (the soak was clean). The fallback half was settled by the owner:
  the 4-bit Qwen3-TTS replaces Kokoro (D13).

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

### S7 — CPU audio models and libraries (A4, 2026-10-23)

Report: `spikes/results/s7_report.md`. Raw numbers: `spikes/results/s7_*.json`. All runs started at a
1-minute load ≤ 4 unless flagged in the report.

**Smart Turn model time** (p50 / p95 ms, spinning off; feature extraction adds ≈ 2 ms):

| File | 1 thread | 2 threads | 4 threads |
|---|---|---|---|
| int8 (`-cpu`) | 51.8 / 53.6 | 28.1 / 29.6 | 17.0 / 18.0 |
| fp32 (`-gpu`, runs on CPU) | 107.9 / 110.6 | 57.1 / 59.2 | 31.8 / 33.2 |

**Smart Turn accuracy, Chinese** (929 clips, threshold 0.5; the test set is TTS speech):

| File | Audio | Accuracy | False-negative rate | False-positive rate |
|---|---|---|---|---|
| int8 | clean | 86.3% | 17.3% | 9.9% |
| int8 | telephone | 87.3% | 9.5% | 16.0% |
| fp32 | clean | 90.5% | 8.3% | 10.8% |
| fp32 | telephone | 89.9% | 6.6% | 13.8% |

Telephone audio pushes the model towards "complete": false-positive rates rise by 3–6 points.

**Feature parity.** The numpy features match transformers to 1.2e-7. The fp32 model output differs by
at most 2.5e-6.

**Silero.** 0.13 ms per 32 ms window. On long telephone streams, native 8 kHz mode drifts: 40% of
onsets come late (p95 +672 ms). Upsampled to 16 kHz there is no drift (onset p95 +64 ms).

**Concurrency, 64 streams.**

- Per-stream Silero on a single-thread executor: event-loop lag p99 ≤ 0.5 ms.
- Smart Turn with fp32 and 2 threads per worker, K=2: total p99 260 ms at 8 evaluations/s (bursts of 8),
  and 301 ms at 16/s. With K=4: 141 / 171 ms. K=1 falls behind.

**soxr.** float32 streaming is bit-identical to one-shot; int16 dithers by up to 2 LSB. Lag with HQ:
24k→16k 21 / 37 ms (mean / max); 8k→16k 64 / 115 ms; LQ halves the 8k lag.

**G.711.** Lookup tables built from `audioop.c`'s algorithm: 0 mismatches over all 65 536 + 256
inputs, for μ-law and A-law. Sun's later `g711.c` disagrees on 381 μ-law and 896 A-law negative
inputs, so do not port that one.

**G3 decision (2026-10-23).**

- Smart Turn **fp32** (`smart-turn-v3.2-gpu.onnx` on CPU), `intra_op_num_threads=2`, spinning off,
  one session per worker. The executor uses **K=2** by default; config allows K=4 for bursty loads. Do
  not batch evaluations.
- Rationale: fp32 costs about 30 ms more per evaluation than int8. But int8 wrongly says "incomplete"
  twice as often (17.3% vs 8.3% on clean Chinese), and each such miss holds the turn until
  `max_silence_ms`. fp32 therefore gives the lower expected latency, and it is also the more accurate
  model.
- Silero: per-stream calls on a single-thread executor. Telephone audio is upsampled to 16 kHz.
- Resampling: soxr HQ, streaming in float32. For M7, consider LQ for 8k→16k to halve the VAD lag.

### S11 part 2 — Echo-cancellation simulation (B4, 2026-10-23)

Report: `spikes/results/s11_report.md`. Raw: `spikes/results/s11_{core,timeline,slices,dt,cpu}.json`.

**Setup.** 3020 engine runs, each in a child process, with 0 crashes. The chain followed A3's recipe:
AICC's playout state machine → downsample ×3 → G.711 → G.168 m1–m8 echo paths → near-end and noise →
AICC's linear upsample ×3 → soxr → 16 kHz → the gateway's Silero.

**Without a canceller**, Silero fires on 75% of echo-only windows: 1005 false `speech_started`
events in 96 calls.

**Core grid** (96 calls per variant; "C1 calls" = calls with Silero ≤ 1% on echo-only audio after
the first 1 s and no false start, at `min_speech` 150 / 400 ms):

| Variant | ERLE steady state p50 | ≥ 20 dB within 1 s (start / after path change) | C1 calls | CPU ms per audio s | 8 threads vs 1 |
|---|---|---|---|---|---|
| AEC3 via pywebrtc-audio, tick phase learned | 31.0 dB | 72% / 51% | 65 / 82 | 7.8 | 6.6× |
| AEC3 via pywebrtc-audio, tick phase known | 31.7 dB | 69% / 50% | 84 / 89 | 7.8 | 6.6× |
| AEC3 via livekit, tick phase known | 32.2 dB | 72% / 41% | 70 / 79 | 9.0 | 1.8× |
| SpeexDSP, linear, oracle delay | 26.4 dB | 3% / 0% | 0 / 0 | 4.4 | 1.1× |
| numpy PBFDAF baseline, oracle delay | 13.2 dB | 0% / 0% | 0 / 0 | 9.2 | 0.9× |

**Timeline error.** With |ε| = 200 ms, engine-only AEC3 passes 6 of 12 calls. With the GCC-PHAT guard
it passes 10 of 12, for 1.4 ms CPU per audio second.

**Double talk.** AEC3's suppressor removes part of the near-end speech: spectral SDR is 6.1 dB, against
4.2 dB with no canceller and 20 dB for Speex linear. ΔCER is pending the GPU, after B2. *(Measured later: see the G7 dual-path finding and DECISIONS
2026-10-23.)*

**G7 decision (2026-10-23, orchestrator under the M0–M7 goal; flagged for owner review).**

- No engine passes all four criteria. C1 and C3 fall short for every candidate; C2 is pending; C4
  passes.
- **Engine:** WebRTC AEC3 via `pywebrtc-audio` 0.2.0:
  - `EchoCanceller(sample_rate=16000, num_channels=1, stream_delay_ms=0)`, one instance per session;
  - AEC only, with no 3.8 kHz pre-filter;
  - `livekit` 1.1.21 as a configurable fallback engine.
- **Alignment:**
  - the AICC playout model, with a 60 ms reference lead;
  - the 20 ms tick phase, learned from GCC-PHAT echo-onset estimates;
  - a GCC-PHAT guard that re-aligns when the residual delay leaves [5, 400] ms, with
    `max_residual_delay_ms` 450.
- **Second line of defence:**
  - `barge_in.echo_guard_ms` 2000 (AEC3 start-up p90 is 1.7 s);
  - `min_speech_ms_while_playing` 400;
  - echo-turn suppression on.

  With the tick phase known, these leave 1 false start in 96 simulated calls; with it learned, 3.
- **Hazards for M4:**
  - livekit aborts the process on frames that are not a whole 10 ms;
  - pywebrtc silently zero-pads partial frames;
  - AEC3 delays capture by a fixed 8 ms, forces a high-pass filter, adds comfort noise, and suppresses
    near-end speech during double talk.
- **Pending:** ΔCER decides whether ASR should get the AEC output (the default) or a less-processed
  signal.
- **Not simulated:** clock drift between AICC's ticker and the uplink clock.

### S0 / S8.1 — Environments, models, smoke checks, SDK probe (A1, 2026-10-23)

**Environments.** uv 0.13.5, CPython 3.12.12, `spikes/{llm,audio,sdk}`, each with its own uv.lock.
Versions: mlx 0.32.4, mlx-lm 0.32.2, mlx-audio 0.6.1, transformers 5.20.2, openai 3.36.1.

**Models.** All 8 are downloaded and sha256-verified against the Hub (51.5 GB). Revisions are in
`spikes/results/models.json`:

- Qwen3.6-35B-A3B-4bit `38740b84`
- Qwen3.5-35B-A3B-4bit `1e20fd8d`
- Qwen3-ASR-1.7B-8bit `a8379a2e`
- Qwen3-ASR-0.6B-8bit `89e96d92`
- Qwen3-TTS-0.6B-CustomVoice-8bit `049ef77f`
- Qwen3-TTS-0.6B-CustomVoice-4bit `08c72cad`
- Qwen3-TTS-1.7B-CustomVoice-8bit `41d3337e`
- Kokoro-82M-bf16 `a71e4d38`

**Download notes.**

- HF Xet downloads through the local proxy failed repeatedly, and huggingface_hub 2.2.0 restarts a
  failed Xet file from zero.
- Plain HTTP (`HF_HUB_DISABLE_XET=1`, same Hub and CDN) resumes with Range requests and ran at
  4–6 MB/s.

**Findings that matter for implementation.**

- `mlx_lm.server`:
  - `/health` answers 200 before the model is loaded, so readiness needs a warmup request;
  - its defaults are temp 0.0, top_p 1.0 and max_tokens 512;
  - `presence_penalty` is accepted;
  - per-request `chat_template_kwargs` override `--chat-template-args`.
- `mlx_audio.server`:
  - no preload flag; the first request loads the model, about 16–20 s from a cold page cache;
  - set `streaming_interval` explicitly: the default 2.0 gives a 2.4 s first chunk, 0.32 gives
    108–118 ms;
  - transcriptions default to ndjson;
  - the ASR `language` must be a model language name (`Chinese`, `English`); ISO codes end up in
    the prompt.
- Kokoro: voice names are resolved in a hardcoded repo (`prince-canuma/Kokoro-82M`). Offline that
  gives HTTP 200 with a truncated body, so pass the absolute path of the voice `.safetensors` inside
  our snapshot.
- Audio smoke: the Qwen3-TTS → Qwen3-ASR round trip was exact; Kokoro zh and en were transcribed
  exactly. All three audio models together use 5.6 GB.
- SDK probe: the official SDK (3.36.1) connects to `ws://…/v1/realtime?model=…` and sends no
  OpenAI-Beta header and nothing unprompted. All 15 scripted GA events parse into the expected
  classes. An unknown event type silently becomes `ConversationCreatedEvent`, so the gateway's tests
  validate every emitted event against the SDK models (done since M1).

### S3 / S4 / S9 — TTS, ASR, normalization (B2, 2026-10-23)

Report: `spikes/results/s3s4s9_report.md`. About 3000 TTS and 2700 ASR requests on stock
`mlx_audio.server` 0.6.1, with 0 HTTP errors.

**Measurement condition.** The machine is in Low Power Mode on AC power (`pmset` powermode 1). Under
sustained load the GPU steps down to ≈ 0.6× after 20–40 s; at 0.32 s chunks, TTFC goes 118 → 189 ms and
RTF 0.30 → 0.50. Each result records whether it ran capped. The owner was asked to switch Energy Mode
to High Power.

**TTS (Qwen3-TTS 0.6B 8-bit; warm, capped; TTFC p50 short / medium / long, RTF):**

| streaming_interval | TTFC | RTF |
|---|---|---|
| 0.16 s | 117 / 119 / 131 ms | 0.57 |
| 0.32 s | 186 / 187 / 200 ms | 0.51 |
| 0.8 s | 394 / 395 / 408 ms | 0.47 |

- **HTTP overhead** over in-process: +2.0 ms median.
- **Cancel:** the server sees the disconnect within 0.3 ms; generation stops 41 / 75 / 241 ms after
  the close at 0.16 / 0.32 / 0.8 s chunks; nothing is sent after the close.
- **Concurrency:** streams are served strictly one at a time. With two concurrent requests the second
  has a TTFC of 2.9 s; with four, the last waits 8.4 s.
- **Warmup:** the model loads at warmup and stays resident (2.77 GB) across idle gaps of up to 120 s.
- **Silence:** every output has ≈ 0.45 s of leading and ≈ 0.5 s of trailing silence.
- **Voice defect:** the slow-speech / runaway defect (mlx-audio#1002) follows the voice `ryan` on
  every model: 5.6% of outputs are longer than 1.5×, up to 20×. vivian 1/590, aiden 2/66, serena
  0/66.
- **Comparison models at 0.32 s:** 0.6B 4-bit 178 ms / RTF 0.48; 1.7B 8-bit 250 ms / 0.65.

**Kokoro vs Qwen3-TTS** (20 sentences per set, back-transcribed):

| Set | Kokoro error | Qwen3-TTS error |
|---|---|---|
| zh | 0% | 0% |
| en | 0% | 0% |
| mixed zh/en | 15.9% (18/20 sentences wrong; English words garbled) | 2.2% |

**ASR (Qwen3-ASR 1.7B 8-bit).**

- Latency p50 with language auto / given: 2 s clips 184–221 / 158–189 ms; 5 s clips 311–350 /
  275–304 ms; 10 s clips 446–521 / 403–499 ms. The 0.6B is about 2× faster.
- Error rate on rows without numbers: zh CER 5.9%, en WER 2.4%, mixed MER 10.3% (0.6B: 6.2 / 2.8 /
  12.7). Telephone copies cost at most 1.4 points.
- Pitfalls:
  - when the model omits its language tag, auto language falls back to "English" (7/42 zh clips,
    text still correct);
  - a wrong language hint can make the model translate;
  - any language string is echoed back.

**Normalization probe (S9; correct readings out of runs, raw text into Qwen3-TTS).**

| Category | Correct | Note |
|---|---|---|
| numbers | 15/18 | "15600" is always read as 1560 |
| money | 18/18 | |
| dates and times | 17/18 | |
| phone numbers | 8/21 | |
| IDs | 12/18 | |
| units | 17/18 | |
| mixed zh/en | 15/18 | the "@" in e-mail addresses is never read |

English segments fail on IDs, numeric dates and amounts of 1000 or more.

**Decisions (2026-10-23, orchestrator under the M0–M7 goal).**

- **G1: keep `mlx_audio.server` for ASR and TTS.**
  - It passes the criteria: 2 ms overhead, reliable warmup and residency, cancellation within one
    chunk, no errors. S6 still checks stability.
  - Serialized TTS streams are a capacity limit for concurrent sessions, scheduled for M6: multiple
    TTS processes, or a batched worker on mlx-audio's streaming `batch_generate`, plus the
    reserve/EDF mitigations.
- **G5: normalize only:**
  - phone numbers in every form (digit by digit, 1 as 幺, including 4-digit 尾号 and extensions);
  - alphanumeric and dashed IDs;
  - e-mail addresses;
  - cardinals of 5 or more digits;
  - in English segments: IDs, numeric dates, and amounts of 1000 or more.

  Money, decimals, percentages, ordinals, dates and times, units, URLs and brand names stay raw.
- **TTS defaults:**
  - `streaming_interval` 0.16 (TTFC ≈ 119 ms capped, ≈ 76 ms uncapped). Fall back to 0.32 if S5 shows
    RTF near 1 under LLM contention.
  - Speakers: vivian for zh and mixed, aiden for en. Never ryan.
  - Cap `max_tokens` in proportion to the text, against runaways.
  - Trim each segment's leading and trailing silence. The 0.45 s lead otherwise adds directly to first
    audio, and edges add ≈ 1 s between sentences.
- **D13 (flagged for owner review): keep Kokoro**, as the brief names it, but only for segments without
  Latin letters. A segment with Latin letters fails rather than being garbled. Config can switch the
  fallback to Qwen3-TTS 0.6B 4-bit (`family: qwen3_tts`, supported since M3) if the owner approves
  the substitution. *(Superseded 2026-10-23: the owner chose the Qwen3-TTS 4-bit fallback; Kokoro
  was removed.)*

**S11 double-talk ΔCER** (B4, 2026-10-23; points of error rate over the clean telephone chain, zh CER /
en WER, 80 utterances per cell, continuous double talk; raw: `spikes/results/s11_cer.json`):

| Canceller | zh 0 dB | zh −15 dB | en 0 dB | en −15 dB |
|---|---|---|---|---|
| none | +13.0 | +68.1 | +5.7 | +61.4 |
| AEC3 (pywebrtc) | +2.0 | +10.9 | +4.9 | +28.4 |
| AEC3 (livekit) | +1.2 | +8.4 | +2.6 | +20.4 |
| Speex linear + GCC-PHAT | +0.9 | +0.6 | +0.7 | −0.8 |
| Speex + preprocessor + GCC-PHAT | +0.1 | +0.2 | +0.6 | −0.6 |

AEC3 fails C2, because its suppressor removes quiet caller speech. Speex passes C2 but fails C1, because
its residual triggers the VAD. Continuous double talk overstates the barge-in case: our output stops
≈ 0.5 s after a barge-in, so only the start of the caller's turn overlaps the echo.

**G7 revision (2026-10-23, flagged for owner review).** The plan is a dual path: AEC3 output drives
Silero and Smart Turn, and a cleaner path feeds the InputBuffer and the ASR. Candidates for that path:
Speex linear or with its preprocessor (only if it can be built reproducibly through uv, e.g. vendored
BSD sources compiled as a project extension), or raw audio. A follow-up task after M5 (effort max)
measures the ASR input on barge-in-realistic double talk and implements it. Until then,
`aec.asr_input` stays `processed` (AEC3). *(Superseded 2026-10-23: the owner chose the dual path;
`aec.asr_input` is `speex`, and `processed` was renamed `aec3`.)*

### S1 / S2 — LLM behaviour and performance (B1, 2026-10-23)

Report: `spikes/results/s1s2_report.md`. Conditions: AC power mode was Automatic (powermode 0); no GPU
step-down was seen (prefill 920–1140 tok/s throughout). An external Go stress test loaded the CPU
for part of the run; every result records the load.

**Thinking.** Off in 20/20 replies on both models; per request it switches back on.

**Tool calls** (20 prompts per run, AICC's 6 tools):

| Sampling | Qwen3.6 | Qwen3.5 |
|---|---|---|
| temp 0.7 | 16–19 | 17–20 |
| temp 0 | 19, 19, 19 | 19, 19, 19 |

**Hard defect in Qwen3.6.** 11% of realistic transfer turns end with `finish_reason: tool_calls` and
no parsed call: the model writes `<transfer_to_agent>` instead of `<function=transfer_to_agent>`.
Rewriting that form for declared tools recovered 7/7 dropped calls offline. Qwen3.5 does not do this,
but it silently says "好的，我查一下" and never calls (6/40 lookups).

**LLM latency, Qwen3.6** (TTFT p50 / p95 s; first segment boundary p95):

| Case | TTFT | Boundary p95 |
|---|---|---|
| warm, a turn appended | 0.14 / 0.18 | 0.33 |
| cold start | 1.49 / 1.63 | 1.72 |
| instructions-only change | 1.50 / 1.64 | 1.84 |
| new per-response override | 0.24 / 1.15 | 1.22 |

- **Cache warming after an instructions change** cuts the real request's TTFT from 1.79 to 0.15 s, but
  only if it finishes first (≈ 1.7 s). Racing it makes things worse.
- **Tool-only `transfer_to_agent`** to `[DONE]` takes 1.41 / 1.68 s warm, and 2.65 / 3.12 s (max 3.69)
  after an instructions change.
- **Concurrency:** streams are batched (58 tok/s each at 2, 39 at 4). A new request can wait up to
  0.5 s behind a decoding stream.
- **Cancellation** stops at the next write, but not inside a tool-call span, and up to 2 more prefill
  steps run during a long prefill.

**G2 decision (2026-10-23).**

- **Model:** `mlx-community/Qwen3.6-35B-A3B-4bit` @ `38740b84`, with a lenient tool-call parser in
  `backends/llm` (rewriting `<NAME>` to `<function=NAME>` for declared tools) and one gateway retry on a
  dropped call.
- **Server:** `--chat-template-args '{"enable_thinking":false}' --max-tokens 2048 --temp 0.7 --top-p 0.8
  --top-k 20 --min-p 0.0`.
- **Per request:** temperature 0 and top_p 1.0 whenever tools are attached; `max_completion_tokens`
  512; no presence_penalty.
- **Gateway follow-ups:**
  - the empty-conversation cue becomes "(The call has connected. Greet the caller as instructed.)";
  - filter JSON-like pseudo-calls out of replies sent without tools;
  - warm the cache after an instructions change, and make the real request wait for the warm-up;
  - a larger `--prompt-cache-size`.

**G4, LLM part — AICC's windows are at risk.**

- First audio after a cold start, an instructions change or a new override: ≈ 2.6 s p95 before GPU
  contention.
- A tool-only transfer after an instructions change misses the 3 s window (p95 3.12 s).
- An LLM-hardening task (effort max, per the owner's rule) addresses these before S5/S6/S10.
- **Outcome:** with the hardening (`mlxrt_llm.serve`, prefix warming, cache checkpoints) every AICC
  case is inside the 3 s windows at p95, one call at a time; the tightest is a tool-only transfer,
  2.17–2.58 s with ASR (`docs/PROGRESS.md`, LLM hardening). The whole of G4 is in the last finding
  below.

### S5 / S6 / S10 — Contention, stability, budget: G4 and G6 (real-stack validation, 2026-10-23/24)

Run through the real gateway and backends (High Power on AC, Qwen3.6 @38740b84, Qwen3-ASR @a8379a2e,
Qwen3-TTS 8-bit @049ef77f and 4-bit @08c72cad). Details: `docs/PROGRESS.md`, "Real-stack validation";
tables: `docs/LATENCY.md`; raw results: `spikes/results/realstack_*`.

**S5, GPU contention** (`spikes/realstack/contention.py`, LATENCY.md §3.7). While the LLM decodes one
or three streams, the TTS's first chunk goes from 56 to 71–81 ms (P50) and its RTF from 0.25 to
0.33–0.38; the ASR from 172 to 194–215 ms. While the TTS streams, the LLM decodes 12 % slower
(77 → 68 tok/s). While the LLM prefills ~16k tokens, the TTS runs at RTF 2.1; AICC's prompts are
~1.6k tokens. RTF stays well below 1 under decode contention, so `streaming_interval` stays 0.16;
it exceeds 1 only during a long prefill, which AICC's prompt sizes do not cause, and
`--prefill-step-size` was not tried.

**S6, stability** (`spikes/realstack/soak.sh 1800`). Three sessions back to back for 30 minutes
through `make dev`: 45 sessions, 378 utterances, 625 TTS requests (fewer than the planned 1000: the
answers are one sentence). No hang, no crash, no backend error status, no IOGPU or GPU kernel message;
7 responses failed with `backend_busy` (the TTS queue timeout). Memory was flat after the LLM's prompt
cache filled in the first 5 minutes, with no swap. Each backend killed with `kill -9` and restarted
recovers on its own; the gateway warms it again (`/readyz` 200 after 1.7 s for the ASR, 10.4 s for
the TTS, 3.0 s for the LLM).

**S10, budget** (`bench/` on 126 utterances, LATENCY.md §3–§4). Server end to end, speech end →
first audio, on turns that end on a complete Smart Turn verdict: 1.37–1.39 / 1.52–1.54 s (P50 / P95)
at one session, 1.42–1.47 / 1.86–2.78 s at two and three. The tails at two and three sessions are the
serialized TTS: a first segment waited up to 1.9 s for another session's segment.

**G4 decision (2026-10-24; proposed, awaiting the owner's approval).**

- **Budget** (P95 per stage from speech end, AICC's 500 ms silence): VAD 550 ms, Smart Turn 100 ms
  (up to `max_silence_ms` when it holds a turn), ASR 400 ms, LLM first token 500 ms warm, first
  sentence 250 ms, TTS first chunk 150 ms alone (2 s when shared), first audio sent 200 ms.
- **Targets** (speech end → first audio, complete verdicts): one call P50 ≤ 1.5 s / P95 ≤ 1.8 s, met;
  up to three calls P50 ≤ 1.6 s / P95 ≤ 2.5 s, P50 met, P95 not met at three calls (2.53–2.78 s).
- **AICC's floor:** one call holds it with margin (`response.created` → first audio 0.94–0.97 s at
  P95; ~2.4 s with a full re-prefill after an instructions change). Three calls do not hold it
  reliably: one first audio at 3.39 s, 7 `backend_busy` failures (1.8 % of the three-session
  responses), and a tool-only transfer at 2.78 s beside two talking sessions.
- **Plan change:** the three-call target needs more TTS capacity (a second TTS process, or batched
  generation) or shorter segments; an owner decision (`docs/DECISIONS.md`, 2026-10-24).

**G6 decision (2026-10-24): keep three model processes.** The soak showed no instability, so the
rule proposes no merge, and recovery works per process. A merge would not widen the TTS stream,
which is the capacity limit. The fallback half of G6 was settled by the owner on 2026-10-23: the
4-bit Qwen3-TTS build replaces Kokoro (D13).
