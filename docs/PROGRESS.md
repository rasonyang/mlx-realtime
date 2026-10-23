# Progress

One section per milestone (ARCHITECTURE §13): what was built, how to verify it, test results, the
gaps left to later milestones, and every deviation from ARCHITECTURE with its reason.

## M1 — Protocol skeleton and mock providers (2026-10-23)

**Goal.** The GA protocol end to end with mock providers (an echoing LLM, a sine-wave TTS, a scripted
ASR), driven by the official `openai` SDK. Landed with it: D1, D2, D12, D18, and `docs/PROTOCOL.md`.

### What was built

The structure of ARCHITECTURE §3, filled to the M1 level. Everything in `src/` passes pyright strict.

| Package | Contents |
|---|---|
| root | `pyproject.toml` (uv project, ruff, pyright, pytest with the `e2e` marker), `uv.lock`, `.python-version` (3.12), `config.yaml`, `Makefile` (`fmt`, `lint`, `typecheck`, `test`, `test-e2e`, `dev`), `README.md` |
| `config.py` | YAML, then `MLXRT__SECTION__KEY` overrides parsed as YAML; strict pydantic validation; every error names its dotted path and, when an override set it, the variable |
| `logs.py` | JSON lines on stdout through stdlib logging (uvicorn included); `session_id`, `turn_id`, `response_id`, `item_id` from context variables that tasks inherit |
| `server/` | Starlette app: `/v1/realtime`, `/healthz`, `/readyz`; Bearer auth (HTTP 401 before the upgrade); readiness (warmup, then `not_ready` → 1013 until first ready); admission (`session_limit_reached` → 1013); per connection one TaskGroup of reader, single writer, actor and a lifetime watch; bounded outbox with producer backpressure; graceful shutdown with 1001 |
| `protocol/` | decoding and validation of every GA client event into commands, with GA errors and dotted `param` paths; encoding of facts into GA server events in the documented orders; ids; the session echo |
| `session/` | the actor (synchronous handlers), conversation, input buffer (24 → 16 kHz), turn manager (§6.2 state machine), response state, per-response pipeline, cancellation handler (§6.5), truncation arithmetic (§6.6), the delivery ledger |
| `providers/` | `base.py` as §5.1 with the §5.2 contracts (`ProviderError` kinds, `ProviderTimeout`); a registry; `EchoLLM`, `SineTTS`, `ScriptedASR`, whose every await point can be delayed, held, observed or made to fail |
| `audio/` | `resample.py`, the only resampler (soxr streams on float32, so chunked equals one-shot); `codec.py`, wire PCM to canonical PCM and back in bounded deltas |
| `text/` | the §7 segmenter (zh/en, first-segment rules, `min_chars`/`max_chars`, never inside numbers, URLs, e-mail addresses or IDs); a pass-through normalizer; the language tag |
| `vad/` | the speech-detector and end-of-turn-detector interfaces; scripted implementations |

### How to run and verify

```sh
make dev                                   # the gateway with mock providers on 127.0.0.1:8000
curl localhost:8000/readyz                 # 200 once the mocks are warm
make lint typecheck test test-e2e          # every check of the done criteria
uv run --locked pytest                     # everything at once
```

`README.md` has an SDK snippet that talks to `make dev`. The e2e suite also starts `mlxrt dev` as a
subprocess with the repository's `config.yaml`, drives a spoken turn with the SDK, and stops it with
SIGINT.

### Test results

All green on 2026-10-23: **459 passed** in about 17 s (`uv run --locked pytest`). `ruff check`,
`ruff format --check` and `pyright` (strict on `src/`, standard on `tests/`) report nothing.

| Suite | Tests | What it covers |
|---|---|---|
| unit | 391 | decoding and validation: 115 rejections, each with its `param` path; every encoded event parsed by the SDK's own pydantic model, with no missing field and no field the SDK does not know; config and env overrides; segmenter; resampler (chunked = one-shot, three rate pairs, four chunk sizes); input buffer; turn manager with scripted detectors; truncation; merge rules; inbox, outbox, ledger; logging; the import graph (98) |
| integration | 64 | 10 golden traces; causal invariants on every golden trace and on a long session with interleaved audio, transcript, a tool call, a cancel and a barge-in; the cancellation matrix (7 phases × 5 triggers); the 5 AICC traces replayed; transport and lifecycle (health, readiness, auth, admission, input limits, frame size, a client that stops reading, maximum duration, shutdown, crash isolation) |
| e2e (`-m e2e`) | 4 | the official SDK: a GA `session.update`, a Chinese `input_text` item, `response.create` with sine audio and echoed transcript to `response.done{completed}`; a text turn; `response.cancel` mid-audio; `mlxrt dev` with `config.yaml` |

The integration client checks every event it receives against the SDK's model, so all tests that talk
to the server also check the wire shapes.

### Known gaps, by milestone

| Milestone | What M1 leaves for it |
|---|---|
| M2 | Silero VAD and Smart Turn behind the `vad/` interfaces, on CPU executors, with the VAD feeder task (M1 calls scripted detectors synchronously in the actor); `MlxLmChatLLM` and `MlxAudioASR`; deadlines (`connect`, `first_item`, `idle`, `total`) in the HTTP providers; the synthetic cue for an empty conversation (the mock LLM greets instead); the LLM message adaptation of §5.2 (one leading system message, mid-conversation system items folded, tool calls merged into one assistant message) |
| M3 | `MlxAudioTTS` and `FallbackTTS`; pacing at most `max_lead_ms` ahead of real time (D6; M1 only bounds the outbox); verbatim lines (D16); mapping an unknown voice to the default and echoing the effective voice (D19 needs the TTS's voice catalog); first AICC acceptance with its `probe` harness and the 3 s deadlines |
| M4 | echo cancellation and its second line of defence (D17); the truncation fallback when the client sends none (D5: M1 keeps the text sent); cancellation tests against real backends |
| M5 | real tool calls from `mlx_lm.server`'s parser; `tool_choice` enforcement in the provider (`none` sends no tools); `required` and forced functions; history replay checked against the Qwen template. M1 already delivers complete calls a provider makes, with the §6.7 events |
| M6 | turn timeline marks, `/metrics`, the benchmark, `/debug/turns` |
| M7 | `audio/pcmu` and `audio/pcma` (8 kHz G.711) |
| not planned | `semantic_vad`, `noise_reduction`, `idle_timeout_ms`, `conversation.item.retrieve`, client-created `function_call` items, out-of-band responses, MCP |

Smaller gaps:

- After a slow-client failure the session's tasks end at once, but uvicorn keeps the socket open until
  the peer goes away: ASGI offers no way to abort a connection whose send buffer is full.
- `conversation.item.input_audio_transcription.delta` is never sent: the ASR returns final transcripts.

### Deviations from ARCHITECTURE, with reasons

1. **Inbound task results are "signals"** (`session/signals.py`); "facts" are what the actor reports
   to the protocol layer (`session/facts.py`). §4.2 calls both facts; one name per direction keeps the
   two apart in the code.
2. **The session scope has a fourth task, a lifetime watch** (shutdown, `max_session_s`), so every way a
   session ends is an exception in one TaskGroup. The VAD feeder of §4.2 arrives with Silero in M2; in
   M1 the scripted detectors are synchronous and run inside the actor's step.
3. **Outbound backpressure** (§4.4, "writer blocks"): producers of audio and text deltas wait while half
   of `outbound_queue_events` are queued, so a fast pipeline never trips the bound. A backlog over the
   bound, or a write stalled beyond `client_write_timeout_ms`, closes the session with 1008 and code
   `slow_client` (the brief's rule; §4.4 only names the stall).
4. **D7 for a client `response.create`.** §6.3 says either trigger creates the response at once and an
   empty transcript completes it. For a client request, the response answers the conversation's last
   item when that is a user audio item, so an empty transcript there also completes it empty. An
   earlier noise item never silences a later request.
5. **`item_not_found` on `conversation.item.truncate` of an unknown item**, as D18 prescribes for
   delete, so both item lookups fail alike.
6. **Truncation applies to assistant messages with audio only**; a text-only assistant message gets
   `invalid_value` (`param: "item_id"`). GA truncates "a previous assistant message's audio".
7. **On a provider failure the `error` event precedes the response's terminal events.** §8 fixes no
   order; this way AICC ties the error to the response it is waiting on. An ASR failure sends
   `conversation.item.input_audio_transcription.failed` and, when it fails a response, the `error`
   too, as every failed response does.
8. **Function-call items are emitted already in M1**, for complete calls a provider makes (the mock
   is scripted). AICC's traces answer calls with `function_call_output`, and the replay needs a call id
   the server minted. `tool_choice` enforcement and real parsing stay in M5.
9. **The response flags are `awaiting_transcript`, `llm_done` and `tts_done`.** `audio_sent` (§6.3) has no
   use before pacing: `response.done` is queued behind the response's audio. It comes back with D6 in
   M3 if pacing needs it.
10. **`websockets` is pinned below 16** (runtime dependency `websockets>=15.0.1,<16`): openai 3.36.1
    with the `realtime` extra requires `websockets<16`, and both resolve in one lockfile.
11. **`session_defaults` is validated by the protocol layer when the gateway is built**, not by the
    config models, so a default can never hold a value a `session.update` would reject. Startup still
    fails with the dotted path (`session_defaults.audio.output.speed`).
12. **Server event ids are `event_<n>`**, counting up per session.
13. **Segmenter details** beyond §7: a CJK full stop at the end of the streamed text ends a segment at
    once (a closing quote that arrives later opens the next segment; waiting for it would cost a token
    of latency); `1.` at the start of a line, initials and common abbreviations are not sentence ends.
14. **Config keys added** to the §10 sketch: `limits.inbox_max_commands` (the "+256 commands" of §4.4),
    `turn.smart_turn.window_s` and `threshold`, `turn.vad.type: mock`, and the mock providers' delays.

### Suggested changes to ARCHITECTURE and DECISIONS

- §4.2: name inbound task results "signals"; list the lifetime watch among the session tasks.
- §4.4: replace "writer blocks" with the rule above (producers wait at half the bound; over the bound
  or a stalled write → 1008 `slow_client`).
- §6.3: state the D7 rule for a client `response.create` (deviation 4).
- §8 / D18: add `item_not_found` for truncate, and the server codes `slow_client` and `internal_error`
  with their close codes; record that a provider failure's `error` precedes `response.done`.
- §5.1: `Message`, `ToolDef`, `ToolChoice`, `Usage`, `Readiness`, `Lang` are now defined in
  `providers/base.py`; the sketch can point there.
- §7 / D19: mapping unknown voices needs the TTS's voice list; suggest a `voices` attribute on the TTS
  provider (or a voice map in `config.yaml`) for M3.
- §3 or `CLAUDE.md`: note that ruff 0.17 formats Python code blocks inside Markdown by default; the
  project's ruff `include` keeps it to Python files, so `make fmt` never rewrites the docs.
- §10: add the keys of deviation 14.

## M2 — VAD, Smart Turn, ASR and LLM (2026-10-23)

**Goal.** The speech-to-transcript path: client audio → Silero VAD and Smart Turn → Qwen3-ASR →
the LLM, with the providers of ARCHITECTURE §5 against the real backends. Landed with it: D4, D7,
D15, and the synthetic cue for an empty conversation. The TTS stays a mock until M3, so real answers
are read with `output_modalities: ["text"]`.

**Not yet run on the real backends.** M0's GPU measurements held the GPU throughout M2, so every
provider was developed against fake servers that copy the backends' wire format, and the tests
against the real backends (marker `backend`) are written but have not been run (see "Pending").

### What was built

| Area | Contents |
|---|---|
| `backends/llm` | uv project (`uv init --no-workspace --bare`, Python 3.12, own `uv.lock`): `mlx-lm==0.32.2`, `huggingface-hub` (its `hf` CLI) |
| `backends/audio` | uv project: `mlx-audio[server]==0.6.1` and what its Qwen3-ASR, Qwen3-TTS and Kokoro paths needed in M0 (`misaki[zh]==0.9.4`, `spacy==3.8.16` with `en_core_web_sm` 3.8.0, `num2words`, `phonemizer-fork`, `espeakng-loader`), `huggingface-hub` |
| `config.yaml` | `backends`: per process the uv project, command, host and port, args (`--chat-template-args '{"enable_thinking":false}'`; max tokens and sampling are G2 placeholders), the served model pinned by repo and revision (`model`) and the models requested by name (`fetch`), and `env` (`HF_HUB_OFFLINE=1`). Providers `mlx_lm_chat` and `mlx_audio_transcription` with their deadlines; Silero and Smart Turn (G3 defaults: fp32 file, 2 intra-op threads, 2 workers). `config.mock.yaml` keeps the M1 setup |
| `cli.py`, `supervisor.py` | `mlxrt dev`: starts each backend with `uv run --locked --project backends/<role> …` (the served model as its local snapshot directory), then the gateway; prefixes output lines with the process name; forwards SIGINT/SIGTERM; stops everything when any child exits (kills a process group still alive after a grace period); refuses to start while a configured port is taken |
| `fetch.py`, `hub.py` | `mlxrt models`: downloads the ONNX files of `models/MANIFEST.json` (sha256 from M0's manifest) and verifies them; `--mlx` downloads the pinned MLX repositories by revision through each backend's `hf` CLI. `HF_TOKEN` goes to huggingface.co only, never to the output. `hub.py` reads the Hugging Face cache layout |
| `vad/` | `silero.py`: pipecat's numpy wrapper (512 + 64 samples at 16 kHz, state [2, 1, 128] per session, pipecat's periodic state reset); `whisper_features.py`: pipecat's numpy log-mel, unchanged; `smart_turn.py`: last ≤ 8 s, front-padded, features, probability; `gate.py`: hysteresis (enter at `threshold`, leave below `threshold − 0.15`), `min_speech_ms`, the silence windows; `executors.py`: one single-thread executor for Silero, K workers with one ONNX session each for Smart Turn; `onnx.py`: the typed boundary to onnxruntime. pipecat's BSD-2 notice and the transformers Apache-2.0 notice are kept |
| `session/` | `feeder.py`: the per-session VAD feeder task (framing, Silero on the executor, the gate, one `VadUpdate` signal per processed chunk); `turn.py`: TurnManager driven by gate events and `TurnVerdict` signals; the actor: VAD epochs (a commit, clear or `turn_detection` change makes late detector results stale), Smart Turn evaluations as tasks, idle trimming on the detector's progress; `timeline.py`: the turn timeline |
| `providers/` | `mlx_lm_chat.py` + `chat_wire.py` + `sse.py`: the LLM provider; `mlx_audio_asr.py`: the ASR provider; `http.py`: deadlines (`connect`, `first_item`, `idle`, `total`), the in-flight limit (`overloaded`), error kinds; the registry builds the configured providers |
| `server/` | the gateway builds the providers and the VAD runtime from the configuration; readiness warms every backend at once (the CPU detectors as `vad`), retrying each with exponential backoff (0.5 s to 5 s) |
| `models/MANIFEST.json` | the three ONNX files: source, revision, license, URL, size, sha256 |

**The LLM provider** (`mlx_lm_chat`):

- Requests: `stream: true`, `stream_options: {include_usage: true}`, `chat_template_kwargs:
  {enable_thinking: false}`, `max_completion_tokens` (the configured cap, or the response's lower
  `max_output_tokens`), `temperature` and `top_p` from the configuration, no `seed`. `model` is left
  out: the server runs on the pinned snapshot (a configured `model` must then be omitted; the
  configuration refuses both).
- Messages for the Qwen template: the leading system messages become one; a later system item becomes
  a user message `(System note: …)`; adjacent assistant messages and calls merge into one message; a
  conversation without a user message gets the cue `(The conversation has just started.)` (both
  configurable).
- `tool_choice: none` sends no tools; `auto` sends the session's (`required` and forced functions are
  rejected by the protocol layer, as in M1).
- The stream: `data:` events, `: keepalive` comments skipped (they are not tokens), `[DONE]`; text
  deltas; reasoning dropped (and counted); tool calls checked to be complete (id, name, arguments a
  JSON object) or the response is `bad_response`; `finish_reason: tool_calls` without a call is
  `bad_response`; the usage chunk, `cached_tokens` included.
- Readiness: `GET /health`. Warmup: a one-token completion with the session defaults' instructions.

**The ASR provider** (`mlx_audio_transcription`): `POST /v1/audio/transcriptions`, multipart, the
turn as a 16 kHz mono 16-bit WAV, `model`, `response_format: verbose_json`, and `language` mapped
from the session's ISO code to Qwen3-ASR's language name (`zh` → `Chinese`); the reported language
(a list of names) comes back as a code. Empty audio is not sent. Readiness: `GET /v1/models` lists the
model. Warmup: a one-second clip, which also makes the server load the model.

**Turn detection** follows §6.2 and D4: one `speech_started` / `speech_stopped` pair per turn; after
`silence_duration_ms` (session default 200) Smart Turn judges the turn; an incomplete verdict holds it
until speech resumes (same turn) or `max_silence_ms` (3 s) passes; without Smart Turn the silence
ends the turn. PROTOCOL.md §11.9 has the client's view.

**The turn timeline**: `speech_end` (when the audio at the acoustic end arrived), `speech_stopped` (the
gate's silence window closed), `turn_confirmed` (the commit), or `committed` for a manual commit, then
`asr_done` and `llm_first_token`; one `turn timeline` log line per turn, in milliseconds from the first
mark, when its response ends (or with its transcript if no response answers it).

### How to run and verify

```sh
make models                                # the ONNX models into models/, verified
make models-mlx                            # the MLX repos pinned in config.yaml (~25 GB; M2 did not run it)
make dev                                   # backends + gateway (config.yaml)
make dev-mock                              # the gateway alone, mock providers (config.mock.yaml)
uv run --locked pytest                     # everything but the backend tests
uv run --locked pytest -m backend          # the real backends; run it with `make dev` up
make lint typecheck
(cd backends/llm && uv sync --locked) && (cd backends/audio && uv sync --locked)
```

`mlxrt models` ran on 2026-10-23: the three files downloaded (83 s through the slow route) and
verified against the manifest.

### Test results

All green on 2026-10-23: **601 passed, 4 skipped** (the `backend` tests) in about 45 s
(`uv run --locked pytest`). `ruff check`, `ruff format --check` and `pyright` (strict on `src/`)
report nothing; both backend projects pass `uv sync --locked`.

| Suite | Tests | New in M2 |
|---|---|---|
| unit | 528 | the speech gate and TurnManager (14, the M1 scenarios rewritten for the split, plus long silence and window cases); the feeder (5: chunking does not change frames, the gate state reaches the detector, epochs, detection off, the backlog bound); Silero and Smart Turn (10: context and state threading, pipecat's reset timing, the 8 s front-padded window, feature parity with `transformers.WhisperFeatureExtractor(chunk_length=8)` below 1e-6, the real models on the fixtures, the executors, and the TurnManager on real Silero over two fixtures with and without Smart Turn); the LLM provider against a fake `mlx_lm.server` (30: request shape, keepalives, usage, reasoning dropped, tool calls, five malformed streams, a cut stream, error statuses, every deadline, cancellation while awaiting headers, before the first token, between tokens, while the consumer holds a delta, and while queued for a slot, the in-flight limit, readiness, warmup); the chat wire format and SSE (15); the ASR provider against a fake `mlx_audio.server` (18); the configuration (24); the supervisor (8, with Python children); `mlxrt models` (7); the timeline (2) |
| integration | 69 | `test_speech_turns.py` (5): 24 kHz audio of the fixtures through real Silero and Smart Turn and the real providers on fake backends: a `server_vad` turn with every event in order, an incomplete verdict then resumed speech in one turn, the `max_silence_ms` fallback, an empty transcript (no LLM call), and a barge-in on a running response |
| e2e (`-m e2e`) | 4 | the `mlxrt dev` test now uses `config.mock.yaml` |
| backend (`-m backend`) | 4 | written, not run: see below |

The fake servers copy the real ones: the fake `mlx_lm.server` runs on `http.server` with HTTP/1.0
as the real one does, writes its chunks in the shapes of `generate_response` and
`completion_usage_response`, and sees a client's disconnect; the fake `mlx_audio.server` runs on
uvicorn and answers `verbose_json` as M0's smoke test recorded it.

Speech fixtures: two FLEURS sentences (CC-BY-4.0), trimmed, 16 kHz; provenance in
`tests/fixtures/README.md`.

### Pending the GPU runs

- `uv run --locked pytest -m backend` with `make dev` running: `test_asr_transcribes_the_fixtures`,
  `test_llm_streams_text_with_thinking_off` (no `reasoning` delta on the wire either),
  `test_llm_makes_a_tool_call`, `test_sdk_server_vad_turn_with_the_real_stack` (the official SDK streams
  a fixture in real time with `server_vad` and gets the transcript and a text answer).
- `make models-mlx` and a first `make dev` with the real backends.
- G2: the LLM repository and revision (the configuration pins the Qwen3.6 revision M0 downloaded, as a
  placeholder) and the sampling and length values.

### Known gaps, by milestone

| Milestone | What M2 leaves for it |
|---|---|
| M3 | the TTS provider (the `tts` backend already starts with `mlxrt dev`; Kokoro's pipeline also looks up `prince-canuma/Kokoro-82M` on the Hub, which M0's smoke test saw fail offline) |
| M4 | the echo canceller (D17); barge-in and cancellation tests against the real backends |
| M5 | tool calls replayed through the Qwen template against the real server (M0 S1.3's history shapes) |
| M6 | the timeline marks become metrics; `first_sentence_ready`, `tts_first_chunk`, `first_audio_sent`; batched Silero calls if loop lag shows |

### Deviations from ARCHITECTURE, with reasons

1. **The feeder sends one `VadUpdate` per processed chunk** (the gate's events plus the earliest
   position a turn not yet ended can start at), not separate start and end signals. The actor trims
   the idle input buffer only from these updates, in the inbox's order, so it can never trim audio of
   a turn whose start is still on its way.
2. **The gate measures `max_silence_ms`** and reports it as a `LongSilence` event: the TurnManager no
   longer sees frames. A pending speech onset defers it.
3. **Resumed speech needs `min_speech_ms`** like any onset (M1: one frame), so a click cannot reset a
   held turn.
4. **Smart Turn sees the turn from `prefix_padding_ms` before the onset to where the silence window
   closed**, trailing silence included, at most 8 s: pipecat's and the model card's usage (§6.2 says
   "the turn's last ≤ 8 s").
5. **Model pinning.** The supervisor starts `mlx_lm.server` on the pinned snapshot's directory, so LLM
   requests leave `model` out. `mlx_audio.server` loads models by the name in each request, which
   resolves the local `refs/main` offline; `mlxrt models --mlx` and `mlxrt dev` warn when that is not
   the pinned revision. Sending a snapshot path as the model name would pin it exactly, but M0 has
   not tried that path.
6. **Configuration shape** (§10 sketch): `base_url` without `/v1` for both providers; deadlines as
   `connect_ms`, `first_item_ms`, `idle_ms`, `total_ms` and `warmup_ms` for both; `max_in_flight` and
   `queue_timeout_ms` (the §4.4 semaphore); `backends.<role>.model`, `fetch` and `env`;
   `turn.vad.type`, `model`, `model_reset_interval_s`; `turn.smart_turn.workers`, `intra_op_threads`.
7. **The feeder's backlog is bounded by `input_audio_buffer_max_ms`**, not `inbox_max_ms`: audio the
   detector has not seen stays in the input buffer anyway, and a client may send faster than real time.
8. **Provider streams are async generators** (`AsyncGenerator` in `providers/base.py`), and the
   pipeline closes them with `aclosing`, so a cancel that lands outside the provider still closes the
   backend request before the task ends.
9. **pipecat's Silero state reset is on by default** (`turn.vad.model_reset_interval_s: 2.0`, `null`
   turns it off), as the M2 brief asks; M0 (S7.4) found it adds a late-onset tail on 16 kHz audio.
10. **pipecat's VAD also gates on volume**; this gate uses Silero's probability alone.
11. **`mlxrt dev` starts the `tts` backend in M2** although the TTS provider is still the mock, and it
    refuses to start while a configured port is in use.
12. **HTTP client**: `httpx2` (the client openai 3.36 and huggingface_hub 2.2 use). `onnxruntime` 1.31.0
    joined the runtime dependencies, `transformers` the dev dependencies (the feature parity test, §11).
13. **M1 tests changed**: `test_turn_manager.py` rewritten for the gate/TurnManager split (every M1
    scenario kept); `test_config.py` (`config.yaml` now names the real backends; `config.mock.yaml`
    is checked instead); the `mlxrt dev` e2e test runs `config.mock.yaml`. Every golden trace is
    unchanged.

### Suggested changes to ARCHITECTURE and DECISIONS

- §4.2, §6.2: the feeder posts `VadUpdate` signals (deviation 1); the gate reports `LongSilence`
  (deviation 2); resumed speech needs `min_speech_ms` (deviation 3); the Smart Turn window
  (deviation 4).
- §5.1: `stream` returns `AsyncGenerator`; consumers close it with `aclosing` (deviation 8).
- §2 table and §10: the LLM runs on its pinned snapshot directory (`--model <snapshot>`), requests
  omit `model`; the configuration shape of deviation 6; `config.mock.yaml`.
- DECISIONS: whether to keep pipecat's Silero state reset on by default, given S7.4 (deviation 9);
  how to pin the audio models (deviation 5).

## M3 — Streaming TTS: one full spoken turn (2026-10-23)

**Goal.** Streaming TTS audio, completing one full spoken conversation turn: speech in → Silero VAD
and Smart Turn → Qwen3-ASR → the LLM → Qwen3-TTS audio out. Landed with it: D6 (pacing, 10 s lead
cap), D13 (the TTS fallback, Kokoro, pending B2's evidence), D16 (verbatim lines), D19 (the voice
map), and AICC acceptance part 1 (connectivity).

**Not yet run on the real backends.** M0's B2 spike held the GPU throughout M3, so the TTS provider
was developed against a fake `mlx_audio.server` that copies the real one's speech route (M0 A1 and
S3), and the real-backend tests (marker `backend`) are written but not run (see "Pending").

### What was built

| Area | Contents |
|---|---|
| `providers/mlx_audio_tts.py` | `MlxAudioTTS`, registry type `mlx_audio_speech`: one model and role (primary, fallback) of the TTS backend; `SpeechServer`, the HTTP client and in-flight slots the models share |
| `providers/fallback.py` | `FallbackTTS` and its circuit breaker (§5.2) |
| `providers/voices.py`, `voices` in config | the voice map (D19): session voice name → Qwen3-TTS speaker, lang policy, Kokoro voice per language |
| `session/pipeline.py` | LLM or verbatim line → segmenter → normalizer → TTS (drains every stream) → paced sender; timeline marks |
| `session/pacing.py`, `session/clock.py` | D6 pacing on an injectable clock |
| `session/actor.py` | voice resolution and lock, verbatim lines, a timeline per response, the first-audio mark |
| `text/normalize/` | the TTS normalizer: numbers, money, dates and times, phones, IDs, units, zh/en spacing, each toggleable |
| `text/segmenter.py` | never cuts between spaced digit groups |
| `session/timeline.py` | marks `response_created`, `first_sentence_ready`, `tts_first_chunk`, `first_audio_sent`; values `tts_queue_ms`, `tts_model`, `tts_fallback`, `verbatim` |
| `config.py`, `config*.yaml` | `output.max_lead_ms`, `lines.verbatim_prefixes`, `voices`, `text.normalizer`, `providers.tts` (`mlx_audio_speech`) |
| `tests/fakes/mlx_audio.py` | the fake server's `POST /v1/audio/speech` |
| `tests/aicc/` | `probe.py` and `zz_mlx_live_test.go`: AICC's own client harness against a gateway; `make aicc-probe`, `make aicc-probe-mock` |

**The TTS provider** (`mlx_audio_speech`, ARCHITECTURE §5.2–§5.3):

- One streamed `POST /v1/audio/speech` per segment: `model`, `input`, `voice`, `lang_code`,
  `response_format: "pcm"`, `stream: true`, `streaming_interval` (0.32), and the sampling fields only
  when configured (unset: the model's `generation_config.json`). The chunked s16le body is passed on
  as it arrives, in whole samples (`PCMChunk(…, 24000)`).
- Per model: Qwen3-TTS gets the voice's speaker and `chinese`/`english` from the segment's language
  tag; Kokoro gets the path of `voices/<name>.safetensors` inside the pinned snapshot (resolved from
  `backends.tts.fetch` in the local HF cache, or `voices_dir`; checked to exist before the request)
  and `z`/`a`.
- Deadlines: `connect`, `first_item` (the first chunk; 1.5 s Qwen3-TTS, 2 s Kokoro), `idle` between
  chunks, `total`; a warmup waits longer, for the model load.
- Errors: 404 (unknown model) and 4xx → `bad_response`; 500 → `internal`; 503 → `unavailable`; a body
  cut after the 200 status (the server failing inside generation, as M0 A1 saw with a Kokoro voice
  name), no audio at all, or an odd byte at the end → `bad_response`; connection failures →
  `unavailable`; a full queue → `overloaded`.
- Cancellation at every await point: closing the response is the cancel; the provider closes it in
  `finally` and yields nothing after.
- The server runs one stream at a time, so `max_in_flight` is 1 and requests queue in the gateway
  (`queue_timeout_ms` 2 s, then `backend_busy`). The wait is measured (`SpeechTiming.queued_s`) and
  logged with the turn (`tts_queue_ms`).
- Readiness: both models listed by `GET /v1/models`. Warmup: one Chinese and one English sentence per
  configured voice on the primary, and per distinct Kokoro voice on the fallback (20 requests with the
  built-in map); a voice that fails keeps the gateway not ready.

**`FallbackTTS`** (D13): a primary failure before its first chunk sends that segment to the fallback,
which maps the session voice to its own voice for the segment's language; a failure after audio
started fails the segment (the response ends `failed`); `overloaded` is not retried (both models share
the backend's queue) and does not count. After `breaker_failures` (3) consecutive primary failures
every segment goes to the fallback for `cooldown_s` (30 s); the first segment after that tries the
primary again, and one more failure reopens the circuit. `fallback.enabled: false` builds the primary
alone.

**Voices (D19).** `voices.map` names the session voices; the built-in map has Qwen3-TTS-0.6B's
Mandarin and English speakers (`vivian`, `serena`, `uncle_fu`, `ryan`, `aiden`; not `dylan` and `eric`,
which turn into dialects with `chinese`/`auto`), each with a Kokoro voice of the same gender per
language. Any other name is accepted, spoken by `voices.default` (`vivian`), logged as a warning, and
the effective name is echoed in `session.updated` and in the response. `session_defaults` must name a
mapped voice (or none: the default). Once the session has produced audio the voice is locked, as in
GA: a different voice in `session.update` or `response.create` gets `cannot_update_voice`; AICC sets
its voice only in its first update and never meets it.

**Verbatim lines (D16).** A `response.create` whose own `instructions` start with one of
`lines.verbatim_prefixes` (AICC's two `SayExactly` directions by default) skips the LLM: the rest,
trimmed, goes through segmenter → normalizer → TTS and becomes the assistant message. It answers no
user turn, so it neither waits for a pending transcript nor makes a tool call: it works on an empty
conversation, mid-call, and while a function call is unanswered (AICC's wait line). Its events are
those of any audio response (golden trace `verbatim_line.jsonl`); `usage` is `null`. PROTOCOL.md
§11.11 documents it as a server extension that is invisible on the wire.

**Pacing (D6) and the pipeline.** The TTS stage makes one request per segment, the next as soon as the
previous finished generating, and reads each stream as fast as the backend produces it, into the
response's audio queue (synthesis stops once it passes `limits.max_response_audio_ms`, and the actor
ends the response `incomplete`). A sender task posts that audio at once but never more than
`output.max_lead_ms` (10 s) ahead of real time, counted from the response's first delta; at the cap it
releases one `max_delta_ms` delta per `max_delta_ms`, so no gap nears AICC's 2 s watchdog while audio
waits. A segment's transcript delta goes out just before its first audio. A barge-in or
`response.cancel` cancels the response scope, so audio held back by the cap is never sent, and the
ledger discards what was queued (M1). The clock is injectable (`session/clock.py`).

**Segmenter and normalizer (§7).** The M1 segmenter already ends the first segment early (a soft
boundary at `first_min_chars`, or the best boundary after `first_max_wait_ms`) and never splits inside
numbers, URLs, e-mail addresses or IDs; M3 adds that whitespace between digit groups ("138 0013 8000")
is never a cut, even before the next group arrives. Each segment is tagged zh (any CJK, mixed
included) or en from its original text; the tag picks each model's lang_code and is the normalizer's
fallback context.

The normalizer (`text/normalize/`, TTS input only; transcripts keep the original text) recognizes
spans in priority order (URLs and e-mail addresses, always left as written; phones; IDs; money; dates
and times; units; free-standing numbers), and a span one category claims is never re-read by a later
one, even when its own category is off. Each span is read in Chinese or English by its
neighbourhood: the nearest letter on either side, CJK winning; the segment's tag decides only when
there is none. Categories (`text.normalizer`):

- `phones` (on): mobiles (`1[3-9]` + 9 digits, optionally `+86`), landlines (area code + 7–8 digits),
  400/800 numbers, and 95xxx/96xxx hotlines after a phone keyword; digit by digit, in groups joined by
  "，" / ", ", 1 read as 幺 in Chinese (`phone_one_as_yao`).
- `ids` (on): alphanumeric tokens with at least 6 characters, 4 digits and a letter (not words, version
  numbers or quantities: `Qwen3`, `iPhone15`, `Qwen2.5-72B`, `1920x1080` stay as written), and digit
  runs of 6 or more after an ID keyword (订单号, 运单号, order number, …); letters as spaced capitals,
  digits one by one (1 as 一), runs over 4 split in fours, each separator a pause.
- `numbers`, `money`, `dates`, `units`, `spacing` (off until S9): cardinals with 万/亿 and 两, decimals,
  percents, ranges, ordinals, negatives; ¥/元/$ amounts (元角分, dollars and cents); dates, years and
  times (上午/下午, "fourteen thirty"); units after a number; one space between CJK and Latin text.

`enabled: false` returns the text unchanged, and text with none of these patterns always comes back
byte for byte.

**AICC acceptance, part 1 (connectivity).**

- The five AICC traces are replayed on the real TTS path (the provider and its fallback against the
  fake backend at M0's pace: first chunk after 110 ms, a 320 ms chunk every 90 ms), and every response
  is checked against AICC's deadlines (`tests/integration/deadlines.py`): `response.created` within
  3 s of its `response.create`, the first audio within 3 s of `response.created`, no gap of 2 s between
  deltas, a tool-only response done within 3 s.
- `tests/integration/test_tts_turns.py`: an AICC-shaped `server_vad` session with real Silero speech
  in gets exactly one reply per speech turn, spoken, within the deadlines.
- AICC's own harness: `tests/aicc/probe.py` runs AICC's `internal/provider` tests with our
  `zz_mlx_live_test.go` added through `go test -overlay` (the AICC checkout is never modified):
  `TestMLXGreeting` (`probe`: handshake, the opening spoken turn on an empty conversation, 24 kHz
  audio), `TestMLXResponseCreatedLatency` (`measureResponseCreated`, 8 samples) and `TestMLXLines` (a
  flow voice, the opening line and a `SpeakText` wait line, both checked word for word). It fails the
  run on any `response.created` sample of 3 s or more, an unanswered request, or AICC's watchdog
  giving up on a turn (`response abandoned`: the first audio later than 3 s after `response.created`,
  or a 2 s gap).

### How to run and verify

```sh
uv run --locked pytest                     # everything but the backend tests
make lint typecheck
make aicc-probe-mock AICC_REPO=<an AICC checkout>   # AICC's client against mock providers
# The real stack (pending the GPU):
make dev                                   # backends and gateway (config.yaml)
uv run --locked pytest -m backend          # in another terminal; AICC_REPO=… adds AICC's harness
make aicc-probe AICC_REPO=<an AICC checkout> AICC_URL=ws://127.0.0.1:8000/v1/realtime
```

The exact harness command, without make:

```sh
uv run --locked python tests/aicc/probe.py --aicc-repo <AICC checkout> \
    --url ws://127.0.0.1:8000/v1/realtime            # or --serve config.mock.yaml
```

It needs Go and AICC's module dependencies (`go mod download` in the checkout). The API key goes to
AICC as `REALTIME_API_KEY` (`--api-key`; the wrapper sends `local` when it is unset).

**Mock run, 2026-10-23** (`--serve config.mock.yaml`, against AICC commit 8fabadc's scratch copy):
all three pass. The greeting's first audio came 321 ms after the session started; 8 of 8
`response.created` within 2 ms; the opening line and the wait line came back word for word ("Good
morning, NovaNet support.", "One moment, let me check that.") with 77 760 and 72 000 bytes of audio.

### Test results

All green on 2026-10-23: **1162 passed, 11 skipped** (the `backend` tests and the AICC run that needs
`AICC_REPO`) in about 70 s (`uv run --locked pytest`). `ruff check`, `ruff format --check` and `pyright`
(strict on `src/`) report nothing.

| Suite | Tests | New in M3 |
|---|---|---|
| unit | 1058 | the normalizer (426: every category on and off, zh, en and mixed, the protection and context rules, the untouched list under 8 option combinations); the TTS provider against the fake server (30: request shape per model and voice, Kokoro voice-pack paths, whole samples across reads, six error shapes, every deadline, cancellation awaiting headers, before the first chunk, between chunks and while the consumer holds a chunk, the queue and its measured wait, a full queue, a cancelled wait, warmup of every voice, readiness); `FallbackTTS` (11: before and after the first chunk, `overloaded`, the breaker and its cooldown, cancellation on either model); pacing on a fake clock (5); the pipeline (10: transcript before audio, the TTS draining while the sender waits at the cap, lines in both modalities, normalized text, unspeakable segments, marks, the audio cap, a TTS failure); the voice map (7); the segmenter (1: spaced digit groups); the configuration (8); the import graph (32, the new modules) |
| integration | 83 | the AICC replay on the real TTS path with every deadline checked (6, one new: a reply of several segments); `test_tts_turns.py` (5: a spoken Chinese `server_vad` turn answered in speech through every real provider on fakes, with the timeline; the fallback; two sessions queueing for the TTS; pacing on the wire; a barge-in flushing paced audio); `test_voices_and_lines.py` (7: D19 echo and warning, the voice lock, lines on an empty conversation, mid-call, during an unanswered call, in text modality, and prefixes switched off); golden `verbatim_line.jsonl` (1) |
| e2e (`-m e2e`) | 4 | the SDK test expects the effective voice |
| tests/aicc | 18 | the runner's parsing of AICC's output and its deadline verdicts (17); AICC's client against `config.mock.yaml` (1, needs `AICC_REPO`; ran: passed) |
| backend (`-m backend`) | 10 | six written, not run: see below |

### Pending the GPU runs

- `uv run --locked pytest -m backend` with `make dev` running, M3's six:
  - `test_tts_streams_with_a_fast_first_chunk_and_faster_than_real_time`: Qwen3-TTS zh and en through
    the provider, first chunk < 500 ms warm, RTF < 1;
  - `test_tts_output_is_heard_right_by_the_asr`: TTS → Qwen3-ASR round trip;
  - `test_the_fallback_speaks_when_the_primary_fails`: a speaker Qwen3-TTS lacks → Kokoro speaks;
  - `test_kokoro_speaks_chinese_and_english_from_voice_pack_paths`;
  - `test_sdk_full_chinese_spoken_turn_with_audio`: the official SDK, only `base_url` changed, streams
    the Chinese fixture in real time; transcript → LLM → TTS audio, first audio ≤ 3 s after
    `response.created`, no 2 s gap;
  - `test_aicc_harness_against_the_real_stack` (needs `AICC_REPO`): `probe.py --url` against `make dev`.
- M2's four backend tests (unchanged).
- Defaults waiting on B2 (`spikes/results/s3s4s9_report.md`), all config values:
  `providers.tts.primary.streaming_interval` (0.32), the sampling fields (unset), the default voice
  and the zh/en speakers (`voices`), the first-chunk deadlines (1.5 s / 2 s), whether Kokoro stays the
  fallback (D13, G6: `providers.tts.fallback`), and which normalizer categories are on (G5,
  `text.normalizer`).
- G4: the real first-audio budget (ASR + LLM first token + first segment + TTS first chunk) under the
  3 s deadline, measured by the backend SDK test and AICC's harness.

### Known gaps, by milestone

| Milestone | What M3 leaves for it |
|---|---|
| M4 | echo cancellation (D17); barge-in against the real TTS; the D5 truncation fallback |
| M6 | timeline marks as metrics; `/metrics`, `/debug/turns` |
| M7 | G.711 output (the encoder still emits 24 kHz PCM16 only) |
| open | under load from several sessions, a segment can wait for the TTS behind other sessions' segments; if a session has sent all its synthesized audio meanwhile, the gap between its deltas grows by the wait, and at 2 s AICC abandons the turn. The wait is logged (`tts_queue_ms`) and bounded (`queue_timeout_ms`), not avoided |

### Deviations from ARCHITECTURE, with reasons

1. **Pacing lives in the pipeline, not in the encoder or the writer.** A sender task between the TTS
   stage and the actor admits each delta (`session/pacing.py`); audio not yet admitted belongs to the
   response scope and dies with it. `response.done` still follows the last delta, so the `audio_sent`
   flag of §6.3 stays unnecessary (M1 deviation 9).
2. **The TTS stage drains every stream into the response's queue**, bounded by
   `limits.max_response_audio_ms`, instead of streaming through to the client: pacing or a slow
   client must never hold the backend's single stream slot that every session shares (§4.2,
   "drains backend streams promptly").
3. **`TTSProvider.stream` takes `timing: SpeechTiming | None`** (keyword-only, like `speed`): the
   provider reports its queue wait, the model and whether the fallback spoke, for the timeline.
   `LLMBackend`, `ASRBackend` and `TTSBackend` moved from the registry to `providers/base.py`.
4. **`FallbackTTS` does not retry `overloaded`** and does not count it: both models share the
   backend's queue. The breaker is half-open after the cooldown (one failure reopens it at once).
5. **A body cut after the 200 status is `bad_response`** (`provider_error` on the wire), not
   `unavailable`: on mlx_audio.server it is the generation failing, as M0 A1 saw with a Kokoro voice
   name. A stream without audio, or ending inside a sample, is `bad_response` too.
6. **The voice map is a top-level `voices` section** (name → speaker, lang policy, Kokoro voice per
   language) instead of `output.default_voice`. `session_defaults` must name a mapped voice (startup
   fails with the dotted path otherwise), and without one the session voice is `voices.default`. The
   session voice therefore echoes `vivian` where M1/M2 echoed the placeholder `default`.
7. **The voice lock also covers `response.create`'s `audio.output.voice`**: after the first audio, a
   per-response voice other than the session's draws `cannot_update_voice`
   (`param: "response.audio.output.voice"`) and creates no response. GA states the lock for the
   session voice; the per-response field would otherwise bypass it.
8. **A verbatim line answers no user turn**: it does not wait for pending transcripts, the D7 empty
   transcript rule does not apply, and it gets a timeline of its own. The line is the instructions
   after the prefix, trimmed. In text modality it is one `output_text` delta.
9. **Every response has a timeline**: one that answers a user audio turn extends that turn's;
   any other (a client `response.create` after text items, a line) gets its own, from
   `response_created`. The marks M2 left for M6 (`first_sentence_ready`, `tts_first_chunk`,
   `first_audio_sent`) and `response_created` landed now, with `tts_queue_ms`, `tts_model`,
   `tts_fallback` and `verbatim`.
10. **Segments with nothing to say** (no letter, digit or CJK character) are not sent to the TTS;
    their transcript still goes out, in order.
11. **A segment's language tag comes from its original text**, not the normalized text.
12. **The TTS warmup speaks every configured voice in both languages on both models**, and a
    fallback that cannot warm keeps the gateway not ready (`/readyz` names it). Switch the fallback
    off with `providers.tts.fallback.enabled: false` rather than run without knowing it is broken.
13. **Kokoro's voice packs are resolved from the gateway's own HF cache** (the backend runs on the
    same machine), at the revision `backends.tts.fetch` pins, and checked to exist before a request.
14. **`output.max_delta_ms` is at most 1000 and `max_lead_ms` at least `max_delta_ms`** (validated):
    at the cap a delta waits as long as it plays, so a delta of 2 s would itself be a 2 s gap.
15. **AICC's harness wrapper ships in this repository** (`tests/aicc/zz_mlx_live_test.go`) and is
    added to AICC's package with `go test -overlay`, so any AICC checkout works unmodified. It adds
    `TestMLXLines` to the scratch wrapper's two tests.
16. **M2 tests changed on purpose**: the speech-turn timeline lists `response_created`; the SDK test
    expects `marin` echoed as `vivian`; the AICC replay runs on the real TTS path, expects lines
    spoken verbatim without the LLM, and no longer sends the waitline trace's delete of a call the
    line's turn made (a verbatim line makes none, and AICC deletes only calls it saw); the repository
    config test checks the TTS. Golden traces: only the echoed voice changed (`default` → `vivian`,
    32 lines in 9 files); `verbatim_line.jsonl` is new.

### Suggested changes to ARCHITECTURE and DECISIONS

- §5.1: `TTSProvider.stream(..., timing: SpeechTiming | None = None)`; the combined
  `LLMBackend`/`ASRBackend`/`TTSBackend` protocols are in `providers/base.py`.
- §5.2 TTS fallback: `overloaded` is raised, not retried; the breaker's half-open rule; a cut body is
  `bad_response`.
- §5.3 TTS row: the model preflight before the 200; a failure inside generation cuts the chunked body;
  `streaming_interval` defaults to 2.0 s server-side when not sent; Kokoro voices as snapshot paths;
  one stream at a time (`max_in_flight` 1, gateway-side queue).
- §7 Pacing: the sender task and the drain-promptly TTS stage (deviations 1–2); at the cap one delta
  per `max_delta_ms`.
- §7 Voices / D19: the voice map shape (deviation 6); the lock, including per-response voices,
  with `cannot_update_voice` (deviation 7).
- §7 Verbatim lines / D16: a line answers no turn, `usage` null, text modality (deviation 8).
- §8 / D18: add `cannot_update_voice` (invalid_request_error) to the error list.
- §9 timeline: the marks and values of deviation 9; one timeline per response.
- §10 sketch: `output.max_lead_ms`; `lines`; `voices`; `text.normalizer` categories; the
  `providers.tts` shape (`primary`, `fallback` with `enabled`, `breaker_failures`, `cooldown_s`,
  `voices_dir`, per-model `timeouts`, `max_in_flight`, `queue_timeout_ms`, `warmup_texts`).
- §11 AICC acceptance: the overlay wrapper and `TestMLXLines` (deviation 15).
- §2 or §7: the multi-session TTS queue can open a gap in a session's audio (Known gaps, "open");
  possible mitigations for the owner: a per-session lead floor that holds back a little audio for the
  next segment, a second TTS process, or prioritizing a segment whose session is about to run dry.
- DECISIONS: record that verbatim lines answer no turn, and the voice lock's extension to
  per-response voices.

## M4 — Barge-in, cancellation and truncate (2026-10-23)

**Goal.** Barge-in, cancellation and truncation, deterministic and covered by tests on every
cancellation path. Landed with it: D5 (the truncation fallback), D17 (echo cancellation in the
service, with the engine G7 chose in M0, the reference on a model of AICC's playout, and the
second line of defence), and AICC acceptance part 2 (barge-in: interruption, the truncate and
delete flows, behaviour under simulated echo).

**Not yet run on the real backends.** M0's spikes held the GPU throughout M4. The echo canceller
runs on the CPU and was exercised for real; everything model-backed ran against mocks and the fake
servers, and the real-backend tests (marker `backend`) are written but not run (see "Pending").

### What was built

| Area | Contents |
|---|---|
| `audio/aec.py` | the engine interface (`EchoEngine`: one 10 ms frame of capture and of reference in, the capture with the echo removed out) and `WebRtcAec3`: `pywebrtc-audio==0.2.0`'s `EchoCanceller(sample_rate=16000, num_channels=1, stream_delay_ms=0)`, AEC only. Every frame is checked to be `int16[160]` before the native call (pywebrtc zero-pads a partial frame silently). livekit can be added behind the interface |
| `audio/playout.py` | the model of AICC's playout (voice/rtp.go `Run`): 20 ms ticks, start at 3 queued frames or 60 ms after the queue was last empty, a re-buffer after more than 2 empty ticks, all in integer 16 kHz samples; and `TickPhaseLearner` |
| `audio/delay.py` | the GCC-PHAT residual-delay estimator: 8 kHz copies, 2 s windows every 0.5 s, lags −300…+1000 ms, PHAT over 200–3600 Hz, only while the reference is active, PSR ≥ 8, two windows within 2 ms to confirm |
| `session/echo.py` | `CaptureClock` (input timeline ↔ loop time), `EchoReference` (the client's modelled playout of our output and everything read from it), `EchoStage` (the canceller on the VAD feeder's path), `EchoRuntime` (the canceller executor), the per-session summary log |
| `session/feeder.py`, `vad/gate.py` | the canceller before Silero; the gate's playback-aware rules; the canceller's output posted to the actor with each `VadUpdate` |
| `session/input_buffer.py` | a second track: the canceller's output over the same positions (Smart Turn reads it; commits give the ASR the track `aec.asr_input` names) |
| `session/actor.py` | the echo reference's lifecycle (response start and end, audio queued, flush on a cancel, truncate), D5, echo-turn suppression, barge-in accounting |
| `session/output.py`, `server/connection.py` | the ledger records when an item's first audio was sent (D5) and tells its listener (the echo reference) of every audio delta sent or discarded; `Delivery` carries the PCM |
| `session/truncation.py`, `text/normalize/` | `normalize_with_map`: for each spoken character its index in the original; the truncation cut is taken in spoken characters and mapped back |
| `text/similarity.py` | the overlap of a transcript with played text, in CJK characters and Latin words (bit-parallel LCS) |
| `config.py`, `config*.yaml` | the `aec` and `barge_in` sections; `config.mock.yaml` now runs the real Silero VAD and the barge-in rules, the canceller off (below) |
| `providers/mock.py` | `SineTTS.queue`: an await point while a segment waits for the TTS backend's slot |
| `tests/phone.py` | a simulated AICC call in real time: AICC's downsampler, framer, playout and barge-in; G.711 μ-law; a synthetic echo path (a seeded, decaying random impulse response of 96 taps at 8 kHz, an echo return loss, a bulk delay; no ITU G.168 coefficients); AICC's per-frame linear upsampler; 20 ms appends |
| `tests/aicc/` | `TestMLXBargeIn` and `TestMLXKeypressCancel` in the overlay; `probe.py` hands them the caller's speech |

**Barge-in and cancellation (§6.5).** A speech start that interrupts runs the cancellation handler
in one synchronous step: `speech_started` (with the item the commit will create), the response
scope cancelled, the generation bumped, queued deltas discarded and audio still held by the
pacer never sent, the terminal events in the GA order, `response.done{cancelled, turn_detected}`,
and the item cut for the next request (D5). The same handler serves `response.cancel`, provider
failures and timeouts; a disconnect cancels the session scope. Speech over the tail of an answer
whose response is done (AICC still playing it) emits `speech_started` alone; AICC's truncate then
stops the modelled playback.

**Truncate (§6.6, D18).** The cut is measured on the audio, which speaks the normalized text: the
segment's share of spoken characters is mapped back through the normalizer's character map, then
snapped to a Latin word start (a phone number read digit by digit counts only as far as its digits
were read). `audio_end_ms` beyond the audio sent is clamped with a warning, never an error.
Completed items truncate in place; a truncate to 0 then a delete removes an unheard answer.

**D5.** When a cancelled audio answer (barge-in or `response.cancel`) gets no client truncate, the
next LLM request sees it cut at `min(audio sent, time since its first audio delta was sent)`,
measured at the cancel on the session clock; a later truncate overrides it. Transcript events keep
reporting the text sent (M1).

**The echo canceller (D17, G7).** One per session, on the canonical 16 kHz input before Silero,
Smart Turn and the input buffer, on its own executor (`aec.workers`, 2), per-session frame order
kept (pywebrtc releases the GIL).

- **Reference.** Every audio delta as the writer hands it to the socket, resampled 24 → 16 kHz per
  item (the project's soxr stream), framed in 20 ms as AICC frames it (the last frame of an item
  padded), and played on the AICC model with the arrival at the send time. The reference for input
  position *n* is the modelled playout at *n* + offset + `reference_lead_ms` (60) − the guard's
  shift: our audio precedes its echo. The offset (`CaptureClock`) is the smallest
  arrival − position over the last 10 s of input: the least delayed arrivals, tracking clock drift.
- **Lock-step feeding.** Exactly one 160-sample reference frame per capture frame, zeros when
  nothing plays, so the delay is carried in the content.
- **Hold-back.** While a response may still add audio and the client model's queue does not cover
  the reference ahead, an input frame waits until its reference can no longer change (now ≥ its
  reference time): about 70 ms. Idle, or while the queue covers it (most of a response, with the
  10 s lead), input goes through at once. A reconfiguration (commit, clear, turn-detection change)
  first releases what is held, under the old epoch.
- **Tick phase.** `TickPhaseLearner` fits AICC's 20 ms grid from the first confirmed GCC-PHAT
  estimate inside each unbroken, prebuffer-started playout run (`learn_tick_phase: true`); the
  model takes a new phase only between runs. In the unit test, 24 runs located to within 1 ms
  put ≥ 90 % of starts on the true tick.
- **GCC-PHAT guard.** On the raw input against the reference fed; a confirmed residual outside
  [`min_residual_delay_ms` 5, `max_residual_delay_ms` 450] moves the reference so the residual
  becomes the lead (the estimator restarts after a move).
- **Re-anchoring.** A barge-in or cancel drops the frames of that response the client has not
  played. A client truncate drops the rest of the item, and when the client's played frames differ
  from the model's by 2 or more, shifts the model's arrivals by the difference (at most ±500 ms).
- **Bypass without echo** (not in G7; deviation 1). With no echo on the line, AEC3's suppressor
  takes a caller talking over our audio for echo and silences them (measured below). Once the
  estimator has evaluated 1.5 s of playing reference without any echo of it
  (`bypass_without_echo_ms`), the canceller gets a silent reference (it then passes the input
  through); the first confirmed echo turns it back on (AEC3 re-converged within 0.5 s offline).
- **ASR input.** `aec.asr_input: processed` (default) or `raw`; speech and end-of-turn detection
  always use the canceller's output. M0's double-talk ΔCER, still pending, sets the default.
- **Observability.** One `echo summary` log line per session: frames, canceller CPU per audio
  second, ERLE (median over 0.5 s windows with an active reference), re-alignments, re-anchorings,
  bypasses, the shift, the anchor, the learned phase, barge-ins, barge-ins whose turn turned out
  empty or echo, and echo turns.

**The second line of defence** (`barge_in.*`, each rule switchable). "Playing" runs from the first
modelled frame of a response's audio to its last (or a flush), plus `max_residual_delay_ms` of echo
tail, on the input timeline.

- An onset in the first `echo_guard_ms` (2000) of a response's playback never interrupts; that
  speech counts only from the guard's end, and a stretch that ends before then is discarded.
- While playing and past the guard, `speech_started` needs `min_speech_ms_while_playing` (400);
  otherwise `turn.vad.min_speech_ms` (150), from the speech's own onset (which the event reports).
- Echo-turn suppression: a VAD-detected turn whose transcript matches the assistant audio played
  around it (from `echo_match_window_ms`, 3 s, before the turn, plus the echo tail; at least
  `echo_match_threshold` 0.7 of its CJK characters and Latin words in order; turns of at least
  `echo_match_min_units` 4) is noise under D7: its response completes with no output and no LLM
  call, and later requests leave it out.

**AICC acceptance, part 2.**

- Replay (`tests/integration/test_aicc_replay.py`, AICC simulated in real time with
  `tests/phone.py`, the real Silero VAD and the canceller as configured by default): the caller
  over a long line (`speech_started` → `response.done{cancelled}` in one step → AICC's truncate
  of its played frames accepted → a reply after `speech_stopped`); the caller over the tail of a
  finished line; the keypress flow (`response.cancel`, a truncate to 0, the delete after
  `response.done`, the keypress note and the owed reply), with no `error`.
- AICC's own client (`tests/aicc/zz_mlx_live_test.go`, against `config.mock.yaml`):
  `TestMLXBargeIn` (the caller's clip 2.5 s into a long line; AICC's played-ms truncate;
  `conversation.item.truncated` seen; a reply after `speech_stopped`; no error, no client warning)
  and `TestMLXKeypressCancel` (a key on the line's first audio: `Interrupt(DTMF, 0)`, the item
  deleted, a spoken reply; a key before any audio: the cut turn's item deleted by AICC).
- Simulated echo end to end (`tests/integration/test_echo.py`): the gateway's own speech loops
  back through AICC's downlink, G.711, a synthetic echo path (ERL 10 dB, 100 ms bulk delay) and
  AICC's uplink. Six seconds of our own echo never interrupt; a caller talking over it later
  does (`speech_started` within 1.5 s of their onset, then `response.done{cancelled}`); the
  session's median ERLE is ≥ 10 dB (about 30–40 dB per 0.5 s window when dumped). With the
  canceller and the rules off, the same echo interrupts the answer.

### How to run and verify

```sh
uv run --locked pytest                     # everything but the backend tests
make lint typecheck
make aicc-probe-mock AICC_REPO=<an AICC checkout>   # AICC's client, barge-in included
uv run --locked pytest -rP tests/unit/test_aec.py   # prints the canceller's CPU per audio second
# The real stack (pending the GPU):
make dev
uv run --locked pytest -m backend          # in another terminal; AICC_REPO=… adds AICC's harness
```

**Mock run, 2026-10-23** (`probe.py --serve config.mock.yaml`, AICC commit 8fabadc): all five
`TestMLX*` pass. `TestMLXBargeIn`: the caller's audio from 2.50 s into the line, `speech_started`
3.15 s into its playback, AICC reported 3140 ms played, the line cancelled in the same step, the
truncate acknowledged; `speech_stopped` 3.8 s later and the reply's audio 0.43 s after it.
`TestMLXKeypressCancel`: the line cancelled 1–18 ms after the key, the reply's audio 250–330 ms
after it. Twelve repeated runs of the two, and a final run of all five: all passed.

### Test results

All green on 2026-10-23: **1283 passed, 12 skipped** (the `backend` tests and the AICC run that
needs `AICC_REPO`) in about 125 s (`uv run --locked pytest`). The real-time tests (echo, barge-in,
AICC replays) and the cancellation matrix also passed three runs in a row under a load average of
about 18 (M0's spikes were running). `ruff check`, `ruff format --check`
and `pyright` (strict on `src/`) report nothing. With `AICC_REPO` set, the AICC e2e test passes too.

| Suite | Tests | New in M4 |
|---|---|---|
| unit | 1147 | the playout model and the phase learner (14); the engine and the estimator on synthetic line echo (13: frame checks, ERLE > 20 dB with the caller kept in double talk, the CPU, residual delays of 20, 160 and 600 ms, a negative lag, no windows without a reference and none confirmed without echo, the restart); the echo reference and stage on a fake clock (19: the capture clock, sample-exact placement with the lead, the hold-back, the flush, truncate re-anchoring, playback and its guard, the guard's keep range, the bypass, lock-step feeding, a gap, the overlap measure); the barge-in rules in the gate (7); the input buffer's second track (3); the feeder with the canceller in front (1); the normalizer's character map (12); truncation over normalized text (2); the echo settings (9); the import graph (the new modules) |
| integration | 114 | the cancellation matrix, now 10 phases × 5 triggers (50, below); truncation on the wire (7); the barge-in rules and echo turns on a virtual clock (3); simulated echo (2); the real-VAD barge-in on the TTS path (1); AICC barge-in replays (3) |
| e2e (`-m e2e`) | 4 | the `mlxrt dev` test needs the ONNX models (config.mock.yaml runs Silero) |
| tests/aicc | 19 | the caller clip it hands the Go tests (1); the AICC run now requires all five tests to pass |
| backend (`-m backend`) | 11 | one new, not run: see below |

**The cancellation matrix** (`tests/integration/test_cancellation.py`). Phases: awaiting the
transcript, before the first token, mid-text, mid-tool-call, waiting for the TTS backend's slot,
before the first TTS chunk, mid-audio, audio held back by pacing, draining after the LLM is done,
and a verbatim line. Triggers: barge-in, `response.cancel`, disconnect, backend error, backend
timeout. Each case checks: no delta after a terminal event, every stream closed once, in order
(the causal invariants); the provider streams closed within 1 s without their gates released;
every stream a mock opened closed exactly once; the pacer's held audio never sent; for a barge-in,
`speech_started` before `response.done{cancelled}` with nothing but terminal events between; and
the next request's history (D5: all that was sent once the session clock ran on, nothing when it
did not).

**The real-VAD barge-in on the TTS path** (`tests/integration/test_tts_turns.py`): real Silero, the
TTS provider on the fake server speaking the English clip, the LLM still writing, a 1 s lead. It
checks §6.5 step by step: `speech_started` names the item the commit creates; the LLM request is
closed; only terminal events lie between `speech_started` and `response.done`; less audio was sent
than synthesized and none after; AICC's truncate is honoured at its played time; the next request
sees the answer cut there. The caller's speech is confirmed about 0.6 s after its onset (the
400 ms rule, the hold-back, framing).

**Measured.** The canceller's CPU: **8.6–9.4 ms per second of audio** (thread time, 20 s of
synthetic line echo, one session, `tests/unit/test_aec.py`; up to 17 ms with a load average of 18
while M0's spikes ran; M0 B4 measured 7.8 ms). AEC3 near-end suppression with no echo on the line
(offline, the fixtures, 0.5 s windows from the caller's onset): a speech reference 0–2 dB or, for
another clip pairing, up to 44 dB for 2 s; a sine reference 45–51 dB throughout; a silent reference
under 0.3 dB. In the gateway, before the bypass, a caller over our speech with no echo was confirmed
2.5–3.4 s late or not at all; with the bypass, about 0.6 s after the onset.

### Pending the GPU runs

- `uv run --locked pytest -m backend` with `make dev`, M4's:
  - `test_sdk_barge_in_with_the_real_stack`: the official SDK streams room tone, asks for a long
    Chinese line, and streams the English fixture over it 2.6 s in: `speech_started`, then
    `response.done{cancelled, turn_detected}` with only terminal events between, the caller
    confirmed within 2 s of the onset, a truncate accepted, the caller's turn transcribed and
    answered, no error;
  - `test_aicc_harness_against_the_real_stack` (needs `AICC_REPO`): now also `TestMLXBargeIn` and
    `TestMLXKeypressCancel` on the real VAD, canceller and TTS.
- M2's and M3's backend tests (unchanged).
- M0's double-talk ΔCER (C2): whether the ASR keeps the canceller's output (`aec.asr_input`).
- Live echo on a real AICC call (deferred by D17): the echo path, ERL and bulk delay, AEC3's
  convergence and the bypass on real lines.

### Known gaps, by milestone

| Milestone | What M4 leaves for it |
|---|---|
| M6 | the echo summary's numbers as metrics (ERLE, bypasses, barge-ins that were echo) |
| M7 | the canceller on G.711 input (8 kHz upsampled; the reference model is rate-independent) |
| open | clock drift between AICC's ticker and its uplink was not simulated (the capture clock tracks drift over 10 s, the learner keeps refining); a segment cut while still being synthesized is measured over the audio made so far, which overstates what was heard of it |

### Deviations from ARCHITECTURE, with reasons

1. **The canceller is bypassed on a line without echo** (`aec.bypass_without_echo_ms`, 1500; null
   turns it off). AEC3's suppressor silences a caller talking over our audio when no echo of it is
   on the line (measurements above): on such calls barge-in came seconds late or never. The bypass
   only follows the estimator's evidence (1.5 s of playing reference without echo); a single
   chance peak never re-engages it, a confirmed estimate does.
2. **The guard's keep range is [5, 450] ms**: `max_residual_delay_ms` is its upper bound, as B4
   recommended over the tested 400 (AEC3 handled 415 ms and failed at 565). The brief quoted
   both [5, 400] and 450; both bounds are configurable.
3. **Echo-turn overlap counts CJK characters and Latin words**, not characters: letter by letter,
   two unrelated English sentences overlap 0.45–0.7, which would drop real English turns as echo.
   `echo_match_min_chars` became `echo_match_min_units`.
4. **The hold-back** (about 70 ms while a response may add audio) makes the reference final before
   it is fed; the alternative, feeding at once, loses the first 40–60 ms of reference at each
   response's start. Idle input is never held.
5. **The reference comes from the writer**: the ledger tells the echo reference of each audio
   delta handed to the socket (and each one discarded), with its PCM in `Delivery`.
6. **"Playing" includes `max_residual_delay_ms` of echo tail** after the modelled end, and the guard
   is per response (from its first modelled frame). The rule for a speech stretch follows its
   onset: one that began before playback needs 400 ms once playback starts; once playback has
   ended, the normal rule applies from its onset.
7. **config.mock.yaml runs the real Silero VAD** (`make dev-mock` and `make aicc-probe-mock` now
   fetch the ONNX models) and the barge-in rules, **with the canceller off**: the mock TTS speaks a
   sine, the pathological reference for AEC3 (deviation 1's measurement), which no bypass can tell
   apart from echo (a periodic reference correlates with anything periodic).
8. **D5 applies to cancelled responses** (barge-in, `response.cancel`), not failed ones; it is
   computed at the cancel; the transcript events still report the text sent.
9. **Echo-turn suppression applies to VAD-detected turns only**, not to manual commits.
10. **A manual commit in VAD mode** takes the canceller's output where it has run; the last
    10–70 ms may still be raw.
11. **AEC3's fixed 8 ms capture delay is not compensated**: the processed track and the VAD's
    positions are consistent with each other, 8 ms behind the raw input.
12. **Test tooling.** `build_gateway` turns the canceller and the barge-in rules off unless a test
    turns them on (scripted detectors and audio sent faster than real time leave no playout to
    model); `ManualClock` gives tests a virtual session clock; `assert_causal(ended=False)` for a
    disconnect; the fake speech server can speak a PCM clip.
13. **M3 tests changed on purpose**: the cancellation matrix was rewritten for the new phases and
    the D5 history rule; the AICC replay counts only client requests in its deadline check (VAD
    turns add responses); the mock-config test checks the new setup. Golden traces: only
    `barge_in.jsonl` changed, in one field: the reply's `usage.input_tokens` went from 28 to 16,
    because the interrupted answer, which had no time to play (the test's frozen clock), is cut
    from the next request (D5). The test now runs on `ManualClock`, so the value is deterministic.

### Suggested changes to ARCHITECTURE and DECISIONS

- §4.3 and §6.5 (Echo): the hold-back; the reference from the writer; the capture clock; the echo
  tail and per-response guard; re-anchoring (±2 frames, ±500 ms); tick-phase learning from confirmed
  estimates inside unbroken runs; the bypass without echo (deviation 1), for the owner's review.
- §6.5 test coverage: the matrix's phases now include the TTS queue wait, audio held by pacing and
  a verbatim line, and the checks include held audio never sent and AICC's ordering rule.
- §6.6: the cut in spoken characters through the normalizer's map; D5 for cancelled responses,
  computed at the cancel; a segment still being synthesized.
- §10: the `aec` and `barge_in` sections as built (`bypass_without_echo_ms`,
  `min_residual_delay_ms`, `echo_match_*`).
- §12 / DECISIONS: D17's bypass; the keep range [5, 450]; echo-turn overlap in characters and
  words; `config.mock.yaml` with the canceller off.
- SPIKE.md S11: AEC3's near-end suppression without echo (not simulated in M0, where echo was
  always present) and this milestone's CPU measurement.

## M5 — Tool calling, and M0's TTS decisions applied (2026-10-23)

**Goal.** Real tool calling: the session's function tools reach the LLM, `tool_choice` is enforced
in the gateway, complete calls from `mlx_lm.server`'s parser become `function_call` items, and the
round trip's history is shaped for the Qwen template in every form AICC produces. With it
(Part B), the G5 normalizer scope and the S3 TTS defaults from DECISIONS (2026-10-23): 0.16 s
chunks, the voice defaults without `ryan`, a proportional `max_tokens`, edge-silence trimming, and
Kokoro limited to segments without Latin letters (D13, owner review pending).

**Not yet run on the real backends.** M0's GPU spikes held the GPU; everything ran against the
mocks and the fake servers, and the real-backend tests are written but not run (see "Pending").

### What was built

| Area | Contents |
|---|---|
| `providers/chat_wire.py` | `Shaping` (cue, system note, `interlude`, `unanswered`, placeholder) and the history rules below; the cue now opens a conversation without a user message |
| `session/history.py` | the output of a deleted call is left out |
| `session/pipeline.py` | calls go to the actor as they complete; a call not offered (`tool_choice: none`, an undeclared function) is dropped with a warning; whitespace at either end of an answer is dropped; per-segment silence trimming before the encoder |
| `protocol/session.py` | `required` and forced functions: `invalid_value` that says why (the server cannot force a call) |
| `audio/trim.py` | `SilenceTrimmer`: an energy gate on 20 ms frames, streaming |
| `providers/mlx_audio_tts.py`, `providers/voices.py` | `token_cap` → `max_tokens` per request (Qwen3-TTS only); a voice can name a speaker per segment language |
| `providers/fallback.py` | `latin_segments` (default off): a segment with Latin letters never goes to the fallback |
| `text/normalize/` | the G5 scope (a forked sub-task; below) |
| `config.py`, `config*.yaml` | `providers.llm.tool_interlude`, `unanswered_calls`, `pending_tool_output`; `output.trim_silence`; `TtsModelConfig.streaming_interval` 0.16 and `token_cap`; `fallback.latin_segments`; `VoiceConfig.speakers`; `ryan` removed; the normalizer keys |
| `tests/fixtures/templates/` | the Qwen3.6 chat template (Apache-2.0, pinned revision); `tests/qwen_template.py` renders a message list with it as `mlx_lm.server` does (`jinja2` joined the dev dependencies) |

**Tool calls on the wire** (PROTOCOL §11.5). Each complete call is its own item: `output_item.added`
→ `conversation.item.added` → `function_call_arguments.delta` (once) → `.done` →
`output_item.done` → `conversation.item.done`; several per response. The wire `call_id` is minted
by the gateway, the server's id kept for history. Text before a call is spoken normally; a call is
delivered when complete, even while that text's audio is still streaming. A call-only response is
done as soon as the LLM is (measured: < 0.5 s after `response.created` with a 2 s TTS that is never
asked). Cancelled before the call is complete: no call; delivered: never retracted, its output
accepted. `tool_choice: auto` sends the session's tools, `none` sends none (session or response);
`required` and forced functions are rejected, because `mlx_lm.server` ignores `tool_choice`.

**History shaping for the Qwen template** (PROTOCOL §11.14; `chat_wire.py`):

1. A call's results follow its assistant `tool_calls` message at once, in call order, whatever
   order the outputs came in (several outputs, one `response.create`).
2. Whatever lies between a call and its result moves after the results, in order: user notes,
   caller speech, later answers. Assistant text before any such user item (AICC's wait line) is
   merged into the call's message, before the call (`tool_interlude: merge`, default), or placed
   after the results (`after_output`). Calls in the interlude before a user item join the call.
3. A call unanswered when the request is built (the caller speaks while the wait line plays)
   gets the placeholder result `pending_tool_output` (`unanswered_calls: pending`, default),
   or none (`keep`), or is left out (`drop`).
4. A deleted call's output is dropped; text-then-call of one response share one message; the cue
   goes first when there is no user message.

Every shape is rendered by the real Qwen3.6 template in the unit tests, so a shape the template
rejects ("No user query found", a system message not first) fails offline. Whether the model
answers well with each setting is for the M0 LLM spike and the backend tests.

**Part B.**

- *Normalizer (G5).* New: `emails` (zh "艾特"/"点", en "at"/"dot"), `cardinals` with
  `min_cardinal_digits: 5`, `en_dates`, `en_amounts` (money ≥ `en_amount_min` 1000 in English);
  phones now include 尾号/last-four, extensions (转, 分机, ext), North American numbers, short
  numbers after a phone keyword, and 10086/12306/12315/12366. Money, decimals, percentages,
  ordinals, other dates and times, units, URLs and brands stay raw (`numbers`, `money`, `dates`,
  `units`, `spacing` stay available, off). To keep 5-digit cardinals from swallowing codes, ID
  detection widened: keyed IDs from 4 digits, new keywords (code, PIN, 取件码, 提货码, 动态码,
  校验码), and a token starting with a letter needs 3 digits (`A1B2C3` is an ID, `500Mbps` is not).
- *TTS defaults.* Qwen3-TTS `streaming_interval` 0.16; voices `vivian` (default; its English
  segments spoken by `aiden`), `serena`, `uncle_fu`, `aiden`; `token_cap` `{base: 25, per_cjk: 6,
  per_other: 2}` codec frames, at most `max_tokens` or the server's 1200 (about 2× normal speech
  plus 2 s for the edges: 91 frames, 7.3 s, for an 11-character Chinese sentence).
- *Silence trimming.* `output.trim_silence` `{threshold_dbfs: -50, frame_ms: 20, max_lead_ms: 60,
  max_tail_ms: 100}` (S3 counted a frame voiced within 35 dB of the loudest and above −55 dBFS;
  a stream needs an absolute gate). The leading silence is held only while silent: the frame that
  turns voiced goes out with the kept 60 ms at once. Silence after a voiced frame waits for the
  next voiced frame (an inner pause, sent whole) or the end (100 ms kept). Trimming happens
  before the audio enters the response's queue, so the segment map, truncation, D5 and the echo
  reference count exactly the audio sent (tested on the wire).
- *Kokoro fallback (D13).* Only for segments without Latin letters (ASCII or full-width), checked
  on the normalized text; such a segment goes to the primary even while the circuit is open, and
  a primary failure fails it. `fallback.latin_segments: true` restores M3's behaviour.

### How to run and verify

```sh
uv run --locked pytest                     # everything but the backend tests
make lint typecheck
uv run --locked pytest tests/integration/test_tool_calls.py tests/unit/test_history_shaping.py
# The real stack (pending the GPU):
make dev
uv run --locked pytest -m backend          # AICC_REPO=<checkout> for AICC's tools and harness
```

### Test results

All green on 2026-10-23: **1491 passed, 19 skipped** (the `backend` tests and the AICC run that
needs `AICC_REPO`) in about 132 s (`uv run --locked pytest`). `ruff check`, `ruff format --check`
and `pyright` (strict on `src/`) report nothing.

| Suite | Tests | New in M5 |
|---|---|---|
| unit | 1339 | history shaping (26: every AICC shape and every setting, rendered by the real template); the LLM provider (3: calls as the real server streams them, with the newline in the call's chunk and two calls in one chunk; a silently dropped second call; shaped history on the wire in both interlude modes); the normalizer (438 → 578); the trimmer (11); the TTS provider (token cap, defaults, the per-language speaker); `FallbackTTS` (6: Latin segments); the pipeline (1: whitespace at the ends) |
| integration | 129 | `test_tool_calls.py` (11: six golden traces, `tool_choice` session and response, an undeclared function, `required`/forced rejected, whitespace before a call, the tool-only deadline); AICC replays through the real LLM provider on the fake `mlx_lm.server` (2: `call_gateway`, `waitline_gateway`, final requests rendered by the template); silence trimming on the wire (2) |
| e2e | 5 | the SDK: a tool round trip with the mock LLM scripted to say a line and call a tool |
| tests/aicc | 19 | unchanged |
| backend | 18 | seven new, not run: see below |

**Golden traces.** Six new: `tool_call_only`, `tool_text_and_call`, `tool_round_trip`,
`tool_cancel_before_call`, `tool_cancel_after_call`, `tool_two_calls`. No existing trace changed:
the mock TTS speaks a sine (nothing to trim), its segments are on the 20 ms grid, and no golden
text has a number, address or ID the new normalizer reads.

### Pending the GPU runs

- `uv run --locked pytest -m backend`, M5's seven:
  - `test_llm_tool_round_trip_with_aiccs_tools`: AICC's `line_session_zh.json` instructions and
    six tools; "RMA1001" → `repair_status` with typed arguments, no markup in the text; the result
    with its hint → a spoken answer, no second call;
  - `test_wait_line_history_on_the_real_template` (4: `merge` / `after_output`, answered /
    unanswered with the pending placeholder): the real template accepts each, the model answers
    in text without repeating the call;
  - `test_sdk_tool_round_trip_with_the_real_stack`: the SDK, AICC's tools, a `function_call`
    item within 3 s of the request, the result, the answer;
  - `test_tts_edges_are_trimmed_and_the_token_cap_leaves_speech_whole`: the real Qwen3-TTS edges
    (S3's ~0.45/0.5 s) trimmed to ≤ 80/120 ms, and normal sentences end well before the cap.
- M2–M4's backend tests (the fallback test speaks Chinese, so the Latin rule does not affect it).
- The M0 LLM spike (S1.3, S1.4): which `tool_interlude` and `unanswered_calls` the model handles
  best, silent drops, and whether Qwen writes text after a call.

### Known gaps, by milestone

| Milestone | What M5 leaves for it |
|---|---|
| M6 | metrics for dropped calls, trimmed silence, token-cap hits; the TTS queue mitigations |
| open | a dropped call (not offered, or rejected by the server's parser next to a valid one) is only logged; a token-cap hit is not detected (the audio just ends) |

### Deviations from ARCHITECTURE, with reasons

1. **History shaping lives in the provider's wire adaptation** (`chat_wire.py`), as §5.2 puts
   backend quirks there; `history.py` stays generic but drops a deleted call's output.
2. **Unanswered calls get a placeholder result by default.** §6.7 says nothing about them; a
   `tool_calls` message followed by a user message is outside what the template's model was
   trained on, and without a result the model may call again or invent one. Configurable.
3. **The cue opens the conversation** (M2 appended it at the end): a greeting or a call the
   assistant already made comes after it, in order.
4. **Calls not offered are dropped**, not delivered: the server parses call markup even with no
   tools sent, so `tool_choice: none` is enforced on the way out too.
5. **Whitespace at either end of an answer is dropped** (M3 kept it): the newline Qwen writes
   around a call would open an empty message or end a transcript. M3's pipeline test changed
   for this (its trailing newline).
6. **A call can be delivered while the message before it is still streaming audio**, so its
   events fall between the message's (GA's own model finishes the message first). AICC tracks
   call items by id and starts the tool at once, which this favours.
7. **`vivian`'s English segments are spoken by `aiden`** (S3's "vivian for zh/mixed, aiden for en",
   read per segment): a female voice in Chinese and a male voice in English within one call.
   Flagged for the owner; `voices.map.vivian.speakers: {}` keeps vivian throughout.
8. **Trimming is an absolute energy gate** (−50 dBFS), not S3's relative one, and keeps up to one
   frame more than the limits when speech starts or ends off the 20 ms grid.
9. **Test changes on purpose.** The AICC replay's fake TTS pace follows 0.16 s chunks (80 ms to
   the first, 50 ms per chunk; M3: 0.32 s every 90 ms), and the flow-voice replay expects
   `aiden` for its English line; the voice-lock test uses `uncle_fu` instead of `ryan`; the TTS
   provider tests pin 0.32 s chunks where they count chunks, and test the new defaults on their
   own; the chat-wire cue test expects the cue first.

### Suggested changes to ARCHITECTURE and DECISIONS

- §5.2 `tool_choice`: `required` and forced functions rejected for good (not "until M5"); calls not
  offered are dropped. §5.2 message shape: the rules of PROTOCOL §11.14 and the cue's position.
- §6.7: the history rules; unanswered calls; delivery interleaved with the message's audio;
  whitespace at the ends of an answer.
- §5.3 TTS row and §7: `streaming_interval` 0.16, `max_tokens` from the text, edge trimming before
  the queue; §10: the new keys (`tool_interlude`, `unanswered_calls`, `pending_tool_output`,
  `output.trim_silence`, `token_cap`, `latin_segments`, `speakers`, the normalizer keys).
- D13: the Latin rule as built (`latin_segments`), pending the owner.
- DECISIONS: the default history shaping (`merge`, `pending`) until the M0 LLM spike reports;
  vivian → aiden for English segments (deviation 7) for the owner; the normalizer's widened ID
  detection.

## M6 — Metrics, the benchmark, and multi-session TTS fairness (2026-10-23)

**Goal.** Measure every turn: the timeline marks of M2–M5 become Prometheus metrics on `/metrics`,
with windowed P50/P95 computed in-process and per-turn records on an opt-in `/debug/turns`; a
benchmark (`bench/`) replays the M0 utterances over `/v1/realtime` and writes the latency and ASR
report (`docs/LATENCY.md`). With it, the multi-session TTS mitigations scheduled by DECISIONS
(2026-10-23): a per-session audio reserve and earliest-deadline-first ordering of the TTS queue.

**Not yet run on the real backends.** M0's LLM spikes hold the GPU. Everything ran on mocks and
the fake servers; the real latency report is pending (LATENCY.md §2 has the command).

### What was built

| Area | Contents |
|---|---|
| `server/metrics.py` | `Metrics` (each gateway's own `CollectorRegistry`; also the sessions' observer), `WindowedQuantiles`, `DebugTurns`, `turn_record`, `quantile` |
| `session/observer.py` | `SessionObserver`, the interface the session layer reports through (turns, responses, barge-ins, echo turns, provider errors, TTS fallback uses), `NullObserver`, `TurnReport` |
| `session/timeline.py` | `STAGES`, `stages_s()`, `e2e_s()`, `marks`, `values` |
| `session/pipeline.py` | `_ReserveSender` (the audio reserve); a deadline and slack on every TTS request; `llm_first_token` marked by the LLM reader when the token arrives; per-segment fallback reporting |
| `session/actor.py` | observer calls next to the M4 echo counters, at each provider error, response end and timeline log |
| `providers/http.py` | `Slots`: a heap of waiters, `fifo` or `edf`, the queue timeout plus the request's slack, `waiting` |
| `providers/base.py`, `mlx_audio_tts.py`, `fallback.py` | `SpeechTiming.deadline`, `slack_s`, `primary_error`; `queue_depth` (`Queued`) |
| `server/app.py`, `gateway.py` | `/metrics`, `/debug/turns`, session and refusal counters, the TTS queue gauge |
| `config.py`, `config*.yaml` | `output.reserve`, `providers.tts.scheduling` / `first_segment_grace_ms`, `observability.quantile_window_turns` / `debug_endpoints` / `debug_sessions` / `debug_turns_per_session`; `config.mock.yaml` turns `/debug/turns` on |
| `bench/` | `bench.py` (CLI, replay, `--serve`), `dataset.py` (manifest or WAV directory, 8 kHz WAVs and G.711 streams to 24 kHz, G.711 codec, speech-end estimate), `scoring.py` (M0's `text_norm`, CER/WER/MER), `report.py` (P50/P95, scores, markdown) |
| `tests/fairness.py` | the fairness harness and its before/after table (`make fairness`) |
| `Makefile`, `pyproject.toml` | `bench`, `bench-mock`, `fairness`; `prometheus-client` (uv add); pyright strict on `bench/` |

**Metrics** (ARCHITECTURE §9; PROTOCOL §1). Histograms: `mlxrt_stage_seconds{stage}` (7 stages),
`mlxrt_e2e_seconds`, `mlxrt_tts_queue_wait_seconds`. Counters: `mlxrt_sessions_total`,
`mlxrt_sessions_refused_total{reason}`, `mlxrt_responses_total{status}`, `mlxrt_barge_ins_total`,
`mlxrt_barge_ins_empty_or_echo_total`, `mlxrt_echo_turns_total`,
`mlxrt_provider_errors_total{backend,kind}` (including primary TTS failures the fallback absorbed),
`mlxrt_tts_fallback_total`. Gauges: `mlxrt_active_sessions`, `mlxrt_backend_ready{backend}` (probed
at each scrape), `mlxrt_tts_queue_depth`, `mlxrt_stage_p50_seconds{stage}`,
`mlxrt_stage_p95_seconds{stage}`, `mlxrt_e2e_p50_seconds`, `mlxrt_e2e_p95_seconds` (linear
interpolation over each series' last `quantile_window_turns`, default 200). Labels are fixed sets:
stage, status, reason, backend (`llm`, `asr`, `tts`, `pipeline`, `vad`), error kind; never an id.

Each stage is named after the mark that ends it and runs from the mark before it:
`speech_stopped` (from `speech_end`), `turn_confirmed`, `asr_done` (from `turn_confirmed` or
`committed`), `llm_first_token` (from the later of `asr_done` and `response_created`),
`first_sentence_ready`, `tts_first_chunk`, `first_audio_sent`. Only timelines that answer a user
audio turn are measured; a verbatim line's or a text-turn response's are not.

**`/debug/turns`** (off by default). One record per finished turn: ids, status, marks in ms after
the earliest, stage durations, end to end, and the logged values; the most recent 64 sessions, at
most 500 turns each. No transcripts, no audio.

**The benchmark.** Replays each utterance in 20 ms frames at real-time speed with `server_vad`
(`silence_duration_ms` 500, as AICC), then streams silence until the turn is answered, so the line
never goes quiet. Per utterance: the client's end to end (from sending the end of the speech to the
first audio delta), the server's stages from `/debug/turns`, the transcript. Report: per-stage and
end-to-end P50/P95, ASR CER (zh) / WER (en) / MER (mix, and all rows), rows with numbers apart;
JSON and markdown. Options: `--concurrency`, `--turns-per-session`, `--sets`, `--ids`,
`--numbers`, `--limit`, `--variant wide|tel16k|tel8k`, `--codec pcm|pcmu|pcma` (G.711 from M7:
the original 8 kHz bytes when the law matches), `--serve CONFIG`. G.711 decode and encode match
`audioop` on every code and every 16-bit value; `text_norm` reproduces all 126 manifest rows.

**Multi-session TTS fairness.**

- *The audio reserve* (`output.reserve`: `hold_ms` 1500, `keepalive_ms` 1000, `min_lead_ms` 400).
  While a response has segments to come, the sender keeps up to 1.5 s of its synthesized audio and
  releases it one delta at a time: after 1 s without a delta, or when the client's modelled buffer
  (the pacer's lead) drops under 400 ms. The response's first delta never waits; once the last
  segment is synthesized the rest goes out as before. Held audio is unsent, so a barge-in discards
  it and truncation and D5 count only what was sent.
- *EDF* (`providers.tts.scheduling: edf`). Each TTS request carries a deadline: when its response's
  synthesized audio runs out (played from when it was made), or for a first segment, when the
  pipeline started (plus `first_segment_grace_ms`, default 0), which puts it ahead of every
  session that still has audio. A request waits at most `queue_timeout_ms` past its deadline, then
  `backend_busy`.

**Fairness, before and after** (`uv run --locked python -m tests.fairness`). The real TTS provider
against the fake `mlx_audio.server` at S3's pace, Mandarin at 220 ms per character, the echoing LLM
mock at 40 tokens/s; every session plays 2 turns of a 2-segment, 5.3 s reply with 3 s between
turns, sessions starting 0.7 s apart (a burst). Rules: delta gap < 2 s, first audio ≤ 3 s after
`response.created`. "before" is `fifo` without the reserve (M5).

Capped pace (TTFC 110 ms, RTF 0.56: Low Power Mode, the pace the M3 tests use):

| sessions | policy | responses: failed / no audio | max delta gap | gaps ≥ 2 s | first audio p50 / max | first audio > 3 s |
|---|---|---|---|---|---|---|
| 2 | before: fifo | 4: 0 / 0 | 2.04 s | 1 | 0.91 / 2.00 s | 0 |
| 2 | fifo + reserve | 4: 0 / 0 | 1.42 s | 0 | 0.92 / 2.00 s | 0 |
| 2 | edf | 4: 0 / 0 | 2.05 s | 1 | 0.91 / 2.00 s | 0 |
| 2 | after: edf + reserve | 4: 0 / 0 | 1.41 s | 0 | 0.94 / 2.06 s | 0 |
| 3 | before: fifo | 6: 2 / 1 | 3.06 s | 3 | 1.30 / 2.07 s | 0 |
| 3 | fifo + reserve | 6: 2 / 1 | 2.43 s | 1 | 1.30 / 2.07 s | 0 |
| 3 | edf | 6: 2 / 0 | 3.07 s | 4 | 1.24 / 2.07 s | 0 |
| 3 | after: edf + reserve | 6: 2 / 0 | 2.43 s | 2 | 1.24 / 2.07 s | 0 |
| 4 | before: fifo | 8: 4 / 3 | 3.06 s | 3 | 1.29 / 2.07 s | 0 |
| 4 | fifo + reserve | 8: 4 / 3 | 2.43 s | 1 | 1.30 / 2.07 s | 0 |
| 4 | edf | 8: 4 / 2 | 3.08 s | 4 | 1.30 / 2.08 s | 0 |
| 4 | after: edf + reserve | 8: 4 / 2 | 2.42 s | 2 | 1.30 / 2.07 s | 0 |
| 8 | before: fifo | 16: 12 / 10 | 3.07 s | 4 | 1.69 / 2.34 s | 0 |
| 8 | fifo + reserve | 16: 12 / 10 | 2.43 s | 2 | 1.69 / 2.35 s | 0 |
| 8 | edf | 16: 13 / 7 | 3.07 s | 2 | 2.24 / 2.49 s | 0 |
| 8 | after: edf + reserve | 16: 13 / 7 | 2.43 s | 2 | 2.23 / 2.47 s | 0 |

Uncapped pace (TTFC 76 ms, RTF 0.33: High Power, `--pace uncapped`):

| sessions | policy | responses: failed / no audio | max delta gap | gaps ≥ 2 s | first audio p50 / max | first audio > 3 s |
|---|---|---|---|---|---|---|
| 2 | before: fifo | 4: 0 / 0 | 1.24 s | 0 | 0.59 / 1.38 s | 0 |
| 2 | after: edf + reserve | 4: 0 / 0 | 0.10 s | 0 | 0.58 / 1.38 s | 0 |
| 3 | before: fifo | 6: 0 / 0 | 2.13 s | 1 | 0.58 / 2.02 s | 0 |
| 3 | after: edf + reserve | 6: 0 / 0 | 1.20 s | 0 | 0.77 / 1.37 s | 0 |
| 4 | before: fifo | 8: 1 / 1 | 2.13 s | 2 | 1.35 / 2.26 s | 0 |
| 4 | after: edf + reserve | 8: 0 / 0 | 2.36 s | 2 | 0.87 / 1.43 s | 0 |
| 8 | before: fifo | 16: 9 / 9 | 2.40 s | 4 | 2.02 / 2.31 s | 0 |
| 8 | after: edf + reserve | 16: 9 / 4 | 2.43 s | 4 | 1.56 / 2.25 s | 0 |

What the numbers say:

- **The reserve fixes the risk DECISIONS named** while the TTS has capacity: the wire gap no longer
  starts when a segment's audio has been sent but when the client's buffer runs out. Two sessions
  capped: 2.04 → 1.41 s; three uncapped: 2.13 → 1.20 s with no gap over 2 s; two uncapped: deltas
  never pause more than 100 ms.
- **EDF alone changes little** in these bursts (fifo already alternates between similar sessions);
  its effect is on who fails: first segments go first, so fewer responses end without any audio
  (8 sessions capped: 10 → 7; uncapped: 9 → 4) and first audio is faster (4 uncapped: p50 1.35 →
  0.87 s, max 2.26 → 1.43 s), while continuations wait longer.
- **One serialized TTS stream is a capacity limit no ordering removes.** At RTF 0.56 it supplies
  about 1.8 sessions speaking at once; at 0.33 about 3. Past that, responses fail with
  `backend_busy` at the queue timeout (the gateway's choice; AICC would abandon them at its own
  2–3 s watchdogs) and gaps reach ~2.4 s (the 2 s queue timeout past the deadline, plus a first
  chunk). No answered response missed the 3 s first-audio rule in any run. More capacity — a second
  TTS process, or a batched worker on mlx-audio's `batch_generate` (S3) — is the remaining lever.
- Tried and not adopted as defaults (measured on 2–4 sessions, a 4-segment reply):
  `first_segment_grace_ms: 1000` (AICC-rule EDF: first segment by 3 s, continuation by 2 s) and
  a 1–1.5 s queue timeout. A shorter timeout keeps every gap under 2 s by failing such requests
  sooner; it trades gaps for failed responses, an owner decision.

### How to run and verify

```sh
uv run --locked pytest                     # everything but the backend tests
make lint typecheck
curl localhost:8000/metrics                # with make dev or make dev-mock running
make bench-mock                            # the benchmark on the mock stack, in-process
make fairness                              # the before/after table (~6 min, no GPU)
# The real stack (pending the GPU):
MLXRT__OBSERVABILITY__DEBUG_ENDPOINTS=true make dev
uv run --locked python -m bench.bench --url ws://127.0.0.1:8000/v1/realtime \
    --input spikes/data/manifest.jsonl --variant tel8k --concurrency 1 --out bench/results/real-tel8k
```

### Test results

All green on 2026-10-23: **1548 passed, 19 skipped** (the `backend` tests and the AICC runs that
need `AICC_REPO`) in about 180 s (`uv run --locked pytest`; +40 s for the real-time fairness
tests). `ruff check`, `ruff format --check` and `pyright` (strict on `src/` and `bench/`) report
nothing.

| Suite | Tests | New in M6 |
|---|---|---|
| unit | 1388 | metrics (12: quantiles against numpy, the window, histograms and gauges from turns, unmeasured timelines, every counter and gauge with bounded labels, separate registries, debug records bounded and only when enabled, the record's shape); `Slots` (5: fifo, edf with ties and arrival-time defaults, a free slot with a zero timeout, timeout plus slack, a cancelled waiter passing on its slot); the reserve and deadlines (4, on a virtual clock: held audio with keepalive and buffer-low releases while the TTS is queued, nothing held at the end, transcript order, each request's deadline and slack, fallback reporting); the bench (15: `text_norm` on the README examples and every manifest row, tokens, alignment, G.711 against `audioop` exhaustively, the speech end, both input layouts with 8 kHz audio, the report's aggregation and markdown); the timeline (3: stages); the configuration (6) |
| integration | 137 | `test_observability.py` (4: every stage of a spoken turn on `/metrics`, provider errors and refusals, `/debug/turns` on and off); `test_bench.py` (2: the bench against `config.mock.yaml` with two sessions and an 8 kHz WAV, an empty selection); `test_tts_fairness.py` (2: two sessions before/after within AICC's rules, eight sessions degrading cleanly) |
| e2e | 5 | unchanged |
| tests/aicc | 19 | unchanged |
| backend | 18 | unchanged, not run |

No golden trace changed: the mocks synthesize faster than real time and finish before the
reserve holds anything back for long, and event order is unchanged by design.

### Pending the GPU runs

- The real latency report (LATENCY.md §2): `tel8k` at concurrency 1, 2 and 4, `wide` at 1.
- The fairness rules on the real TTS: the same burst through `bench --concurrency 2..4`, and the
  owner's choice of `queue_timeout_ms` / `first_segment_grace_ms` under overload.
- M2–M5's backend tests.

### Known gaps, by milestone

| Milestone | What M6 leaves for it |
|---|---|
| M7 | `--codec pcmu/pcma` in the bench is written but untested end to end (the gateway rejects G.711 until M7) |
| open | TTS capacity beyond ~2 (capped) or ~3 (uncapped) concurrently speaking sessions; metrics for dropped tool calls, trimmed silence, token-cap hits and the echo canceller's ERLE and bypasses (M4/M5 asked; not in the M6 brief) |

### Deviations from ARCHITECTURE, with reasons

1. **Stage names are the marks that end them** (`asr_done`, not `asr`), and `llm_first_token` starts
   at the later of `asr_done` and `response_created`: a client may create the response after the
   transcript, and only then can the LLM start.
2. **`llm_first_token` is marked by the pipeline's LLM reader** when the token arrives (the actor
   marked it when it handled the token, which could fall after `first_sentence_ready`). The
   pipeline's marks test lists it.
3. **Extra series beyond §9**: `mlxrt_tts_queue_wait_seconds`, `mlxrt_sessions_refused_total`,
   `mlxrt_barge_ins_empty_or_echo_total`, the end-to-end P50/P95 gauges.
4. **`/metrics` and `/debug/turns` are unauthenticated** (like `/healthz`); the debug endpoint is off
   by default and never carries transcripts. `config.mock.yaml` turns it on.
5. **The queue timeout counts from the request's deadline** under `edf` (from arrival under `fifo`,
   as before): a continuation whose session still has audio may wait that much longer.
6. **The fake `mlx_audio.server` builds its tone with numpy**: the per-sample Python loop blocked
   the event loop ~50 ms per long request, which distorted the gap measurements. Same samples.
7. **`first_segment_grace_ms`** is a knob the brief did not ask for (default 0 keeps "a first
   segment is most urgent"); it lets the owner try rule-based deadlines.

### Suggested changes to ARCHITECTURE and DECISIONS

- §9: the stage definitions above; the extra series; `/debug/turns`'s record and bounds; the
  observer interface between session and server.
- §7 Pacing / §4.4: the audio reserve, and the TTS queue's EDF with deadline-relative timeout.
- §10: `output.reserve`, `providers.tts.scheduling`, `first_segment_grace_ms`, the observability keys.
- §11 Benchmark: `bench/` as built (layouts, replay, the client and server end to end, scoring).
- DECISIONS: the reserve and EDF are on by default; one TTS stream supports about 2 concurrently
  speaking sessions at S3's capped pace (3 uncapped), so more TTS capacity is needed for more; the
  overload trade-off (`queue_timeout_ms`) is for the owner.

## M7 — G.711 8 kHz telephony (2026-10-23)

**Goal.** `audio/pcmu` and `audio/pcma` (G.711 at 8 kHz, D2, D3) in both directions, on the same
canonical path as 24 kHz PCM, so AICC can pass a call's G.711 through untouched with a dedicated
profile (`AcceptsG711: true`, written on the AICC side, D11).

**Not yet run on the real backends.** M0's GPU spikes hold the GPU. Everything ran on mocks, the
fake servers, the real Silero VAD and the real echo canceller; the G.711 backend tests are
written but not run (see "Pending").

### What was built

| Area | Contents |
|---|---|
| `audio/g711.py` | μ-law and A-law from `audioop.c`'s algorithm: 256-entry decode and 65 536-entry encode tables per law, built with numpy at import; `decode`/`encode` (bytes) and `decode_samples`/`encode_samples` (arrays); silence codes |
| `audio/formats.py` | `AudioFormat` (`audio/pcm` 24 kHz 2 bytes a sample; `audio/pcmu`, `audio/pcma` 8 kHz 1 byte): rate, sample width, law, bytes ↔ ms. Every wire byte ↔ ms conversion goes through it |
| `audio/resample.py` | a soxr `quality` per stream (HQ unless asked) |
| `audio/codec.py` | `InputDecoder(format, g711_quality)`: G.711 → LUT decode → 8 → 16 kHz; `OutputEncoder(max_delta_ms, format, quality)`: one per response, resample to the format's rate, encode, ≤ `max_delta_ms` deltas, flushed per segment |
| `config.py`, `config.yaml` | `g711: {input_quality: LQ, output_quality: HQ}` |
| `protocol/` | `decode_audio_format` returns the format (`rate` absent or the format's own, else `invalid_value` on `.rate`); `render_format` (G.711 echoed without `rate`, as the SDK types it); the response's own `audio.output.format`; appends are only base64-decoded here |
| `session/settings.py`, `commands.py`, `response.py` | `input_format` / `output_format` in the session and its patch; `ResponseParams.output_format`; `Response.output_format` |
| `session/actor.py` | the whole-sample check of appends (by the input format); format changes mid-session; each response's format into the pipeline, its message, segments and deliveries |
| `session/input_buffer.py` | decodes in the session's input format; `set_format` flushes the old decoder's tail onto the timeline |
| `session/inbox.py` | queued audio charged by its duration in the format it will be decoded in |
| `session/pipeline.py` | one encoder per response in its format; deltas carry their duration, so pacing, the reserve, the TTS deadline horizon and `max_response_audio_ms` count 8 kHz audio |
| `session/output.py`, `conversation.py` | the ledger, segments and messages count bytes in their own format (truncation, its clamp, D5) |
| `session/echo.py` | the reference decodes what was sent in its format (G.711 → 8 → 16 kHz at `input_quality`), framed per 20 ms of that format |
| `bench/` | `--codec pcmu|pcma` sets both directions; a response in another format stops the run; the dataset's G.711 is the gateway's codec |
| `tests/phone.py` | `codec=`: AICC's G.711 passthrough (no 24 ↔ 8 kHz conversion, the line's own 160-byte frames) |
| `tests/aicc/` | `zz_mlx_live_test.go`: `MLX_REALTIME_G711` turns `GatewayProfile` into its `AcceptsG711: true` variant (format-aware durations; the caller clip converted by AICC's `media.Converter`); `probe.py --g711` |

**Resampling quality** (M0 S7, measured again here with 20 ms appends). 8 → 16 kHz: LQ lags
32 ms on average (61 max), HQ 63 ms (115). LQ is flat to 3 kHz (-0.3 dB), -8 dB at 3.4 kHz;
HQ is flat to 3.6 kHz. Input defaults to LQ (every speech event of a G.711 call comes ~30 ms
sooner); the ASR's CER on LQ vs HQ input is pending the GPU and decides whether it stays.
Output (24 → 8 kHz) defaults to HQ: its lag delays no delta, because each segment's tail is
flushed when the segment ends, and LQ would cut the caller's 3–3.4 kHz band by up to 8 dB.

### How to run and verify

```sh
uv run --locked pytest tests/unit/test_g711.py tests/integration/test_g711.py   # the codec, goldens
uv run --locked pytest                                                           # everything
uv run --locked python tests/aicc/probe.py --aicc-repo ~/src/ai-native-callcenter \
    --serve config.mock.yaml --g711                 # AICC's harness on AcceptsG711: true
uv run --locked python -m bench.bench --serve config.mock.yaml --codec pcmu --limit 4
# The real stack (pending the GPU):
uv run --locked python -m bench.bench --url ws://127.0.0.1:8000/v1/realtime \
    --input spikes/data/manifest.jsonl --variant tel8k --codec pcmu --out bench/results/real-g711
uv run --locked pytest -m backend -k g711
```

### Test results

On 2026-10-23, the whole suite: **1618 passed, 3 failed, 22 skipped** (about 250 s). The three
failures are in `tests/unit/test_mlx_lm_chat.py` (the empty-conversation cue and the warmup's
request), from the LLM-hardening work in progress in the same tree, not from M7; every other
test passes. `ruff check`, `ruff format --check` and `pyright` (strict on `src/` and `bench/`)
report nothing in M7's files (the in-progress `providers/chat_wire.py` and `mlx_lm_chat.py`
have findings of their own).

| Suite | New in M7 (72 tests) |
|---|---|
| unit | `test_g711.py` (22: decode on all 256 codes and encode on all 65 536 samples against `audioop`, both laws; round trips; the formats' arithmetic; G.711 in → 16 kHz and 24 kHz → G.711 deltas, with 20 ms appends and any byte count; LQ vs HQ lag and passband; the `g711` config); the session (5: merges, the ledger at 8 kHz, segments' ms, the inbox's duration charge across format updates); the input buffer (4: G.711 held at 16 kHz on the same timeline, LQ's smaller hold-back, the flush at a format change); the protocol (+13: G.711 accepted with no rate or 8000, input ≠ output, the response's own format; rejected: a wrong `rate` per format, a string rate, beta names, unknown keys) |
| integration | `test_g711.py` (13: golden `g711_turn` and `g711_barge_in`; the same speech-event ms at 8 and 24 kHz; D5 and the truncate clamp at 8 kHz; pacing and the paced flush; `max_response_audio_ms`; format changes after audio flowed; odd byte counts; G.711 session defaults); the AICC replays in the G.711 shape (+8); the simulated echo in μ-law and A-law passthrough (+2: the echo never interrupts, the caller still does, ERLE ≥ 10 dB); the bench with `--codec pcmu` and `pcma` (+2) |
| tests/aicc | the harness on the `AcceptsG711: true` variant (+1, needs `AICC_REPO`) |
| backend | +2, not run (below) |

Golden traces: two new (`g711_turn`, `g711_barge_in`); `protocol_errors` changed in one place,
because `audio/pcmu` is no longer an error: its `e7` is now a μ-law format at 16 kHz, refused
on `.rate`. Every other golden is unchanged.

AICC's own harness ran here against `config.mock.yaml`, both profiles: `gateway` (24 kHz PCM)
and its `AcceptsG711: true` variant (μ-law passed through): all five `TestMLX*` pass
(greeting `0.66s of audio interpreted as PCMU@8000`; barge-in: speech_started 3.2 s into the
line, cancelled at once, the truncate acknowledged, a reply after speech_stopped; both
keypresses), `response.created` worst 2 ms.

### Pending the GPU runs

- `test_sdk_g711_spoken_turn_with_the_real_stack`: the official SDK in μ-law both ways, 160-byte
  frames in real time: transcript, a spoken answer within AICC's deadlines, and that answer's
  μ-law decoded and heard right by the ASR.
- `test_aicc_harness_with_g711_against_the_real_stack`: AICC's harness with `--g711`.
- The bench on `tel8k` with `--codec pcmu` (the original μ-law streams, untouched): latency and
  CER, once with `MLXRT__G711__INPUT_QUALITY=LQ` and once with `HQ`, which settles the default.

### Known gaps, by milestone

| Milestone | What M7 leaves |
|---|---|
| open | the AICC-side profile itself (AICC's repository); a live G.711 call; the input-quality default (CER above) |

### Deviations from ARCHITECTURE, with reasons

1. **One output stream per response, flushed at each segment's end** (the brief: one stateful
   stream per response). Without the flush a segment's last few ms would wait in the resampler
   for the next segment's synthesis, which under multi-session load can be seconds (M6), and a
   segment the fallback speaks may come at another rate. Segments start and end in trimmed
   near-silence, so the joins are inaudible; every count stays exact per segment.
2. **The first G.711 delta needs ≥ ~41 ms of TTS audio** (24 → 8 kHz HQ lets nothing out of a
   shorter first chunk). Qwen3-TTS's first chunk is 160 ms (S3), from which 122 ms goes out at
   once; no delta waits.
3. **G.711 is echoed without `rate`.** The SDK's `AudioPCMU`/`AudioPCMA` have no `rate`; a
   `rate` of 8000 is accepted on input.
4. **The whole-sample check of an append moved from the protocol layer to the session** (only
   the session knows the input format). Same error, same position in the event order
   (`protocol_errors.jsonl` unchanged for it).
5. **Format changes after audio has flowed are accepted**, as GA allows: the input decoder's
   tail is flushed onto the timeline first; output changes apply from the next response (a
   response may set its own, GA's `response.audio.output.format`).
6. **The inbox bound counts duration**, not bytes: an append is charged in the format of the
   latest queued `session.update` that set one, until the actor has handled it.
7. **A missing format `type` means `audio/pcm`** (as before), not "keep the current format".
8. **Golden traces show audio as `<format:bytes>`** (`<pcmu:160>`); PCM traces are unchanged.

### Suggested changes to ARCHITECTURE and DECISIONS

- §4.3: the per-step soxr quality (`g711.input_quality` LQ, `output_quality` HQ; 24 kHz PCM stays
  HQ), the per-response output encoder flushed per segment, and that the echo reference decodes
  G.711 output.
- §4.4: the inbox bound in milliseconds of the input format.
- §10: the `g711` section.
- §2 / D11: a dedicated AICC profile can now set `AcceptsG711: true` (with `tool_choice`).
- DECISIONS: LQ for the 8 → 16 kHz input step, pending the CER comparison; formats may change
  mid-session.
- Makefile (outside M7's write scope): an `aicc-probe-g711` target (`probe.py --g711`).

## LLM hardening — Qwen3.6's tool calls and AICC's 3 s windows (2026-10-23)

**Goal.** G2 accepted Qwen3.6-35B-A3B-4bit on conditions (B1, M0 S1/S2): a lenient tool-call
parser in `backends/llm`, one gateway retry on a dropped call, greedy sampling for requests with
tools, B1's empty-conversation cue, a filter for JSON pseudo-calls, and prompt-cache warming after
instructions changes. S2 also found AICC's two 3 s windows at risk (first audio after
`response.created`; `response.done` of a tool-only response after `response.created`) after a cold
start and after AICC's instructions-only `session.update`. This task builds the conditions,
measures AICC's cases on the real LLM through the gateway, and adds the mitigations the numbers
called for.

**Run on the GPU** (Qwen3.6 @38740b84, mlx-lm 0.32.2, AC powermode 0 (Automatic), M7 code-only in
the same tree): ~3 hours of measurements, every server stopped after each run.

### What was built

| Area | Contents |
|---|---|
| `backends/llm` (now the package `mlxrt_llm`) | `serve.py`: `python -m mlxrt_llm.serve`, `mlx_lm.server`'s own CLI plus `--tools-checkpoint` and `--shared-prefix-checkpoint`; `lenient.py`: the lenient qwen3_coder parser; `checkpoints.py`: prompt-cache checkpoints inside the system message. Both patches refuse to start on anything but mlx-lm 0.32.2's internals (version, the hooks' source, a self-test). `tests/` (73); `pyproject.toml`: a uv_build package, pytest (dev), pyright strict, ruff extending the root config |
| `providers/chat_wire.py` | per-request sampling (`Sampling.of`); `warmup_messages`, `prefix_key`; `PseudoCallFilter`, `is_pseudo_call`; `ChunkReader.dropped_call` |
| `providers/mlx_lm_chat.py` | the dropped-call retry; the pseudo-call filter on replies without tools; `warm_prefix` (one warm-up per prefix in flight, at most `max_warmups_in_flight`, none once a real request for the prefix went out, a sent one runs to its end); a request waits for the warm-up of its prefix |
| `providers/base.py`, `http.py` | `PrefixWarmer` (an optional capability: mocks need nothing); `RequestClock(total_at=…)` |
| `session/actor.py` | a `session.update` that changes the instructions or the tools warms the prefix; while the session's LLM is generating, when it finishes |
| `config.py`, `config.yaml` | `providers.llm`: `tool_temperature` 0, `tool_top_p` 1.0, `retry_dropped_call`, `warm_on_update`, `max_warmups_in_flight`, the cue; `backends.llm`: `python -m mlxrt_llm.serve` with the G2 flags (`--max-tokens 2048 --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.0`) and `--prompt-cache-size 64 --prompt-cache-bytes 8GB --tools-checkpoint --shared-prefix-checkpoint` |
| `tests/fakes/mlx_lm.py` | a one-token request is a warm-up: its own scripts (`warmup()`), recorded in `warmups` |
| `spikes/llm_hardening/` | `run.py` (the measurements), `probe_parser.py` (B1's transfer probe against the wrapper), `summarize.py` (the tables below) |
| `Makefile` | `make test-llm-backend` |

### The LLM backend: `python -m mlxrt_llm.serve`

- **The lenient parser.** mlx-lm picks a tool parser when it loads the model: `tokenizer_utils.load`
  imports `mlx_lm.tool_parsers.<type>` (qwen3_coder for Qwen3.6's template) and reads its
  `parse_tool_call` then, and every request's `ToolCallFormatter` calls that function. So the hook
  is one module attribute, replaced before the server starts. The replacement runs the stock
  parser; only when it fails, the call text has no `<function=` tag and opens with `<NAME>` for a
  NAME the request declared, is that tag rewritten to `<function=NAME>` (and a closing `</NAME>`
  to `</function>`) and the stock parser run again. Everything else parses, or fails, exactly as
  before; an undeclared name is never turned into a call.
- **Guards.** `install()` refuses to start (exit 2, with what changed) unless mlx-lm is 0.32.2,
  `tokenizer_utils.load` still imports the parser module and reads `tool_module.parse_tool_call`,
  `ToolCallFormatter` still calls `self._tool_parser(tool_text, self._tools)`, the markers are
  still `<tool_call>`/`</tool_call>`, the stock parser still rejects a `<NAME>` call and the
  lenient one parses it to the expected call. The checkpoint patch checks
  `ResponseGenerator._tokenize`'s signature and the lines it builds on, and the
  `LRUPromptCache` lines the refresh relies on.
- **Tests** (`make test-llm-backend`, 73): every call B1 captured as dropped (7) is recovered with
  typed arguments (`slots` → `{}`); the 23 it parsed parse identically through the wrapper; an
  undeclared name, a call cut short, prose and an empty text fail as before; mlx-lm's own
  `ToolCallFormatter` drops the captured text with the stock parser and streams it with the
  lenient one; loading the real Qwen3.6 tokenizer after `install()` yields the lenient parser;
  each guard refuses a changed mlx-lm; the CLI flags are taken out before `mlx_lm.server` parses
  its arguments.
- **Cache checkpoints** (`checkpoints.py`). Qwen3.6 is hybrid (Gated DeltaNet + attention): its
  cache cannot be trimmed back, so the server reuses only prefixes it saved whole, at the end of
  the system message (plus the next role token), of the last user message and of the whole
  sequence, and only for requests that end with a user message. The template renders the tools
  first, inside the system message. `--tools-checkpoint` splits the system segment where the
  instructions begin (827 of AICC's 1519 system tokens are its six tools);
  `--shared-prefix-checkpoint` also where the system message stops matching the most similar of
  the last 16 (AICC's instructions up to `当前环节：`, 575 more tokens, which its phase changes
  keep: the first change saves it, every later change of any call of that flow starts there).
  Prompts and outputs are unchanged; only where the cache is saved changes. With either flag a
  cache hit also makes its entry the newest of its type: mlx-lm evicts the oldest *insertion*
  first, so the checkpoints every request starts from (and never re-inserts) aged out first.
  Tested on the real tokenizer: the tools segment ends at `</IMPORTANT>`, the shared one at
  `当前环节：\n`, the segments always concatenate to the prompt, and requests without tools, without
  a system message or ending with a tool result keep the stock segments; and on mlx-lm's own
  `LRUPromptCache`: stock, a checkpoint used by every request is evicted after 3 of 5 one-off
  entries; refreshed, it survives all 5.

### The gateway

- **Sampling (G2).** A request with tools attached goes at `tool_temperature` 0 / `tool_top_p`
  1.0; one without tools at `temperature` 0.7 / `top_p` 0.8, the server's own flags (with
  `--top-k 20 --min-p 0.0`). `max_completion_tokens` 512.
- **The cue** for a conversation without a user message: "(The call has connected. Greet the
  caller as instructed.)" (B1: the full greeting 5/5).
- **A dropped call** (`finish_reason: tool_calls`, no call: the server's parser still rejected
  it) is asked for once more inside the provider, in the same stream, when nothing visible was
  produced yet: the consumer sees one answer whose call comes later. The retry runs at the reply
  sampling (0.7 / 0.8): at temperature 0 the model would write the same call again. It keeps the
  first request's absolute total deadline and is cancelled like any request. After visible text,
  or a second drop, the response fails as before (`bad_response` → `provider_error`), logged
  with the reason. `providers.llm.retry_dropped_call`.
- **Pseudo-calls.** A reply made without tools passes through `PseudoCallFilter`: text flows
  through untouched until a `{` whose next non-blank character is `"` (or a code fence) starts;
  that stretch is held until it closes, then dropped if it is valid JSON shaped like a call (a
  name key — `name`, `tool_name`, `function`, `tool`, `action` — with an arguments key, or such an
  object under `tool_call`/`function_call`/`function`), else released as it was. An object that
  never closes is released at the end (or past 4000 characters). M0's case,
  `好的，我查一下。\n{"tool_name": "repair_status", "tool_input": {...}}`, is spoken as "好的，我查一下。"
  whichever way the stream splits it.
- **Prefix warming.** A `session.update` that changes the instructions or the tools (AICC's phase
  changes, and every session's first update) makes the session send a one-token request with the
  prefix of its next request: the new system message and tools, and the conversation up to its
  last user message (with no user message yet, `[system, assistant "", cue]`, so that the system
  checkpoint ends before the next role token and serves a history that opens with the greeting).
  While this session's LLM is generating — AICC sends the update as soon as a call's arguments are
  done, before `response.done` — it starts when the LLM finishes; while a response only speaks,
  at once. A request whose system message and tools match a warm-up in flight waits for it (at
  most the first-token deadline): racing it prefills the same prompt twice (B1: 2.93 s). The
  server cannot drop a request it is prefilling (it notices a closed connection at its next
  write), so warm-ups must not pile up: at most one per prefix in flight (others are skipped),
  `max_warmups_in_flight` (1) in all, none once a real request for the prefix has gone out, and
  a sent one runs to its end even when its session no longer needs it. The first measurement
  showed why: the 20 calls of the `line` case each sent their warm-up and hung up, and the next
  call's request waited 6.2 s behind 19 abandoned prefills.

### Measurements

**How.** `spikes/llm_hardening/run.py` starts the backend (`python -m mlxrt_llm.serve`, the G2
flags plus each run's own) on :8101 and a gateway on :8103 (`config.mock.yaml` with the real LLM
provider; mock ASR and TTS at zero delay, so the LLM is what is measured), and replays AICC's
cases over `/v1/realtime` with AICC's `line_session_zh.json` (its instructions, announce and 6
tools), one call at a time, n = 20 per case (40 for R2). Spoken replies: the first segment
boundary, the gateway's `first_sentence_ready` after `response_created`. Tool-only replies:
`response.done` after `response.created` at the client. p50 / p95 in s (p95 nearest-rank, which
at n = 20 is the maximum). Caller turns are text items: their ASR is not in the numbers.

**First audio and AICC's windows** are estimated as the brief says: ASR 0.3–0.5 s when a caller
turn created the response (the gateway creates it at the commit, before the transcript, so the
ASR runs inside AICC's window) + the boundary + the TTS's first chunk (0.12 s); a tool-only reply
to a caller turn adds the ASR to its `response.done`. A reply to `response.create` (R2, an
override, a line) has no ASR.

**Cases** (B1's S2, through the gateway). *cold*: a new call whose instructions the server has
never seen (a nonce in front); its first caller turn comes 3 s after the `session.update` (the
greeting line plays first). *warm*: that call's next two turns. *B1 case 3* and *B1's transfer*:
after two turns, an unseen instructions-only update, 3 s, then a caller turn ("好的，没有别的问题了。"
alternating with a question: the former mostly ends in a `hangup` call) or "你说的我都试过了没用，给我转人工客服，
把我的情况跟他们说清楚。". *AICC's own sequence*: after one turn the caller gives an RMA number; the
model calls `repair_status`; as soon as its arguments are done the client sends the update (an
unseen phase) and the `function_call_output`, and `response.create` once `response.done`
arrives (AICC's `SendToolResult` and owed turn): **R2** is that reply, **R3** the caller's next
turn (a question, or "转人工…"). *override*: a new per-response instructions text after a turn.
*line*: "Say exactly this…" (D16).

**Runs** (`spikes/results/llm_hardening_<name>.json`): `baseline` = G2 alone (the lenient parser,
greedy tool sampling, no warming, the stock 10-entry prompt cache); `warm` = + warming;
`cache_both` = the checkpoints and a 64-entry cache, no warming; `final` = warming + checkpoints
+ 64 entries (config.yaml). Prefilled tokens (p50): cold 1625 → 47; R2 1668 → 74; R3 1711 → 123;
override 1025 → 163.

| case | baseline | warm | cache_both | final | est. p95 vs 3 s (final) |
|---|---|---|---|---|---|
| cold: a call whose instructions are new (first caller turn 3 s after the update) | 1.32 / 1.38 | 0.22 / 0.28 | 1.12 / 1.66 | 0.41 / 0.49 | 0.91–1.11 |
| warm: the next caller turns | 0.19 / 0.28 | 0.18 / 0.24 | 0.18 / 0.28 | 0.18 / 0.25 | 0.67–0.87 |
| B1 case 3: unseen instructions update, 3 s, a caller turn (spoken replies) | 1.37 / 1.93 | 0.22 / 0.26 | 0.60 / 1.20 | 0.45 / 0.49 | 0.91–1.11 |
|   … the same turns answered by a hangup call | 1.70 / 2.27 | 0.50 / 0.57 | 0.89 / 1.57 | 0.68 / 1.43 | 1.73–1.93 |
| B1's transfer: unseen update, 3 s, "转人工…" (tool-only) | 2.89 / 3.46 | 1.33 / 1.56 | 1.67 / 1.92 | 1.50 / 1.88 | 2.17–2.38 |
| the same transfer turn without an update | 1.29 / 1.70 | 1.30 / 1.71 | 1.29 / 1.66 | 1.31 / 1.66 | 1.96–2.16 |
| AICC R2: update + tool result → response.create (spoken) | 1.48 / 1.80 | 1.74 / 1.89 | 0.35 / 0.46 | 0.52 / 0.64 | 0.76 |
| AICC R3: the next caller turn (spoken) | 1.45 / 1.49 | 0.28 / 0.33 | 0.38 / 0.47 | 0.30 / 0.32 | 0.74–0.94 |
| AICC R3: the next caller turn is "转人工…" (tool-only) | 3.02 / 4.29 | 1.65 / 2.47 | 1.69 / 3.15 | 1.53 / 2.08 | 2.38–2.58 |
| a new per-response instructions override | 1.38 / 1.56 | 1.24 / 1.45 | 0.32 / 0.38 | 0.40 / 1.15 | 1.27 |
| a verbatim line (no LLM) | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.12 |

The intermediate runs (same cases) isolate each step: `warm_cache` (+64 entries: no change for
one call at a time), `warm_cache_tools` (+ the tools checkpoint: R2 0.94 / 1.22, override 0.37 /
1.10, R3 transfer 1.58 / 2.15), `warm_cache_both` (+ the shared prefix: R2 0.55 / 0.68, R3
transfer 1.59 / 1.98), `final_nowarm` (the final flags without warming), `final_run1` (the final
flags, a run during which the GPU slowed: decode fell from ~75 to ~52 tok/s and R2 drifted from
0.57 to 0.98 s; the same flags without warming, run right after, were normal, and `final` was
re-run; kept as evidence of the risk below).

**No lead time** (`gap0_both`, `gap0_both_nowarm`: the caller turn right after the update, final
flags; tool-only for the transfer):

| case | warming | no warming |
|---|---|---|
| cold (first turn right after the first update) | 0.81 / 1.59 | 1.08 / 1.90 |
| B1 case 3, spoken | 0.50 / 0.95 | 0.50 / 1.40 |
| B1's transfer | 1.74 / 3.28 (one outlier; the rest ≤ 2.02) | 1.77 / 2.23 |

A response that meets its warm-up in flight waits for it and then sends its own small request,
which costs about 0.2 s against one request doing both; racing would prefill twice (B1: 2.93 s).

**Contention** (`contended`, final flags, n = 10): the warm transfer turn while 1 or 2 other
calls decode a long answer (batched): 1.88 / 2.26 and 2.07 / 2.31 s (2.56–2.81 s p95 with ASR),
against 1.31 / 1.66 alone. The transfer call is 96–176 tokens (`queue`, `reason`, a 40–90
character `summary`, `slots`) decoded at 75–90 tok/s alone.

**The prompt cache and memory.** Four calls of different flows taking turns (5 rounds,
`interleaved_*`): with the stock 10 entries each turn re-prefills more of its history as the
others evict it (p50 by round 49, 84, 139, 175, 260 tokens); with 64 it stays flat (35–75 tokens;
under the final flags p50 49, max 164). Footprint of the backend after a run: 21.1 GB with 10
entries, 26.2 GB with 64 (64 cached prompts, 6.25 GB: ~100 MB each, 63 MB of it the linear
layers' fixed state). `--prompt-cache-bytes 8GB` is not the limit that binds: mlx-lm 0.32.2 builds
the cache without it and trims to it only when a request joins a batch. With 64 entries and many
one-off system messages, mlx-lm's eviction (oldest insertion first, hits not counted) dropped the
tools and shared-prefix checkpoints after ~40 phase changes (`cache_both`'s last iteration
re-prefilled 1660 and 1713 tokens: a 3.16 s transfer); with hits counted (`final_nowarm`, same
flags) that iteration prefilled 173.

**The lenient parser, live** (`llm_hardening_parser.json`, B1's transfer probe against the wrapper,
60 turns each): at B1's sampling (0.7 / 0.8) 60/60 transfers delivered, 4 of them written as
`<transfer_to_agent>` and recovered (B1's stock server dropped 11%); at the G2 tool sampling
(greedy) 60/60, none malformed. No call was dropped.

**Ending a tool-only response at its call** instead of at `[DONE]` would save 10.6 ms p50 (13.5
p95, 135 tool-only responses): the stream ends one or two tokens after the call. Not worth losing
a second, parallel call; not done.

### What the numbers say

- **G2 alone misses AICC's windows** in two places: a transfer after an instructions change
  (B1's sequence: 2.89 / 3.46 s, ~3.8–4.0 s with ASR) and, in AICC's own sequence, the caller's
  next turn after the tool-result reply: R2 ends with a tool result, so the server saves no
  checkpoint for the new instructions and R3 re-prefills them all again (1711 tokens; a transfer
  3.02 / 4.29 s, ~4.6–4.8 s with ASR).
- **Warming** fixes every case with lead time (cold, B1's cases) and R3 (its warm-up saves the
  checkpoints R2 cannot), but not R2: AICC's update and `response.create` arrive together, so R2
  waits for the warm-up's full prefill (1.74 / 1.89 s, 0.26 s slower than without warming).
- **The checkpoints** fix what warming cannot: an instructions change re-prefills only the
  changed phase text and the history (R2 0.35–0.55 s), a new override only its own text.
- **Together** (`final`, config.yaml): every case is inside AICC's 3 s windows at p95 one call at
  a time; the tightest is a tool-only transfer on a caller turn, 2.17–2.58 s with ASR.
- Measured but small or not adopted: the 64-entry cache does nothing for one call but keeps several
  calls' turns cheap; ending at the call saves 11 ms. All 5,260 measured responses completed
  without an `error`; the lenient parser recovered 9 `<NAME>` calls among them (tool requests
  greedy) and 4 in the probe, so the retry never fired. It guards correctness, not the deadline
  (B1: a retry does not fit 3 s).

### Remaining risks

1. **Tool-only transfers are decode-bound.** AICC's `transfer_to_agent` call is ~100–176 tokens;
   alone it decodes in 1.2–2.0 s, so a transfer on a caller turn is 2.0–2.6 s p95 with ASR, 2.6–
   2.8 s with one or two other calls decoding, and more with more calls or with S5's TTS/ASR GPU
   contention, which these numbers do not include. Nothing in the gateway shortens the model's
   arguments; AICC can (a short `reason`, the `summary` and `slots` filled after the transfer or
   from the transcript).
2. **Throttling.** `final_run1` lost ~30% of decode speed mid-run with no code change (AC power
   mode 0, "Automatic"; DECISIONS records High Power, `pmset` reads 0; a VM and `syspolicyd` were
   busy); its R3 transfer reached 2.83 s (3.1–3.3 s with ASR).
3. **The first phase change of a flow after a backend start** has no shared-prefix checkpoint yet:
   R2 re-prefills ~1650 tokens once (`final`: 1.76 s, the first iteration).
4. **Calls of different flows starting together** queue their warm-ups (one at a time): in the
   4-call burst some first turns waited 0.9–2.1 s (2.7 s worst with ASR and TTS).
5. **Small prefills that resume from a warm-up's checkpoint** took 0.30–0.40 s instead of 0.15 s
   with the checkpoint flags on (same ~50 tokens prefilled, server pickup unchanged at ~75 ms
   p50); not isolated. It costs the cold and B1 case-3 turns ~0.2 s, well inside their budget.
6. **The patches are version-pinned** to mlx-lm 0.32.2 (they refuse to start otherwise); the
   checkpoints and the LRU refresh change the server's cache bookkeeping, not its outputs.
7. A call dropped **after** visible text still fails the response (as before, now logged).

### How to run and verify

```sh
make test-llm-backend                      # backends/llm: the parser, the checkpoints, the guards
uv run --locked pytest -m "not e2e"        # the gateway (the hardening: test_chat_wire.py,
                                           # test_mlx_lm_chat.py, test_prefix_warming.py,
                                           # test_llm_hardening.py)
make lint typecheck
make dev                                   # the LLM backend is now python -m mlxrt_llm.serve
# The measurements (each run starts its own backend on :8101 and gateway on :8103, ~10 min):
SPIKE_LOG_DIR=<scratch> uv run --project spikes/llm_hardening python spikes/llm_hardening/run.py \
    --name final --warm on --server-args "--prompt-cache-size 64 --prompt-cache-bytes 8GB \
    --tools-checkpoint --shared-prefix-checkpoint"
uv run --project spikes/llm_hardening python spikes/llm_hardening/summarize.py baseline final
SPIKE_LOG_DIR=<scratch> uv run --project spikes/llm_hardening python spikes/llm_hardening/probe_parser.py
```

### Test results

On 2026-10-23, after M7 finished, on the whole tree: the gateway suite **1756 passed, 20
skipped** (`uv run --locked pytest -m "not e2e"`, about 260 s; the skips need the real backends
or an AICC checkout), e2e **5 passed, 2 skipped**; `backends/llm` **73 passed** (`make
test-llm-backend`, the tokenizer tests on the local Qwen3.6 snapshot). `ruff check`, `ruff format
--check`, `pyright` (strict on `src/` and `bench/`) and `pyright -p backends/llm` (strict on
`src/`) report nothing; every `uv.lock` checks. M7's section above lists three failures in
`test_mlx_lm_chat.py` (the cue and the warmup request): they came from this work in progress and
pass now.

New and changed tests:

- `backends/llm/tests/` (73): the lenient parser on B1's captured outputs (7 dropped, 23 parsed)
  and on edge cases; mlx-lm's `ToolCallFormatter` with either parser; the real tokenizer picking
  the lenient parser; every guard; the checkpoints on the real tokenizer and template with AICC's
  instructions and tools; the LRU refresh on mlx-lm's own `LRUPromptCache` (stock: the checkpoint
  is evicted after 3 hits; patched: kept through all 5).
- `tests/unit/test_chat_wire.py` (+112 cases): sampling by tools; the warm-up prefix with and
  without a user message; the prefix key; `dropped_call`; the pseudo-call filter on M0's reply
  split at each of its 81 positions, on 11 call shapes, and on 12 texts it must leave alone (one
  character per delta too).
- `tests/unit/test_mlx_lm_chat.py` (+19): the retry (transparent, twice dropped, after text, off,
  under the first total deadline, cancelled); the filter in a stream, and its release at the end;
  the warm-up request; a request waiting for its prefix's warm-up, another prefix not waiting,
  cancelling the wait; warm-ups failing, off, deduplicated, one at a time, dropped once their
  request went out, dropped when cancelled while queued, outliving their caller once sent, stopped
  by `aclose`.
- `tests/integration/test_prefix_warming.py` (6) and `test_llm_hardening.py` (4): through the
  gateway: when the session warms and with what, the deferral while the LLM generates, a failing
  warm-up; the retry, the filter and the wait on the wire with the real provider; a burst of 20
  calls warming once.
- Changed: `test_mlx_lm_chat.py`'s request-shape test (tool requests at 0 / 1.0), its warmup test
  (the new cue; the fake records it as a warm-up), its malformed-streams test (a dropped call is
  now retried: tested on its own), and its between-tokens cancellation test, which raced (it
  cancelled on the server's hold, sometimes before the client had read the token; it now waits for
  the token first); `test_config.py` (the backend command and the G2 sampling). The fake
  `mlx_lm.server` answers a one-token request as a warm-up, from its own scripts, so tests that
  script their real replies are unaffected by warm-ups. **No golden trace changed** (the goldens
  run the mock LLM).

### Deviations from the brief, with reasons

1. **The retry samples at the reply temperature** (0.7 / 0.8), not the tools' 0: greedy decoding
   would write the same malformed call again (B1: identical output 4/4 at temperature 0).
2. **Warming is not only for instructions-only updates and idle sessions.** Any update that
   changes the instructions or the tools warms (the first update's warm-up is what fixes the cold
   case); and an update that arrives while a response is running does not skip it: it waits until
   this session's LLM is done (AICC sends its update right after a call's arguments, before
   `response.done`, so "no response running" would never warm AICC's own sequence), and starts at
   once while the response is only speaking.
3. **Warm-ups are deduplicated and serialized** (`max_warmups_in_flight`), and a sent one runs to
   its end: mlx_lm.server cannot drop a request it is prefilling, and the first measurement lost a
   cold turn to 6.2 s of queued, abandoned prefills.
4. **Two server-side cache checkpoints and a true LRU** (`--tools-checkpoint`,
   `--shared-prefix-checkpoint`) were not asked for: they are task 8's mitigation (and are what
   keep AICC's own sequence fast; warming alone cannot, because AICC's update and response.create
   come together). Opt-in flags, version-guarded like the parser.
5. **`--prompt-cache-bytes` only binds when a request joins a batch** in mlx-lm 0.32.2
   (`LRUPromptCache` is built without it); `--prompt-cache-size 64` is the limit that holds.

### Suggested changes to ARCHITECTURE and DECISIONS

- §2 (process table, Startup): the llm process is `python -m mlxrt_llm.serve` (mlx_lm.server plus
  the lenient parser and the cache checkpoints) with `--max-tokens 2048 --temp 0.7 --top-p 0.8
  --top-k 20 --min-p 0.0 --prompt-cache-size 64 --prompt-cache-bytes 8GB --tools-checkpoint
  --shared-prefix-checkpoint`; sessions warm the prompt cache on `session.update`.
- §5.1: `PrefixWarmer`, an optional provider capability (`warm_prefix`).
- §5.2 (the LLM provider): per-request sampling (tools → 0 / 1.0); the cue; tool calls: the
  backend's lenient parser, the retry of a dropped call before visible text, `bad_response`
  after; the pseudo-call filter; prefix warming and the wait for it, with its rules.
- §5.3: the hybrid model's cache (whole prefixes only, at segment ends, only for requests ending
  with a user message); the server notices a closed connection only at its next write, so
  prefills are never cancelled; the checkpoints and the LRU refresh; ~100 MB per cached prompt.
- §10: the new `providers.llm` keys.
- DECISIONS: the warming policy (any instructions/tools change; deferred while the session's LLM
  generates; one per prefix, one at a time, a sent one runs to its end); the retry's sampling;
  the cache checkpoints and refresh, opt-in flags on; `--prompt-cache-size 64` (+6 GB resident);
  AICC's transfer call is decode-bound (Remaining risks, 1): a request to AICC, not a gateway fix.

## G7 dual path, D13, capacity (2026-10-23)

**Goal.** The three owner decisions of 2026-10-23 (DECISIONS):

- **G7 final, dual path.** AEC3 keeps cleaning what Silero and Smart Turn hear; a SpeexDSP linear
  canceller, aligned by GCC-PHAT, feeds the input buffer and the ASR. SpeexDSP's BSD sources are
  vendored and built through uv, with no Homebrew library.
- **D13 final.** The TTS fallback is Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit @08c72cad, with the
  primary's speakers and Latin segments allowed. Kokoro is dropped.
- **Capacity.** At most 3 concurrent calls (`max_sessions: 3`), with the other defaults sized to
  match.

**Run on the GPU** (High Power on AC): the double-talk ΔCER with the real ASR
(Qwen3-ASR-1.7B-8bit @a8379a2, `HF_HUB_OFFLINE=1`, 5120 requests, 0 errors), and the real TTS and
ASR servers for D13. Every server was stopped afterwards.

### What was built

| Area | Contents |
|---|---|
| `vendor/speexdsp-aec` (new) | A uv project (`uv init --no-workspace --lib --build-backend setuptools`) holding a setuptools-built CPython extension. **Sources:** `speexdsp/` holds the 17 files of the SpeexDSP 1.2.1 release tarball that the echo canceller (`mdf.c`), the preprocessor (`preprocess.c`, `filterbank.c`) and the smallft FFT need, copied unchanged, with `COPYING`. **Provenance:** `speexdsp/SOURCE.json` records the version, the URL https://downloads.xiph.org/releases/speex/speexdsp-1.2.1.tar.gz, the tarball's sha256 `8c777343…aeb54d3d` (as Xiph's `SHA256SUMS.txt` publishes it) and every file's sha256. **Build:** `setup.py` defines what SpeexDSP's own `configure` defines on macOS (`FLOATING_POINT`, `USE_SMALLFT`, `VAR_ARRAYS`), adds `OS_SUPPORT_CUSTOM`, and builds with `-fvisibility=hidden`, so only `PyInit__speexdsp` is exported and only libSystem is linked. `include/os_support_custom.h` counts SpeexDSP's warnings per thread instead of printing them. **Binding:** `src/speexdsp_aec/_speexdsp.c` is `EchoCanceller(frame_size, filter_length, sample_rate, *, preprocess=False)`, plus `process`, `reset`, `close`, the `frames` and `warnings` counters, and a `.pyi`. |
| `pyproject.toml`, `uv.lock` | `speexdsp-aec` is a path dependency (`uv add --no-workspace`). `build-constraint-dependencies = ["setuptools==84.0.0"]` is recorded in `uv.lock`'s manifest, and the package pins the same in its `build-system`. `cache-keys` make uv rebuild it when a C or header file changes, and only then. |
| `audio/aec.py` | `SpeexLinear`, the MDF filter of `filter_ms` with an optional preprocessor. Frames are checked before the native call, as for AEC3. `LowPass` is the 63-tap, 3.8 kHz linear-phase FIR from M0 S11, streamed with its history. |
| `session/echo.py` | **EchoReference.** It places Speex's reference from confirmed GCC-PHAT estimates (`_align_speex`), and `render_speex` reads the model at Speex's own shift. The hold-back waits for whichever reference reads furthest ahead. Re-anchoring leaves Speex's reference where the echo was measured, and ignores estimates of audio placed before the correction. **AsrPath** (the pre-filter, then the engine) runs in the same executor work as AEC3. **EchoStage** returns `Processed(start, pcm, asr)`, and its estimate loop stops at a re-alignment. **EchoRuntime** builds both engines from `AecConfig`. The `echo summary` log adds `speex_*` fields. |
| `session/input_buffer.py`, `feeder.py`, `signals.py`, `actor.py` | The buffer has a third track, the ASR path's output. `asr_input` (`speex`, `aec3` or `raw`) picks which track a commit takes, and Smart Turn keeps reading AEC3's track. `VadUpdate.processed_asr` carries the ASR path's output. |
| `config.py`, `config*.yaml` | `aec.asr_input` takes `speex`, `aec3` or `raw` (default `speex`; M4's `processed` is now `aec3`). `aec.speex` holds `filter_ms` 64, `target_delay_ms` 20, `max_delay_ms` 44, `prefilter_hz` 3800 and `preprocess` true, validated as target ≤ max < filter. `speex` needs `delay_guard`. Capacity is set in the "Capacity" section below. |
| D13: `config.py`, `providers/{mlx_audio_tts,fallback,registry,voices}.py`, `server/gateway.py`, `bench/bench.py`, `config.yaml` | **Fallback model:** the 4-bit build, asked exactly like the primary (same speaker, `chinese`/`english`, 0.16 s chunks, the token cap) for any segment, Latin letters included. **Removed:** the Kokoro paths (`family`, `voices_dir`, `latin_segments`, the voice map's `fallback`, the voice-pack lookup through the HF cache, `BackendsConfig.pins()`, `Gateway(environ=…)`). **Warmup:** the primary speaks every voice; the fallback speaks only the default voice (1.4 s against 6.9 s). **Pins:** `backends.tts.fetch` pins 049ef77f (8-bit) and 08c72cad (4-bit). |
| `backends/audio` | `uv remove` takes out what only Kokoro needed (misaki[zh], spacy with en_core_web_sm, num2words, phonemizer-fork, espeakng-loader); the lock goes from 113 to 55 packages. In mlx_audio, only kokoro, kitten_tts and chatterbox import them. |
| `spikes/g7dual` (new uv project) | The measurement harness. `g7_chain.py` builds B4's calls; `g7_gateway.py` drives the gateway's own echo path on a manual clock; `g7_run.py` generates the calls, runs the variants, sends the clips to the ASR and measures CPU. |
| `README.md`, `docs/PROTOCOL.md` | The dual path, the build prerequisite (a C compiler), capacity, and D13 (status line, admission, §11.12, §11.13). |

### How the ASR path works

The canceller executor runs both paths on every 10 ms frame.

- **Detection path.** AEC3 runs on AEC3's reference (unchanged from M4), and Silero and Smart Turn
  hear its output.
- **ASR path.** The capture and Speex's reference go through the 3.8 kHz pre-filter, then Speex.
  Its output is written over the buffer's ASR track.

**Alignment.**

- Speex's reference is the same playout model, read at Speex's own shift. A confirmed estimate puts
  the echo 20 ms behind the reference; the shift moves again only when an estimate leaves
  [0, 44] ms (M0 S11).
- Until the first confirmed estimate, the ASR track is the input as received. In the runs below this
  happened 2.5 s into the call (median; 5 s at most). A line without echo never aligns, so the ASR
  keeps its raw input.

**Re-anchoring.** M4's truncate re-anchoring moves the model by whole frames for the audio that
follows it. Moving Speex's reference along with it pushed the echo out of the 44 ms keep range
after almost every barge-in. So:

- Speex's shift absorbs the move;
- estimates of audio placed before the move are ignored for 2 s.

Effect on Speex's re-alignments in 32 barge-in calls:

| Handling | Re-alignments |
|---|---|
| No compensation | 47 |
| Compensation, no hold | 85 |
| Both (the build) | 7 |

**No bypass.** AEC3's no-echo bypass (M4) does not apply to the ASR path. A linear filter cannot
silence a caller, and the bypass misfires in long double talk: in 16 of the 32 continuous
double-talk calls, GCC-PHAT found no distinct peak for 1.5 s.

### Measurements

**Setup.**

- **Calls.** B4's double-talk grid, rebuilt with B4's own chain (`spikes/aec`), stops at AICC's
  24 kHz uplink. Soxr of this uplink is bit-identical to B4's cached capture.
- **Grid.** 40 zh and 40 en FLEURS utterances × ERL 6/15 dB × near end 0/−15 dB re the far end;
  G.168 m1–m8; 32 calls per layout.
- **Two layouts:**
  - `dt`: B4's continuous double talk, with each utterance entirely under the far end;
  - **barge-in realistic**: the far end stops 0.5 s after each onset. AICC flushes and truncates to
    what it played, and the next answer starts 1 s after the utterance. Onsets land past the 2 s
    echo guard.
- **The gateway's path.** Each call goes through the gateway's own classes on a manual clock: the
  InputBuffer's decoder, `EchoReference` (AICC playout model, learned tick phase, truncates),
  `EchoStage` (AEC3, Speex, GCC-PHAT, hold-back) and the tracks the InputBuffer would commit.
- **Scoring.** Each utterance ±300 ms is transcribed by the gateway's `MlxAudioASR` provider (with
  the language), scored as B4 scored it (CER zh, WER en), and paired against the same chain without
  echo. Each cell has 80 utterances, with a 95 % bootstrap CI.
- Raw results: `spikes/results/g7dual_{cer,runs,cpu}.json`.

**ΔCER, points over the clean chain** (C2 limits: ≤ 2 at 0 dB, ≤ 5 at −15 dB):

| ASR input | zh 0 dB | zh −15 dB | en 0 dB | en −15 dB |
|---|---|---|---|---|
| *Continuous double talk* | | | | |
| raw | +13.0 [8.3, 18.6] | +68.1 [56.4, 80.5] | +5.7 [2.8, 9.4] | +61.4 [50.6, 71.7] |
| aec3 | +2.4 [1.3, 3.7] | +13.7 [10.4, 17.5] | +6.5 [3.7, 9.9] | +33.2 [27.2, 39.4] |
| speex, linear | +0.6 [−0.1, 1.7] | +2.9 [1.1, 5.1] | +0.4 [−0.2, 1.0] | +3.6 [0.6, 7.4] |
| **speex + preprocessor (default)** | **+0.5** [−0.3, 1.4] | **+1.8** [0.3, 3.5] | **+0.6** [−0.4, 1.9] | **+1.6** [−0.6, 4.4] |
| speex, no pre-filter | +0.7 | +3.5 | +0.7 | +3.1 |
| speex + preprocessor, no pre-filter | +0.1 | +2.0 | +0.8 | +2.3 |
| speex, 128 ms filter (keep 0–108 ms) | +0.7 | +2.6 | +0.1 | +3.6 |
| *Barge-in realistic* | | | | |
| raw | +7.5 [5.3, 9.9] | +10.3 [8.0, 12.6] | −0.2 [−0.8, 0.5] | +2.1 [0.7, 3.8] |
| aec3 | +1.1 [0.0, 2.4] | +0.2 [−0.2, 0.7] | +0.9 [0.2, 1.6] | +1.9 [0.4, 3.5] |
| speex, linear | +1.9 [0.6, 3.4] | +2.5 [1.1, 4.0] | −0.5 [−1.1, 0.1] | −0.7 [−1.4, −0.1] |
| **speex + preprocessor (default)** | **+0.8** [0.0, 1.9] | **+0.3** [−0.9, 1.4] | **+0.0** [−0.7, 0.6] | **−0.2** [−1.1, 0.6] |
| speex, no pre-filter | +1.4 | +2.3 | +0.0 | −0.1 |
| speex + preprocessor, no pre-filter | +0.6 | +1.5 | +0.0 | +0.1 |
| speex, 128 ms filter | +1.8 | +1.6 | −0.4 | −0.7 |

Clean error rates were zh 9.6–10.7 % and en 3.3–5.1 %. The raw rows of continuous double talk equal
B4's no-canceller row exactly.

**What the numbers say.**

- **The default meets C2 in all eight cells.** The point estimates are within the limits with room
  to spare; the largest upper CI bound is 4.4 against the limit of 5.
- **AEC3 fails continuous double talk** (3 of 4 cells; +33 en at −15 dB) **but passes the barge-in
  realistic case.** Our output stops 0.5 s after the onset, so its suppressor only eats the start.
- **Raw fails both layouts in Chinese.** Even 0.5 s of echo over the onset costs 7–10 points.
- **The preprocessor shows a gain.** Paired against linear Speex it is significantly better in 3 of
  8 cells:
  - continuous en −15: −2.0 [−3.9, −0.4];
  - barge-in zh 0: −1.0 [−2.1, −0.1];
  - barge-in zh −15: −2.2 [−3.9, −0.7].

  It is worse only in barge-in en 0 (+0.5 [0.0, 1.0]), where linear Speex scored below the clean
  chain. So it is on by default, as the brief's rule asks.
- **The pre-filter shows no ASR gain, and is kept.**
  - Without it, linear Speex is no different (all CIs include 0).
  - With the preprocessor, dropping it costs barge-in zh −15: +1.2 [0.1, 2.5].
  - B4 measured with it, and it raises Speex's ERLE: median 17.9 against 14.7 dB in the barge-in
    calls.
  - The input passes through unfiltered until Speex aligns, so a line without echo keeps its full
    band.
- **A 128 ms filter buys nothing:** no cell differs significantly.
- **Versus B4's offline Speex** (+0.9/+0.6/+0.7/−0.8 linear; +0.1/+0.2/+0.6/−0.6 with the
  preprocessor), the gateway path loses 1–4 points at −15 dB. That is the price of a learned rather
  than known tick phase: Speex re-aligned 23 times in the 32 continuous calls.

**The echo path's counters** (default configuration; 32 calls per layout):

- Speex aligned at 2.5 s (median; 5.0 s at most).
- Re-alignments: 23 (continuous) and 7 (barge-in).
- 0 filter resets.
- 43 re-anchorings in the 32 barge-in calls.
- ERLE medians (all reference-active windows, double talk included):
  - AEC3: 8.1 dB (continuous) and 29.9 dB (barge-in);
  - Speex with its preprocessor: 5.4 dB (continuous) and 31.9 dB (barge-in).

**CPU per session-second** (`spikes/results/g7dual_cpu.json`; thread time in the canceller
executor; default configuration; mean of 8 calls, 4 per layout):

| Part | ms of CPU per second of audio |
|---|---|
| AEC3 | 5.44 |
| Speex path (pre-filter, MDF 64 ms, preprocessor) | 2.01 |
| Whole executor work (both engines, GCC-PHAT, ERLE) | **9.25** |

- Other Speex paths, from the run summaries:
  - linear: 1.62 ms;
  - without the pre-filter: 1.00 ms;
  - 128 ms filter: 1.83 ms.
- The engines alone, in the unit tests: AEC3 6.05 ms, Speex 0.94 ms.
- On the gateway's executor (`aec.workers` 2), as fast as possible:
  - 1 call runs at 87× real time;
  - 3 calls at 131× aggregate (11.0 ms per session-second);
  - 8 calls at 135×.

  Three calls use about 3 % of one core.

### D13: the 4-bit fallback on the real TTS server

**Primary forced to fail** (`pytest -m backend -k "asr_transcribes or tts or fallback or 4bit"`
against `mlx_audio.server` from `backends/audio`, TTS on :8103 and ASR on :8102):

| How it failed | What spoke | First chunk after the request | RTF |
|---|---|---|---|
| A model the server cannot load (500) | the 4-bit model | 203 ms | 0.26 |
| A 1 ms first-chunk deadline | the 4-bit model | 102 ms | 0.24 |

**The 4-bit model speaking directly.**

- zh, en and the mixed line 您好，请登录 App 查看您的订单。: first chunk 45–51 ms, RTF 0.22–0.23.
- The ASR heard the mixed line right, "App" included.
- The 8-bit primary, for comparison: first chunk 54 ms, RTF 0.25.

**Runs.**

- 9 passed, twice in a row: once warm, once on freshly started servers.
- The run before them, the first after the backend environment changed, had 1 failure on a cold page
  cache (34 s of model loading) that did not recur. Its test was not recorded.
- An earlier copy of the same change passed the same tests three times.

### Capacity

| Setting | Value | Why |
|---|---|---|
| `gateway.max_sessions` | 8 → **3** | `config.py`, `config.yaml` and `config.mock.yaml` |
| `providers.llm.max_in_flight` | 8 → **4** | one response per call, plus the prefix warm-up, which takes the same slots |
| `providers.asr.max_in_flight` | 4 → **3** | one turn per call |
| `providers.tts.max_in_flight` | 1 | the server is serialized; M6 measured about 3 speaking sessions at full speed |
| `aec.workers` | 2 | 3 calls need about 3 % of a core (above) |
| `turn.smart_turn.workers` | 2 | S7 sized K = 2 for up to 64 sessions |

`--prompt-cache-size 64` stays (memory only).

### How to run and verify

```sh
uv sync --locked                    # builds vendor/speexdsp-aec (needs a C compiler)
uv run --locked pytest              # everything but the backend tests
make lint typecheck
cd spikes/g7dual                    # the measurements (calls cached in git-ignored spikes/data)
uv run --locked python g7_run.py gen && uv run --locked python g7_run.py run
uv run --locked python g7_run.py asr    # the real ASR on :8112 (GPU), stopped at the end
uv run --locked python g7_run.py cpu && uv run --locked python g7_run.py runs
```

### Test results

On 2026-10-23 `ruff check`, `ruff format --check` and `pyright` (strict on `src/` and `bench/`)
report nothing. `uv run --locked pytest`: **1807 passed, 25 skipped** in about 260 s (before: 1761
and 22). The skips are the backend tests (marked `backend`, 3 of them new) and the AICC runs that
need `AICC_REPO`.
No golden trace changed.

**New and changed tests.**

- `test_aec.py`:
  - both engines refuse partial frames;
  - Speex cancels an aligned line echo (> 20 dB) and keeps the caller at full level in double talk;
  - Speex cannot cancel echo outside its filter;
  - CPU;
  - the C binding's own buffer checks (length, item type, contiguity) and size bounds;
  - a busy or closed canceller refuses calls (a re-entrant `__buffer__`);
  - Speex runs without the GIL;
  - the pre-filter's band;
  - the vendored files against `SOURCE.json`.
- `test_echo.py`:
  - Speex's alignment from confirmed estimates, its keep range, and AEC3's guard shift;
  - nothing is aligned without the `speex` input;
  - `render_speex`;
  - the hold-back takes the smaller shift;
  - re-anchoring compensation and the hold;
  - the stage before and after alignment, with AEC3's bypass;
  - the pre-filter only once Speex cancels;
  - the summary fields.
- `test_input_buffer.py`: the Speex track in commits, the tracks through trim, clear and commit, and
  every input without cancellers.
- `test_feeder.py`: `processed_asr` is carried.
- `test_config.py`: the dual path, capacity and the validation errors.
- `test_echo.py` (integration): Speex aligns within 5 s with no resets, on PCM and on G.711
  passthrough.
- D13: the fallback is asked like the primary; warmup and readiness; Latin segments fall back; a
  mixed segment on the wire; the backend tests above.

### Deviations from the brief, with reasons

1. **The package is `speexdsp-aec` (`import speexdsp_aec`).** An `mlxrt_*` name trips the
   import-graph rule that nothing in `src/` imports `mlx*`.
2. **The preprocessor is vendored and on by default.** The data shows a gain (above). It is a config
   switch (`aec.speex.preprocess`).
3. **Re-anchoring leaves Speex's reference alone.** Speex's shift absorbs the move, and pre-move
   estimates are held off for 2 s. Not in B4; the numbers are in "How the ASR path works".
4. **The ASR path never takes AEC3's bypass.** It misfires in long double talk.
5. **The ASR track passes the input through unfiltered until Speex aligns.** The pre-filter applies
   only once Speex cancels.
6. **Configuration changes.**
   - `aec.asr_input: processed` is now `aec3`; an old config fails at startup, naming the key.
   - `speex` without `delay_guard` is refused, because nothing would align it.
7. **EchoStage's estimate loop stops after a re-alignment.** Later windows of the same batch were
   measured against the old reference. This affects AEC3's guard too.
8. **The ΔCER cuts utterances by ground truth (±300 ms), as B4 did.** It does not use the VAD's
   turns. The tracks are exactly what the InputBuffer commits.

### Remaining risks

- **The echo is simulated.** No live echo has been measured: G.168 paths, AICC's model, no clock
  drift.
- **The start of a call.** For the first ~2.5 s of a call's first answer the ASR hears echo, and
  Speex keeps converging for a few seconds after that.
- **The preprocessor's denoiser is unmeasured on noisy lines.** The runs had −65 dBm0 of noise.
- **M4's two-frame re-anchoring fires often** (43 in 32 barge-in calls), partly spuriously. Speex is
  now immune, but AEC3's reference still moves.
- **M4's bypass misfires in long double talk** (16 of 32 continuous calls). AEC3 then passes echo to
  Silero until the next confirmed estimate.
- **A wideband client with echo** gets a 3.8 kHz ASR input.
- **The build.**
  - It needs a C compiler.
  - The build backend is pinned by version but not by hash: uv locks no build environment.
  - SpeexDSP does not check its own allocations. Sizes are bounded, so only exhaustion of memory
    could crash it.

### Suggested changes to ARCHITECTURE and DECISIONS

- **ARCHITECTURE.**
  - §3: `vendor/speexdsp-aec` (vendored BSD sources, a setuptools extension built by uv, setuptools
    pinned in `uv.lock`'s build constraints).
  - §4.3: the dual path in the diagram, and the InputBuffer's third track.
  - §6.5: Speex's alignment (target 20 ms, keep [0, 44] ms, first estimate ≈ 2.5 s, pass-through
    until then), re-anchoring compensation and its hold, no bypass for the ASR path, the
    preprocessor and the pre-filter.
  - §10: `aec.asr_input` (`speex | aec3 | raw`), `aec.speex.*`, `gateway.max_sessions` 3, and
    LLM 4 / ASR 3 in flight.
  - §2: 3 concurrent calls.
  - §5.3 and §12 D13: the 4-bit Qwen3-TTS fallback.
- **DECISIONS.**
  - G7 dual path implemented, with the measured ΔCER and CPU.
  - The preprocessor on (it shows a gain); the pre-filter kept.
  - `processed` renamed to `aec3`.
  - Capacity: max_sessions 3, LLM in flight 4, ASR in flight 3, executors unchanged.
  - D13 implemented: Kokoro-only dependencies removed from backends/audio; the fallback warms up
    with the default voice only.

## Real-stack validation (2026-10-23/24)

**Goal.** Run the whole service on the real backends for the first time: every milestone's
pending backend tests, AICC's own harness in both profiles, the benchmark (the first real
latency report, `docs/LATENCY.md`), M0's S5, S6 and S10 through the real gateway (contention, a
30-minute soak, the budget), backend recovery, the G.711 input-quality default, and gates G4 and
G6. Fix what breaks; report the rest.

**Conditions.**

- Apple M3 Max (16 cores), 128 GB, macOS 26.6.2 (25G83), AC power, `pmset` AC `powermode 2`
  (High Power; battery 80 %, not charging). A `caffeinate` assertion kept the machine awake;
  nothing else used the GPU. `iogpu.wired_limit_mb` 0 (the default); `mx.device_info()`:
  `applegpu_g15s`, `max_recommended_working_set_size` 115.4 GB.
- Python 3.12.12, uv 0.13.5, Go 1.27.3. Gateway: onnxruntime 1.31.0, openai 3.36.1, soxr 1.1.0,
  numpy 2.5.4. `backends/llm`: mlx 0.32.4, mlx-lm 0.32.2. `backends/audio`: mlx 0.32.4,
  mlx-audio 0.6.1, transformers 5.20.2.
- Models from the HF cache (`HF_HUB_OFFLINE=1`): Qwen3.6-35B-A3B-4bit @38740b84,
  Qwen3-ASR-1.7B-8bit @a8379a2e, Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit @049ef77f and its 4-bit
  build @08c72cad.
- mlx-realtime bb75826 plus this working tree; AICC 8ea1f95 (read only).
- Every server was stopped at the end.

### Startup

`MLXRT__OBSERVABILITY__DEBUG_ENDPOINTS=true make dev` started the four processes; the gateway was
ready 13.3 s after it started (a warm page cache): the ASR warm after 4.0 s, the LLM after 7.7 s,
the TTS after 13.3 s (8 requests on the 8-bit primary, every voice, and 2 on the 4-bit fallback).
Footprints after warmup: LLM 22.9 GB, TTS 3.8 GB, ASR 2.5 GB, gateway 0.25 GB. Each backend prints
uv's `VIRTUAL_ENV=.venv does not match the project environment` warning once (the variable is
inherited through `uv run`); harmless.

### Backend tests (`uv run --locked pytest -m backend`, `AICC_REPO` set)

The 23 tests M2–M7 left pending. First run, before any fix: **22 passed, 1 failed**
(`test_aicc_harness_against_the_real_stack`, below), and pytest exited 1 even when every test
passed (bug 8). Final run, on the final tree: **23 passed**, exit 0. Measured on the way: TTS first
chunk 50–56 ms at RTF 0.25 (8-bit, zh and en); the 4-bit fallback 56–74 ms at RTF 0.26–0.39, and
48–125 ms after a forced primary failure; warmup 7.1–7.2 s; the SDK's Chinese turn first audio
676–705 ms after `response.created` (PCM) and 702–748 ms (G.711, the answer heard back right by
the ASR); the SDK's barge-in confirmed 1503 ms after the onset in the first run (the echo
canceller wrongly back on, bug 4) and 539 ms in the final one.

### AICC's own harness (`tests/aicc/probe.py`)

Every `TestMLX*` test in the `gateway` profile (24 kHz PCM) and with `--g711` (`AcceptsG711`,
μ-law passed through), directly and inside the backend tests:

| Run | Greeting (first audio) | `response.created` | Lines | Barge-in | Keypresses |
|---|---|---|---|---|---|
| PCM, first backend run | pass (902 ms) | **fail**: "response 6 never finished" | pass | pass | pass |
| G.711, first backend run | pass (562 ms) | pass (8/8, worst 3 ms) | pass | pass | pass |
| PCM, direct | pass (579 ms) | pass (8/8, worst 3 ms) | pass | pass | pass |
| G.711, direct | pass (370 ms) | pass (8/8, worst 13 ms) | pass | pass | pass |
| PCM, final backend run | pass (530 ms) | pass (8/8, worst 2 ms) | pass | pass | pass |
| G.711, final backend run | pass (526 ms) | pass (8/8, worst 2 ms) | pass | pass | pass |

**The failure is the model, not the gateway.** `measureResponseCreated` sends eight bare
`response.create` in a row after the greeting, with no caller turn between them, and waits 30 s
for each to finish. The sixth answer was a monologue (13 TTS segments in 28 s, still speaking at
30 s); `response.created` came within 1–2 ms every time. Fourteen more runs of the test through a
logging proxy passed: 112 such answers, the greeting repeated, plus three odd ones ("I'm sorry, but
I am an AI assistant and cannot answer the phone.", "Sure! Here's a short response for when the
call connects: …", a bare quote). AICC asks for a turn after a tool result or a note, never after
the model's own words alone, so this is the test's conversation at temperature 0.7 (1 run in 16).
`max_completion_tokens` 512 still allows ~2 minutes of speech in one answer.

**Barge-in**, `TestMLXBargeIn` 20 more times per profile (`spikes/realstack/bargein.sh`; the caller
starts 2.74 s into a long line): `speech_started` 0.40–0.59 s after the onset in 35 of 40 runs and
0.86–1.52 s in 5, which then lost the first word or two ("People don't think…"). In those 5 the
echo canceller's no-echo bypass engaged 5.4–5.9 s into the call, 0.2–0.9 s after the caller had
started; in the fast runs it was on before the onset or within 0.2 s of it. Until it engages,
AEC3 treats a caller talking over our audio as echo (M4: up to 44 dB). It engages late because of
the wrapper's listening "room tone", a pure 182 Hz sine at −61 dBFS copied from AICC's
`feedRoomTone`: GCC-PHAT finds chance peaks on it in 60 % of its windows (PSR median 10.3), and
only windows without a peak count toward the bypass; on line noise it finds one peak in 418
windows. Counting weak unconfirmed peaks (PSR < 12) as well engaged the bypass 1.5 s sooner on the
room tone offline, but raised its misfires on G7's real-echo calls (continuous double talk:
16 → 23), so it was not adopted (Remaining risks). Keypresses: the line cancelled 1–2 ms after
the key and the reply's audio 340–520 ms after it, every run.

### Benchmark (`bench/`; the tables are in `docs/LATENCY.md`)

126 utterances per run: `wide`, `tel8k` PCM and `tel8k` μ-law, each with 1, 2 and 3 sessions (the
three-session runs are the soak's three passes). Server end to end, speech end → first audio
(P50 / P95): 1.38–1.40 / 3.78–3.87 s at one session, 1.43–1.44 / 3.81–3.87 s at two, 1.48–1.52 /
3.88–3.98 s at three. The P95 is Smart Turn holding 11–17 turns per run as incomplete until
`max_silence_ms` (mostly ASCEND excerpts that really stop mid-sentence); turns that ended on a
complete verdict: 1.37–1.39 / 1.52–1.54 s at one session, 1.42–1.47 / 1.86–2.78 s at two and
three. The tails at two and three sessions are the serialized TTS: a first segment waited up to
1.9 s for another session's segment. AICC's windows: no miss at one session; at two, one delta
gap of 2.09 s (bug 5); at three (bug 5 fixed by then), one first audio of 3.39 s. Eight responses
failed with `backend_busy` (one at two sessions, seven at three), 2.4–3.2 s after
`response.created`. ASR on the rows without numbers: zh CER 5.9–6.0 %, en WER 2.4–3.6 %, mix MER
8.6–13.2 %, all MER 5.1–6.8 % (M0 S4: 5.9 / 2.4 / 10.3 %).

The fixes went in while the runs went on: the one-session runs had bugs 1–3 fixed; the
two-session runs and the HQ comparison also 4 and 6; the soak (the three-session runs) also 5; the
final backend tests everything. The benchmark never talks over an answer, so bug 4 does not
touch its numbers.

**AICC's call flow, with tools** (`spikes/realstack/aiccflow.sh`, LATENCY.md §3.5): ten calls of
AICC's `line_session_zh.json` with spoken caller turns, alone and beside two bench sessions. Alone,
the `transfer_to_agent` tool-only response was done 1.56 / 1.72 s (P50 / P95) after
`response.created`, `repair_status` 0.83 / 0.91 s, the spoken replies' first audio ≤ 0.73 s.
Beside two sessions: transfer 1.93 / 2.78 s, a question's first audio 0.83 / 2.72 s. Every
window was met, with 0.2–0.3 s to spare at worst.

### G.711 input quality (task 5)

`tel8k` μ-law at two sessions through the gateway: all MER on the rows without numbers 5.9 % with
`MLXRT__G711__INPUT_QUALITY=LQ` (the default), 6.6 % with `HQ` (a second gateway on the same
backends). Paired on the ASR alone (`spikes/realstack/g711_quality.py`: the same 126 streams,
upsampled both ways by the gateway's own decoder, language detected): HQ − LQ = +0.12 points
(95 % CI −0.42 to +0.60), zh +0.09, en 0.00, mix +0.30; 19 of 126 transcripts differ. HQ is no
better; **LQ stays the default** (DECISIONS below).

### GPU contention (S5)

On the backends directly (`spikes/realstack/contention.py`, LATENCY.md §3.7): while the LLM decodes
one or three streams, the TTS's first chunk goes from 56 to 71–81 ms (P50) and its RTF from 0.25
to 0.33–0.38; the ASR from 172 to 194–215 ms; while the TTS streams, the LLM decodes 12 % slower
(77 → 68 tok/s) with an unchanged TTFT. While the LLM prefills ~16k tokens (22.8 s), the TTS runs
at **RTF 2.1** (first chunk 203 / 375 ms): a long prefill starves every speaking session. AICC's
prompts are ~1.6k tokens (≈ 1.5 s cold). Through the gateway, the same effects show in the
two- and three-session runs (ASR P95 +0.1–0.2 s, LLM first token P95 +0.05–0.15 s).

### Stability: the 30-minute soak (S6)

`zsh spikes/realstack/soak.sh 1800`: three sessions back to back through `make dev`, 30 min 9 s,
three bench passes (`wide` PCM, `tel8k` PCM, `tel8k` μ-law), 45 sessions, 378 utterances; 425
LLM, 391 ASR and 625 TTS requests (fewer TTS calls than S6's 1000: the answers are one sentence).

- **Hangs: none.** Every utterance was answered or failed: no bench timeout, no response left
  open, no delta gap over 1.0 s.
- **Errors:** 7 responses failed with `backend_busy` (TTS queue timeout, above); no backend
  returned an error status. The LLM log has 7 `BrokenPipeError` tracebacks: mlx_lm.server writing
  to a response the gateway had cancelled (a barge-in), harmless.
- **Crashes: none.** The four processes kept their PIDs throughout.
- **GPU kernel messages** (`log show --last 40m --predicate 'process == "kernel"'`): no IOGPU or
  GPU error, fault or hang; the only GPU line is a CoreAnalytics rate-policy counter
  (`com.apple.agx.ctxswitchlatency count 3`).
- **Memory** (`footprint`, GB, every 5 min; `spikes/results/realstack_soak_memory.jsonl`): flat
  after the LLM's prompt cache filled in the first 5 minutes. No swap at any time.

| time | LLM (peak) | TTS (peak) | ASR (peak) | gateway (peak) |
|---|---|---|---|---|
| start | 18.73 (19.05) | 3.78 (4.44) | 2.46 (3.49) | 0.20 (0.20) |
| +5 min | 22.88 (23.51) | 4.24 (4.58) | 2.48 (3.85) | 0.26 (0.26) |
| +10 min | 22.90 (23.51) | 3.78 (4.59) | 2.49 (3.85) | 0.26 (0.27) |
| +15 min | 22.85 (23.51) | 3.78 (4.59) | 2.49 (3.85) | 0.26 (0.28) |
| +20 min | 22.98 (23.51) | 3.78 (4.90) | 2.50 (3.85) | 0.26 (0.28) |
| +25 min | 22.83 (23.62) | 4.26 (4.90) | 2.50 (3.85) | 0.25 (0.28) |
| +30 min | 22.81 (23.62) | 3.81 (4.90) | 2.49 (4.03) | 0.26 (0.28) |

### Recovery (task 7)

Each backend killed with `kill -9` while the stack ran process by process
(`zsh spikes/realstack/recovery.sh`; `mlxrt dev` stops everything when a child exits), a spoken
turn sent while it was down, the backend started again with `mlxrt dev`'s command, and a turn sent
after it:

| Backend | While it is down | After the restart, before the fix | After the restart, with the fix |
|---|---|---|---|
| ASR | `/readyz` 503 at once; the turn: `transcription.failed`, `error` `backend_unavailable`, `response.done{failed}`, all 1 ms after the commit | `/readyz` 503 ("model not loaded") for 30 s and more, until a caller's turn loaded the model in its own request (transcript 964 ms instead of ~160) | 200 again 1.7 s after the restart; the next turn's first audio 0.55 s after `response.created` |
| TTS | `error` `backend_unavailable` and `response.done{failed}` at the first segment, 0.44 s after `response.created` | the next response loaded the primary in its first segment (first audio 1.86 s); the fallback was never loaded, so `/readyz` stayed 503 for good | 200 again after 10.4 s (both models, every voice); next turn 0.61 s |
| LLM | `error` `backend_unavailable` and `response.done{failed}` at the LLM request, 0.16 s after `response.created` | `/health` back 1 s after the start; the next turn normal (0.62 s) | 200 again after 3.0 s (`/readyz` waits for the warmup too); next turn 0.60 s |

Nothing hung and every session stayed open. The fix is bug 1; its last column is the final tree
(`recovery.sh kill asr tts llm`, 01:58); an earlier run of the fix, before `/readyz` required
warm, gave 3.7 s, 10.3 s and 0.9 s.

### Bugs found and fixed

1. **A restarted backend was never warmed again** (`server/readiness.py`, `server/app.py`,
   `server/metrics.py`). See Recovery. The monitor now keeps the backends warm: after the startup
   warmup it probes each warm backend every 5 s (`WATCH_S`); one that answers "not ready"
   (unreachable, or its model no longer loaded) is warmed again with the startup retries until it
   is ready. A probe that times out (a busy backend) changes nothing; one that raises is logged
   and skipped. `/readyz` and `mlxrt_backend_ready` also require the backend to be warm (the LLM
   answers `/health` before the gateway's warmup has run). Tests: `tests/unit/test_readiness.py`
   (6), `test_metrics.py` (+1 case).
2. **The benchmark's replay was double talk** (`bench/bench.py`). It sent the next utterance 300 ms
   after `response.done`, but the answer goes out up to 10 s ahead of real time and the gateway
   models the client playing it (D17's reference, the barge-in rules): every turn after the first
   started while the gateway believed our answer was still playing. In the first run, five
   sessions had 8 spurious "echo on the line" events (there is no echo in the bench), AEC3 was put
   back on over the caller (27 dB of "ERLE" on a caller's speech), and Speex aligned to an echo
   that was not there; that run was discarded. The replay now waits until the answer would have
   played out, as a caller who listens; `--overlap` keeps the old timing. Tests:
   `tests/unit/test_bench.py` (+1).
3. **The benchmark did not check AICC's windows** (`bench/bench.py`, `bench/report.py`). Every
   response now records `response.created` → first audio, its longest delta gap, last delta →
   `response.done`, and `response.created` → `response.done` without audio; the report counts
   those at or over AICC's limits. Tests: `test_bench.py` (+2), `tests/integration/test_bench.py`.
4. **The echo canceller's bypass was left on chance peaks** (`session/echo.py`). On a line without
   echo, a caller talking over our audio turned the canceller back on whenever two consecutive
   GCC-PHAT windows peaked at the same lag; the windows overlap by 1.5 s, so a chance peak confirms
   itself. Offline (`spikes/realstack/echo_bypass.py`, FLEURS speech over FLEURS speech): 29
   resumes in 4.5 min of double talk, 0 on line noise; with real echo (ERL 6/15 dB, the caller at
   0/−15 dB) at most 1 of ~410 confirmations at a wrong lag. On the real stack it slowed the SDK's
   barge-in to 1.5 s (a "residual" of 891 ms confirmed during the caller's speech). Leaving the
   bypass now takes three consecutive valid windows that agree within 2 ms, each with a PSR ≥ 10:
   false resumes 29 → 2 (24 → 1 with a quieter caller), a real echo resumes one window (0.5 s)
   later. Alignment while the canceller is on is unchanged. On G7's cached G.168 calls
   (`spikes/realstack/g7_counters.py`): bypasses 16 → 14 and AEC3/Speex re-alignments 8 → 6 /
   23 → 21 in continuous double talk, the barge-in layout identical, Speex aligned at 2.5 s
   (median) either way; the ASR ΔCER was not measured again. No "echo on the line" in the 20
   harness barge-ins after the change (3 before). Tests: `tests/unit/test_echo.py` (+1, 1
   changed).
5. **A 2 s delta gap, so AICC would abandon the turn** (`session/pipeline.py`, `session/signals.py`,
   `session/conversation.py`, `session/truncation.py`, `session/actor.py`, `config.py`,
   `config.yaml`). With two sessions, one answer's second segment waited 2.3 s in the TTS queue
   behind the other session's long segment. M6's audio reserve keeps the wire busy only until
   shortly before the audio sent so far has played out, so the wire went quiet for 2092 ms and
   AICC's 2 s watchdog would have ended the turn. Now, once the reserve has nothing left and more
   audio is to come, a 20 ms silence delta goes out every `keepalive_ms` (1 s), never before the
   response's first audio: the caller hears the same pause, and the TTS queue's own timeout still
   bounds the wait (`output.reserve.keepalive_silence`, on by default). The silence belongs to the
   segment before it but is not its speech: truncation and D5 measure a segment's text over its
   speech only (`SpokenSegment.speech_ms`), so a cut in the silence heard the segment whole
   (without that, the cancellation matrix's barge-in cases heard "你好" of "你好。"). In the
   three-session soak 23 responses were bridged this way and no gap exceeded 1.0 s. Tests:
   `tests/unit/test_reserve.py` (+3, 1 changed), `test_session_state.py` (+1).
6. **`make dev` refused to start again right after a stop** (`supervisor.py`): "127.0.0.1:8101 is
   already in use" for ~30 s, because the free-port check's plain `bind` fails on connections the
   stopped servers left in TIME_WAIT. The check now binds with `SO_REUSEADDR`, as uvicorn and
   `http.server` do, and also counts a port busy when it accepts connections (a server on every
   interface). Tests: `tests/unit/test_supervisor.py` (+2).
7. **Failed responses were not logged** (`session/actor.py`). The eight `backend_busy` failures
   reached the client and `/metrics`, but the log said nothing. A response that fails on a provider
   and a transcription that fails now log a warning with the stage, the kind and the message.
   Tests: `tests/integration/test_observability.py` (+1).
8. **The backend tests left an ASR connection open** (`tests/backend/test_real_backends.py`), and
   with warnings as errors pytest exited 1 after 23 passes. The `llm` and `asr` fixtures now close
   their providers.

### Gates

- **G4, the latency budget and targets** (proposed for approval; LATENCY.md §4). Budget per stage
  (P95, from speech end, AICC's 500 ms silence): VAD 550 ms, Smart Turn 100 ms (up to
  `max_silence_ms` when it holds a turn), ASR 400 ms, LLM first token 500 ms warm, first sentence
  250 ms, TTS first chunk 150 ms alone, first audio sent 200 ms. Targets, speech end → first audio
  on turns that end on a complete verdict: one call P50 ≤ 1.5 s / P95 ≤ 1.8 s (measured 1.37–1.39 /
  1.52–1.54); up to three calls P50 ≤ 1.6 s / P95 ≤ 2.5 s (measured 1.42–1.47 / 1.86–2.78). AICC's
  hard floor: one call holds it with margin, including a full re-prefill after an instructions
  change (~2.4 s worst case from `response.created`). **Three calls do not hold it reliably**: the
  one TTS stream queues first segments for up to 1.9 s (one first audio at 3.39 s, seven
  `backend_busy` failures at three sessions, 1.8 % of their responses; eight in all), and a tool-only transfer
  reaches 2.78 s beside two talking sessions. Reaching the three-call target needs more TTS
  capacity (a second TTS process, or batched generation) or shorter segments; the owner decides.
- **G6, 3 model processes or merged: keep 3.** The soak showed no instability: no hang, crash,
  backend error or GPU kernel message, flat memory; recovery works per process (bug 1). There is
  nothing a merge would fix; the capacity limit is the TTS stream itself, which a merge would not
  widen. (D13's fallback half of G6 was settled by the owner: the 4-bit Qwen3-TTS.)
- **G.711 input quality: LQ stays the default** (`g711.input_quality`): HQ is no more accurate
  (+0.12 points MER, CI −0.42 to +0.60) and LQ reports the caller's speech ~30 ms sooner.

### Remaining risks

1. **TTS capacity at two and three calls** (the main one). See G4. Options, all needing the owner:
   a second TTS server (the provider takes one `base_url` today, so a pool is a code change);
   mlx-audio's batched generation; capping continuation segments (`text.segmenter.max_chars`)
   so no segment holds the TTS for more than ~1 s; letting continuation segments wait longer
   (`providers.tts.queue_timeout_ms`, now that the keepalive silence keeps AICC's watchdog fed).
   AICC's `AICC_BOT_MAX_CALLS` should match whatever is decided.
2. **Tool-only transfers are decode-bound:** 2.78 s (of 3 s) beside two talking sessions. The
   request to AICC to keep `transfer_to_agent`'s arguments short stands.
3. **A long prefill starves the TTS** (RTF 2.1 during a 16k-token prefill). Not seen with AICC's
   prompt sizes; `--prefill-step-size` is untried.
4. **Long answers:** `max_completion_tokens` 512 allows ~2 minutes of speech; seen once, in AICC's
   harness, on a conversation AICC does not produce.
5. **The bypass engages late on a tonal background** (the harness's room tone: barge-in 0.9–1.5 s
   in 5 of 40 runs, the first word lost). Real lines carry broadband noise. Counting weak
   unconfirmed peaks fixes it but raised misfires on G7's real-echo calls; a test-side fix would
   be broadband room tone in `tests/aicc/zz_mlx_live_test.go`'s listener (left as it is).
6. **Smart Turn holds incomplete turns** for up to `max_silence_ms` (3 s): 10–13 % of this
   dataset, mostly ASCEND excerpts cut mid-sentence; this is the P95 of the end to end.
7. **The readiness probes log** a request line every 5 s per backend in the backends' own logs
   (and the gateway's httpx line), ~0.8 lines a second in `make dev`'s output.
8. Live echo on a real AICC call is still unmeasured (D17, deferred).

### How to run and verify

```sh
MLXRT__OBSERVABILITY__DEBUG_ENDPOINTS=true make dev                  # terminal 1
AICC_REPO=<checkout> uv run --locked pytest -m backend                # everything M2–M7 left
uv run --locked python tests/aicc/probe.py --aicc-repo <checkout> \
    --url ws://127.0.0.1:8000/v1/realtime [--g711]                     # AICC's own harness
uv run --locked python -m bench.bench --url ws://127.0.0.1:8000/v1/realtime \
    --input spikes/data/manifest.jsonl --variant tel8k --codec pcmu --concurrency 1 \
    --out bench/results/real-tel8k-pcmu-c1                             # LATENCY.md's runs
python3 -I spikes/realstack/bench_tables.py bench/results/real-*      # LATENCY.md's tables
zsh spikes/realstack/soak.sh 1800                                     # S6, then:
log show --last 40m --predicate 'process == "kernel"' | grep -iE 'iogpu|gpu'
AICC_REPO=<checkout> zsh spikes/realstack/aiccflow.sh 10              # AICC's flow with tools
AICC_REPO=<checkout> zsh spikes/realstack/bargein.sh 10               # 2 × 10 harness barge-ins
uv run --locked --project spikes/realstack python spikes/realstack/contention.py out.json  # S5
uv run --locked --project spikes/realstack python spikes/realstack/g711_quality.py out.json
uv run --locked --project spikes/realstack python spikes/realstack/echo_bypass.py out.json
(cd spikes/g7dual && uv run --locked python ../realstack/g7_counters.py out.json)
# Recovery: stop make dev first; the stack runs process by process.
zsh spikes/realstack/recovery.sh start && zsh spikes/realstack/recovery.sh kill asr tts llm \
    && zsh spikes/realstack/recovery.sh stop
```

`spikes/realstack` is a uv project (`uv init --no-workspace`, its own `uv.lock`, the gateway as an
editable path dependency, like `spikes/g7dual`). Raw results: `spikes/results/realstack_*`, the
benchmark's reports in `spikes/results/realstack_bench/`.

### Test results

On 2026-10-24, the final tree: `uv run --locked pytest` **1825 passed, 25 skipped** (before:
1807 and 25; the skips are the backend tests and the AICC runs that need `AICC_REPO`), about 265 s;
`uv run --locked pytest -m backend` with `AICC_REPO` against `make dev`: **23 passed**. `ruff
check`, `ruff format --check` and `pyright` report nothing; `uv lock --check` passes for the root
and `spikes/realstack`. No golden trace changed.

New and changed tests: `tests/unit/test_readiness.py` (6, new); `test_bench.py` (+3);
`test_echo.py` (+1, the bypass test changed to the new rule); `test_reserve.py` (+3, the first
test now also checks the keepalive); `test_session_state.py` (+1); `test_supervisor.py` (+2);
`test_metrics.py` (one more case); `tests/integration/test_bench.py` (the windows);
`test_observability.py` (+1); `tests/backend/test_real_backends.py` (fixtures close).

### Deviations from the brief, with reasons

1. **The three-session benchmark runs are the soak's passes**, not separate runs: the same command
   (`bench.bench --concurrency 3`) on each variant, back to back, which is what the soak needed.
2. **Two behaviour changes beyond plain bug fixes**, both flagged for the owner: the keepalive
   silence (bug 5) and the stricter bypass resume (bug 4). Each is a config switch or a constant,
   and each was measured before and after.
3. **The c1 and c2 runs predate bug 5's fix**, and the c3 runs follow it; the first c1 run, with the
   old replay timing (bug 2), was discarded.
4. **S6's 1000 TTS calls**: the soak made 625 in 30 minutes (one-sentence answers).
5. **New measurement code lives in `spikes/realstack`**, as the earlier spikes do; nothing in it is
   imported by the service.

### Suggested changes to ARCHITECTURE and DECISIONS

- **ARCHITECTURE.**
  - §2: the figure and process table still show Kokoro (D13 replaced it with the 4-bit
    Qwen3-TTS) and `mlx_lm.server` (now `python -m mlxrt_llm.serve`); add the 3-call capacity and
    that it is TTS-bound.
  - §2 (Startup, warmup, readiness) and §8 (Backend health): the readiness monitor probes every
    5 s and warms a backend again when it comes back; `/readyz` requires warm.
  - §6.5 (D17): leaving the no-echo bypass needs three agreeing windows with PSR ≥ 10.
  - §6.6: truncation and D5 measure a segment's text over its speech, not its keepalive silence.
  - §7 Pacing / §4.4: the reserve's keepalive silence.
  - §10: `output.reserve.keepalive_silence`.
  - §11 Benchmark: the replay waits for each answer's playback; AICC's windows; `spikes/realstack`.
- **DECISIONS** (dated entries to add).
  - G4: the budget and targets above, for the owner's approval; three calls miss AICC's floor on
    TTS capacity.
  - G6: keep three model processes (soak clean).
  - G.711 input stays LQ (CER no better with HQ).
  - The reserve's keepalive silence, on by default (flagged for owner review).
  - The bypass resume rule (flagged for owner review).
  - TTS capacity for three calls: a second TTS process, batched TTS or shorter segments (owner
    decision); size `AICC_BOT_MAX_CALLS` to it.
