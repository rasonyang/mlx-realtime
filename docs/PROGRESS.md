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
