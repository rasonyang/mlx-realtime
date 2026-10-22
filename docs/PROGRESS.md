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
