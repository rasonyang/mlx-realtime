# mlx-realtime — Realtime GA protocol profile

> **Status: M7 (2026-10-23).** What the gateway accepts and emits on `/v1/realtime`, event by event and
> field by field, and in which milestone each unsupported part arrives. M2 made `server_vad` real
> (Silero VAD + Smart Turn) and the transcripts real (Qwen3-ASR); answers come from the LLM. M3 made
> the audio real: Qwen3-TTS streamed per segment (with a 4-bit fallback), paced at most 10 s ahead
> of real time (D6), with configured voices (D19) and verbatim lines (D16). M4 made barge-in,
> cancellation and truncation complete (§11.2, §11.3, §11.13): the D5 truncation fallback, and
> echo cancellation in the service (D17) with its second line of defence, which gates the speech
> events while the client is modelled as playing our audio. M5 made tool calling real: calls from
> the LLM server's parser, `tool_choice` enforced in the gateway, and the round trip's history
> shaped for the Qwen template (§11.5, §11.14). M6 added `/metrics` and the opt-in `/debug/turns`
> (§1), and the multi-session TTS mitigations, which change only when audio deltas are sent (§11.8).
> M7 added G.711: `audio/pcmu` and `audio/pcma` at 8 kHz, in either direction or both (§3, §11.15).
> After M7 (G7 dual path, D13, capacity): the ASR transcribes a linear echo canceller's output
> (§11.13) and the TTS fallback is Qwen3-TTS's 4-bit build. Issue #2 (2026-10-24) added
> transcription sessions (`session.type: "transcription"`, §3.1) and raised the limit to 4 sessions.

**Sources.** The protocol source of truth is **openai-python 3.36.1**: the types in
`openai/types/realtime/` name every event and field below (SDK type names are given in `code`). Where
the types are silent, the GA reference and guides decide (ARCHITECTURE §14). The decisions D1–D19 are
in ARCHITECTURE §12; where they differ from GA, this document says so.

**Status labels.**

| Label | Meaning |
|---|---|
| **Supported** | accepted and implemented as documented |
| **Partial** | accepted; part of it works now, the rest arrives in the milestone named |
| **Unsupported (Mn)** | rejected with an `error` until milestone Mn |
| **Unsupported** | rejected with an `error`; not planned |
| **Never** | not applicable to this service (WebRTC/SIP only, beta, or replaced by another event) |

Contents: [1 Transport](#1-transport) · [2 Client events](#2-client-events) ·
[3 Session](#3-session-object) · [4 response.create](#4-responsecreate-parameters) ·
[5 Items](#5-conversation-items) · [6 Server events](#6-server-events) ·
[7 Response object](#7-the-response-object) · [8 Event orders](#8-golden-event-orders) ·
[9 Errors](#9-errors) · [10 Ids](#10-identifiers) · [11 Behaviour notes](#11-behaviour-notes) ·
[12 AICC](#12-aicc-compatibility)

## 1. Transport

| Item | Behaviour |
|---|---|
| Endpoint | `GET /v1/realtime`, upgraded to WebSocket. The SDK derives it from `base_url="http://<host>:<port>/v1"`. |
| Auth | `Authorization: Bearer <gateway.api_key>`. With an empty key, auth is off. A missing or wrong key gets HTTP 401 before the upgrade. |
| `?model=` | Echoed as `session.model`; never used for routing. Without it, `session_defaults.model` is echoed. A transcription session has no `model` to echo. |
| `?intent=` | `transcription` makes the session a transcription session from `session.created` on (§3.1). Any other value is ignored. |
| `OpenAI-Beta` header | Ignored: only the GA shape is served. |
| Frames | One JSON object per text frame. A binary frame or invalid JSON draws an `error` (`invalid_event`); the session stays open. |
| Frame size | At most `limits.client_max_message_bytes` (16 MiB; GA allows 15 MiB of base64 per append). A larger frame closes the connection with 1009. |
| First events | `session.created`, then `conversation.created`, before any client event is processed. |
| Admission | Before the first successful readiness: `error` `not_ready`, close 1013. Over `gateway.max_sessions` (4, the Mac's capacity target; realtime and transcription sessions count alike): `error` `session_limit_reached`, close 1013. Both after the upgrade, as the SDK expects. |
| `/healthz` | 200 while the event loop answers. No backend calls. |
| `/readyz` | 200 once every backend has been warmed and reports ready, else 503; per-backend detail in the body: `llm`, `asr`, `tts`, and `vad` (the CPU detectors) when Silero runs. |
| `/metrics` | Prometheus text format (0.0.4), always served and unauthenticated (ARCHITECTURE §9). Backend readiness is probed at each scrape. No label carries an id. |
| `/debug/turns` | Only with `observability.debug_endpoints: true` (off by default; 404 otherwise). `?session_id=sess_…`: `{"session_id", "turns": [...]}`, one record per finished turn with `turn_id`, `item_id`, `response_id`, `status`, `marks_ms` (ms after the earliest mark), `stages_ms`, `e2e_ms` and `values` (`tts_queue_ms`, `tts_model`, ...); 404 for a session not kept. Without `session_id`: `{"sessions": [...]}`. Kept for the most recent `debug_sessions` sessions, at most `debug_turns_per_session` turns each; transcripts and audio are never included. For `bench/` (D8). |

**Close codes.**

| Code | When |
|---|---|
| 1000 | `gateway.max_session_s` elapsed (GA sessions last at most 60 minutes) |
| 1001 | server shutdown: queued events are flushed first (bounded) |
| 1008 | the client stopped reading: a write stalled longer than `client_write_timeout_ms`, or more than `outbound_queue_events` events are queued |
| 1009 | a frame larger than `client_max_message_bytes` (closed by the WebSocket layer) |
| 1011 | a fatal session error (`input_queue_overflow`, `input_audio_buffer_overflow`, `internal_error`), after the `error` event |
| 1013 | admission refused (`not_ready`, `session_limit_reached`) |

## 2. Client events

Members of `RealtimeClientEvent`. Every client event may carry `event_id` (a string); any `error` it
causes echoes it in `error.event_id`.

| Event | SDK type | Status | Notes |
|---|---|---|---|
| `session.update` | `SessionUpdateEvent` | **Supported** | Validated as a whole (§3), then merged field by field; answered by `session.updated` with the full effective session. A rejected update changes nothing. |
| `input_audio_buffer.append` | `InputAudioBufferAppendEvent` | **Supported** | `audio`: base64 of the session's input format: `audio/pcm` 24 kHz mono s16le, whole samples (an odd byte count is `invalid_value`, `param: "audio"`), or G.711 at 8 kHz, one byte a sample (any byte count; AICC sends 160 bytes every 20 ms). No confirmation event. Never dropped: an inbox over `limits.inbox_max_ms` of queued audio (counted in the input format's duration) is fatal. |
| `input_audio_buffer.commit` | `InputAudioBufferCommitEvent` | **Supported** | Commits the whole buffer, the resampler's tail included → `input_audio_buffer.committed`, `conversation.item.added`. Creates no response. Allowed in `server_vad` mode too (a speech turn in progress ends without `speech_stopped`). An empty buffer: `invalid_event`. In a transcription session an empty or silent buffer is never an error (§3.1). |
| `input_audio_buffer.clear` | `InputAudioBufferClearEvent` | **Supported** | → `input_audio_buffer.cleared`; a speech turn in progress is dropped. |
| `conversation.item.create` | `ConversationItemCreateEvent` | **Partial** | Not in a transcription session (§3.1). Messages and `function_call_output` (§5); `previous_item_id` absent (append), `"root"` (first) or an existing id. → `conversation.item.added`, `conversation.item.done`. Client-created `function_call` items: Unsupported. |
| `conversation.item.truncate` | `ConversationItemTruncateEvent` | **Supported** | Assistant messages with audio, `content_index: 0`, in progress or completed (AICC truncates after `response.done`). An `audio_end_ms` beyond the audio sent is clamped to it with a warning in the log, never an error (D18). → `conversation.item.truncated` (§11.3). |
| `conversation.item.delete` | `ConversationItemDeleteEvent` | **Supported** | Any item, including one a response is still producing (§11.4). → `conversation.item.deleted`. Unknown id: `item_not_found`, `param: "item_id"` (D18). |
| `conversation.item.retrieve` | `ConversationItemRetrieveEvent` | **Unsupported** | `invalid_event`, `param: "type"`: user audio is not retained in the wire format, and an item without its audio would silently diverge from GA. |
| `response.create` | `ResponseCreateEvent` | **Partial** | Not in a transcription session (§3.1). §4. While a response is active: `conversation_already_has_active_response` (not queued). `response.created` is immediate (D7). |
| `response.cancel` | `ResponseCancelEvent` | **Supported** | With or without `response_id`. Nothing active, or another id: `response_cancel_not_active` (D18). → the terminal events, `response.done{cancelled, client_cancelled}`. |
| `output_audio_buffer.clear` | `OutputAudioBufferClearEvent` | **Never** | WebRTC/SIP only. `invalid_event`, `param: "type"`; over WebSocket, send `response.cancel` (in a realtime session). |
| `transcription_session.update` | — | **Never** | The beta event. `invalid_event`, `param: "type"`: send `session.update` with `session.type: "transcription"` (§3.1). |
| any other `type` | — | — | `invalid_value`, `param: "type"` (as GA's `scooby.dooby.doo` example). A missing `type`: `invalid_event`. |

## 3. Session object

`session.update.session` is a `RealtimeSessionCreateRequest`, or a `RealtimeTranscriptionSessionCreateRequest`
when `type` is `"transcription"` (§3.1; the table below is the realtime session's). `session.created` and `session.updated` echo the full effective session in
the same shape, plus `id` (`sess_…`) and `object: "realtime.session"`, which the SDK reads as extra
fields. `config.yaml`'s `session_defaults` uses this shape and the same validation.

**Merge rules.** Only fields present change. Nested objects merge field by field. `null` clears a
nullable field (`transcription`, `turn_detection`). A nested object set after being `null` starts from
the configured session defaults, so its absent fields take their configured values.

| Field (SDK type) | Status | Accepted | Echo | Rejection `param` |
|---|---|---|---|---|
| `type` | **Supported** | `"realtime"` or `"transcription"`, required in `session.update`; a change of type only before any input audio, item or response (§3.1) | `"realtime"` or `"transcription"` | `session.type` |
| `model` | **Supported** | any string; echoed only | last value set, else `?model=`, else the default | `session.model` |
| `instructions` | **Supported** | any string (`""` clears) | effective | `session.instructions` |
| `output_modalities` | **Supported** | `["audio"]` or `["text"]` (D12) | effective | `session.output_modalities` |
| `audio.input.format` (`RealtimeAudioFormats`) | **Supported** | `{type: "audio/pcm", rate: 24000}`, `{type: "audio/pcmu"}` or `{type: "audio/pcma"}` (G.711 at 8 kHz, D3). `type` defaults to `audio/pcm`; `rate`, when present, must be the format's (24000, or 8000 for G.711), else `invalid_value` on `.rate`. Beta names (`g711_ulaw`, …) are rejected (D2). May change at any time (GA): later appends decode in the new format (§11.15) | `{type: "audio/pcm", rate: 24000}`; G.711 as `{type: "audio/pcmu"}` / `{type: "audio/pcma"}` (the SDK's `AudioPCMU`/`AudioPCMA` have no `rate`) | `session.audio.input.format.type` / `.rate` |
| `audio.input.transcription` (`AudioTranscription`) | **Partial** | object or `null`; `model` (echoed only: the ASR is Qwen3-ASR, `providers.asr.model`), `language` (an ISO 639 code such as `zh` or `en`, passed to the ASR as its language name; absent: the ASR detects it), `prompt` (words to expect, such as brand and product names: sent to the ASR as the turn's context, which biases Qwen3-ASR toward them; at most 1024 characters, longer is `invalid_value` on `.prompt` and the session is unchanged; `""` or `null` is no prompt; the prompt in force when a turn is committed is the one its transcription gets, so a change applies from the next turn on and never re-transcribes a committed one; absent, the deployment's `providers.asr.context` applies, if set, and a session prompt replaces it, it does not add to it). `delay`, `keywords`, `languages`: Unsupported | effective object (unset keys omitted), or `null`. On by default (D15) | `session.audio.input.transcription.<key>` |
| `audio.input.turn_detection` (`RealtimeAudioInputTurnDetection`) | **Supported** | `null` (manual turns), or `ServerVad`: Silero VAD v6.2.3 plus Smart Turn v3.2 (D4, §11.9). `threshold` [0, 1] (speech enters at it, leaves below it − 0.15), `prefix_padding_ms` ≥ 0, `silence_duration_ms` ≥ 0 (the silence before an end-of-turn evaluation; session default 200), `create_response`, `interrupt_response`; `idle_timeout_ms` only `null`. `SemanticVad`: **Unsupported** | every field filled, `idle_timeout_ms: null`; or `null` | `…turn_detection.type` (`semantic_vad`), `….idle_timeout_ms` (non-null), `….<field>` |
| `audio.input.noise_reduction` (`NoiseReduction`) | **Unsupported** | `null` only (echo cancellation is internal and always configured server-side: §11.13) | `null` | `session.audio.input.noise_reduction` |
| `audio.output.format` | **Supported** | as `audio.input.format`, independently of it (μ-law in and A-law out is accepted). A change applies from the next response; a response in progress keeps its format | as input | `session.audio.output.format.type` / `.rate` |
| `audio.output.voice` (`Voice`) | **Partial** | any string, never rejected (D19): a name in `voices.map` is that voice, any other name is spoken by `voices.default` and logged as a warning (§11.10). Locked once the session has produced audio, as in GA: a different voice then gets `cannot_update_voice`; the same voice, or no voice field, passes. `VoiceID` objects: Unsupported | the effective voice (default `vivian`) | `session.audio.output.voice` (object; or `cannot_update_voice`) |
| `audio.output.speed` | **Partial** | `1.0` only (D14: the TTS has no speed control) | `1.0` | `session.audio.output.speed` |
| `tools` (`RealtimeToolsConfig`) | **Supported** | `RealtimeFunctionTool`: `type: "function"` and `name` required, `description`, `parameters` (an object, passed through). Duplicate names rejected. `Mcp`: Unsupported | effective list; `[]` when unset | `session.tools[i].type` / `.name` / `.parameters` / `.<key>` |
| `tool_choice` (`RealtimeToolChoiceConfig`) | **Partial** | `"auto"` (the session's tools are offered), `"none"` (none are: the LLM request carries no tools, and a call the model makes anyway is dropped). `"required"`, `ToolChoiceFunction` and `ToolChoiceMcp`: **Unsupported**, `invalid_value`: `mlx_lm.server` ignores `tool_choice`, so nothing can make the model call a tool, and accepting them would promise what cannot be enforced (§11.5) | effective; `"auto"` by default | `session.tool_choice` |
| `max_output_tokens` | **Supported** | `"inf"` or 1–4096 | effective | `session.max_output_tokens` |
| `truncation` (`RealtimeTruncation`) | **Partial** | `"auto"` only | `"auto"` | `session.truncation` |
| `tracing` (`RealtimeTracingConfig`) | **Partial** | `null` only | `null` | `session.tracing` |
| `prompt` (`ResponsePrompt`) | **Partial** | `null` only | `null` | `session.prompt` |
| `include` | **Partial** | `null` or `[]` | `null` | `session.include` |
| `parallel_tool_calls` | **Unsupported** | — (one call at a time) | not echoed | `session.parallel_tool_calls` |
| `reasoning` (`RealtimeReasoning`) | **Unsupported** | — (reasoning is always off) | not echoed | `session.reasoning` |
| unknown field | — | rejected: "Unknown parameter" | — | dotted path, e.g. `session.audio.input.foo` |
| beta field (`modalities`, `voice`, `turn_detection`, `input_audio_format`, `output_audio_format`, `input_audio_transcription`, `temperature`, `max_response_output_tokens` at the top level) | **Never** (D2) | rejected, with a hint that the GA shape is required | — | `session.<field>` |

`expires_at`, present in GA's `session.created` example, is not emitted: sessions are capped by
`gateway.max_session_s` but no expiry is advertised.

### 3.1 Transcription sessions

A transcription session turns audio into transcripts and nothing else (issue #2; AICC's human-call
transcription, one session per leg). It runs no LLM, no TTS, no Silero or Smart Turn and no echo
canceller: the audio goes to the ASR as it arrived, after the format conversion of §11.15. The
client owns endpointing: it appends audio continuously, silence included, and commits after each
utterance.

**Selecting the type.** `?intent=transcription` on the handshake makes the session a transcription
session from `session.created` on. Otherwise a `session.update` with `session.type: "transcription"`
(or `"realtime"`) changes the type, while the session has had no input audio, no item and no
response; the new type starts from its own defaults, and only the input format and the
transcription settings carry over. After that, a type change is `invalid_value`, `param:
"session.type"`, and the session stays as it was. An update of the type the session already has is
an ordinary update.

**The session object** is `RealtimeTranscriptionSessionCreateRequest` in `session.update`, and in
`session.created` and `session.updated` it is echoed as `{type: "transcription", object:
"realtime.transcription_session", id, include: null, audio: {input: {format, transcription,
noise_reduction: null, turn_detection: null}}}`. Nothing else is rendered: no `model`,
`instructions`, `voice`, `output`, `tools`.

| Field | Behaviour | Rejection `param` |
|---|---|---|
| `audio.input.format` | as in realtime sessions (§3): `audio/pcm` at 24 kHz, or G.711 at 8 kHz | `session.audio.input.format.type` / `.rate` |
| `audio.input.transcription` | as in realtime sessions: `model` is echoed and never judged (an unknown model is accepted, because the ASR is `providers.asr.model`; DECISIONS 2026-10-24), `language`, `prompt` (the same context path as D15's: the prompt in force at commit is the turn's). On by default; `null` still transcribes, with no language or prompt | `session.audio.input.transcription.<key>` |
| `audio.input.turn_detection` | `null` only, and `null` is the default (a realtime session defaults to `server_vad`). `server_vad` and `semantic_vad` are `invalid_value` | `session.audio.input.turn_detection` |
| `audio.input.noise_reduction` | `null` only | `session.audio.input.noise_reduction` |
| `include` | `null` or `[]` | `session.include` |
| `audio.output`, `model`, `instructions`, `output_modalities`, `tools`, `tool_choice`, `max_output_tokens`, `truncation`, `tracing`, `prompt` | not part of the object: rejected | `session.<field>` |

**Events a transcription session does not take.** `response.create`, `response.cancel`,
`conversation.item.create`, `.truncate` and `.delete` get an `error` (`invalid_request_error`,
`invalid_event`, `param: "type"`) whose message names the event; `output_audio_buffer.clear` is
refused everywhere (§2). The session stays open. `input_audio_buffer.append`, `.commit` and `.clear`
and `session.update` work as in realtime sessions.

**One commit, one transcript.** Each `input_audio_buffer.commit` → `input_audio_buffer.committed`,
`conversation.item.added`, then `conversation.item.input_audio_transcription.completed` (`item_id`,
`transcript`, `usage`, and `language` plus GA's `languages: [{code}]` when the ASR reported one) and
`conversation.item.done`. `previous_item_id` chains the items. No `.delta` events are sent: the
final transcript is the only one (AICC's final replaces any accumulated deltas). A failed
transcription is `…transcription.failed` for that item, as in realtime sessions. Items of
different commits finish in the order the ASR answers; an empty commit, which needs no ASR, may
overtake a transcript still being made. Items are not kept after their transcript is out, so a session
of hours holds no audio and no history (a realtime session keeps its items as the LLM's context).

**An empty or silent commit is never an error** (GA answers an empty buffer with
`input_audio_buffer_commit_empty`, and realtime sessions here with `invalid_event`). The client
cannot always know: AICC commits after speech, and a leg can go quiet before the commit lands.
The commit gets the same events with `transcript: ""` and no ASR request. "Silent" means no 20 ms
frame with an RMS of `transcription.voiced_rms` (400, on 16-bit samples) in the buffer, or, in a
buffer that has one, less than `transcription.min_speech_ms` (224) of speech by Silero
(`transcription.speech_threshold` 0.5): hold music and DTMF are loud, AICC's pump commits them, and
the ASR would answer with its prompt or run to its token limit. Such a commit is answered
`transcript: ""` too, and counted in `mlxrt_transcription_skipped`. `min_speech_ms: 0` turns the
check off. A second commit straight after a first is such an empty commit.

**Silence costs nothing.** While no frame in the buffer is voiced, it keeps only the last
`transcription.idle_keep_ms` (1 s), so a 90 s hold is 1 s of memory and the next utterance reaches
the ASR with at most that much silence in front of it (AICC commits after speech only, so a silent hold
is never committed on its own; hold music is above AICC's RMS 500 and is committed when it ends: it reaches the ASR as
"speech", `docs/PROGRESS.md`). Speech longer than `limits.input_audio_buffer_max_ms` (60 s)
without a commit loses its oldest audio, with a warning in the log; the session
lives, where a realtime session ends with `input_audio_buffer_overflow` (§9). The other limits
apply as before: `limits.inbox_max_ms` bounds audio queued faster than the actor reads it, not
audio over time; no idle timer exists, so pings and silence keep a session open for
`gateway.max_session_s` (3600 s); AICC reconnects after that.

**Metrics.** A voiced commit has a turn timeline (`committed` → `asr_done`) like a realtime turn,
with `session_type: "transcription"` among its logged values and in `/debug/turns`; its `asr_done`
stage is in `mlxrt_stage_seconds` with the realtime turns'. Transcription sessions share the
ASR's `max_in_flight` (4) and `queue_timeout_ms` with the realtime ones.

## 4. `response.create` parameters

`response.create.response` is a `RealtimeResponseCreateParams`; every field overrides the session for
this response only. `response: null` or absent means the session's settings.

| Field | Status | Notes | Rejection `param` |
|---|---|---|---|
| `instructions` | **Supported** | replaces the session instructions for this response (GA). Server extension, invisible on the wire: instructions that start with a configured direction are a verbatim line, spoken as written without the LLM (D16, §11.11) | `response.instructions` |
| `output_modalities` | **Supported** | `["audio"]` or `["text"]` | `response.output_modalities` |
| `max_output_tokens` | **Supported** | `"inf"` or 1–4096; reaching it ends the response `incomplete` (`max_output_tokens`) | `response.max_output_tokens` |
| `tool_choice` | **Partial** | `"auto"`, `"none"`, for this response only; `"required"` and forced functions **Unsupported** (`invalid_value`, as for the session) | `response.tool_choice` |
| `audio.output.voice` (`OutputVoice`) | **Partial** | this response's voice, resolved as the session's (D19); once the session has produced audio only the session voice is accepted, any other draws `cannot_update_voice` and no response. `OutputVoiceID`: Unsupported | `response.audio.output.voice` |
| `audio.output.format` | **Supported** | this response's format, as the session's (`audio/pcm` 24 kHz, `audio/pcmu`, `audio/pcma`); echoed in its `audio.output.format` | `response.audio.output.format.type` / `.rate` |
| `metadata` | **Supported** | at most 16 pairs; keys ≤ 64 characters, values strings ≤ 512; echoed on the response | `response.metadata` / `response.metadata.<key>` |
| `conversation` | **Partial** | `"auto"` (or `null`) only; out-of-band responses (`"none"`) Unsupported | `response.conversation` |
| `input` | **Unsupported** | out-of-band context | `response.input` |
| `tools` | **Unsupported** | tools are set on the session | `response.tools` |
| `prompt`, `reasoning`, `parallel_tool_calls` | **Unsupported** | — | `response.<field>` |
| unknown field | — | "Unknown parameter" | `response.<field>` |

## 5. Conversation items

Members of `ConversationItem`.

**Created by the client** (`conversation.item.create`):

| Item | SDK type | Status | Notes |
|---|---|---|---|
| system message | `RealtimeConversationItemSystemMessage` | **Supported** | parts `input_text` only |
| user message | `RealtimeConversationItemUserMessage` | **Partial** | parts `input_text` only; `input_audio` (audio enters through the input buffer) and `input_image`: Unsupported, `param: "item.content[i].type"` |
| assistant message | `RealtimeConversationItemAssistantMessage` | **Partial** | parts `output_text` only; `output_audio` cannot be created by a client (GA) |
| function call output | `RealtimeConversationItemFunctionCallOutput` | **Supported** | `call_id` must name a function call in the conversation (`param: "item.call_id"`); `output` is opaque text |
| function call | `RealtimeConversationItemFunctionCall` | **Unsupported** | made by the server only |
| MCP items | `RealtimeMcp*` | **Unsupported** | — |

Several parts per message are accepted; the LLM sees them joined by newlines. A client `id` must be
unique (`param: "item.id"`). `object: "realtime.item"` and a valid `status` are accepted and ignored.

**Emitted** (in `conversation.item.*`, `response.output_item.*` and `response.done`), always with
`id`, `object: "realtime.item"`, `type` and `status`; audio is never included:

| Item | Shape |
|---|---|
| client message | `{type: "message", role, status: "completed", content: [{type: "input_text" \| "output_text", text}]}` |
| committed user audio | `{type: "message", role: "user", content: [{type: "input_audio", transcript}]}`; `in_progress` with `transcript: null` until the ASR result, then `completed`. With transcription off the transcript stays `null`. |
| assistant message | `content: []` and `in_progress` in `*.added`; then `[{type: "output_audio", transcript}]` or `[{type: "output_text", text}]`, `completed` or `incomplete` |
| function call | `{type: "function_call", name, call_id, arguments}`; `in_progress` with `arguments: ""` in `*.added`, `completed` after |
| function call output | `{type: "function_call_output", call_id, output, status: "completed"}` |

## 6. Server events

Members of `RealtimeServerEvent`. Every event carries `event_id` (`event_<n>`, increasing within the
session).

| Event | SDK type | Status | Emitted |
|---|---|---|---|
| `session.created` | `SessionCreatedEvent` | **Supported** | first, with the full effective session |
| `session.updated` | `SessionUpdatedEvent` | **Supported** | after each accepted `session.update` |
| `conversation.created` | `ConversationCreatedEvent` | **Supported** | right after `session.created`: `{id: "conv_…", object: "realtime.conversation"}` |
| `error` | `RealtimeErrorEvent` | **Supported** | §9 |
| `input_audio_buffer.committed` | `InputAudioBufferCommittedEvent` | **Supported** | client commit, or the end of a detected speech turn |
| `input_audio_buffer.cleared` | `InputAudioBufferClearedEvent` | **Supported** | after `input_audio_buffer.clear` |
| `input_audio_buffer.speech_started` | `InputAudioBufferSpeechStartedEvent` | **Supported** | `server_vad` only, once speech has lasted `turn.vad.min_speech_ms` (150), or `barge_in.min_speech_ms_while_playing` (400) while the client is modelled as playing our audio, and never for an onset in the first `barge_in.echo_guard_ms` (2000) of a response's playback (§11.13); while the client is modelled as playing our audio it comes only once the ASR has heard the caller, so later than the onset, at the pause or the turn end for a short utterance (§11.13); `audio_start_ms` includes `prefix_padding_ms` and is the onset's, not the event's time; `item_id` of the item the commit will create. One per turn: speech that resumes while the turn is held continues it (§11.9) |
| `input_audio_buffer.speech_stopped` | `InputAudioBufferSpeechStoppedEvent` | **Supported** | `server_vad` only, when the turn ends (§11.9), immediately before `input_audio_buffer.committed`; `audio_end_ms` is the acoustic end of speech (§11.6) |
| `conversation.item.added` | `ConversationItemAdded` | **Supported** | client items, committed audio, response output items |
| `conversation.item.done` | `ConversationItemDone` | **Supported** | client items (at once), user audio (when the transcript is final), response items (when they close) |
| `conversation.item.created` | `ConversationItemCreatedEvent` | **Never** (D1) | replaced by `conversation.item.added` + `conversation.item.done` |
| `conversation.item.input_audio_transcription.completed` | `ConversationItemInputAudioTranscriptionCompletedEvent` | **Supported** | one per committed utterance when transcription is on (the default, D15): the whole turn's transcript from Qwen3-ASR, trimmed; `usage: {type: "duration", seconds}` |
| `conversation.item.input_audio_transcription.failed` | `ConversationItemInputAudioTranscriptionFailedEvent` | **Supported** | the ASR request failed; the turn's response ends `failed` |
| `conversation.item.input_audio_transcription.delta` | `ConversationItemInputAudioTranscriptionDeltaEvent` | **Unsupported** | the ASR returns final transcripts only |
| `conversation.item.input_audio_transcription.segment` | `ConversationItemInputAudioTranscriptionSegment` | **Never** | diarization is not offered |
| `conversation.item.truncated` | `ConversationItemTruncatedEvent` | **Supported** | after a truncate, with the clamped `audio_end_ms` |
| `conversation.item.deleted` | `ConversationItemDeletedEvent` | **Supported** | after a delete |
| `conversation.item.retrieved` | `ConversationItemRetrieved` | **Never** | retrieve is unsupported |
| `response.created` | `ResponseCreatedEvent` | **Supported** | at once on `response.create`, or at a VAD commit with `create_response` (D7) |
| `response.output_item.added` / `.done` | `ResponseOutputItemAddedEvent` / `ResponseOutputItemDoneEvent` | **Supported** | per output item; the message item is created lazily, on the first text delta |
| `response.content_part.added` / `.done` | `ResponseContentPartAddedEvent` / `ResponseContentPartDoneEvent` | **Supported** | `part: {type: "audio", transcript}` or `{type: "text", text}` |
| `response.output_audio.delta` / `.done` | `ResponseAudioDeltaEvent` / `ResponseAudioDoneEvent` | **Supported** | base64 of the response's output format (`audio/pcm` 24 kHz whole samples, or G.711 8 kHz: 800 bytes per 100 ms), at most `output.max_delta_ms` (100 ms) per delta: Qwen3-TTS speech (from its 4-bit build, in the same voice, when the fallback speaks a segment), sent as soon as synthesized but at most `output.max_lead_ms` (10 s) ahead of real time (D6, §11.8) |
| `response.output_audio_transcript.delta` / `.done` | `ResponseAudioTranscriptDeltaEvent` / `ResponseAudioTranscriptDoneEvent` | **Supported** | one delta per synthesized segment, sent just before the segment's first audio; `.done` carries the text whose deltas reached the client |
| `response.output_text.delta` / `.done` | `ResponseTextDeltaEvent` / `ResponseTextDoneEvent` | **Supported** | text modality (D12): one delta per LLM delta; TTS is never opened |
| `response.function_call_arguments.delta` / `.done` | `ResponseFunctionCallArgumentsDeltaEvent` / `ResponseFunctionCallArgumentsDoneEvent` | **Supported** | complete calls only, the arguments in one delta (§11.5); several calls per response, each its own item |
| `response.done` | `ResponseDoneEvent` | **Supported** | always, last for its response (§7) |
| `rate_limits.updated` | `RateLimitsUpdatedEvent` | **Never** | there are no rate limits to report |
| `input_audio_buffer.timeout_triggered` | `InputAudioBufferTimeoutTriggered` | **Never** | `idle_timeout_ms` accepts only `null` (D4) |
| `input_audio_buffer.dtmf_event_received` | `InputAudioBufferDtmfEventReceivedEvent` | **Never** | SIP only |
| `output_audio_buffer.started` / `.stopped` / `.cleared` | `OutputAudioBufferStarted` / `Stopped` / `Cleared` | **Never** | WebRTC/SIP only |
| `mcp_list_tools.*`, `response.mcp_call*` | `McpListTools*`, `ResponseMcpCall*` | **Never** | MCP tools are not served |

## 7. The response object

`response.created` and `response.done` carry a `RealtimeResponse`:

```
{object: "realtime.response", id: "resp_…", status, status_details, output, conversation_id,
 output_modalities, max_output_tokens, audio: {output: {format, voice}}, usage, metadata}
```

- In `response.created`: `status: "in_progress"`, `status_details: null`, `output: []`, `usage: null`.
- `status_details` (`RealtimeResponseStatus`):

  | Status | `status_details` | Cause |
  |---|---|---|
  | `completed` | `null` | the LLM finished and all audio was handed to the writer, or nothing was to be said (D7) |
  | `cancelled` | `{type: "cancelled", reason: "turn_detected"}` | speech started with `interrupt_response` (barge-in) |
  | `cancelled` | `{type: "cancelled", reason: "client_cancelled"}` | `response.cancel` |
  | `incomplete` | `{type: "incomplete", reason: "max_output_tokens"}` | `max_output_tokens` reached, or the response's audio exceeded `limits.max_response_audio_ms` |
  | `incomplete` | `{type: "incomplete", reason: "content_filter"}` | the LLM reported `content_filter` |
  | `failed` | `{type: "failed", error: {type: "server_error", code, message}}` | a backend failed or timed out (§9) |

- `output`: every item the response produced, without audio; a message ended early is `incomplete`.
- `usage` (`RealtimeResponseUsage`): LLM text tokens; `audio_tokens` are always 0; `null` when the LLM
  reported none (a cancelled or empty response, or a verbatim line, which asks no LLM).
- `audio.output.voice`: the effective voice that spoke the response (D19).

## 8. Golden event orders

Each order below is pinned by a golden trace in `tests/golden/` (both directions, ids normalized),
and every golden run is also checked against the causal invariants of `tests/integration/invariants.py`.

**Connect** (`connect.jsonl`)
```
session.created → conversation.created
```

**Client item, text response** (`text_turn.jsonl`)
```
conversation.item.added → conversation.item.done
response.created → response.output_item.added → conversation.item.added → response.content_part.added
→ response.output_text.delta × n → response.output_text.done
→ response.content_part.done → response.output_item.done → conversation.item.done → response.done
```

**Audio response** (`audio_turn.jsonl`; a verbatim line, `verbatim_line.jsonl`, has exactly these
events); per segment, its transcript delta immediately precedes its first audio delta:
```
response.created → response.output_item.added → conversation.item.added → response.content_part.added
→ { response.output_audio_transcript.delta → response.output_audio.delta × k } per segment
→ response.output_audio.done → response.output_audio_transcript.done
→ response.content_part.done → response.output_item.done → conversation.item.done → response.done
```

**Manual commit, then `response.create`** (`audio_turn.jsonl`); the response waits for the transcript:
```
input_audio_buffer.committed → conversation.item.added
response.created → conversation.item.input_audio_transcription.completed → conversation.item.done
→ response.output_item.added → … (audio response)
```

**`server_vad` turn** (D7; `barge_in.jsonl`, second turn; with the real VAD, ASR and LLM providers on
fake backends: `tests/integration/test_speech_turns.py`, text modality):
```
input_audio_buffer.speech_started → input_audio_buffer.speech_stopped
→ input_audio_buffer.committed → conversation.item.added → response.created
→ conversation.item.input_audio_transcription.completed → conversation.item.done
→ response.output_item.added → … (audio response, or response.output_text.* → response.done)
```
With `create_response: false`, or while another response is active, no response is created; the
commit still happens.

**Empty or whitespace transcript** (D7; `empty_transcript.jsonl`): the reply is completed at once with no
output and no LLM call, so a client waiting for one reply per turn is released:
```
… → response.created → conversation.item.input_audio_transcription.completed{""}
→ conversation.item.done → response.done{completed, output: []}
```

**Barge-in** (`barge_in.jsonl`):
```
input_audio_buffer.speech_started
→ response.output_audio.done → response.output_audio_transcript.done → response.content_part.done
→ response.output_item.done{incomplete} → conversation.item.done → response.done{cancelled, turn_detected}
```
These events come out of one synchronous step: nothing else lies between `speech_started` and
`response.done` (ARCHITECTURE §6.5 steps 1–7). While the response still awaits its transcript (no
output item yet): `speech_started → response.done{cancelled}`. Speech over the tail of an answer
whose response is already done (its audio all sent, the client still playing it) emits
`speech_started` alone: there is nothing to cancel, and the client's truncate stops our model of
its playback. In every case the reply follows `speech_stopped`.

**G.711** (`g711_turn.jsonl`: μ-law in, A-law out, a `server_vad` turn and a clamped truncate;
`g711_barge_in.jsonl`: μ-law both ways): the same orders as with PCM; only the audio payloads
(8 kHz G.711, at most 800 bytes a delta) and the echoed formats differ.

**`response.cancel`** (`response_cancel.jsonl`): as barge-in without `speech_started`, with
`reason: "client_cancelled"`. A second cancel: `error` `response_cancel_not_active`.

**Backend failure:** `error{server_error, backend_*}` → the terminal events of the open item →
`response.done{failed}`. The `error` comes first, so a client that ties an error to its in-flight request
ties it to this response.

**`response.create` while active** (`response_while_active.jsonl`):
`error{conversation_already_has_active_response}`; the active response continues.

**Function call** (the long-session invariant test):
```
response.output_item.added{function_call} → conversation.item.added
→ response.function_call_arguments.delta (the complete arguments, once)
→ response.function_call_arguments.done → response.output_item.done → conversation.item.done
```
A message item that started before the call keeps output index 0 (the call takes 1), and its closing
events come after the call's. A call-only response has no message item.

**Truncate and delete** (`truncate_and_delete.jsonl`):
`conversation.item.truncate → conversation.item.truncated`;
`conversation.item.delete → conversation.item.deleted`.

**Errors** (`protocol_errors.jsonl`): an `error` replaces the events the rejected client event would
have produced; nothing else changes and the session stays open.

## 9. Errors

Shape (`RealtimeErrorEvent`, `RealtimeError`):

```
{type: "error", event_id: "event_…", error: {type, code, message, param, event_id}}
```

`error.event_id` is the client event's `event_id` (or `null`). An `error` is sent only where the
protocol requires one or a client request cannot be honoured (D18): AICC ties any error to the request
it has in flight.

**Client errors** (`type: "invalid_request_error"`; the session continues):

| Code | When | `param` |
|---|---|---|
| `invalid_event` | invalid JSON, a binary frame, a missing `type`, an unsupported event type, an empty buffer on commit (realtime sessions only), a realtime-only event in a transcription session (§3.1) | `"type"` for event types, else `null` |
| `invalid_value` | a field missing, mistyped, out of range, unsupported or unknown; an unknown `previous_item_id` or `call_id`; a duplicate item id; a truncate of something other than an assistant audio message, or with `content_index` ≠ 0 | the dotted path: `session.audio.input.turn_detection.type`, `item.content[0].type`, `response.metadata.k`, … |
| `conversation_already_has_active_response` | `response.create` while a response is active (D18) | `null` |
| `response_cancel_not_active` | `response.cancel` with no active response, or naming another one (D18) | `null`, or `"response_id"` |
| `item_not_found` | `conversation.item.delete` or `.truncate` of an unknown item (D18) | `"item_id"` |
| `cannot_update_voice` | a different voice once the session has produced audio (GA's voice lock; the one voice error, D19) | `session.audio.output.voice` or `response.audio.output.voice` |

**Server errors** (`type: "server_error"`):

| Code | When | Effect |
|---|---|---|
| `not_ready` | connection before the backends were first ready | close 1013 |
| `session_limit_reached` | over `gateway.max_sessions` | close 1013 |
| `backend_unavailable` / `backend_timeout` / `backend_busy` / `provider_error` | a provider raised `ProviderError` of kind `unavailable` / `timeout` / `overloaded` / `bad_response` or `internal` | the response ends `failed`; the session continues |
| `input_queue_overflow` | more than `limits.inbox_max_ms` of audio, or `limits.inbox_max_commands` events, queued | fatal: close 1011 |
| `input_audio_buffer_overflow` | the input buffer of a realtime session exceeds `limits.input_audio_buffer_max_ms` (a transcription session drops its oldest audio instead, §3.1) | fatal: close 1011 |
| `slow_client` | the client stopped reading (§1) | close 1008; the `error` is sent if the socket still accepts it |
| `internal_error` | an unexpected exception in the session | fatal: close 1011; other sessions are unaffected |

Clamping is preferred to errors where the client could do nothing useful with one: a truncate beyond
the audio sent is clamped (D18), and unknown voices are accepted (D19).

## 10. Identifiers

`sess_`, `conv_`, `item_`, `resp_` and `call_` followed by 16 random lowercase alphanumerics; logs also
carry `turn_…`. Server event ids are `event_1`, `event_2`, … in each session. Every conversation item,
function calls included, gets an `item_` id; GA example prefixes such as `fc_` are not copied. The wire
`call_id` is minted by the gateway; the LLM provider's own call id stays internal and links a
`function_call_output` back to its call in the LLM history. Client-supplied item ids are kept.

## 11. Behaviour notes

1. **Response creation (D7).** `response.created` is sent immediately, for a client `response.create`
   and for a VAD commit alike. The LLM starts once every pending transcript is final. If the response
   answers a user audio item (the item a VAD commit created, or the conversation's last item when the
   client asked) and that transcript is empty or whitespace, the response completes with no output and
   no LLM call.
2. **What a cancelled response leaves.** The transcript of a cancelled audio message is the text whose
   transcript deltas reached the client. Audio and transcript deltas still queued at the cancel are
   discarded unsent (§6.5 step 4), and so is audio the pacer was holding back; the writer counts
   what it handed to the socket. For later LLM requests a cancelled audio message (barge-in or
   `response.cancel`) is cut to what the client can have heard when it sends no truncate (D5):
   `min(audio sent, time since its first audio delta was sent)` at the cancel, by the rule of §11.3.
   A failed response keeps the text sent.
3. **Truncation (§6.6).** `audio_end_ms` is clamped to the audio sent for the item (with a warning in
   the log, never an `error`: D18). The item's text for later LLM requests becomes every segment that
   ended by then plus a proportional prefix of the one playing: the audio speaks the segment's
   normalized text, so the share is taken in spoken characters and mapped back through the
   normalizer's character map (a phone number read digit by digit counts as heard only as far as
   its digits were read), then snapped back to a word start in Latin text. A segment still being
   synthesized at the cut counts over the audio made so far. The item is truncated in place, also
   after `response.done`; a truncate overrides the D5 cut of §11.2, and a later truncate overrides an
   earlier one. A truncate to 0 followed by `conversation.item.delete` (AICC's flow for an answer
   nobody heard) leaves no trace of the item in later requests. The truncate also tells the gateway
   the client stopped playing the item there (§11.13).
4. **Deleting an item a response is producing** is accepted: the item leaves the conversation, the
   response's own events continue, and its closing `conversation.item.done` is skipped.
5. **Function calls (§6.7, M5).** Calls come from `mlx_lm.server`'s own parser (Qwen3.6:
   qwen3_coder XML, arguments typed by the tool's JSON schema), one complete call at a time.
   - Each call is delivered as soon as it is complete, as its own item: `output_item.added` →
     `conversation.item.added` → `function_call_arguments.delta` (all of it, once) → `.done` →
     `output_item.done` → `conversation.item.done`. A response may hold several. The wire
     `call_id` (`call_…`) is minted by the gateway; the server's own id is kept for history.
   - Text before a call is spoken (or streamed) normally; call markup never reaches the TTS or a
     transcript (the server streams nothing while the model is inside a call). Whitespace at
     either end of an answer is dropped, so the newline a model writes before a call opens no
     empty message. A message started before the call keeps output index 0, and its audio may
     still be streaming when the call is delivered (the call's events come in between).
   - A call-only response ends with `response.done{completed}` as soon as the LLM is done:
     nothing waits on the TTS (AICC's 3 s tool-only deadline).
   - A response cancelled before a call is complete delivers no call; a delivered call is never
     retracted (it stays in the cancelled response's `output` and in the conversation, and its
     `function_call_output` is accepted). `.done` means "delivered", not "valid JSON": the gateway
     never parses arguments, the server's parser already did.
   - `tool_choice: "none"` sends no tools to the LLM. A call the model makes anyway (the server
     parses call markup whether or not tools were sent), or a call to a function the session does
     not declare, is dropped with a warning in the log, never delivered. A JSON "call" the model
     writes into the text of such a reply (`{"tool_name": …, "tool_input": …}`, `{"name": …,
     "arguments": …}`, alone or in a code fence) is removed from the reply before the
     segmenter, so it is neither spoken nor in the transcript; any other text, other JSON
     included, is left as written (LLM hardening).
   - Qwen3.6 sometimes opens a call with the bare function name (`<transfer_to_agent>` for
     `<function=transfer_to_agent>`); the LLM backend's parser accepts that form for a declared
     function (M0: 11% of transfer turns), so such a call is delivered like any other.
   - A call the server's parser still rejects is dropped by the server itself. When that leaves
     the response with no call at all (`finish_reason: tool_calls` and nothing parsed) before
     any text of the response was produced, the gateway asks the LLM once more, within the same
     response (sampled at the reply temperature, since the tools' greedy setting would repeat
     the same call); the client sees one response whose call is simply later. If that also
     fails, or text was already produced, the response fails (`server_error`,
     `provider_error`); when another call was delivered, the dropped one is invisible.
6. **`audio_end_ms`** of `speech_stopped` is the acoustic end of speech (ARCHITECTURE §6.2), not GA's
   end of the silence window; the committed audio ends there. With Smart Turn deciding, the commit
   comes some time after it; logs record that delay as the turn's `turn_confirmed` stage.
7. **Input timeline.** `audio_start_ms` and `audio_end_ms` count milliseconds of audio since the
   session's first append (GA). Audio is held at 16 kHz internally, whatever the input format (24 kHz
   PCM or 8 kHz G.711), so the milliseconds are the same for the same audio; the resampler's tail is
   flushed at a manual commit, and at an input format change, so no trailing audio is lost.
8. **Output pacing (D6).** Audio is synthesized as fast as the TTS allows (one request per segment,
   the next as soon as the previous has finished generating) and sent as soon as it is ready, but
   never more than `output.max_lead_ms` (10 s) ahead of real time, counted from the response's first
   audio delta. At the cap the deltas go out at the playback rate, one per `max_delta_ms`, so no gap
   between deltas comes near 2 s while audio is waiting. Audio held back by the cap stays in the
   gateway: a barge-in or `response.cancel` discards it unsent (§11.2). `response.done` follows the
   last delta. The writer's backpressure still applies: producers wait whenever half of
   `outbound_queue_events` are queued.

   **Multi-session TTS (M6).** Sessions share one serialized TTS stream, so a response's next
   segment can wait behind other sessions. While a response still has segments to come, the
   gateway holds back up to `output.reserve.hold_ms` (1.5 s) of its synthesized audio and sends it
   a delta at a time: after `keepalive_ms` (1 s) without a delta, or when the client's modelled
   buffer falls below `min_lead_ms` (400 ms). The client sees the same deltas in the same order,
   only spread out, so the gap between deltas stays under AICC's 2 s watchdog while the client
   still has audio to play. The TTS queue serves the request whose session runs out of audio
   soonest first (`providers.tts.scheduling: edf`); a first segment counts from when its response
   started. A request waits at most `queue_timeout_ms` (2 s) past that point, then the response
   fails with `backend_busy` (§9).
9. **Turn detection (`server_vad`, D4).** Speech detection runs off the session's event loop, on
   CPU executors; positions are on the input timeline, so the same audio always yields the same
   turns.
   - Speech starts once Silero's probability has been at least `threshold` and stayed above
     `threshold − 0.15` for `turn.vad.min_speech_ms` (150): clicks and short noises never start a
     turn or interrupt a response.
   - After `silence_duration_ms` of silence, Smart Turn judges the turn's last ≤ 8 s, trailing
     silence included. Complete (probability > `turn.smart_turn.threshold`, 0.5): the turn ends.
     Incomplete: the turn is held; speech that resumes continues it (no second `speech_started`),
     and after `turn.smart_turn.max_silence_ms` (3000) of silence it ends anyway.
   - With `turn.smart_turn.enabled: false`, `silence_duration_ms` of silence ends the turn.
   - While no turn is in progress the input buffer keeps only `prefix_padding_ms` plus one frame,
     so a caller who streams silence never fills it.
   - A manual `input_audio_buffer.commit` or `.clear`, and any change to `turn_detection`, drop the
     turn in progress.
   - Input may arrive faster than real time, but at most `limits.inbox_max_ms` (2 s) of it may wait
     unread in the session's queue (§9, `input_queue_overflow`).
10. **Voices (D19).** `voices.map` in `config.yaml` names the voices a session can ask for (default:
    the Qwen3-TTS speakers `vivian`, `serena`, `uncle_fu`, `aiden`; `vivian` is the default;
    `ryan` is not offered since M5, because of the runaway defect M0 found). Each name says which
    Qwen3-TTS speaker speaks it (optionally a different speaker per segment language: by default
    `vivian`'s English segments are spoken by `aiden`, as M0 S3 chose) and how its lang_code is
    chosen (each segment's zh/en tag, or a fixed code). The fallback model has the same speakers
    (D13), so a segment it takes is spoken in the same voice. Any other name is accepted, spoken by
    the default voice, logged as a warning, and `session.updated` and the response echo the
    effective name, never the one asked for. Once the session has produced audio its voice is
    locked (GA); clients that set the voice in their first `session.update` only, as AICC does,
    never meet the lock.
11. **Verbatim lines (D16), a server extension invisible on the wire.** A `response.create` whose
    own `instructions` start with a configured direction (`lines.verbatim_prefixes`; by default
    AICC's "Say exactly this, word for word, and add nothing else:\n" and
    "请一字不差地说出下面这句话，不要添加任何其它内容：\n") skips the LLM: the rest of the instructions,
    trimmed, is segmented, normalized for the TTS and spoken, and becomes the assistant message (its
    transcript, and what later LLM requests see). The events are exactly those of any audio response
    (text modality: one `output_text` delta); `usage` is `null`. A line needs no conversation: it
    works on an empty one, mid-call, and while a function call is unanswered, and it neither waits
    for pending transcripts nor makes a tool call. Any other `instructions` override replaces the
    session instructions as GA specifies. A client that does not send these exact prefixes never
    sees a difference.
12. **What the TTS hears.** Each segment is normalized for the TTS only (transcripts keep the
    original text), within the scope M0's S9 probe chose (G5): phone numbers in every form, digit
    by digit (1 as 幺 in Chinese, 尾号 and extensions included); alphanumeric and dashed IDs
    character by character; e-mail addresses (with "@" and "." read out); cardinals of 5 or more
    digits; and in English text numeric dates and amounts of 1000 or more. Money, decimals,
    percentages, ordinals, other dates and times, units, URLs and brand names stay as written
    (`text.normalizer`). A segment is tagged zh (any CJK, mixed text included) or en, which picks
    each model's lang_code. Qwen3-TTS streams 0.16 s chunks, with `max_tokens` capped in
    proportion to the text, so a runaway stops early. Each segment's edge silence is trimmed as
    it streams (at most 60 ms before its first voiced frame and 100 ms after its last; inner
    pauses kept; `output.trim_silence`): the client receives only the trimmed audio, and every
    count (transcript timing, truncation, D5, the echo reference) is of that audio. A segment the
    primary TTS (Qwen3-TTS 0.6B 8-bit) fails before its first audio is spoken by the fallback, the
    4-bit build of the same model, in the same voice and whatever its language (D13); a failure
    after audio started fails the response (`provider_error` / `backend_*`), so no segment is
    spoken partly by each model.
13. **Echo (D17), invisible on the wire.** Neither AICC nor FreeSWITCH cancels echo, so the caller's
    audio carries our own speech back. The gateway handles it, configured in `config.yaml`'s `aec`
    and `barge_in` sections (`config.mock.yaml` turns the canceller off: its sine-tone TTS is no
    speech for the canceller):
    - **Echo cancellers (G7 dual path).** Each session runs WebRTC AEC3 (pywebrtc-audio) on the
      16 kHz input before speech detection and Smart Turn. What the ASR transcribes is
      `aec.asr_input`: `speex` (the default), the output of a SpeexDSP linear canceller run on the
      same frames and of SpeexDSP's preprocessor (noise and residual-echo suppression), which
      removes the echo but keeps a caller who talks over us; `aec3`, AEC3's output, whose
      suppressor also removes quiet caller speech in long double talk; or `raw`. Speex's output is
      band-limited to 3.8 kHz, and until its reference is aligned (about 2.5 s into the call's
      first answer) the ASR hears the caller's audio as received, uncancelled. Both references are
      our own audio as the writer sent it, placed on a model of AICC's playout (20 ms ticks, a
      60 ms prebuffer, a re-buffer after more than 2 empty ticks) 60 ms ahead of where it plays. A
      GCC-PHAT delay estimator re-aligns AEC3's reference when the residual delay leaves
      [5, 450] ms, places Speex's 20 ms ahead of the echo (again when it leaves [0, 44] ms), and
      learns AICC's tick phase. A barge-in or `response.cancel` ends the modelled playback of what
      the client had not played; a client truncate ends it at the truncated position and corrects
      the model's timing (Speex's reference stays where the echo was measured). On a line where
      the estimator finds no echo of our playing audio (1.5 s of it), AEC3 is bypassed until echo
      shows up, because it would otherwise take a caller talking over us for echo and silence them
      (Speex, a linear filter, needs no bypass). While a response may still add audio, input waits
      about 70 ms for its reference, so speech events during a response come that much later.
    - **Speech events while the client is modelled as playing** (from the first frame of a
      response's audio to its last, or a flush, plus 450 ms of echo tail): speech must last
      `barge_in.min_speech_ms_while_playing` (400) before `speech_started`, and an onset in the
      first `barge_in.echo_guard_ms` (2000) of a response's playback counts only from the guard's
      end: a stretch that ends before then is discarded, never a turn. Speech that began before
      the playback, or continues after it, follows the usual rules.
    - **Echo turns.** A detected turn whose transcript matches the assistant audio the client was
      playing around it (at least 70 % of its CJK characters and Latin words lie in runs of 3 or more
      that follow each other in that audio's text; turns of at least 4 of them) is treated like noise (D7): its response completes at once with no output
      and no LLM call, and later LLM requests leave the turn out. One reply per speech turn still
      holds.
    - **A caller who speaks into the echo** (issue #8) lands in the same turn as the echo, which
      is the bulk of the transcript. The turn is judged without its echo prefix, the leading
      stretch of the transcript that follows the assistant text in runs of 3 words or more. If
      what comes after it has at least 4 words and is no echo itself (under 70 % in such runs), the
      turn is the caller's, and the remainder is its transcript everywhere the client and the LLM
      see it: `conversation.item.input_audio_transcription.completed` and the item's
      `content[0].transcript` carry "等一下，我想问问天气", not the echo before it, and the LLM
      request holds only that. The turn's audio, and its `audio_start_ms` and `audio_end_ms`,
      still cover the echo. A held turn is released at the end this way (`speech_started` then
      `speech_stopped`) if 2 of the remainder's words are also not ours. A remainder of fewer
      than 4 words, or one that is ours, leaves the turn an echo turn.
    - **Held `speech_started`** (`barge_in.confirm_while_playing`). A turn that starts while the
      client is modelled as playing, with or without a response active, gets no `speech_started`
      at its onset: the ASR transcribes the turn so far and the event goes out once at least 2
      of its words are not part of the assistant text played around it, so a client that
      flushes its queue on any `speech_started` (AICC) is not cut by echo that passed its AEC.
      The event carries the onset's `audio_start_ms` and the turn's item id, then the barge-in
      rule applies. A single word is enough once the caller pauses (the transcript of the whole
      utterance has one word that is not ours), so it comes at the pause, not at the turn end;
      it also comes when the ASR fails. The end of the playback does not release a turn whose
      checks match our words. Once a turn's checks have matched our words (3 words or more, none
      outside them), 2 words that are not ours release it only if they come before the
      transcript's last word, or if the next check has them too: a mishearing at the edge of
      a check does not release echo, and a caller who joins ongoing echo waits about 250 ms more. On a line with no steady echo (AEC3 bypassed), a turn whose raw
      input correlates strongly with our own audio (GCC-PHAT PSR of 15 or more,
      `barge_in.confirm_echo_psr`) is not released on the ASR's word alone, and one that ends with a
      PSR of 15 or more (`confirm_echo_drop_psr`) is dropped as echo even if the ASR heard a caller.
      The correlation never releases a turn the words call echo: a live line's echo correlates as
      weakly as callers. A caller who only paraphrases what we were saying, reusing short
      fragments of it, is no echo and gets `speech_started` at the pause or the turn end.
      A turn that ends held and is our echo (4 words or more) or empty gets
      no event at all (no `speech_started`, `speech_stopped`, commit, item or response); any
      other is released at its end, `speech_started` then `speech_stopped`. One pair per turn, in
      order, as ever.
    - Each rule can be switched off (`echo_guard_ms: 0`, `min_speech_ms_while_playing: null`,
      `echo_turn_suppression: false`, `confirm_while_playing: false`, `aec.enabled: false`).
14. **Tool calls in later LLM requests (M5), invisible on the wire.** The round trip is GA's:
    `conversation.item.create {type: function_call_output, call_id, output}` (any text; AICC's JSON
    with a `hint` reaches the model verbatim), answered by `conversation.item.added` → `.done`, then
    `response.create`. The conversation is replayed for the Qwen template by these rules
    (`providers.llm` in `config.yaml`):
    - A call's result always follows its call: the call is an assistant message with
      `tool_calls` (text the same response spoke before the call is its content), followed at once
      by the results of its calls, in call order, however many outputs arrived and in whatever
      order.
    - What the conversation holds between a call and its result is moved after the results,
      keeping its order: user notes (keypress, dead air, "(System note …)"), caller speech, later
      answers. An assistant text spoken before any such user item (AICC's wait line) is merged
      into the call's message, before the call (`tool_interlude: merge`, the default), or placed
      after the results (`after_output`).
    - A call still unanswered when a request is built (the caller speaks while the wait line plays
      and the tool runs) gets a placeholder result, `pending_tool_output` (`unanswered_calls:
      pending`, the default), so the model neither repeats the call nor invents its result;
      `keep` replays it without a result, `drop` leaves it out.
    - The output of a deleted call is left out; a deleted call leaves nothing.
    - A conversation with no user message opens with the synthetic cue, before anything the
      assistant said.

    How the real model behaves with each setting is checked by M0's LLM spike and the backend
    tests; the defaults may change with their findings.
15. **G.711 (M7, D2, D3).** `{type: "audio/pcmu"}` (μ-law) and `{type: "audio/pcma"}` (A-law),
    8 kHz mono, one byte a sample, in either direction; the two directions are independent.
    - **Codec.** Lookup tables built from CPython `audioop.c`'s algorithm (256 entries to decode,
      65 536 to encode), equal to `audioop` on every code and every sample (M0 S7: not Sun's later
      `g711.c`).
    - **Input.** Decoded, then resampled 8 → 16 kHz (`g711.input_quality`, default soxr LQ: 32 ms
      mean lag instead of HQ's 64, flat to 3 kHz, -8 dB at 3.4 kHz); from there the path is the
      PCM path's (echo canceller, Silero on 16 kHz audio, Smart Turn, input buffer, ASR). The
      timeline stays in milliseconds of audio, so `audio_start_ms`, `audio_end_ms` and the
      transcription's `usage.seconds` are those of the same audio sent as PCM.
    - **Output.** The TTS's 24 kHz resampled to 8 kHz (`g711.output_quality`, default HQ; one
      stream per response, flushed at each segment's end), then encoded; deltas of at most
      `max_delta_ms` (800 bytes). Pacing (D6), the reserve, the delivery ledger, truncation and
      its clamp (D18), D5 and `max_response_audio_ms` all count 8 kHz milliseconds (bytes / 8).
    - **Echo (D17).** The canceller's reference is the G.711 we sent, decoded and resampled to
      16 kHz the way the caller's audio is; AICC's playout model frames it in 160-byte frames.
    - **Format changes** after audio has flowed are accepted, as GA allows: later appends decode
      in the new input format (what the old decoder held goes onto the timeline first), later
      responses speak the new output format, and `response.create` may set its own.
16. **LLM requests and the prompt cache (LLM hardening), invisible on the wire but for timing.**
    - **Sampling (G2).** A request with tools attached is greedy (temperature 0, top_p 1.0:
      Qwen3.6 makes 19 of 20 expected calls, 16–19 at 0.7); one without tools is sampled at
      `providers.llm.temperature` / `top_p` (0.7 / 0.8, the server's own `--temp`/`--top-p`, with
      its `--top-k 20 --min-p 0.0`). A reply has at most 512 tokens; `max_output_tokens` can only
      lower that.
    - **Prefix warming.** When a `session.update` changes the instructions or the tools (AICC's
      phase changes, and every session's first update), the gateway sends the LLM a one-token
      request with the prefix of the session's next request: the new system message and tools,
      and the conversation up to its last user message. The model's prompt cache then holds that
      prefix when the request comes, so its first token does not wait for a full prefill. While
      the session's LLM is still generating (AICC sends the update as soon as a call's arguments
      are done, before `response.done`), the warm-up starts when it finishes. A response whose
      instructions and tools match a warm-up still in flight waits for it, at most the
      first-token deadline (5 s): racing it would prefill the same prompt twice (M0 S2). The
      LLM cannot drop a request it is prefilling, so warm-ups never pile up: one per
      instructions-and-tools at a time (a burst of new calls with the same flow warms it
      once), at most `max_warmups_in_flight` (1) in all, and none once a real request with
      those instructions has gone out. A failed warm-up is logged and costs nothing else
      (`providers.llm.warm_on_update`).
    - **Cache checkpoints.** The LLM backend also keeps its prompt cache where the tool
      definitions end and where new instructions stop matching recent ones (AICC's instructions
      before `当前环节：`), so a phase change or a per-response override prefills only what changed
      (`--tools-checkpoint`, `--shared-prefix-checkpoint` in `backends.llm.args`). Outputs are
      unchanged.

## 12. AICC compatibility

ai-native-callcenter attaches through its existing `gateway` profile with no change (D11).
**Since M7 a dedicated AICC profile can set `AcceptsG711: true`**: the gateway then gets the call's
G.711 untouched (`{"type": "audio/pcmu"}` or `audio/pcma` both ways, 160-byte appends every
20 ms) and answers in it, so neither AICC nor the gateway converts telephone audio to 24 kHz
(and back). The five traces are also replayed in that shape, and AICC's harness runs on a
`GatewayProfile` with `AcceptsG711: true` (`tests/aicc/probe.py --g711`); `tool_choice`
(`none`/`auto`) can be enabled in the same profile (D11). Its five
client traces in `spikes/aicc/traces/` are replayed in `tests/integration/test_aicc_replay.py`, on the
real TTS path (the TTS provider and its fallback against a fake `mlx_audio.server` paced as M0
measured): every frame is sent as captured, AICC's fake-server ids are swapped for this server's ids,
no frame draws an `error`, and every response meets AICC's deadlines. AICC's own client harness
(`probe`, `measureResponseCreated`, a lines check, and since M4 the barge-in and keypress
scenarios) runs against the gateway with `tests/aicc/probe.py` (`make aicc-probe`). AICC's
requirements (ARCHITECTURE §2) at M5:

| Area | M5 |
|---|---|
| Handshake | `session.updated` answers the first `session.update` at once; every AICC `session.update` (tools, `server_vad` with only `silence_duration_ms: 500`, flow voice) is accepted. |
| Session shape | GA, `audio/pcm` 24 kHz both ways (the `gateway` profile), or G.711 passed through both ways (a profile with `AcceptsG711: true`, M7); flat function tools; a flow-chosen voice is accepted, spoken by the default voice when not configured, and the effective voice echoed (D19); instructions-only updates merge and never meet the voice lock. |
| Turn-taking | `server_vad` is Silero VAD + Smart Turn (§11.9): AICC's `silence_duration_ms: 500` is the silence before Smart Turn decides. One response per commit, at the commit (D7); an empty transcript completes it empty. |
| Deadlines | `response.created` is immediate. First audio: ASR + LLM first token + first segment + TTS first chunk (~110 ms warm in M0); checked on every replayed response and by AICC's own watchdog in its harness. Gaps between deltas: one TTS request per segment, back to back, and paced deltas at the playback rate (D6). The real budget on the real backends is pending (G4). |
| One response at a time | `conversation_already_has_active_response`, not queued. |
| Barge-in | the server cancels: `speech_started`, then the terminal events and `response.done{cancelled, turn_detected}`, in one step; the reply follows `speech_stopped`; audio held back by pacing is discarded. Speech over the tail of a finished answer gets `speech_started`, AICC's flush and truncate, and a reply. The echo of our own speech is cancelled in the service, and gated while AICC is modelled as playing (D17, §11.13); AICC's own 800 ms guard stays as it is. |
| Client edits | `response.cancel` without an id, truncate with played milliseconds (clamped, D18; also after `response.done`), a truncate to 0 then the delete of an unheard answer, delete of calls, user `input_text` notes, `function_call_output` with a `hint`: all replayed. |
| Tools | flat function tools offered to the LLM; calls delivered complete, each its own item, with a minted `call_id`; a tool-only response done as soon as the LLM is (its 3 s deadline checked on every replay); the result, with its `hint`, replayed right after its call, the wait line merged into the call's message (§11.14). `tool_choice` stays `auto` (AICC's gateway profile sends none). |
| Lines | "Say exactly this…" lines are spoken verbatim, without the LLM (D16): on an empty conversation, mid-call, and as the wait line while a call is unanswered; their latency is the TTS's first chunk. |
| Transcripts | one `conversation.item.input_audio_transcription.completed` per utterance, without being asked (D15). |
| Errors | only the codes of §9, and only where required (D18); AICC never meets `cannot_update_voice`. |
| Audio | 20 ms appends at any time, also while the bot speaks (960 bytes of PCM, or 160 of G.711); output deltas of whole samples, at most 100 ms, at most 10 s ahead of real time (D6; AICC queues at most 30 s). |
| Human-call transcription (issue #2) | `?intent=transcription` and AICC's `session.update` (`type: "transcription"`, `turn_detection: null`, `transcription` with `language` and `prompt`) are accepted; its continuous appends, silence included, and its commit after each utterance give `committed`, `item.added`, `…transcription.completed` (with `language`) and `item.done`; the session never meets `response.create` (§3.1). |

Trace by trace (what each replay checks beyond "no error"):

| Trace | Checked |
|---|---|
| `call_gateway` | tools and `server_vad` accepted; a bare opening `response.create` on an empty conversation completes; caller silence creates no turn; truncate of the greeting; instructions-only update; keypress note; a scripted tool call answered by `function_call_output` (the LLM history replays it with the provider's call id); the closing line spoken verbatim, without the LLM. Since M5 also through the real LLM provider on a fake `mlx_lm.server` streaming the call as the real one does: the session's tools in every request, text then the call delivered, and the final request's messages (the keypress note, the call with its text, the result with its hint) rendered by the real Qwen3.6 template |
| `opening_gateway_with_voice` | the flow voice accepted and the effective voice (`vivian`) echoed; the English opening line sent to the TTS as `vivian`'s English speaker, `aiden` (M5, S3); the opening line spoken verbatim (no LLM request) |
| `waitline_gateway` | the wait line, asked for while the first call is unanswered, spoken verbatim: its turn makes no second call, so the trace's delete of one is not sent (AICC deletes only calls a line's turn made); the first call answered; the owed `response.create` completes and sees the line in its history. Since M5 also through the real LLM provider: a tool-only opening within the deadline, and the owed request's messages are cue, the call with the wait line merged in, and its result, accepted by the real template |
| `unheard_keypress_gateway` | an unheard answer deleted; a system note; the keypress `response.cancel` of a reply in progress (started by the test: the trace's reply came from caller speech); the truncate clamped to the audio sent; the keypress note and the owed response |
| `dropped_line_retry_gateway` | the line, the direction re-sent as a user item, and the retry all complete; both lines spoken verbatim |
| barge-in (M4, from `call_gateway`'s session) | AICC simulated in real time (20 ms appends, its playout and barge-in) with the real Silero VAD and canceller: the caller talks over a long line → `speech_started` → `response.done{cancelled}` in one step → AICC's truncate (its played frames) accepted → a reply after `speech_stopped`; and over the tail of a finished line → `speech_started`, truncate, reply |
| G.711 (M7) | every trace above, the barge-in and keypress replays included, again in AICC's G.711 passthrough shape: `{"type": "audio/pcmu"}` both ways with no rate, 160-byte appends; every response in μ-law, deltas of at most 800 bytes, the same checks passing |
| keypress (M4, from `unheard_keypress_gateway`) | `response.cancel` of an answer nobody heard, a truncate to 0, its delete after `response.done`, the keypress note and the owed reply, with no `error`; the next request has no trace of the answer |
