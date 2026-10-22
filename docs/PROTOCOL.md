# mlx-realtime — Realtime GA protocol profile

> **Status: M2 (2026-10-23).** What the gateway accepts and emits on `/v1/realtime`, event by event and
> field by field, and in which milestone each unsupported part arrives. M2 made `server_vad` real
> (Silero VAD + Smart Turn) and the transcripts real (Qwen3-ASR); answers come from the LLM. Audio
> output is still a sine tone until M3.

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
| `?model=` | Echoed as `session.model`; never used for routing. Without it, `session_defaults.model` is echoed. |
| `OpenAI-Beta` header | Ignored: only the GA shape is served. |
| Frames | One JSON object per text frame. A binary frame or invalid JSON draws an `error` (`invalid_event`); the session stays open. |
| Frame size | At most `limits.client_max_message_bytes` (16 MiB; GA allows 15 MiB of base64 per append). A larger frame closes the connection with 1009. |
| First events | `session.created`, then `conversation.created`, before any client event is processed. |
| Admission | Before the first successful readiness: `error` `not_ready`, close 1013. Over `gateway.max_sessions`: `error` `session_limit_reached`, close 1013. Both after the upgrade, as the SDK expects. |
| `/healthz` | 200 while the event loop answers. No backend calls. |
| `/readyz` | 200 once every backend has been warmed and reports ready, else 503; per-backend detail in the body: `llm`, `asr`, `tts`, and `vad` (the CPU detectors) when Silero runs. |

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
| `input_audio_buffer.append` | `InputAudioBufferAppendEvent` | **Supported** | `audio`: base64 of `audio/pcm` 24 kHz mono s16le, whole samples (an odd byte count is `invalid_value`, `param: "audio"`). No confirmation event. Never dropped: an inbox over `limits.inbox_max_ms` of queued audio is fatal. |
| `input_audio_buffer.commit` | `InputAudioBufferCommitEvent` | **Supported** | Commits the whole buffer, the resampler's tail included → `input_audio_buffer.committed`, `conversation.item.added`. Creates no response. Allowed in `server_vad` mode too (a speech turn in progress ends without `speech_stopped`). An empty buffer: `invalid_event`. |
| `input_audio_buffer.clear` | `InputAudioBufferClearEvent` | **Supported** | → `input_audio_buffer.cleared`; a speech turn in progress is dropped. |
| `conversation.item.create` | `ConversationItemCreateEvent` | **Partial** | Messages and `function_call_output` (§5); `previous_item_id` absent (append), `"root"` (first) or an existing id. → `conversation.item.added`, `conversation.item.done`. Client-created `function_call` items: Unsupported. |
| `conversation.item.truncate` | `ConversationItemTruncateEvent` | **Supported** | Assistant messages with audio, `content_index: 0`. An `audio_end_ms` beyond the audio sent is clamped to it (D18). → `conversation.item.truncated` (§11.3). |
| `conversation.item.delete` | `ConversationItemDeleteEvent` | **Supported** | Any item, including one a response is still producing (§11.4). → `conversation.item.deleted`. Unknown id: `item_not_found`, `param: "item_id"` (D18). |
| `conversation.item.retrieve` | `ConversationItemRetrieveEvent` | **Unsupported** | `invalid_event`, `param: "type"`: user audio is not retained in the wire format, and an item without its audio would silently diverge from GA. |
| `response.create` | `ResponseCreateEvent` | **Partial** | §4. While a response is active: `conversation_already_has_active_response` (not queued). `response.created` is immediate (D7). |
| `response.cancel` | `ResponseCancelEvent` | **Supported** | With or without `response_id`. Nothing active, or another id: `response_cancel_not_active` (D18). → the terminal events, `response.done{cancelled, client_cancelled}`. |
| `output_audio_buffer.clear` | `OutputAudioBufferClearEvent` | **Never** | WebRTC/SIP only. `invalid_event`, `param: "type"`; over WebSocket, send `response.cancel`. |
| `transcription_session.update` | — | **Unsupported** | Transcription sessions are not served: `invalid_event`, `param: "type"`. |
| any other `type` | — | — | `invalid_value`, `param: "type"` (as GA's `scooby.dooby.doo` example). A missing `type`: `invalid_event`. |

## 3. Session object

`session.update.session` is a `RealtimeSessionCreateRequest` (the `RealtimeTranscriptionSessionCreateRequest`
alternative is rejected). `session.created` and `session.updated` echo the full effective session in
the same shape, plus `id` (`sess_…`) and `object: "realtime.session"`, which the SDK reads as extra
fields. `config.yaml`'s `session_defaults` uses this shape and the same validation.

**Merge rules.** Only fields present change. Nested objects merge field by field. `null` clears a
nullable field (`transcription`, `turn_detection`). A nested object set after being `null` starts from
the configured session defaults, so its absent fields take their configured values.

| Field (SDK type) | Status | Accepted | Echo | Rejection `param` |
|---|---|---|---|---|
| `type` | **Supported** | `"realtime"`, required in `session.update` | `"realtime"` | `session.type` (`"transcription"` included) |
| `model` | **Supported** | any string; echoed only | last value set, else `?model=`, else the default | `session.model` |
| `instructions` | **Supported** | any string (`""` clears) | effective | `session.instructions` |
| `output_modalities` | **Supported** | `["audio"]` or `["text"]` (D12) | effective | `session.output_modalities` |
| `audio.input.format` (`RealtimeAudioFormats`) | **Partial** | `{type: "audio/pcm", rate: 24000}`; both keys optional. `audio/pcmu`, `audio/pcma`: **Unsupported (M7)** | `{type: "audio/pcm", rate: 24000}` | `session.audio.input.format.type` / `.rate` |
| `audio.input.transcription` (`AudioTranscription`) | **Partial** | object or `null`; `model` (echoed only: the ASR is Qwen3-ASR, `providers.asr.model`), `language` (an ISO 639 code such as `zh` or `en`, passed to the ASR as its language name; absent: the ASR detects it), `prompt` (echoed only). `delay`, `keywords`, `languages`: Unsupported | effective object (unset keys omitted), or `null`. On by default (D15) | `session.audio.input.transcription.<key>` |
| `audio.input.turn_detection` (`RealtimeAudioInputTurnDetection`) | **Supported** | `null` (manual turns), or `ServerVad`: Silero VAD v6.2.3 plus Smart Turn v3.2 (D4, §11.9). `threshold` [0, 1] (speech enters at it, leaves below it − 0.15), `prefix_padding_ms` ≥ 0, `silence_duration_ms` ≥ 0 (the silence before an end-of-turn evaluation; session default 200), `create_response`, `interrupt_response`; `idle_timeout_ms` only `null`. `SemanticVad`: **Unsupported** | every field filled, `idle_timeout_ms: null`; or `null` | `…turn_detection.type` (`semantic_vad`), `….idle_timeout_ms` (non-null), `….<field>` |
| `audio.input.noise_reduction` (`NoiseReduction`) | **Unsupported** | `null` only (echo cancellation is internal, M4) | `null` | `session.audio.input.noise_reduction` |
| `audio.output.format` | **Supported** | `audio/pcm` at 24000 only | as input | `session.audio.output.format.type` / `.rate` |
| `audio.output.voice` (`Voice`) | **Partial** | any string, never rejected (D19). M1 echoes it as given; M3 maps an unknown name to the default voice, logs a warning and echoes the effective voice. `VoiceID` objects: Unsupported | effective string | `session.audio.output.voice` (object) |
| `audio.output.speed` | **Partial** | `1.0` only (D14: the TTS has no speed control) | `1.0` | `session.audio.output.speed` |
| `tools` (`RealtimeToolsConfig`) | **Supported** | `RealtimeFunctionTool`: `type: "function"` and `name` required, `description`, `parameters` (an object, passed through). Duplicate names rejected. `Mcp`: Unsupported | effective list; `[]` when unset | `session.tools[i].type` / `.name` / `.parameters` / `.<key>` |
| `tool_choice` (`RealtimeToolChoiceConfig`) | **Partial** | `"auto"`, `"none"`. `"required"`, `ToolChoiceFunction`, `ToolChoiceMcp`: **Unsupported (M5)** | effective; `"auto"` by default | `session.tool_choice` |
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

## 4. `response.create` parameters

`response.create.response` is a `RealtimeResponseCreateParams`; every field overrides the session for
this response only. `response: null` or absent means the session's settings.

| Field | Status | Notes | Rejection `param` |
|---|---|---|---|
| `instructions` | **Supported** | replaces the session instructions for this response (GA). Verbatim lines (D16) arrive in M3: until then a "Say exactly this…" line goes through the LLM | `response.instructions` |
| `output_modalities` | **Supported** | `["audio"]` or `["text"]` | `response.output_modalities` |
| `max_output_tokens` | **Supported** | `"inf"` or 1–4096; reaching it ends the response `incomplete` (`max_output_tokens`) | `response.max_output_tokens` |
| `tool_choice` | **Partial** | `"auto"`, `"none"`; the rest **Unsupported (M5)** | `response.tool_choice` |
| `audio.output.voice` (`OutputVoice`) | **Partial** | as the session voice (D19); `OutputVoiceID`: Unsupported | `response.audio.output.voice` |
| `audio.output.format` | **Supported** | `audio/pcm` at 24000 only | `response.audio.output.format.type` |
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
| `input_audio_buffer.speech_started` | `InputAudioBufferSpeechStartedEvent` | **Supported** | `server_vad` only, once speech has lasted `turn.vad.min_speech_ms` (150); `audio_start_ms` includes `prefix_padding_ms`; `item_id` of the item the commit will create. One per turn: speech that resumes while the turn is held continues it (§11.9) |
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
| `response.output_audio.delta` / `.done` | `ResponseAudioDeltaEvent` / `ResponseAudioDoneEvent` | **Supported** | base64 `audio/pcm` 24 kHz, whole samples, at most `output.max_delta_ms` (100 ms) per delta. M1 audio is a sine tone |
| `response.output_audio_transcript.delta` / `.done` | `ResponseAudioTranscriptDeltaEvent` / `ResponseAudioTranscriptDoneEvent` | **Supported** | one delta per synthesized segment, sent just before the segment's first audio; `.done` carries the text whose deltas reached the client |
| `response.output_text.delta` / `.done` | `ResponseTextDeltaEvent` / `ResponseTextDoneEvent` | **Supported** | text modality (D12): one delta per LLM delta; TTS is never opened |
| `response.function_call_arguments.delta` / `.done` | `ResponseFunctionCallArgumentsDeltaEvent` / `ResponseFunctionCallArgumentsDoneEvent` | **Partial** | complete calls only, the arguments in one delta (§11.5). Calls the LLM server parsed are delivered since M2 (the provider passes the session tools when `tool_choice` is `auto`); real tool calling, with history replay checked, is M5 |
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
  reported none (a cancelled or empty response).

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

**Audio response** (`audio_turn.jsonl`); per segment, its transcript delta immediately precedes its
first audio delta:
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
While the response still awaits its transcript (no output item yet):
`speech_started → response.done{cancelled}`.

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
| `invalid_event` | invalid JSON, a binary frame, a missing `type`, an unsupported event type, an empty buffer on commit | `"type"` for event types, else `null` |
| `invalid_value` | a field missing, mistyped, out of range, unsupported or unknown; an unknown `previous_item_id` or `call_id`; a duplicate item id; a truncate of something other than an assistant audio message, or with `content_index` ≠ 0 | the dotted path: `session.audio.input.turn_detection.type`, `item.content[0].type`, `response.metadata.k`, … |
| `conversation_already_has_active_response` | `response.create` while a response is active (D18) | `null` |
| `response_cancel_not_active` | `response.cancel` with no active response, or naming another one (D18) | `null`, or `"response_id"` |
| `item_not_found` | `conversation.item.delete` or `.truncate` of an unknown item (D18) | `"item_id"` |

**Server errors** (`type: "server_error"`):

| Code | When | Effect |
|---|---|---|
| `not_ready` | connection before the backends were first ready | close 1013 |
| `session_limit_reached` | over `gateway.max_sessions` | close 1013 |
| `backend_unavailable` / `backend_timeout` / `backend_busy` / `provider_error` | a provider raised `ProviderError` of kind `unavailable` / `timeout` / `overloaded` / `bad_response` or `internal` | the response ends `failed`; the session continues |
| `input_queue_overflow` | more than `limits.inbox_max_ms` of audio, or `limits.inbox_max_commands` events, queued | fatal: close 1011 |
| `input_audio_buffer_overflow` | the input buffer exceeds `limits.input_audio_buffer_max_ms` | fatal: close 1011 |
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
   discarded unsent (§6.5 step 4); the writer counts what it handed to the socket.
3. **Truncation (§6.6).** `audio_end_ms` is clamped to the audio sent for the item. The item's text for
   later LLM requests becomes every segment that ended by then plus a proportional prefix of the one
   playing (characters, snapped back to a word start in Latin text). The item is truncated in place,
   even after `response.done`. Without a client truncate, the item keeps the text sent (the D5
   fallback arrives in M4).
4. **Deleting an item a response is producing** is accepted: the item leaves the conversation, the
   response's own events continue, and its closing `conversation.item.done` is skipped.
5. **Function calls (§6.7).** A call is delivered when complete: arguments in one delta, then `.done`.
   A response cancelled before a call is complete delivers no call; a delivered call is never retracted.
   `.done` therefore means "delivered", not "valid JSON" (the gateway never parses arguments).
6. **`audio_end_ms`** of `speech_stopped` is the acoustic end of speech (ARCHITECTURE §6.2), not GA's
   end of the silence window; the committed audio ends there. With Smart Turn deciding, the commit
   comes some time after it; logs record that delay as the turn's `turn_confirmed` stage.
7. **Input timeline.** `audio_start_ms` and `audio_end_ms` count milliseconds of audio since the
   session's first append (GA). Audio is held at 16 kHz internally; the resampler's tail is flushed at
   a manual commit, so no trailing audio is lost.
8. **Output flow control.** Audio is produced as fast as the backends allow, while the writer keeps
   up: producers wait whenever half of `outbound_queue_events` are queued. The real-time lead cap
   (D6) arrives in M3.
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

## 12. AICC compatibility

ai-native-callcenter attaches through its existing `gateway` profile with no change (D11). Its five
client traces in `spikes/aicc/traces/` are replayed in `tests/integration/test_aicc_replay.py`: every
frame is sent as captured, AICC's fake-server ids are swapped for this server's ids, and no frame draws an
`error`. AICC's requirements (ARCHITECTURE §2) at M2:

| Area | M2 |
|---|---|
| Handshake | `session.updated` answers the first `session.update` at once; every AICC `session.update` (tools, `server_vad` with only `silence_duration_ms: 500`, flow voice) is accepted. |
| Session shape | GA, `audio/pcm` 24 kHz both ways; flat function tools; a flow-chosen voice is accepted and echoed (D19; mapping in M3); instructions-only updates merge. |
| Turn-taking | `server_vad` is Silero VAD + Smart Turn (§11.9): AICC's `silence_duration_ms: 500` is the silence before Smart Turn decides. One response per commit, at the commit (D7); an empty transcript completes it empty. |
| Deadlines | `response.created` is immediate. The ASR and LLM are real since M2; the first audio delta is still the mock TTS's, so the real budget (ASR + LLM first token + first segment + TTS first chunk ≤ 3 s, G4) is measured from M3. |
| One response at a time | `conversation_already_has_active_response`, not queued. |
| Barge-in | the server cancels: `speech_started`, then the terminal events and `response.done{cancelled, turn_detected}`; the reply follows `speech_stopped`. Echo cancellation (D17) is M4. |
| Client edits | `response.cancel` without an id, truncate with played milliseconds (clamped, D18), delete of unheard answers and of calls, user `input_text` notes, `function_call_output` with a `hint`: all replayed. |
| Lines | `response.create` with "Say exactly this…" instructions, also on an empty conversation, is accepted; until M3 the line goes through the LLM, verbatim playback (D16) is M3. |
| Transcripts | one `conversation.item.input_audio_transcription.completed` per utterance, without being asked (D15). |
| Errors | only the codes of §9, and only where required (D18). |
| Audio | 20 ms appends at any time, also while the bot speaks; output deltas of whole samples, at most 100 ms, faster than real time. The 10 s lead cap (D6) arrives in M3. |

Trace by trace (what each replay checks beyond "no error"):

| Trace | Checked |
|---|---|
| `call_gateway` | tools and `server_vad` accepted; a bare opening `response.create` on an empty conversation completes; caller silence creates no turn; truncate of the greeting; instructions-only update; keypress note; a scripted tool call answered by `function_call_output` (the LLM history replays it with the provider's call id); the closing line |
| `opening_gateway_with_voice` | the flow voice echoed and used by TTS; the per-response instructions reach the LLM as its system message |
| `waitline_gateway` | a wait line whose turn made a second call; the first call answered; the second call's item deleted; the owed `response.create` completes |
| `unheard_keypress_gateway` | an unheard answer deleted; a system note; the keypress `response.cancel` of a reply in progress (started by the test: the trace's reply came from caller speech); the truncate clamped to the audio sent; the keypress note and the owed response |
| `dropped_line_retry_gateway` | the line, the direction re-sent as a user item, and the retry all complete |
