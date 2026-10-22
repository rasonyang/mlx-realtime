# AICC client traces (gateway profile)

Client → server frames that ai-native-callcenter's Realtime client (`internal/provider`, repo commit
8fabadc) sends when configured with its `gateway` profile (GA style, 24 kHz PCM both ways,
`server_vad` with `silence_duration_ms: 500`). Captured on 2026-10-22 by running the real client
against AICC's own fake server in a scratch copy of that repository; the repository itself was not
modified. Instructions and tools are AICC test fixtures, not customer data.

| File | Scenario |
|---|---|
| `call_gateway.jsonl` | Covers, in order: `session.update` with tools; a bare opening `response.create`; caller audio; a truncate; an instructions-only `session.update`; a keypress note and `response.create`; a tool result and `response.create`; a "Say exactly" line. |
| `opening_gateway_with_voice.jsonl` | An opening line sent as a per-response `instructions` override, with a flow-chosen voice. |
| `waitline_gateway.jsonl` | A wait line while an HTTP tool runs. Then the `function_call_output`, the delete of the call made in the line's turn, and the owed bare `response.create`. |
| `unheard_keypress_gateway.jsonl` | A tool result's answer is deleted unheard. Then a "(System note …)" user item, a keypress `response.cancel`, a truncate, a keypress note, and `response.create`. |
| `dropped_line_retry_gateway.jsonl` | `response.created` never arrives for a line. After 3 s the client retries, with the direction first added as a user item. |

These are M0/M1 reference inputs for protocol tests; see `docs/ARCHITECTURE.md` §2.
