# mlx-realtime

A local voice service for Apple Silicon that speaks a documented subset of the
[OpenAI Realtime API (GA)](https://developers.openai.com/api/reference/resources/realtime) over
WebSocket at `/v1/realtime`. Inside, it is a cascade: VAD → turn detection → ASR → LLM → TTS, for
Chinese, English and code-switched speech. The official `openai` Python SDK connects by changing only
`base_url`.

**Status: milestone M1.** The protocol skeleton runs end to end on mock providers: an LLM that echoes
the user, a sine-wave TTS and a scripted ASR. Real models arrive in M2 (ASR, LLM) and M3 (TTS); see
[`docs/PROGRESS.md`](docs/PROGRESS.md).

| Document | What it holds |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | the design and the decisions D1–D19 |
| [`docs/PROTOCOL.md`](docs/PROTOCOL.md) | what the gateway supports, event by event and field by field |
| [`docs/PROGRESS.md`](docs/PROGRESS.md) | what each milestone delivered, and what is left |
| [`CLAUDE.md`](CLAUDE.md) | the repository rules (uv only) |

## Requirements

- macOS on Apple Silicon (M1 itself runs anywhere Python 3.12 does).
- [uv](https://docs.astral.sh/uv/) 0.13.5 or later. uv installs Python 3.12 and every dependency from
  `uv.lock`; nothing is installed with pip.

## Run

```sh
make dev            # uv sync --locked, then: uv run --locked mlxrt dev
```

The gateway listens on `127.0.0.1:8000` (`config.yaml`). Check it:

```sh
curl localhost:8000/healthz   # the event loop answers
curl localhost:8000/readyz    # every backend is warm; 503 until then
```

Talk to it with the official SDK:

```python
import asyncio
from openai import AsyncOpenAI

async def main() -> None:
    client = AsyncOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
    async with client.realtime.connect(model="gpt-realtime") as connection:
        await connection.session.update(session={"type": "realtime", "output_modalities": ["text"]})
        await connection.conversation.item.create(
            item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "你好！"}]}
        )
        await connection.response.create()
        async for event in connection:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)
            elif event.type == "response.done":
                break

asyncio.run(main())
```

Leave out `output_modalities` to get audio: `response.output_audio.delta` events carry base64 PCM16 at
24 kHz (a sine tone in M1) alongside `response.output_audio_transcript.delta`.

## Configure

All configuration lives in [`config.yaml`](config.yaml). Any key can be overridden with an environment
variable `MLXRT__<SECTION>__<KEY>`, whose value is parsed as YAML:

```sh
MLXRT__GATEWAY__PORT=9000 MLXRT__GATEWAY__API_KEY=secret make dev
```

An unknown key or a bad value stops startup with the dotted path of the problem.
`session_defaults` is GA-shaped and is validated exactly like a `session.update`.
`uv run --locked mlxrt --config other.yaml serve` runs the gateway with another file.

## Develop

```sh
make fmt         # ruff: sort imports, format
make lint        # ruff check, ruff format --check
make typecheck   # pyright (strict on src/)
make test        # unit and integration tests
make test-e2e    # the official SDK against the gateway (marker e2e)
```

`uv run --locked pytest` runs everything. Golden traces live in `tests/golden/`; after an intended
protocol change, regenerate them with `uv run --locked pytest tests/integration/test_golden.py
--update-golden` and review the diff.

## Layout

```
src/mlx_realtime/
  server/      Starlette app, WebSocket transport (reader, single writer), admission, readiness
  protocol/    GA wire layer: client events → commands, facts → server events, ids, errors
  session/     the session actor: conversation, input buffer, turn manager, responses, pipeline
  providers/   ASR, LLM and TTS interfaces, registry, mocks
  audio/       resampling and wire formats      text/  segmenter, normalizer, language tag
  vad/         speech and end-of-turn detector interfaces, scripted detectors
tests/unit · tests/integration · tests/golden · tests/e2e
```
