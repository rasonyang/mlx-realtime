# mlx-realtime

A local voice service for Apple Silicon that speaks a documented subset of the
[OpenAI Realtime API (GA)](https://developers.openai.com/api/reference/resources/realtime) over
WebSocket at `/v1/realtime`. Inside, it is a cascade: VAD → turn detection → ASR → LLM → TTS, for
Chinese, English and code-switched speech. The official `openai` Python SDK connects by changing only
`base_url`.

**Status: milestone M2.** Speech turns are detected with Silero VAD and Smart Turn, transcribed by
Qwen3-ASR (`mlx_audio.server`) and answered by the LLM (`mlx_lm.server`); the TTS is still a
sine-wave mock until M3, so use `output_modalities: ["text"]` for real answers. See
[`docs/PROGRESS.md`](docs/PROGRESS.md).

| Document | What it holds |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | the design and the decisions D1–D19 |
| [`docs/PROTOCOL.md`](docs/PROTOCOL.md) | what the gateway supports, event by event and field by field |
| [`docs/PROGRESS.md`](docs/PROGRESS.md) | what each milestone delivered, and what is left |
| [`CLAUDE.md`](CLAUDE.md) | the repository rules (uv only) |

## Requirements

- macOS on Apple Silicon. The model backends need about 25 GB of unified memory; the gateway alone
  (`make dev-mock`) runs anywhere Python 3.12 does.
- [uv](https://docs.astral.sh/uv/) 0.13.5 or later. uv installs Python 3.12 and every dependency from
  the `uv.lock` files; nothing is installed with pip.

## Run

Three processes serve the models, each in its own uv project (`backends/llm`, `backends/audio`), and
the gateway talks to them over HTTP on 127.0.0.1 (ports 8101–8103, `config.yaml` → `backends`).

```sh
make models-mlx     # once: the MLX models pinned in config.yaml (~25 GB), through each backend's hf
make dev            # the ONNX models (mlxrt models), then the backends and the gateway (mlxrt dev)
make dev-mock       # or: the gateway alone with mock providers (config.mock.yaml)
```

`mlxrt dev` prefixes every line with the process name, stops everything on Ctrl-C, and stops
everything when any process exits. The backends run with `HF_HUB_OFFLINE=1`: models are fetched ahead
of time, never during a call. The gateway admits sessions once every backend has answered a warmup
request (the first one also loads the model, so this can take a minute).

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
24 kHz (a sine tone until M3) alongside `response.output_audio_transcript.delta`.

Speak to it: append `audio/pcm` at 24 kHz with `input_audio_buffer.append` (20 ms per event is what
telephony clients send). With the default `server_vad`, the gateway detects the turn, commits it,
transcribes it (`conversation.item.input_audio_transcription.completed`) and answers it.

## Configure

All configuration lives in [`config.yaml`](config.yaml) ([`config.mock.yaml`](config.mock.yaml) is the
mock setup the tests use). Any key can be overridden with an environment variable
`MLXRT__<SECTION>__<KEY>`, whose value is parsed as YAML:

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
make test        # unit and integration tests (fake backends, the real VAD models)
make test-e2e    # the official SDK against the gateway (marker e2e)
make test-backend  # the real backends (marker backend): run `make dev` first
```

`uv run --locked pytest` runs everything except the `backend` tests, which are skipped unless
selected with `-m backend`. Golden traces live in `tests/golden/`; after an intended
protocol change, regenerate them with `uv run --locked pytest tests/integration/test_golden.py
--update-golden` and review the diff.

## Layout

```
src/mlx_realtime/
  server/      Starlette app, WebSocket transport (reader, single writer), admission, readiness
  protocol/    GA wire layer: client events → commands, facts → server events, ids, errors
  session/     the session actor: conversation, input buffer, turn manager, responses, pipeline
  providers/   ASR, LLM and TTS interfaces, registry, the mlx_lm and mlx_audio HTTP providers, mocks
  audio/       resampling and wire formats      text/  segmenter, normalizer, language tag
  vad/         Silero VAD, Smart Turn, Whisper features (numpy), the speech gate, CPU executors
  cli.py · supervisor.py (mlxrt dev) · fetch.py (mlxrt models) · hub.py (the HF cache layout)
backends/llm · backends/audio      the model processes, each its own uv project
models/MANIFEST.json               the ONNX models: source, revision, license, sha256
tests/unit · tests/integration · tests/golden · tests/e2e · tests/backend · tests/fakes
```
