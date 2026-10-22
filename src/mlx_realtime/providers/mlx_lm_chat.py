"""``MlxLmChatLLM``: the LLM provider for ``mlx_lm.server`` (ARCHITECTURE §5.2, §5.3).

One streamed ``POST /v1/chat/completions`` per response. The SSE stream is read under the
request's deadlines: the first token, then the idle gap between tokens, and the total.
``: keepalive`` comments during prefill are not tokens and extend nothing. Closing the
response is the cancel signal: the server stops at its next token once a write fails.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing

import httpx2

from mlx_realtime.config import MlxLmChatConfig
from mlx_realtime.providers.base import (
    Delta,
    Message,
    ProviderError,
    Readiness,
    ToolChoice,
    ToolDef,
)
from mlx_realtime.providers.chat_wire import ChunkReader, Json, Sampling, request_body
from mlx_realtime.providers.http import (
    Deadlines,
    RequestClock,
    Slots,
    new_client,
    status_error,
    transport_error,
)
from mlx_realtime.providers.sse import sse_data

COMPLETIONS = "/v1/chat/completions"
HEALTH = "/health"


class MlxLmChatLLM:
    def __init__(
        self,
        config: MlxLmChatConfig,
        *,
        warmup_instructions: str = "",
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = config
        self._sampling = Sampling(config.max_completion_tokens, config.temperature, config.top_p)
        self._deadlines = Deadlines.of(config.timeouts)
        self._warmup_deadlines = Deadlines.warmup(config.timeouts)
        self._slots = Slots(config.max_in_flight, config.queue_timeout_ms / 1000)
        self._warmup_instructions = warmup_instructions
        self._transport = transport
        self._http: httpx2.AsyncClient | None = None

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDef],
        *,
        tool_choice: ToolChoice = "auto",
        max_output_tokens: int | None = None,
    ) -> AsyncGenerator[Delta]:
        body = self._body(messages, tools, tool_choice, max_output_tokens)
        return self._complete(body, self._deadlines)

    async def warmup(self) -> None:
        """A one-token completion with the default instructions; it also fills the prompt cache."""
        instructions = self._warmup_instructions
        messages = [Message("system", instructions)] if instructions else []
        body = self._body(messages, (), "none", 1)
        async with aclosing(self._complete(body, self._warmup_deadlines)) as stream:
            async for _ in stream:
                pass

    async def ready(self) -> Readiness:
        """``/health`` answers 503 once the server's generation thread has died."""
        try:
            response = await self._client().get(HEALTH)
        except httpx2.HTTPError as error:
            return Readiness(ready=False, detail=str(transport_error(error)))
        if response.status_code != 200:
            return Readiness(ready=False, detail=f"{HEALTH} answered {response.status_code}")
        return Readiness(ready=True, detail="ok")

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()

    def _body(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDef],
        tool_choice: ToolChoice,
        max_output_tokens: int | None,
    ) -> Json:
        return request_body(
            messages,
            tools,
            tool_choice=tool_choice,
            max_output_tokens=max_output_tokens,
            sampling=self._sampling,
            model=self._config.model,
            cue=self._config.empty_conversation_cue,
            system_note=self._config.system_note,
        )

    async def _complete(self, body: Json, deadlines: Deadlines) -> AsyncGenerator[Delta]:
        async with self._slots.slot():
            clock = RequestClock(deadlines)
            client = self._client()
            content = json.dumps(body, ensure_ascii=False).encode()
            request = client.build_request(
                "POST", COMPLETIONS, content=content, headers={"content-type": "application/json"}
            )
            try:
                async with clock.waiting():
                    response = await client.send(request, stream=True)
            except httpx2.HTTPError as error:
                raise transport_error(error) from error
            try:
                if response.status_code != 200:
                    async with clock.waiting():
                        detail = await response.aread()
                    raise status_error(response.status_code, detail)
                reader = ChunkReader()
                async with aclosing(sse_data(response.aiter_bytes())) as events:
                    while (data := await _next(events, clock)) != "[DONE]":
                        for delta in reader.read(data):
                            clock.item()
                            yield delta
                yield reader.finish()
            except httpx2.HTTPError as error:
                raise transport_error(error) from error
            finally:
                await response.aclose()

    def _client(self) -> httpx2.AsyncClient:
        if self._http is None:
            self._http = new_client(
                self._config.base_url, self._deadlines.connect_s, self._transport
            )
        return self._http


async def _next(events: AsyncGenerator[str], clock: RequestClock) -> str:
    async with clock.waiting():
        data = await anext(events, None)
    if data is None:
        raise ProviderError("bad_response", "llm stream: it ended before [DONE]")
    return data
