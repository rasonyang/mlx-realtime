"""``MlxLmChatLLM``: the LLM provider for ``mlx_lm.server`` (ARCHITECTURE §5.2, §5.3).

One streamed ``POST /v1/chat/completions`` per response. The SSE stream is read under the
request's deadlines: the first token, then the idle gap between tokens, and the total.
``: keepalive`` comments during prefill are not tokens and extend nothing. Closing the
response is the cancel signal: the server stops at its next token once a write fails.

LLM hardening (M0 S1/S2, G2):

- **A dropped call** (``finish_reason: tool_calls`` with no call: the server's parser rejected
  the call) before any visible text is retried once, in the same stream, so the consumer sees
  one answer. The retry uses the sampling of requests without tools (at temperature 0 the same
  malformed call would come back) and what is left of the first request's total deadline; it is
  cancelled like any request. After visible text the call is not retried: the response fails as
  before (``bad_response``), and the log says so.
- **A reply made without tools** passes through ``chat_wire.PseudoCallFilter``, so a JSON
  "call" in its text never reaches the TTS.
- **Prefix warming.** ``warm_prefix`` sends the one-token request of
  ``chat_wire.warmup_messages`` (the session calls it when its instructions or tools change). A
  request whose system message and tools match a warm-up still in flight waits for that warm-up
  first, at most the first-token deadline: racing it would prefill the same prompt twice (M0 S2).
  Warm-ups never pile up on the server, which cannot drop a request it is prefilling: one per
  prefix at a time, at most ``max_warmups_in_flight`` in all, and none once a real request for
  the prefix has gone out (a burst of new calls with the same instructions warms them once). A
  warm-up that has been sent runs to its end even when its session no longer wants it: the
  server does the work anyway, and only so do the rules see it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections import OrderedDict
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import cast

import httpx2

from mlx_realtime.config import MlxLmChatConfig
from mlx_realtime.providers.base import (
    Delta,
    Message,
    ProviderError,
    Readiness,
    TextDelta,
    ToolCall,
    ToolChoice,
    ToolDef,
)
from mlx_realtime.providers.chat_wire import (
    ChunkReader,
    Json,
    PseudoCallFilter,
    Sampling,
    Shaping,
    prefix_key,
    request_body,
    warmup_messages,
)
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
_REQUESTS_KEPT = 256  # prefixes whose last request time is remembered

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _Attempt:
    """One request of a response's stream: what its reader saw, and its absolute total deadline
    (a retry runs under the first attempt's)."""

    reader: ChunkReader = field(default_factory=ChunkReader)
    total_at: float | None = None


class MlxLmChatLLM:
    def __init__(
        self,
        config: MlxLmChatConfig,
        *,
        warmup_instructions: str = "",
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = config
        self._sampling = Sampling(
            config.max_completion_tokens,
            config.temperature,
            config.top_p,
            config.tool_temperature,
            config.tool_top_p,
        )
        self._shaping = Shaping(
            cue=config.empty_conversation_cue,
            system_note=config.system_note,
            interlude=config.tool_interlude,
            unanswered=config.unanswered_calls,
            pending_output=config.pending_tool_output,
        )
        self._deadlines = Deadlines.of(config.timeouts)
        self._warmup_deadlines = Deadlines.warmup(config.timeouts)
        self._slots = Slots(config.max_in_flight, config.queue_timeout_ms / 1000)
        self._warmup_instructions = warmup_instructions
        self._transport = transport
        self._http: httpx2.AsyncClient | None = None
        self._warming: dict[str, asyncio.Future[None]] = {}  # warm-ups in flight, by prefix_key
        self._warmup_slots = asyncio.Semaphore(config.max_warmups_in_flight)
        self._warmups: set[asyncio.Task[None]] = set()  # sent: they run to their end
        self._requested: OrderedDict[str, float] = OrderedDict()  # prefix_key: last request sent

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDef],
        *,
        tool_choice: ToolChoice = "auto",
        max_output_tokens: int | None = None,
    ) -> AsyncGenerator[Delta]:
        body = self._body(messages, tools, tool_choice, max_output_tokens)
        return self._stream(body)

    async def warm_prefix(
        self, messages: Sequence[Message], tools: Sequence[ToolDef], *, tool_choice: ToolChoice
    ) -> None:
        """Put the next request's prefix in the server's prompt cache (the module docstring).

        At most ``max_warmups_in_flight`` warm-ups run at once, the rest queue. A warm-up is
        skipped when one for the same prefix is in flight (it puts the shared part in the
        cache), and when a real request for the prefix was sent while it queued (that request
        puts it there). Cancelling the caller drops a queued warm-up; one already sent runs to
        its end, because the server finishes a prefill it has started (it notices a closed
        connection only at its next write), and so it stays in flight for the rules above. A
        failed warm-up is logged, never raised."""
        if not self._config.warm_on_update:
            return
        body = self._body(messages, tools, tool_choice, 1)
        body["messages"] = warmup_messages(cast(list[Json], body["messages"]), self._shaping.cue)
        key = prefix_key(body)
        if key in self._warming:
            return
        loop = asyncio.get_running_loop()
        queued = loop.time()
        await self._warmup_slots.acquire()
        if key in self._warming or self._requested.get(key, -math.inf) >= queued:
            self._warmup_slots.release()
            return
        self._warming[key] = loop.create_future()
        warmup = asyncio.create_task(self._warm(key, body))
        self._warmups.add(warmup)
        warmup.add_done_callback(self._warmups.discard)
        await asyncio.shield(warmup)

    async def _warm(self, key: str, body: Json) -> None:
        """A warm-up request, to its end; then its slot and its prefix are free."""
        loop = asyncio.get_running_loop()
        started = loop.time()
        attempt = _Attempt()
        try:
            async with aclosing(self._complete(body, self._deadlines, attempt)) as stream:
                async for _ in stream:
                    pass
        except ProviderError as error:
            logger.warning("prefix warm-up failed", extra={"error": str(error)})
            return
        finally:
            self._warming.pop(key).set_result(None)
            self._warmup_slots.release()
        usage = attempt.reader.usage
        logger.info(
            "prefix warmed",
            extra={
                "warmup_ms": round((loop.time() - started) * 1000),
                "prompt_tokens": None if usage is None else usage.input_tokens,
                "cached_tokens": None if usage is None else usage.cached_tokens,
            },
        )

    async def warmup(self) -> None:
        """A one-token completion with the default instructions; it also fills the prompt cache."""
        instructions = self._warmup_instructions
        messages = [Message("system", instructions)] if instructions else []
        body = self._body(messages, (), "none", 1)
        async with aclosing(self._complete(body, self._warmup_deadlines, _Attempt())) as stream:
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
        for warmup in list(self._warmups):
            warmup.cancel()
        await asyncio.gather(*self._warmups, return_exceptions=True)
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
            shaping=self._shaping,
        )

    async def _stream(self, body: Json) -> AsyncGenerator[Delta]:
        """One answer: the request, a retry of a dropped call, the pseudo-call filter."""
        await self._after_warmups(body)
        self._sent(prefix_key(body))
        texts = None if "tools" in body else PseudoCallFilter()
        visible = False  # text the consumer will speak or show has been yielded
        attempt = _Attempt()
        request = body
        while True:
            async with aclosing(self._complete(request, self._deadlines, attempt)) as deltas:
                async for delta in deltas:
                    if isinstance(delta, TextDelta) and texts is not None:
                        delta = TextDelta(texts.push(delta.text))
                        if not delta.text:
                            continue
                    if not isinstance(delta, TextDelta) or delta.text.strip():
                        visible = True
                    yield delta
            if not attempt.reader.dropped_call:
                break
            if visible or request is not body or not self._config.retry_dropped_call:
                logger.warning(
                    "the server dropped a malformed tool call; not retried",
                    extra={"after_text": visible, "retried": request is not body},
                )
                break  # reader.finish() fails the response
            logger.warning("the server dropped a malformed tool call; retrying once")
            request = {**body, "temperature": self._sampling.temperature}
            request["top_p"] = self._sampling.top_p
            attempt = _Attempt(total_at=attempt.total_at)
        if texts is not None:
            if rest := texts.flush():
                yield TextDelta(rest)
            if texts.dropped:
                logger.warning(
                    "pseudo-call removed from a reply made without tools",
                    extra={"count": len(texts.dropped), "chars": sum(map(len, texts.dropped))},
                )
        yield attempt.reader.finish()

    async def _after_warmups(self, body: Json) -> None:
        """Wait for the warm-up of this request's prefix if one is in flight, at most the
        first-token deadline."""
        pending = self._warming.get(prefix_key(body)) if self._warming else None
        if pending is None:
            return
        loop = asyncio.get_running_loop()
        started = loop.time()
        await asyncio.wait({pending}, timeout=self._deadlines.first_item_s)
        logger.info(
            "waited for a prefix warm-up",
            extra={"waited_ms": round((loop.time() - started) * 1000)},
        )

    def _sent(self, key: str) -> None:
        """A real request for this prefix goes out now (a queued warm-up of it is redundant)."""
        self._requested[key] = asyncio.get_running_loop().time()
        self._requested.move_to_end(key)
        while len(self._requested) > _REQUESTS_KEPT:
            self._requested.popitem(last=False)

    async def _complete(
        self, body: Json, deadlines: Deadlines, attempt: _Attempt
    ) -> AsyncGenerator[TextDelta | ToolCall]:
        """Text and calls until ``[DONE]``; ``attempt.reader`` keeps what ``finish`` needs."""
        async with self._slots.slot():
            clock = RequestClock(deadlines, total_at=attempt.total_at)
            attempt.total_at = clock.total_at
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
                reader = attempt.reader
                async with aclosing(sse_data(response.aiter_bytes())) as events:
                    while (data := await _next(events, clock)) != "[DONE]":
                        for delta in reader.read(data):
                            clock.item()
                            yield delta
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
