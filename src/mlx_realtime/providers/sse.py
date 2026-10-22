"""Server-sent events: the ``data`` of each event in a byte stream.

Comment lines (``: keepalive 512/2048``, which mlx_lm.server writes during prefill) and fields
other than ``data`` are skipped; a blank line ends an event. As the SSE specification says, an
event the stream ends inside of is never dispatched. Lines end with ``\\n`` or ``\\r\\n``.
Splitting happens on bytes, before decoding, so no character inside the data can split a
line.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterable


async def sse_data(chunks: AsyncIterable[bytes]) -> AsyncGenerator[str]:
    pending = bytearray()
    data: list[str] = []
    async for chunk in chunks:
        pending += chunk
        while (end := pending.find(b"\n")) >= 0:
            line = bytes(pending[:end]).rstrip(b"\r").decode("utf-8", errors="replace")
            del pending[: end + 1]
            if line:
                if not line.startswith(":"):
                    field, _, value = line.partition(":")
                    if field == "data":
                        data.append(value.removeprefix(" "))
            elif data:
                yield "\n".join(data)
                data = []
