"""S5, GPU contention between the three model processes, measured on the backends directly.

    uv run --locked --project spikes/realstack python spikes/realstack/contention.py OUT.json

TTS: time to first chunk (TTFC) and RTF of a ~5 s Chinese sentence (the gateway's request shape:
vivian, 0.16 s chunks, streamed PCM). ASR: latency of a 6 s FLEURS clip. LLM: TTFT and decode
tokens/s of a long answer. Each alone, then under load:

- TTS and ASR while the LLM decodes a long answer (1 stream, and 3 batched streams);
- TTS while the LLM prefills ~16k tokens;
- the LLM's TTFT and decode rate while the TTS streams back to back;
- ASR while the TTS streams.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
import uuid
import wave
from pathlib import Path

import httpx2

LLM = "http://127.0.0.1:8101"
ASR = "http://127.0.0.1:8102"
TTS = "http://127.0.0.1:8103"
TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
ASR_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
SENTENCE = "您好，您的维修单已经受理，预计三个工作日内完成，完成后我们会短信通知您。"
CLIP = Path(__file__).resolve().parents[2] / "spikes" / "data" / "asr" / "16k" / "zh_002.wav"
STORY = "讲一个很长很长的关于灯塔守护人的故事，至少两千字。"


def wav_bytes(path: Path) -> bytes:
    return path.read_bytes()


async def tts_once(client: httpx2.AsyncClient) -> dict[str, float]:
    body = {"model": TTS_MODEL, "input": SENTENCE, "voice": "vivian", "lang_code": "chinese",
            "response_format": "pcm", "stream": True, "streaming_interval": 0.16, "max_tokens": 400}
    start = time.perf_counter()
    first = None
    size = 0
    async with client.stream("POST", f"{TTS}/v1/audio/speech", json=body, timeout=120) as response:
        async for chunk in response.aiter_bytes():
            if chunk and first is None:
                first = time.perf_counter()
            size += len(chunk)
    end = time.perf_counter()
    audio_s = size / 48000
    return {"ttfc_ms": round(1000 * ((first or end) - start), 1), "rtf": round((end - start) / audio_s, 3),
            "audio_s": round(audio_s, 2)}


async def asr_once(client: httpx2.AsyncClient, data: bytes) -> dict[str, float]:
    start = time.perf_counter()
    response = await client.post(f"{ASR}/v1/audio/transcriptions",
                                 data={"model": ASR_MODEL, "response_format": "verbose_json", "language": "Chinese"},
                                 files={"file": ("a.wav", data, "audio/wav")}, timeout=60)
    response.raise_for_status()
    return {"asr_ms": round(1000 * (time.perf_counter() - start), 1)}


async def llm_once(client: httpx2.AsyncClient, prompt: str, max_tokens: int,
                   stop_after: asyncio.Event | None = None) -> dict[str, float]:
    body = {"messages": [{"role": "user", "content": prompt}], "stream": True,
            "max_completion_tokens": max_tokens, "temperature": 0.7, "top_p": 0.8,
            "chat_template_kwargs": {"enable_thinking": False}}
    start = time.perf_counter()
    first = None
    tokens = 0
    async with client.stream("POST", f"{LLM}/v1/chat/completions", json=body, timeout=600) as response:
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices") or []
            if choices and (choices[0].get("delta") or {}).get("content"):
                tokens += 1
                if first is None:
                    first = time.perf_counter()
            if stop_after is not None and stop_after.is_set():
                break
    end = time.perf_counter()
    decode_s = end - (first or end)
    return {"ttft_ms": round(1000 * ((first or end) - start), 1), "tokens": tokens,
            "tok_s": round((tokens - 1) / decode_s, 1) if decode_s > 0 and tokens > 1 else None}


def summary(rows: list[dict[str, float]], key: str) -> dict[str, float | int | None]:
    values = sorted(r[key] for r in rows if r.get(key) is not None)
    if not values:
        return {"n": 0}
    p95 = values[max(0, min(len(values) - 1, round(0.95 * len(values) + 0.5) - 1))]
    return {"n": len(values), "p50": round(statistics.median(values), 3), "p95": p95, "max": values[-1]}


async def background_llm(client: httpx2.AsyncClient, streams: int, stop: asyncio.Event) -> None:
    async def one() -> None:
        while not stop.is_set():
            await llm_once(client, STORY + f" ({uuid.uuid4().hex[:6]})", 1500, stop)

    await asyncio.gather(*(one() for _ in range(streams)))


async def background_tts(client: httpx2.AsyncClient, stop: asyncio.Event) -> None:
    while not stop.is_set():
        await tts_once(client)


async def main() -> None:
    out = Path(sys.argv[1])
    results: dict[str, object] = {}
    data = wav_bytes(CLIP)
    limits = httpx2.Limits(max_connections=20)
    async with httpx2.AsyncClient(trust_env=False, limits=limits) as client:
        await tts_once(client)
        await asr_once(client, data)  # warm
        # alone
        results["tts_alone"] = [await tts_once(client) for _ in range(10)]
        results["asr_alone"] = [await asr_once(client, data) for _ in range(10)]
        results["llm_alone"] = [await llm_once(client, STORY + f" ({uuid.uuid4().hex[:6]})", 300) for _ in range(3)]
        print("alone done", flush=True)
        # while the LLM decodes
        for streams in (1, 3):
            stop = asyncio.Event()
            bg = asyncio.create_task(background_llm(client, streams, stop))
            await asyncio.sleep(2.0)
            results[f"tts_llm_decode_{streams}"] = [await tts_once(client) for _ in range(8)]
            results[f"asr_llm_decode_{streams}"] = [await asr_once(client, data) for _ in range(8)]
            stop.set()
            await bg
            print(f"decode {streams} done", flush=True)
        # while the LLM prefills ~16k tokens
        rows = []
        for _ in range(4):
            filler = "".join(f"第{i}条：系统日志记录，状态正常，无需处理。" for i in range(1400))
            prompt = f"({uuid.uuid4().hex})\n{filler}\n请总结以上内容。"
            prefill = asyncio.create_task(llm_once(client, prompt, 8))
            await asyncio.sleep(0.5)  # into the prefill
            rows.append(await tts_once(client))
            llm = await prefill
            rows[-1]["llm_ttft_ms"] = llm["ttft_ms"]
        results["tts_llm_prefill_16k"] = rows
        print("prefill done", flush=True)
        # the LLM while the TTS streams; ASR while the TTS streams
        stop = asyncio.Event()
        bg = asyncio.create_task(background_tts(client, stop))
        await asyncio.sleep(1.0)
        results["llm_while_tts"] = [await llm_once(client, STORY + f" ({uuid.uuid4().hex[:6]})", 300) for _ in range(3)]
        results["asr_while_tts"] = [await asr_once(client, data) for _ in range(8)]
        stop.set()
        await bg
    table: dict[str, object] = {}
    for name, rows in results.items():
        assert isinstance(rows, list)
        keys = [k for k in ("ttfc_ms", "rtf", "asr_ms", "ttft_ms", "tok_s", "llm_ttft_ms") if any(k in r for r in rows)]
        table[name] = {key: summary(rows, key) for key in keys}
    out.write_text(json.dumps({"raw": results, "summary": table}, ensure_ascii=False, indent=1))
    for name, value in table.items():
        print(name, json.dumps(value))


asyncio.run(main())
