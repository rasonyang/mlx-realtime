"""S8.1 probe (M0 work package A1): what openai-python 3.36.1 sends to a GA Realtime endpoint, and how
it parses the GA events we plan to emit.

A local `websockets` stub on 127.0.0.1 records the opening handshake and every frame the client sends.
After the client's session.update it plays a scripted GA sequence:

    session.created, session.updated, conversation.item.added, conversation.item.done,
    (waits for response.create)
    response.created, response.output_item.added, response.content_part.added,
    response.output_audio.delta, response.output_audio_transcript.delta, response.output_audio.done,
    response.output_audio_transcript.done, response.content_part.done, response.output_item.done,
    response.done, error{invalid_request_error, conversation_already_has_active_response}

The client is the official SDK:

    AsyncOpenAI(base_url="http://127.0.0.1:<port>/v1", api_key="test").realtime.connect(model="gpt-realtime")

It sends a GA session.update and a response.create. For every received event it records the SDK class
it was parsed into (`parse_event`, which `recv()` and `async for` use), the classes of nested objects,
and whether the raw payload also validates against the SDK model declared for that event type. Quiet
windows before the first send and after the last event show whether the SDK sends anything unprompted.

Run from the repository root (port 0 = any free port):

    uv run --project spikes/sdk python spikes/sdk/probe_sdk.py --port 8103

Writes spikes/results/sdk_probe.json.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import datetime as dt
import importlib.metadata as md
import json
import math
import os
import struct
import sys
import time
import typing
import urllib.parse
import urllib.request
from pathlib import Path

# Make the run independent of the caller's OpenAI settings, and keep any real key out of the output.
for _key in [k for k in os.environ if k.startswith("OPENAI_")]:
    del os.environ[_key]

import openai  # noqa: E402
import openai.lib._websocket as sdk_websocket  # noqa: E402
import pydantic  # noqa: E402
import websockets  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from openai.types.realtime.realtime_server_event import RealtimeServerEvent  # noqa: E402
from websockets.asyncio.server import serve  # noqa: E402
from websockets.uri import get_proxy, parse_uri  # noqa: E402

RESULT = Path(__file__).resolve().parents[1] / "results" / "sdk_probe.json"
QUIET_S = 0.5
T0 = time.perf_counter()

SESSION_ID = "sess_probe0001"
CONVERSATION_ID = "conv_probe0001"
RESPONSE_ID = "resp_probe0001"
USER_ITEM_ID = "item_user0001"
ASSISTANT_ITEM_ID = "item_asst0001"
TRANSCRIPT = "您好，请提供一下订单号。"

# AICC-shaped GA session.update: pcm 24 kHz both ways, server_vad with only silence_duration_ms,
# flat function tools, a voice.
SESSION_UPDATE = {
    "type": "realtime",
    "instructions": "你是电话客服助理，回答要简短。",
    "output_modalities": ["audio"],
    "audio": {
        "input": {
            "format": {"type": "audio/pcm", "rate": 24000},
            "turn_detection": {"type": "server_vad", "silence_duration_ms": 500},
        },
        "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "alloy"},
    },
    "tools": [
        {
            "type": "function",
            "name": "lookup_order",
            "description": "Look up an order by its id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        }
    ],
}


def now_ms() -> float:
    return round((time.perf_counter() - T0) * 1000, 1)


# --- instrumentation: record the URI and options the SDK hands to websockets.connect -------------
CONNECT_CALLS: list[dict] = []
_SdkConnect = sdk_websocket._WebSocketConnect


class _RecordingConnect(_SdkConnect):
    def __init__(self, uri: str, *args, **kwargs):
        plain = (int, float, str, bool, type(None))
        CONNECT_CALLS.append(
            {
                "uri": uri,
                "kwargs": {
                    k: (v if isinstance(v, plain) else repr(v))
                    for k, v in kwargs.items()
                    if k != "additional_headers"
                },
                "additional_headers": redact(dict(kwargs.get("additional_headers") or {})),
            }
        )
        super().__init__(uri, *args, **kwargs)


sdk_websocket._WebSocketConnect = _RecordingConnect  # imported by the SDK at connect time


def redact(headers: dict[str, str]) -> dict[str, str]:
    return {
        k: (v if k.lower() != "authorization" or v == "Bearer test" else "<redacted>")
        for k, v in headers.items()
    }


# --- the scripted GA server events --------------------------------------------------------------
def pcm_tone_b64(ms: int = 40, rate: int = 24000, hz: float = 440.0) -> str:
    n = rate * ms // 1000
    samples = (int(8000 * math.sin(2 * math.pi * hz * i / rate)) for i in range(n))
    return base64.b64encode(struct.pack(f"<{n}h", *samples)).decode()


def deep_merge(base: dict, update: dict) -> dict:
    out = dict(base)
    for k, v in update.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def session_object(update: dict | None = None) -> dict:
    session = {
        "object": "realtime.session",
        "id": SESSION_ID,
        "type": "realtime",
        "model": "gpt-realtime",
        "output_modalities": ["audio"],
        "instructions": "",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": None,
                "noise_reduction": None,
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200,
                    "idle_timeout_ms": None,
                    "create_response": True,
                    "interrupt_response": True,
                },
            },
            "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "alloy", "speed": 1.0},
        },
        "tools": [],
        "tool_choice": "auto",
        "max_output_tokens": "inf",
        "tracing": None,
        "truncation": "auto",
        "prompt": None,
        "expires_at": 0,
        "include": None,
    }
    return deep_merge(session, update or {})


def user_item(transcript: str | None) -> dict:
    return {
        "id": USER_ITEM_ID,
        "object": "realtime.item",
        "type": "message",
        "status": "completed",
        "role": "user",
        "content": [{"type": "input_audio", "transcript": transcript}],
    }


def assistant_item(status: str, transcript: str | None) -> dict:
    content = [] if transcript is None else [{"type": "output_audio", "transcript": transcript}]
    return {
        "id": ASSISTANT_ITEM_ID,
        "object": "realtime.item",
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": content,
    }


def response_object(status: str, output: list[dict], usage: dict | None) -> dict:
    return {
        "object": "realtime.response",
        "id": RESPONSE_ID,
        "conversation_id": CONVERSATION_ID,
        "status": status,
        "status_details": None,
        "output": output,
        "output_modalities": ["audio"],
        "max_output_tokens": "inf",
        "audio": {"output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "alloy"}},
        "usage": usage,
        "metadata": None,
    }


def session_events(update: dict) -> list[dict]:
    return [
        {"type": "session.created", "session": session_object()},
        {"type": "session.updated", "session": session_object(update)},
        {"type": "conversation.item.added", "previous_item_id": None, "item": user_item(None)},
        {"type": "conversation.item.done", "previous_item_id": None, "item": user_item("我想查一下订单。")},
    ]


def response_events() -> list[dict]:
    ids = {"response_id": RESPONSE_ID, "item_id": ASSISTANT_ITEM_ID, "output_index": 0, "content_index": 0}
    usage = {
        "total_tokens": 30,
        "input_tokens": 20,
        "output_tokens": 10,
        "input_token_details": {"text_tokens": 20, "audio_tokens": 0, "cached_tokens": 0},
        "output_token_details": {"text_tokens": 10, "audio_tokens": 0},
    }
    return [
        {"type": "response.created", "response": response_object("in_progress", [], None)},
        {
            "type": "response.output_item.added",
            "response_id": RESPONSE_ID,
            "output_index": 0,
            "item": assistant_item("in_progress", None),
        },
        {
            "type": "response.content_part.added",
            **{k: v for k, v in ids.items()},
            "part": {"type": "audio", "transcript": ""},
        },
        {"type": "response.output_audio.delta", **ids, "delta": pcm_tone_b64()},
        {"type": "response.output_audio_transcript.delta", **ids, "delta": TRANSCRIPT},
        {"type": "response.output_audio.done", **ids},
        {"type": "response.output_audio_transcript.done", **ids, "transcript": TRANSCRIPT},
        {"type": "response.content_part.done", **ids, "part": {"type": "audio", "transcript": TRANSCRIPT}},
        {
            "type": "response.output_item.done",
            "response_id": RESPONSE_ID,
            "output_index": 0,
            "item": assistant_item("completed", TRANSCRIPT),
        },
        {
            "type": "response.done",
            "response": response_object("completed", [assistant_item("completed", TRANSCRIPT)], usage),
        },
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": "conversation_already_has_active_response",
                "message": f"Conversation already has an active response in progress: {RESPONSE_ID}.",
                "param": None,
                "event_id": None,
            },
        },
    ]


# --- the stub server ----------------------------------------------------------------------------
class Stub:
    def __init__(self) -> None:
        self.handshake: dict | None = None
        self.client_frames: list[dict] = []
        self.sent: list[dict] = []
        self.inbox: asyncio.Queue[dict | None] = asyncio.Queue()
        self.close: dict | None = None
        self._event_seq = 0

    async def handler(self, ws) -> None:
        req = ws.request
        split = urllib.parse.urlsplit(req.path)
        self.handshake = {
            "t_ms": now_ms(),
            "request_line_path": req.path,
            "path": split.path,
            "query": urllib.parse.parse_qs(split.query, keep_blank_values=True),
            "headers": [[k, v if k.lower() != "authorization" or v == "Bearer test" else "<redacted>"]
                        for k, v in req.headers.raw_items()],
            "subprotocol_selected": ws.subprotocol,
        }
        reader = asyncio.create_task(self._read(ws))
        try:
            update = await self._expect("session.update")
            for event in session_events(update.get("session") or {}):
                await self._send(ws, event)
            await self._expect("response.create")
            for event in response_events():
                await self._send(ws, event)
            await asyncio.sleep(2 * QUIET_S)  # anything the client sends now is unprompted
        finally:
            await ws.close(1000, "probe done")
            await reader
            self.close = {"t_ms": now_ms(), "code": ws.close_code, "reason": ws.close_reason}

    async def _read(self, ws) -> None:
        try:
            async for message in ws:
                data = None
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                    except ValueError:
                        pass
                self.client_frames.append(
                    {
                        "t_ms": now_ms(),
                        "frame": "text" if isinstance(message, str) else "binary",
                        "bytes": len(message.encode() if isinstance(message, str) else message),
                        "type": data.get("type") if isinstance(data, dict) else None,
                        "json": data,
                    }
                )
                await self.inbox.put(data if isinstance(data, dict) else {})
        except websockets.ConnectionClosed:
            pass
        finally:
            await self.inbox.put(None)

    async def _expect(self, event_type: str, timeout: float = 10.0) -> dict:
        while True:
            data = await asyncio.wait_for(self.inbox.get(), timeout)
            if data is None:
                raise RuntimeError(f"connection closed while waiting for {event_type}")
            if data.get("type") == event_type:
                return data

    async def _send(self, ws, event: dict) -> None:
        self._event_seq += 1
        event = {"event_id": f"event_{self._event_seq:04d}", **event}
        await ws.send(json.dumps(event, ensure_ascii=False))
        self.sent.append({"t_ms": now_ms(), "type": event["type"], "json": event})


# --- the SDK client -----------------------------------------------------------------------------
def event_classes() -> dict[str, type]:
    union = typing.get_args(RealtimeServerEvent)[0]  # Annotated[Union[...], PropertyInfo(...)]
    mapping: dict[str, type] = {}
    for cls in typing.get_args(union):
        for value in typing.get_args(cls.model_fields["type"].annotation):
            mapping[value] = cls
    return mapping


def qualname(obj: object) -> str:
    return f"{type(obj).__module__}.{type(obj).__name__}"


def describe(raw: bytes, event: object, classes: dict[str, type]) -> dict:
    data = json.loads(raw)
    etype = data.get("type")
    expected = classes.get(etype)
    out: dict = {
        "t_ms": now_ms(),
        "type": etype,
        "sdk_class": qualname(event),
        "expected_class": f"{expected.__module__}.{expected.__name__}" if expected else None,
        "class_matches": expected is not None and type(event) is expected,
    }
    try:
        validated = expected.model_validate_json(raw) if expected else None
        out["validates_against_expected_model"] = validated is not None
    except pydantic.ValidationError as exc:
        validated = None
        out["validates_against_expected_model"] = False
        out["validation_errors"] = [
            {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()
        ]
    nested: dict = {}
    for attr in ("session", "item", "part", "response", "error"):
        value = getattr(event, attr, None)
        if value is None:
            continue
        info: dict = {"sdk_class": type(value).__name__}
        if validated is not None and getattr(validated, attr, None) is not None:
            info["validated_class"] = type(getattr(validated, attr)).__name__
        for field in ("id", "object", "type", "role", "status", "code"):
            if getattr(value, field, None) is not None:
                info[field] = getattr(value, field)
        if attr == "session":
            extra = getattr(value, "model_extra", None) or {}
            info["extra_fields_kept"] = sorted(extra)
        if attr == "response":
            info["output_item_classes"] = [type(i).__name__ for i in (value.output or [])]
        nested[attr] = info
    if nested:
        out["nested"] = nested
    if etype == "response.output_audio.delta":
        out["delta_pcm_bytes"] = len(base64.b64decode(event.delta))  # type: ignore[attr-defined]
    if etype == "error":
        out["error_code"] = event.error.code  # type: ignore[attr-defined]
        out["error_type"] = event.error.type  # type: ignore[attr-defined]
    return out


async def run_client(port: int, sent_calls: list[dict], received: list[dict], conn_info: dict) -> None:
    classes = event_classes()
    client = AsyncOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="test")
    async with client.realtime.connect(model="gpt-realtime") as conn:
        ws = conn._connection  # the websockets ClientConnection, for introspection only
        conn_info.update(
            {
                "connected_t_ms": now_ms(),
                "client_ping_interval": getattr(ws, "ping_interval", None),
                "client_ping_timeout": getattr(ws, "ping_timeout", None),
                "client_close_timeout": getattr(ws, "close_timeout", None),
                "client_max_queue": repr(getattr(ws, "max_queue", None)),
                "client_max_size": getattr(getattr(ws, "protocol", None), "max_size", None),
            }
        )
        await asyncio.sleep(QUIET_S)  # the SDK should send nothing on its own
        sent_calls.append({"t_ms": now_ms(), "call": "conn.session.update(session=SESSION_UPDATE)"})
        await conn.session.update(session=SESSION_UPDATE)  # type: ignore[arg-type]
        response_sent = False
        while True:
            raw = await asyncio.wait_for(conn.recv_bytes(), 10)
            event = conn.parse_event(raw)  # what recv() and `async for event in conn` return
            received.append(describe(raw, event, classes))
            if event.type == "session.updated" and not response_sent:
                sent_calls.append({"t_ms": now_ms(), "call": "conn.response.create()"})
                await conn.response.create()
                response_sent = True
            if event.type == "error":
                break
        await asyncio.sleep(QUIET_S)
        conn_info["offline_parse_checks"] = offline_parse_checks(conn)
    conn_info["client_closed_t_ms"] = now_ms()


def offline_parse_checks(conn) -> list[dict]:
    """parse_event on payloads we must never emit, to show what a client would get (no network).

    For unions the SDK first runs full validation; only if that fails does it fall back to the
    `type` discriminator map, and then to the first union variant that constructs.
    """
    bad_item = user_item("x") | {"status": "done"}  # invalid status literal
    cases = {
        "user item that fails validation": {"type": "conversation.item.added", "event_id": "e1", "item": bad_item},
        "unknown event type": {"type": "x.custom_event", "event_id": "e2", "foo": 1},
    }
    out = []
    for name, payload in cases.items():
        event = conn.parse_event(json.dumps(payload))
        item = getattr(event, "item", None)
        out.append(
            {
                "case": name,
                "sdk_class": type(event).__name__,
                "item_sdk_class": type(item).__name__ if item is not None else None,
                "type_attr": getattr(event, "type", None),
            }
        )
    return out


async def main_async(port: int) -> dict:
    stub = Stub()
    async with serve(stub.handler, "127.0.0.1", port) as server:
        port = server.sockets[0].getsockname()[1]
        sent_calls: list[dict] = []
        received: list[dict] = []
        conn_info: dict = {}
        await run_client(port, sent_calls, received, conn_info)
        for _ in range(50):  # let the handler record the close
            if stub.close is not None:
                break
            await asyncio.sleep(0.05)

    uri = CONNECT_CALLS[0]["uri"] if CONNECT_CALLS else ""
    headers = {k.lower(): v for k, v in (stub.handshake or {}).get("headers", [])}
    frames = stub.client_frames
    first_send_t = sent_calls[0]["t_ms"] if sent_calls else None
    proxy = get_proxy(parse_uri(uri)) if uri else None
    checks = {
        "url": uri,
        "url_scheme_is_ws": uri.startswith("ws://"),
        "path_is_v1_realtime": (stub.handshake or {}).get("path") == "/v1/realtime",
        "query": (stub.handshake or {}).get("query"),
        "authorization_is_bearer_test": headers.get("authorization") == "Bearer test",
        "openai_beta_header_sent": "openai-beta" in headers,
        "permessage_deflate_offered": "sec-websocket-extensions" in headers,
        "subprotocol_requested": "sec-websocket-protocol" in headers,
        "user_agent": headers.get("user-agent"),
        "x_stainless_headers": sorted(k for k in headers if k.startswith("x-stainless")),
        "client_frame_types": [f["type"] for f in frames],
        "client_frames_are_exactly_the_two_calls": [f["type"] for f in frames]
        == ["session.update", "response.create"],
        "no_frame_before_first_explicit_send": all(
            first_send_t is not None and f["t_ms"] >= first_send_t for f in frames
        ),
        "binary_frames_from_client": sum(f["frame"] == "binary" for f in frames),
        # The stub sends nothing until it gets session.update, so connect() returning before the first
        # server event means the SDK does not wait for session.created.
        "connect_returned_before_any_server_event": bool(
            stub.sent and conn_info.get("connected_t_ms", 1e18) < stub.sent[0]["t_ms"]
        ),
        "all_events_parsed_into_expected_class": all(r["class_matches"] for r in received),
        "all_events_validate_against_expected_model": all(
            r.get("validates_against_expected_model") for r in received
        ),
        "events_received": len(received),
        "events_sent": len(stub.sent),
        "websockets_proxy_for_this_uri": None if proxy is None else urllib.parse.urlsplit(proxy).scheme,
        "close": stub.close,
    }
    return {
        "probe": "S8.1 openai-python Realtime client against a GA stub (M0 A1)",
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "versions": {
            "python": sys.version.split()[0],
            "openai": openai.__version__,
            "websockets": websockets.__version__,
            "pydantic": pydantic.VERSION,
            "openai_realtime_extra": [r for r in (md.requires("openai") or []) if "realtime" in r],
        },
        "client_call": 'AsyncOpenAI(base_url="http://127.0.0.1:<port>/v1", api_key="test")'
        '.realtime.connect(model="gpt-realtime")',
        "stub": {"host": "127.0.0.1", "port": port, "quiet_window_s": QUIET_S},
        "checks": checks,
        "sdk_connect_calls": CONNECT_CALLS,
        "client_connection": conn_info,
        "handshake": stub.handshake,
        "client_sent_calls": sent_calls,
        "client_frames_received_by_stub": frames,
        "received_events": received,
        "server_events_sent": stub.sent,
        "proxy_env": {
            "getproxies_schemes": sorted(urllib.request.getproxies()),
            "proxy_bypass_127.0.0.1": bool(urllib.request.proxy_bypass(f"127.0.0.1:{port}")),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=0, help="stub port on 127.0.0.1 (0 = any free port)")
    parser.add_argument("--out", type=Path, default=RESULT, help=f"result file (default {RESULT})")
    args = parser.parse_args()
    result = asyncio.run(main_async(args.port))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
    c = result["checks"]
    print(f"url={c['url']}")
    for r in result["received_events"]:
        flag = "ok " if r["class_matches"] and r.get("validates_against_expected_model") else "!! "
        item = r.get("nested", {}).get("item", {}).get("sdk_class", "")
        print(f"{flag}{r['type']:42s} -> {r['sdk_class'].rsplit('.', 1)[-1]} {item}")
    print(json.dumps({k: v for k, v in c.items() if k not in ("query",)}, ensure_ascii=False, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
