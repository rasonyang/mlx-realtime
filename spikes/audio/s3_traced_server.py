"""mlx_audio.server 0.6.1 with trace lines, for the S3 cancellation and pinning checks (B2).

Behaviour is unchanged: the hooks below only print `[trace <perf_counter_ns>] ...` lines (the same
mach_absolute_time clock the client uses) and then call the original methods. After patching, this
runs `mlx_audio.server.main()`, which is what `python -m mlx_audio.server` runs; uvicorn imports
"mlx_audio.server:app", which is the already-imported, patched module.

Traced points:
- TTSExecutionAdapter.run_serial: start and end of one request on the broker thread, and whether
  its cancel_event was set when it returned;
- TTSExecutionAdapter._emit_audio: every audio chunk handed to the HTTP side;
- InferenceHandle.cancel: the first call per request (the HTTP side's disconnect, or normal end);
- ModelProvider.load_model: only when a model is actually loaded (not already cached).

Usage (same flags as mlx_audio.server):
    HF_HUB_OFFLINE=1 uv run --project spikes/audio python spikes/audio/s3_traced_server.py --host 127.0.0.1 --port 8103
"""

import sys
import time

import mlx_audio.server as srv
from mlx_audio.server_inference import InferenceHandle


def trace(msg: str) -> None:
    sys.stdout.write(f"[trace {time.perf_counter_ns()}] {msg}\n")
    sys.stdout.flush()


_run_serial = srv.TTSExecutionAdapter.run_serial


def run_serial(self, request):
    p = request.payload.request
    rid = request.request_id[:8]
    trace(f"tts.start rid={rid} chars={len(p.input)} stream={p.stream} interval={p.streaming_interval}")
    try:
        return _run_serial(self, request)
    finally:
        trace(f"tts.end rid={rid} cancelled={request.cancel_event.is_set()}")


_emit_audio = srv.TTSExecutionAdapter._emit_audio


def emit_audio(self, request, speech_request, audio, sample_rate):
    trace(f"tts.emit rid={request.request_id[:8]} samples={audio.shape[0]} "
          f"cancelled={request.cancel_event.is_set()}")
    return _emit_audio(self, request, speech_request, audio, sample_rate)


_cancel = InferenceHandle.cancel


def cancel(self):
    if not self.cancel_event.is_set():
        trace(f"handle.cancel rid={self.context.request_id[:8]} kind={self.context.endpoint_kind}")
    return _cancel(self)


_load_model = srv.ModelProvider.load_model


def load_model(self, model_name):
    fresh = model_name not in self.models
    if fresh:
        trace(f"model.load.start {model_name}")
    try:
        return _load_model(self, model_name)
    finally:
        if fresh:
            trace(f"model.load.end {model_name}")


srv.TTSExecutionAdapter.run_serial = run_serial
srv.TTSExecutionAdapter._emit_audio = emit_audio
InferenceHandle.cancel = cancel
srv.ModelProvider.load_model = load_model

if __name__ == "__main__":
    trace("traced server: hooks installed")
    srv.main()
