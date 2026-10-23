"""Echo-canceller engines behind one interface (D17; the engine was chosen in M0, G7).

An engine takes exactly one 10 ms frame of capture (the client's audio, 160 samples at 16 kHz)
and the reference frame for the same capture positions (our own output audio as the client is
modelled to play it), and returns the capture frame with the echo removed.

The frame size is checked before every native call. pywebrtc-audio silently zero-pads a partial
frame, which changes its output by thousands of LSB (M0 S11), and livekit's AEC3 aborts the
whole process on one; the session layer therefore only ever hands over whole frames.

Engines:

- ``webrtc_aec3``: WebRTC AEC3 through pywebrtc-audio 0.2.0 (Apache-2.0), configured as G7 chose:
  ``EchoCanceller(sample_rate=16000, num_channels=1, stream_delay_ms=0)``, AEC only. AEC3
  estimates the residual delay itself (up to about 500 ms), forces a high-pass filter, adds
  comfort noise, and delays the capture by a fixed 8 ms. It releases the GIL while it runs, and
  one instance must serve one stream, one call at a time.

livekit's AEC3 wrapper (G7's fallback) can be added behind the same interface.
"""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt
import pywebrtc_audio

RATE = 16_000
"""Every engine runs on canonical audio."""

FRAME_SAMPLES = 160
"""One engine frame: 10 ms at 16 kHz."""

type EngineName = Literal["webrtc_aec3"]
type Pcm = npt.NDArray[np.int16]


class EchoEngine(Protocol):
    """One stream's canceller. Not thread-safe: one call at a time, frames in order."""

    def process(self, capture: Pcm, reference: Pcm) -> Pcm:
        """The capture frame with the echo of ``reference`` removed (both ``FRAME_SAMPLES``)."""
        ...


def check_frame(frame: Pcm, name: str) -> None:
    """Refuse anything but one whole int16 frame: engines must never pad or abort."""
    if frame.dtype != np.int16 or frame.shape != (FRAME_SAMPLES,):
        raise ValueError(
            f"an echo-canceller {name} frame is int16[{FRAME_SAMPLES}] (10 ms at 16 kHz), "
            f"not {frame.dtype}{frame.shape}"
        )


class WebRtcAec3:
    """WebRTC AEC3 through pywebrtc-audio."""

    def __init__(self) -> None:
        self._canceller = pywebrtc_audio.EchoCanceller(
            sample_rate=RATE, num_channels=1, stream_delay_ms=0
        )

    def process(self, capture: Pcm, reference: Pcm) -> Pcm:
        check_frame(capture, "capture")
        check_frame(reference, "reference")
        output = self._canceller.process(capture, reference)
        if output.shape != (FRAME_SAMPLES,):
            raise RuntimeError(f"the echo canceller returned {output.shape} samples")
        return output


def make_engine(name: EngineName) -> EchoEngine:
    match name:
        case "webrtc_aec3":
            return WebRtcAec3()
