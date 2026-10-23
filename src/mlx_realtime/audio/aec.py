"""Echo-canceller engines behind one interface (D17; the engines were chosen in M0, G7).

An engine takes exactly one 10 ms frame of capture (the client's audio, 160 samples at 16 kHz)
and the reference frame for the same capture positions (our own output audio as the client is
modelled to play it), and returns the capture frame with the echo removed.

The frame size is checked before every native call. pywebrtc-audio silently zero-pads a partial
frame, which changes its output by thousands of LSB (M0 S11), and livekit's AEC3 aborts the
whole process on one; the session layer therefore only ever hands over whole frames.

Engines (G7 dual path: AEC3 for speech detection, Speex for the ASR):

- ``webrtc_aec3``: WebRTC AEC3 through pywebrtc-audio 0.2.0 (Apache-2.0), configured as G7 chose:
  ``EchoCanceller(sample_rate=16000, num_channels=1, stream_delay_ms=0)``, AEC only. AEC3
  estimates the residual delay itself (up to about 500 ms), forces a high-pass filter, adds
  comfort noise, and delays the capture by a fixed 8 ms. It releases the GIL while it runs, and
  one instance must serve one stream, one call at a time. Its suppressor also removes quiet
  near-end speech in double talk (M0 S11: +11 zh / +28 en points of ASR error at -15 dB).
- ``SpeexLinear``: SpeexDSP 1.2.1's MDF canceller (BSD-3-Clause), vendored and compiled by the
  ``speexdsp-aec`` package (``vendor/speexdsp-aec``). A linear adaptive filter: it leaves the
  near end intact but cancels only echo that falls inside its ``filter_ms``, so the session
  aligns its reference (GCC-PHAT). The binding checks every buffer again in C, copies it into
  its own scratch memory, and releases the GIL while it runs.

``LowPass`` is the 3.8 kHz pre-filter the Speex path puts on both its inputs (M0 S11, A3 design
point 8).

livekit's AEC3 wrapper (G7's fallback) can be added behind the same interface.
"""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt
import pywebrtc_audio
import speexdsp_aec

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


class SpeexLinear:
    """SpeexDSP's MDF canceller of ``filter_ms`` (vendored), optionally followed by SpeexDSP's
    preprocessor (noise and residual-echo suppression)."""

    def __init__(self, filter_ms: int = 64, *, preprocess: bool = False) -> None:
        self._canceller = speexdsp_aec.EchoCanceller(
            FRAME_SAMPLES, filter_ms * RATE // 1000, RATE, preprocess=preprocess
        )

    @property
    def resets(self) -> int:
        """How often SpeexDSP reset a filter that diverged (its only warning here)."""
        return self._canceller.warnings

    def process(self, capture: Pcm, reference: Pcm) -> Pcm:
        check_frame(capture, "capture")
        check_frame(reference, "reference")
        output = np.frombuffer(self._canceller.process(capture, reference), dtype="<i2")
        if output.shape != (FRAME_SAMPLES,):
            raise RuntimeError(f"the echo canceller returned {output.shape} samples")
        return output.astype(np.int16)


class LowPass:
    """A streaming linear-phase FIR low-pass on 16 kHz int16 audio: a 63-tap Blackman-windowed
    sinc (M0 S11's pre-filter), with its history carried from call to call (any length). It
    delays the signal by 31 samples (1.94 ms)."""

    TAPS = 63

    def __init__(self, cutoff_hz: float) -> None:
        index = np.arange(self.TAPS, dtype=np.float64) - (self.TAPS - 1) / 2
        fraction = cutoff_hz / RATE
        taps = 2 * fraction * np.sinc(2 * fraction * index) * np.blackman(self.TAPS)
        self._taps = taps / taps.sum()
        self._history = np.zeros(self.TAPS - 1, dtype=np.float64)

    @property
    def delay(self) -> int:
        return (self.TAPS - 1) // 2

    def __call__(self, pcm: Pcm) -> Pcm:
        signal = np.concatenate((self._history, pcm.astype(np.float64)))
        self._history = signal[len(signal) - (self.TAPS - 1) :]
        filtered = np.convolve(signal, self._taps, mode="valid")
        return np.clip(np.rint(filtered), -32768, 32767).astype(np.int16)


def make_engine(name: EngineName) -> EchoEngine:
    match name:
        case "webrtc_aec3":
            return WebRtcAec3()
