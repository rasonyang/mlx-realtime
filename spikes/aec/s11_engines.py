"""S11 part 2: the echo-canceller candidates behind one interface. Throwaway.

Every engine takes exactly one 10 ms frame (160 int16 samples at 16 kHz) of capture and of reference
and returns 160 int16 samples. `check_frame` runs before every native call: livekit 1.1.21 aborts the
whole process (Rust panic -> SIGABRT) on any frame that is not a whole number of 10 ms
(a3_aec_install_check.json), so a size error must never reach it. The driver also runs every engine
condition in a child process.

Variants (A3 section 5 "Variants", plus the pre-filter of A3 design point 8):
  none        passthrough (the "before" signal)
  lk          WebRTC AEC3 via livekit 1.1.21 APM: AEC on, NS/AGC/HPF flags off (the HPF is forced on by
              AEC3's enforce_high_pass_filtering), stream delay hint 0 set once
  pw          WebRTC AEC3 via pywebrtc-audio 0.2.0 EchoCanceller(16000, 1, stream_delay_ms=0)
  sx<taps>    SpeexDSP 1.2.1 MDF via speex_ctypes.py, frame 160, filter_length <taps>, preprocessor off
  sx<taps>pp  ... with the preprocessor linked to the echo state, denoise off (A3's variant). Measured: a
              no-op apart from 10 ms of delay, because speexdsp applies no spectral gain with denoise off,
              so its residual echo suppression never acts
  sx<taps>ppd ... preprocessor with denoise on: residual echo suppression and noise suppression active
  bl<mu>      numpy PBFDAF + Geigel (baseline.py), 130 ms tail, step mu (bl03 = 0.3, bl05 = 0.5)
  <any>_pf    the 3.8 kHz low-pass pre-filter (63-tap FIR) on both capture and reference first
"""

from __future__ import annotations

import s11_common as c

import numpy as np  # noqa: E402


def check_frame(near: np.ndarray, far: np.ndarray) -> None:
    if not (isinstance(near, np.ndarray) and isinstance(far, np.ndarray)):
        raise TypeError("frames must be numpy arrays")
    if near.dtype != np.int16 or far.dtype != np.int16 or near.shape != (c.FRAME,) or far.shape != (c.FRAME,):
        raise ValueError(f"engine frames must be int16[{c.FRAME}] (10 ms at 16 kHz); got {near.dtype}{near.shape} "
                         f"and {far.dtype}{far.shape}")


class Passthrough:
    def process(self, near, far):
        check_frame(near, far)
        return near.copy()


class LivekitAec3:
    """delay_hint: "once" = set_stream_delay_ms(0) once after creation (A3's install check);
    "none" = never call it; "every" = call it before every process_stream (the APM forgets it after
    one frame)."""

    def __init__(self, delay_hint: str = "once"):
        from livekit import rtc

        self.rtc = rtc
        self.hint = delay_hint
        self.apm = rtc.AudioProcessingModule(echo_cancellation=True, noise_suppression=False,
                                             high_pass_filter=False, auto_gain_control=False)
        if delay_hint == "once":
            self.apm.set_stream_delay_ms(0)

    def process(self, near, far):
        check_frame(near, far)
        rtc = self.rtc
        r = rtc.AudioFrame(far.tobytes(), c.SR, 1, c.FRAME)
        self.apm.process_reverse_stream(r)
        if self.hint == "every":
            self.apm.set_stream_delay_ms(0)
        cap = rtc.AudioFrame(near.tobytes(), c.SR, 1, c.FRAME)
        self.apm.process_stream(cap)
        return np.frombuffer(cap.data, dtype=np.int16).copy()


class PywebrtcAec3:
    def __init__(self):
        import pywebrtc_audio

        self.ec = pywebrtc_audio.EchoCanceller(sample_rate=c.SR, num_channels=1, stream_delay_ms=0)

    def process(self, near, far):
        check_frame(near, far)
        out = self.ec.process(near, far)
        if out.shape != (c.FRAME,):
            raise RuntimeError(f"pywebrtc returned {out.shape}")
        return np.asarray(out, dtype=np.int16)


class Speex:
    def __init__(self, taps: int, preprocess: bool, denoise: bool = False):
        from speex_ctypes import SpeexAec

        self.s = SpeexAec(frame_size=c.FRAME, filter_length=taps, sample_rate=c.SR, preprocess=preprocess, denoise=denoise)

    def process(self, near, far):
        check_frame(near, far)
        return self.s.process(near, far)


class Baseline:
    def __init__(self, mu: float, reg_rel: float):
        from baseline import PbfdafGeigel

        self.b = PbfdafGeigel(sample_rate=c.SR, block=c.FRAME, tail_ms=128.0, mu=mu, reg_rel=reg_rel)

    def process(self, near, far):
        check_frame(near, far)
        return c.to_i16(self.b.process(near, far))


class Prefiltered:
    """3.8 kHz low-pass on both signals before the engine (adds 1.94 ms to both paths)."""

    def __init__(self, inner):
        self.inner = inner
        self.fn = c.StreamFir(c.PREFILTER_TAPS)
        self.ff = c.StreamFir(c.PREFILTER_TAPS)

    def process(self, near, far):
        check_frame(near, far)
        return self.inner.process(c.to_i16(self.fn(near)), c.to_i16(self.ff(far)))


def make(name: str):
    base, pf = (name[:-3], True) if name.endswith("_pf") else (name, False)
    if base == "none":
        e = Passthrough()
    elif base == "lk":
        e = LivekitAec3()
    elif base == "lknd":
        e = LivekitAec3("none")
    elif base == "lkev":
        e = LivekitAec3("every")
    elif base == "pw":
        e = PywebrtcAec3()
    elif base.startswith("sx"):
        if base.endswith("ppd"):  # preprocessor with denoise on (residual echo suppression engaged)
            e = Speex(int(base[2:-3]), True, denoise=True)
        else:
            pp = base.endswith("pp")
            e = Speex(int(base[2:-2] if pp else base[2:]), pp)
    elif base.startswith("bl"):
        mu, _, reg = base[2:].partition("r")  # bl05 = mu 0.5 (A3's file as is); bl05r2 = reg_rel 1e-2
        e = Baseline(int(mu) / 10.0, 10.0 ** -int(reg) if reg else 0.0)
    else:
        raise ValueError(name)
    return Prefiltered(e) if pf else e


DESCRIPTIONS = {
    "none": "no canceller (capture as received)",
    "lk": "AEC3 via livekit 1.1.21",
    "lk_pf": "AEC3 via livekit 1.1.21 + 3.8 kHz pre-filter",
    "pw": "AEC3 via pywebrtc-audio 0.2.0",
    "pw_pf": "AEC3 via pywebrtc-audio 0.2.0 + 3.8 kHz pre-filter",
    "sx1024_pf": "SpeexDSP MDF 1024 taps (64 ms) + pre-filter",
    "sx2048_pf": "SpeexDSP MDF 2048 taps (128 ms) + pre-filter",
    "sx1024pp_pf": "SpeexDSP MDF 1024 + preprocessor (residual echo suppression) + pre-filter",
    "sx2048pp_pf": "SpeexDSP MDF 2048 + preprocessor (residual echo suppression) + pre-filter",
    "sx1024": "SpeexDSP MDF 1024 taps, no pre-filter",
    "sx1024ppd_pf": "SpeexDSP MDF 1024 + preprocessor with denoise on (residual echo suppression active) + pre-filter",
    "sx2048ppd_pf": "SpeexDSP MDF 2048 + preprocessor with denoise on (residual echo suppression active) + pre-filter",
    "bl03_pf": "numpy PBFDAF + Geigel, mu 0.3, 130 ms + pre-filter",
    "bl05_pf": "numpy PBFDAF + Geigel, mu 0.5, 130 ms + pre-filter",
}
