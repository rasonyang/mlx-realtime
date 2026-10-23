"""Minimal ctypes binding to the system SpeexDSP echo canceller, for S11 part 2.

Why: the PyPI bindings are unsuitable as the SpeexDSP vehicle (see
spikes/results/a3_aec_research.md). pyaec never sets SPEEX_ECHO_SET_SAMPLING_RATE and does not
check buffer lengths; speexdsp 0.1.1 and speexaec 0.1.3 need build flags and hold the GIL.

This module loads Homebrew's float build of libspeexdsp 1.2.1 at run time (a system prerequisite:
`brew install speexdsp`), so the uv environment stays pure Python. ctypes.CDLL releases the GIL
during each call. Constants and prototypes are from speex/speex_echo.h and
speex/speex_preprocess.h (SpeexDSP 1.2.1).

Status: smoke-tested only in the A3 install check.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

_LIB_PATH = os.environ.get("SPEEXDSP_LIB", "/opt/homebrew/opt/speexdsp/lib/libspeexdsp.1.dylib")

SPEEX_ECHO_SET_SAMPLING_RATE = 24
SPEEX_PREPROCESS_SET_DENOISE = 0
SPEEX_PREPROCESS_SET_AGC = 2
SPEEX_PREPROCESS_SET_VAD = 4
SPEEX_PREPROCESS_SET_DEREVERB = 8
SPEEX_PREPROCESS_SET_ECHO_SUPPRESS = 20
SPEEX_PREPROCESS_SET_ECHO_SUPPRESS_ACTIVE = 22
SPEEX_PREPROCESS_SET_ECHO_STATE = 24

_lib = ctypes.CDLL(_LIB_PATH)
_i16p = ctypes.POINTER(ctypes.c_int16)
_lib.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.speex_echo_state_init.restype = ctypes.c_void_p
_lib.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]
_lib.speex_echo_state_reset.argtypes = [ctypes.c_void_p]
_lib.speex_echo_cancellation.argtypes = [ctypes.c_void_p, _i16p, _i16p, _i16p]
_lib.speex_echo_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
_lib.speex_echo_ctl.restype = ctypes.c_int
_lib.speex_preprocess_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.speex_preprocess_state_init.restype = ctypes.c_void_p
_lib.speex_preprocess_state_destroy.argtypes = [ctypes.c_void_p]
_lib.speex_preprocess_run.argtypes = [ctypes.c_void_p, _i16p]
_lib.speex_preprocess_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
_lib.speex_preprocess_ctl.restype = ctypes.c_int


class SpeexAec:
    """One SpeexDSP MDF echo canceller (one per session), optionally followed by the preprocessor.

    preprocess=False (default) gives the linear canceller only: no residual-echo suppression, no
    denoise, which is what ASR wants. preprocess=True links the preprocessor to the echo state
    (residual echo suppression); `denoise` then controls the noise suppressor separately.
    """

    def __init__(
        self,
        frame_size: int = 160,
        filter_length: int = 2048,
        sample_rate: int = 16000,
        preprocess: bool = False,
        denoise: bool = False,
    ) -> None:
        self.frame_size = frame_size
        self._st = _lib.speex_echo_state_init(frame_size, filter_length)
        if not self._st:
            raise RuntimeError("speex_echo_state_init failed")
        rate = ctypes.c_int(sample_rate)
        _lib.speex_echo_ctl(self._st, SPEEX_ECHO_SET_SAMPLING_RATE, ctypes.byref(rate))
        self._pp = None
        if preprocess:
            self._pp = _lib.speex_preprocess_state_init(frame_size, sample_rate)
            _lib.speex_preprocess_ctl(self._pp, SPEEX_PREPROCESS_SET_ECHO_STATE, ctypes.c_void_p(self._st))
            on = ctypes.c_int(1 if denoise else 0)
            _lib.speex_preprocess_ctl(self._pp, SPEEX_PREPROCESS_SET_DENOISE, ctypes.byref(on))
        self._out = np.zeros(frame_size, dtype=np.int16)

    def process(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        """Cancel one frame: `near` (capture) and `far` (reference), int16, exactly frame_size each."""
        n = self.frame_size
        near = np.ascontiguousarray(near, dtype=np.int16)
        far = np.ascontiguousarray(far, dtype=np.int16)
        if near.shape != (n,) or far.shape != (n,):
            raise ValueError(f"frames must have shape ({n},)")  # the C code reads frame_size samples
        out = np.empty(n, dtype=np.int16)
        _lib.speex_echo_cancellation(
            self._st, near.ctypes.data_as(_i16p), far.ctypes.data_as(_i16p), out.ctypes.data_as(_i16p)
        )
        if self._pp is not None:
            _lib.speex_preprocess_run(self._pp, out.ctypes.data_as(_i16p))
        return out

    def reset(self) -> None:
        _lib.speex_echo_state_reset(self._st)

    def __del__(self) -> None:
        if getattr(self, "_pp", None):
            _lib.speex_preprocess_state_destroy(self._pp)
            self._pp = None
        if getattr(self, "_st", None):
            _lib.speex_echo_state_destroy(self._st)
            self._st = None
