"""Edge-silence trimming of one synthesized segment (M0 S3: Qwen3-TTS starts every output with
about 0.45 s of silence and ends it with about 0.5 s).

An energy gate on fixed frames (20 ms by default): a frame is voiced when its RMS reaches
``threshold_dbfs``. The M0 measurement counted a frame voiced within 35 dB of the loudest frame
and above −55 dBFS; a stream cannot know its loudest frame in advance, so the gate is absolute.

- Leading silence: at most ``max_lead_ms`` of it is kept, the frames just before the first
  voiced one. Nothing is held back once a voiced frame has arrived: that frame and the kept
  lead go out at once.
- Silence after a voiced frame is held until the next voiced frame (an inner pause, sent
  whole) or the end of the stream, where at most ``max_tail_ms`` of it is kept.
- A stream with no voiced frame yields nothing.

Only what ``push`` and ``flush`` return is audio; whatever the segment map, truncation, D5 and
the echo reference count is therefore the audio actually sent.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from mlx_realtime.audio.pcm import SAMPLE_WIDTH


class SilenceTrimmer:
    def __init__(
        self,
        sample_rate: int,
        *,
        threshold_dbfs: float,
        frame_ms: int = 20,
        max_lead_ms: int = 60,
        max_tail_ms: int = 100,
    ) -> None:
        self._frame_bytes = max(sample_rate * frame_ms // 1000, 1) * SAMPLE_WIDTH
        self._lead_frames = math.ceil(max_lead_ms / frame_ms) if max_lead_ms > 0 else 0
        self._tail_bytes = sample_rate * max_tail_ms // 1000 * SAMPLE_WIDTH
        # Mean square of a frame at the threshold, in s16 units squared.
        self._threshold = (32768.0 * 10 ** (threshold_dbfs / 20)) ** 2
        self._partial = b""
        self._lead: deque[bytes] = deque(maxlen=self._lead_frames or None)
        self._silence: list[bytes] = []  # silent frames since the last voiced one
        self.voiced = False  # a voiced frame has been seen
        self.trimmed_bytes = 0  # silence dropped so far (for logs and tests)

    def push(self, pcm: bytes) -> bytes:
        """The audio that can be sent now."""
        data = self._partial + pcm
        whole = len(data) - len(data) % self._frame_bytes
        self._partial = data[whole:]
        out: list[bytes] = []
        for start in range(0, whole, self._frame_bytes):
            self._frame(data[start : start + self._frame_bytes], out)
        return b"".join(out)

    def flush(self) -> bytes:
        """The end of the stream: the last partial frame, then at most ``max_tail_ms`` of the
        trailing silence."""
        out: list[bytes] = []
        if self._partial:
            self._frame(self._partial, out)
            self._partial = b""
        if self.voiced:
            tail = b"".join(self._silence)
            kept = tail[: self._tail_bytes - self._tail_bytes % SAMPLE_WIDTH]
            out.append(kept)
            self.trimmed_bytes += len(tail) - len(kept)
        else:
            self.trimmed_bytes += sum(len(frame) for frame in self._lead)
        self._silence.clear()
        self._lead.clear()
        return b"".join(out)

    def _frame(self, frame: bytes, out: list[bytes]) -> None:
        if self._is_voiced(frame):
            if not self.voiced:
                self.voiced = True
                out.extend(self._lead)
                self._lead.clear()
            else:
                out.extend(self._silence)
                self._silence.clear()
            out.append(frame)
        elif not self.voiced:
            if self._lead_frames == 0:
                self.trimmed_bytes += len(frame)
                return
            if len(self._lead) == self._lead_frames:
                self.trimmed_bytes += len(self._lead[0])
            self._lead.append(frame)  # the deque drops the oldest frame
        else:
            self._silence.append(frame)

    def _is_voiced(self, frame: bytes) -> bool:
        samples = np.frombuffer(frame[: len(frame) - len(frame) % SAMPLE_WIDTH], dtype="<i2")
        if samples.size == 0:
            return False
        power = float(np.mean(samples.astype(np.float64) ** 2))
        return power >= self._threshold
