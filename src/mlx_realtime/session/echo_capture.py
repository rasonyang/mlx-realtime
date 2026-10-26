"""Opt-in capture of one session's echo path for offline analysis (``aec.capture_dir``; dev only).

A realtime session whose cancellers run writes
``<capture_dir>/<UTC start to the ms>_<session id>/``:

- mono int16 WAVs at 16 kHz, all on the session's input timeline and sample-aligned to each
  other (sample i of every file is input sample i): ``input.wav`` (as received), ``ref_aec3.wav``
  (the modelled reference of AEC3, before the bypass silences it), ``aec3.wav`` (AEC3's output),
  ``ref_speex.wav`` (the ASR path's reference at its own shift; zeros until it is aligned) and
  ``asr.wav`` (the ASR path's output; the input where the ASR hears it as received);
- ``events.jsonl``: one JSON object per line, ``t`` its time in input-timeline ms (the input
  position being received when it happened), ``kind`` and the kind's fields;
- ``meta.json``: session id, start, formats, the ``aec`` and ``barge_in`` configuration and the
  final echo summary.

Nothing here may hurt the session. The event loop only queues (bounded) work for a writer
thread; the first I/O error, or a writer that falls behind, logs one warning and stops the
capture. The files hold the caller's voice and transcripts.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

import numpy as np
import numpy.typing as npt

from mlx_realtime.audio.pcm import CANONICAL_RATE, SAMPLE_WIDTH

logger = logging.getLogger(__name__)

TRACKS = ("input", "ref_aec3", "aec3", "ref_speex", "asr")
_QUEUE_ITEMS = 512  # batches and events waiting for the writer thread
_POLL_S = 0.2

type Pcm = npt.NDArray[np.int16]


@dataclass(frozen=True, slots=True)
class _Audio:
    start: int  # input sample
    tracks: tuple[bytes, ...]  # int16 little-endian, one per name in TRACKS


@dataclass(frozen=True, slots=True)
class _Event:
    line: str


class EchoCapture:
    """One session's capture. ``audio`` and ``event`` are called on the event loop and never
    block; ``close`` ends the capture (the writer thread finishes the files)."""

    def __init__(
        self,
        root: str | Path,
        session_id: str,
        *,
        max_s: int,
        meta: Mapping[str, object],
        started: datetime | None = None,
    ) -> None:
        self._started = started or datetime.now(UTC)
        self.directory = (
            Path(root)
            / f"{self._started:%Y%m%dT%H%M%S}{self._started.microsecond // 1000:03d}Z_{session_id}"
        )
        self._max_samples = max_s * CANONICAL_RATE
        self._meta = dict(meta)
        self._meta["session_id"] = session_id
        self._meta["started_utc"] = self._started.isoformat(timespec="milliseconds")
        self._queue: queue.Queue[_Audio | _Event] = queue.Queue(_QUEUE_ITEMS)
        self._active = True  # producers queue work while this holds
        self._failed = False
        self._capped = False
        self._closing = threading.Event()
        self._summary: Mapping[str, object] = {}
        self._submitted = 0  # input samples queued (the cap)
        self._files: dict[str, wave.Wave_write] = {}
        self._events_file: TextIO | None = None
        self._written = 0  # input samples written to every track
        self._first_audio_utc: str | None = None
        self._thread = threading.Thread(target=self._run, name="echo-capture", daemon=True)
        self._thread.start()

    @property
    def active(self) -> bool:
        return self._active

    # Producers (the event loop)

    def audio(
        self,
        start: int,
        capture: Pcm,
        reference: Pcm,
        output: Pcm,
        speex_reference: Pcm | None = None,
        asr: Pcm | None = None,
    ) -> None:
        """Input samples from ``start`` and what the cancellers made of them (equal lengths).
        ``speex_reference`` None: zeros; ``asr`` None: the ASR hears the input as received."""
        if not self._active:
            return
        count = len(capture)
        if start + count > self._max_samples:
            keep = max(self._max_samples - start, 0)
            if keep == 0:
                self._cap()
                return
            capture, reference, output = capture[:keep], reference[:keep], output[:keep]
            speex_reference = None if speex_reference is None else speex_reference[:keep]
            asr = None if asr is None else asr[:keep]
            count = keep
        silence = np.zeros(count, dtype=np.int16)
        tracks = (
            capture,
            reference,
            output,
            silence if speex_reference is None else speex_reference,
            capture if asr is None else asr,
        )
        self._submit(_Audio(start, tuple(track.astype("<i2").tobytes() for track in tracks)))
        self._submitted = max(self._submitted, start + count)
        if self._submitted >= self._max_samples:
            self._cap()

    def event(self, t_ms: float | None, kind: str, **fields: object) -> None:
        """An event at input-timeline ``t_ms`` (None: not known yet)."""
        if not self._active:
            return
        record = {"t": None if t_ms is None else round(t_ms, 1), "kind": kind, **fields}
        try:
            self._submit(_Event(json.dumps(record, default=str)))
        except (TypeError, ValueError) as error:
            self._fail(error)

    def close(self, summary: Mapping[str, object]) -> None:
        """The session is over: ``summary`` goes into ``meta.json``."""
        self._summary = dict(summary)
        self._active = False
        self._closing.set()

    def wait(self, timeout: float = 5.0) -> None:
        """Until the writer thread has finished the files (after ``close``)."""
        self._thread.join(timeout)

    # Internals

    def _submit(self, item: _Audio | _Event) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self._fail(RuntimeError("the writer fell behind"))

    def _cap(self) -> None:
        if not self._capped:
            self._capped = True
            logger.info(
                "echo capture reached its duration cap",
                extra={"directory": str(self.directory), "max_samples": self._max_samples},
            )
        self._active = False

    def _fail(self, error: Exception) -> None:
        if not self._failed:
            self._failed = True
            logger.warning(
                "echo capture stopped: %s", error, extra={"directory": str(self.directory)}
            )
        self._active = False

    # The writer thread

    def _run(self) -> None:
        try:
            while not self._failed:
                try:
                    item = self._queue.get(timeout=_POLL_S)
                except queue.Empty:
                    if self._closing.is_set():
                        break
                    continue
                self._handle(item)
        except Exception as error:  # nothing may reach the session
            self._fail(error)
        finally:
            self._finish()

    def _open(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=False)
        for name in TRACKS:
            file = wave.open(str(self.directory / f"{name}.wav"), "wb")  # noqa: SIM115
            self._files[name] = file
            file.setnchannels(1)
            file.setsampwidth(SAMPLE_WIDTH)
            file.setframerate(CANONICAL_RATE)
        self._events_file = (self.directory / "events.jsonl").open("w", encoding="utf-8")
        self._first_audio_utc = datetime.now(UTC).isoformat(timespec="milliseconds")
        self._write_meta(final=False)

    def _handle(self, item: _Audio | _Event) -> None:
        if self._events_file is None:
            self._open()
        match item:
            case _Event(line=line):
                assert self._events_file is not None
                self._events_file.write(f"{line}\n")
            case _Audio(start=start, tracks=tracks):
                lead = start - self._written
                for name, data in zip(TRACKS, tracks, strict=True):
                    if lead > 0:  # the stage skipped input (a commit or clear): zeros keep time
                        data = bytes(lead * SAMPLE_WIDTH) + data
                    elif lead < 0:
                        data = data[-lead * SAMPLE_WIDTH :]
                    self._files[name].writeframes(data)
                self._written = max(self._written, start + len(tracks[0]) // SAMPLE_WIDTH)

    def _finish(self) -> None:
        try:
            for file in self._files.values():
                file.close()
            if self._events_file is not None:
                self._events_file.close()
                self._write_meta(final=True)
        except OSError as error:
            self._fail(error)

    def _write_meta(self, *, final: bool) -> None:
        meta = {
            **self._meta,
            "first_audio_utc": self._first_audio_utc,
            "sample_rate": CANONICAL_RATE,
            "files": [f"{name}.wav" for name in TRACKS] + ["events.jsonl"],
            "samples": self._written,
            "capped": self._capped,
            "failed": self._failed,
            "complete": final,
            "echo_summary": self._summary,
        }
        text = json.dumps(meta, indent=2, default=str)
        (self.directory / "meta.json").write_text(text + "\n", encoding="utf-8")
