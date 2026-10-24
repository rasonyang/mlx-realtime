"""The CPU executors for speech and end-of-turn detection (ARCHITECTURE §4.2; G3 in M0).

- Silero runs on one single-thread executor shared by every session: one call per 32 ms frame
  and stream, each stream with its own recurrent state.
- Smart Turn runs on an executor of K workers, each with its own ONNX session. Evaluations are
  never batched: in M0 (S7) batching a burst saved 2–13 % per item, while every verdict waited
  for the whole batch.

Feature extraction and inference both run in the worker threads, never on the event loop.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from concurrent.futures import Executor, ThreadPoolExecutor
from functools import partial
from pathlib import Path

from mlx_realtime.vad.base import FRAME_SAMPLES
from mlx_realtime.vad.onnx import OnnxModel
from mlx_realtime.vad.silero import SileroStream
from mlx_realtime.vad.smart_turn import RATE, SmartTurnModel


class SileroDetector:
    """A session's speech detector: its own Silero state, the shared Silero executor."""

    def __init__(self, executor: Executor, stream: SileroStream) -> None:
        self._executor = executor
        self.stream = stream

    async def speech_probability(self, frame: bytes, *, start_ms: int, in_speech: bool) -> float:
        call = partial(self.stream.probability, frame, in_speech=in_speech)
        return await asyncio.get_running_loop().run_in_executor(self._executor, call)

    def reset(self) -> None:
        self.stream.reset_states()


class _WorkerModel(threading.local):
    model: SmartTurnModel | None = None


class SmartTurnDetector:
    """End-of-turn verdicts from the Smart Turn workers; each worker owns one session."""

    def __init__(self, executor: Executor, worker: _WorkerModel) -> None:
        self._executor = executor
        self._worker = worker

    async def end_of_turn_probability(self, pcm: bytes) -> float:
        return await asyncio.get_running_loop().run_in_executor(self._executor, self._run, pcm)

    def _run(self, pcm: bytes) -> float:
        model = self._worker.model
        if model is None:
            raise RuntimeError("a Smart Turn worker without a model")
        return model.probability(pcm)


class VadRuntime:
    """Process-wide: the models and their executors."""

    def __init__(
        self,
        *,
        silero_model: Path,
        reset_interval_s: float | None,
        smart_turn_model: Path | None,
        workers: int,
        intra_op_threads: int,
    ) -> None:
        self._silero = OnnxModel(silero_model)
        self._reset_interval_s = reset_interval_s
        self._silero_executor = ThreadPoolExecutor(1, thread_name_prefix="silero")
        self._turn_executor: ThreadPoolExecutor | None = None
        self._worker = _WorkerModel()
        self._workers = workers
        if smart_turn_model is not None:
            models: queue.SimpleQueue[SmartTurnModel] = queue.SimpleQueue()
            for _ in range(workers):
                models.put(SmartTurnModel(smart_turn_model, intra_op_threads=intra_op_threads))
            self._turn_executor = ThreadPoolExecutor(
                workers,
                thread_name_prefix="smart-turn",
                initializer=self._bind_worker,
                initargs=(models,),
            )

    def _bind_worker(self, models: queue.SimpleQueue[SmartTurnModel]) -> None:
        self._worker.model = models.get_nowait()

    def speech_detector(self) -> SileroDetector:
        stream = SileroStream(self._silero, reset_interval_s=self._reset_interval_s)
        return SileroDetector(self._silero_executor, stream)

    def turn_end_detector(self) -> SmartTurnDetector | None:
        if self._turn_executor is None:
            return None
        return SmartTurnDetector(self._turn_executor, self._worker)

    async def warmup(self) -> None:
        """One inference each: a Silero frame, and a Smart Turn window on every worker."""
        await self.speech_detector().speech_probability(
            bytes(2 * FRAME_SAMPLES), start_ms=0, in_speech=False
        )
        detector = self.turn_end_detector()
        if detector is not None:
            second = bytes(2 * RATE)
            await asyncio.gather(
                *(detector.end_of_turn_probability(second) for _ in range(self._workers))
            )

    async def aclose(self) -> None:
        self._silero_executor.shutdown(wait=False, cancel_futures=True)
        if self._turn_executor is not None:
            self._turn_executor.shutdown(wait=False, cancel_futures=True)
