"""A typed boundary around onnxruntime, which ships no type information.

Sessions run on the CPU execution provider, sequentially, with one inter-op thread: the
settings of the Smart Turn reference code and of pipecat's Silero wrapper. ``run`` releases
the GIL, so the executors of ``vad/executors.py`` keep model time off the event loop.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime  # pyright: ignore[reportMissingTypeStubs]

_ort: Any = onnxruntime  # untyped C extension: everything below it is Any


class ModelFileMissing(Exception):
    """A model file is not there: ``mlxrt models`` downloads and verifies them."""


class OnnxModel:
    """One inference session. It is used by one thread at a time."""

    def __init__(self, path: Path, *, intra_op_threads: int = 1, spinning: bool = False) -> None:
        if not path.is_file():
            raise ModelFileMissing(f"{path} is missing; run `mlxrt models` to download it")
        options: Any = _ort.SessionOptions()
        options.execution_mode = _ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = intra_op_threads
        options.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        flag = "1" if spinning else "0"
        options.add_session_config_entry("session.intra_op.allow_spinning", flag)
        options.add_session_config_entry("session.inter_op.allow_spinning", flag)
        self._session: Any = _ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def run(self, inputs: Mapping[str, npt.NDArray[Any]]) -> list[npt.NDArray[np.float32]]:
        outputs: list[Any] = self._session.run(None, dict(inputs))
        return [np.asarray(output, dtype=np.float32) for output in outputs]
