"""``python -m mlxrt_audio.serve``: ``mlx_audio.server`` with the ASR ``context`` field working.

The command line is ``mlx_audio.server``'s own, passed through unchanged. Before the server
starts, ``install()`` replaces ``mlx_audio.server._load_model_for_inference`` with a function
that hands every model the STT adapter loads to ``context.with_context``, so a transcription
request's ``context`` reaches Qwen3-ASR as its ``system_prompt``. That is the one hook it
needs: ``STTExecutionAdapter.run_serial`` calls it for each request, reads the model's
``generate`` signature to decide which request fields to pass, and calls ``generate`` with them.
The TTS and speech routes are untouched, so the same entry point can serve either backend.

The patch is written against mlx-audio 0.6.1 and refuses to run on anything else: a different
version, a hook site that no longer reads the model the same way, or a patch that does not
behave as expected stops the backend at start with an error that says what changed.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from types import SimpleNamespace
from typing import Any, cast

from mlxrt_audio import context

SUPPORTED_MLX_AUDIO = "0.6.1"
_MARK = "_mlxrt_context"  # on the replaced loader, so a second install() does nothing

# What the patch relies on, as it reads in mlx-audio 0.6.1's source.
_LOADS_MODEL = "return model_provider.load_model(model_name)"
_ADAPTER_LOADS = "_load_model_for_inference(request.model_name)"
_ADAPTER_READS_SIGNATURE = "inspect.signature(stt_model.generate)"
_ADAPTER_FILTERS = "if key in signature.parameters or key in _STT_EXTRA_KWARGS"
_ADAPTER_CALLS = "stt_model.generate(tmp_path, **gen_kwargs)"
_BUILD_PROMPT_USES = 'system_content = f"{system_prompt}\\n" if system_prompt else ""'


class PatchError(RuntimeError):
    """mlx-audio is not what the patch was written against; the backend must not start."""


def install() -> None:
    """Make ``mlx_audio.server`` hand a request's ``context`` to the models that need it."""
    # mlx-audio ships no type information: its modules are used as Any.
    server: Any = importlib.import_module("mlx_audio.server")
    qwen3_asr: Any = importlib.import_module("mlx_audio.stt.models.qwen3_asr.qwen3_asr")
    version: str = importlib.import_module("mlx_audio.version").__version__

    if getattr(server._load_model_for_inference, _MARK, False):
        return
    problems: list[str] = []
    if version != SUPPORTED_MLX_AUDIO:
        problems.append(
            f"mlx-audio is {version}; the patch was written against {SUPPORTED_MLX_AUDIO}"
        )
    if _LOADS_MODEL not in _source(server._load_model_for_inference):
        problems.append("server._load_model_for_inference no longer returns the loaded model")
    adapter_source = _source(server.STTExecutionAdapter.run_serial)
    for expected, what in (
        (_ADAPTER_LOADS, "loads the model through _load_model_for_inference"),
        (_ADAPTER_READS_SIGNATURE, "reads the model's generate signature"),
        (_ADAPTER_FILTERS, "passes a request field only if generate names it"),
        (_ADAPTER_CALLS, "calls generate with the filtered fields"),
    ):
        if expected not in adapter_source:
            problems.append(f"STTExecutionAdapter.run_serial no longer {what}")
    if "context" not in server.TranscriptionRequest.model_fields:
        problems.append("TranscriptionRequest no longer has a context field")
    if _BUILD_PROMPT_USES not in _source(qwen3_asr.Qwen3ASRModel._build_prompt):
        problems.append(
            "Qwen3-ASR's _build_prompt no longer puts system_prompt in the system block"
        )
    generate_signature = inspect.signature(qwen3_asr.Qwen3ASRModel.generate)
    if not context.applies_to(generate_signature):
        problems.append("Qwen3-ASR's generate no longer takes system_prompt without context")
    if problems:
        raise PatchError(_refusal(problems))

    current = server._load_model_for_inference
    problems.extend(_self_test(generate_signature))
    if problems:
        raise PatchError(_refusal(problems))

    def load_model_for_inference(model_name: str) -> object:
        return context.with_context(current(model_name))

    setattr(load_model_for_inference, _MARK, True)
    server._load_model_for_inference = load_model_for_inference


def _self_test(generate_signature: inspect.Signature) -> list[str]:
    """What the patched ``generate`` of a model with Qwen3-ASR's signature does with ``context``."""
    problems: list[str] = []
    received: dict[str, Any] = {}

    def generate(*args: Any, **kwargs: Any) -> None:
        received.clear()
        received.update(kwargs)

    own = inspect.Signature(list(generate_signature.parameters.values())[1:])  # without self
    cast(Any, generate).__signature__ = own
    probe = SimpleNamespace(generate=generate)
    context.with_context(probe)
    seen = inspect.signature(probe.generate)
    kept = [name for name in seen.parameters if name != context.CONTEXT]
    if context.CONTEXT not in seen.parameters or kept != [*own.parameters]:
        problems.append("the patched generate does not keep Qwen3-ASR's signature plus context")
    probe.generate("clip.wav", language="Chinese", context="NovaNet")
    if received != {"language": "Chinese", context.SYSTEM_PROMPT: "NovaNet"}:
        problems.append(f"context did not reach system_prompt: {received!r}")
    probe.generate("clip.wav", context="NovaNet", system_prompt="own")
    if received != {context.SYSTEM_PROMPT: "own"}:
        problems.append(f"context replaced a system_prompt the call set: {received!r}")
    probe.generate("clip.wav")
    if received:
        problems.append(f"a call without context was changed: {received!r}")
    return problems


def _source(function: object) -> str:
    """The source of ``function``, as installed (mlx-audio ships its .py files)."""
    return inspect.getsource(cast(Any, function))


def _refusal(problems: list[str]) -> str:
    return (
        "mlxrt_audio: refusing to start mlx_audio.server with the ASR context patch:\n"
        + "\n".join(f"  - {problem}" for problem in problems)
        + "\nReview backends/audio/src/mlxrt_audio/ against the installed mlx-audio."
    )


def main() -> None:
    try:
        install()
    except PatchError as error:
        print(error, file=sys.stderr, flush=True)
        raise SystemExit(2) from None
    print(
        f"mlxrt_audio: ASR context field -> system_prompt (mlx-audio {SUPPORTED_MLX_AUDIO})",
        file=sys.stderr,
        flush=True,
    )
    importlib.import_module("mlx_audio.server").main()


if __name__ == "__main__":
    main()
