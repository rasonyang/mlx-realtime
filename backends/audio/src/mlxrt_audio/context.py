"""The ``context`` form field of ``/v1/audio/transcriptions``, handed to Qwen3-ASR.

``mlx_audio.server`` accepts ``context`` (and ``prompt``) on a transcription request, but its
STT adapter passes a request field to ``model.generate`` only if ``generate``'s signature names
it. Qwen3-ASR's ``generate`` takes ``system_prompt`` (the text of its ``<|im_start|>system``
block, which is how the model is biased toward expected words) and no ``context``, so the field
is dropped without a word.

``with_context(model)`` replaces a loaded model's ``generate`` with a wrapper that takes
``context`` and passes it on as ``system_prompt``, unless the call sets a ``system_prompt``
itself. The wrapper's ``__signature__`` is the original's plus ``context``, so the server's
signature filter keeps every other field it keeps today (``language``, ``max_tokens``,
``stream``, …) and now lets ``context`` through. A model whose ``generate`` already takes
``context``, or has no ``system_prompt`` to put it in, is left alone.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, cast

CONTEXT = "context"
SYSTEM_PROMPT = "system_prompt"
_MARK = "_mlxrt_context"


def applies_to(signature: inspect.Signature) -> bool:
    """``generate`` takes a ``system_prompt`` but no ``context``."""
    return SYSTEM_PROMPT in signature.parameters and CONTEXT not in signature.parameters


def is_wrapped(generate: object) -> bool:
    return getattr(generate, _MARK, False) is True


def wrap_generate(generate: Callable[..., Any]) -> Callable[..., Any]:
    """``generate`` with a ``context`` keyword that fills ``system_prompt`` when it is unset."""
    signature = inspect.signature(generate)

    @functools.wraps(generate)
    def generate_with_context(*args: Any, **kwargs: Any) -> Any:
        context = kwargs.pop(CONTEXT, None)
        if context and not kwargs.get(SYSTEM_PROMPT):
            kwargs[SYSTEM_PROMPT] = context
        return generate(*args, **kwargs)

    parameters = list(signature.parameters.values())
    added = inspect.Parameter(CONTEXT, inspect.Parameter.KEYWORD_ONLY, default=None)
    variadic = [i for i, p in enumerate(parameters) if p.kind is inspect.Parameter.VAR_KEYWORD]
    parameters.insert(variadic[0] if variadic else len(parameters), added)
    cast(Any, generate_with_context).__signature__ = signature.replace(parameters=parameters)
    setattr(generate_with_context, _MARK, True)
    return generate_with_context


def with_context(model: object) -> object:
    """``model`` with its ``generate`` taking ``context`` if it needs the wrapper (idempotent)."""
    generate = cast(Callable[..., Any], getattr(model, "generate", None))
    if not callable(generate) or is_wrapped(generate):
        return model
    if applies_to(inspect.signature(generate)):
        setattr(model, "generate", wrap_generate(generate))  # noqa: B010 (model is any object)
    return model
