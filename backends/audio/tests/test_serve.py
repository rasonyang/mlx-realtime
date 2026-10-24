"""The ASR context patch (mlx-audio 0.6.1): hook sites, signatures and the run path.

No model is loaded. The run path is ``STTExecutionAdapter.run_serial`` itself, with a stand-in
model that has Qwen3-ASR's ``generate`` signature and records what it is called with.
"""

from __future__ import annotations

import inspect
import threading
from types import SimpleNamespace
from typing import Any, cast

import mlx_audio.server as server
import numpy as np
import pytest
from mlx_audio.stt.models.qwen3_asr import qwen3_asr

from mlxrt_audio import context, serve

QWEN3_GENERATE = inspect.signature(qwen3_asr.Qwen3ASRModel.generate)


class QwenLike:
    """A model whose ``generate`` has Qwen3-ASR's signature."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        audio: str,
        *,
        max_tokens: int = 8192,
        language: str | None = None,
        stream: bool = False,
        system_prompt: str | None = None,
        hotwords: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        call = {
            "language": language,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
            "stream": stream,
        }
        self.calls.append(call)
        return {"text": "ok"}


class WithContext:
    """A model that already takes ``context`` (VibeVoice's way)."""

    def generate(
        self, audio: str, *, context: str | None = None, language: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        return {"text": context}


class Plain:
    """A model with neither field (Whisper's way)."""

    def generate(self, audio: str, *, language: str | None = None, **kwargs: Any) -> dict[str, Any]:
        return {"text": "plain"}


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "_load_model_for_inference", server._load_model_for_inference)
    serve.install()


def test_the_installed_version_is_the_supported_one() -> None:
    from mlx_audio.version import __version__

    assert __version__ == serve.SUPPORTED_MLX_AUDIO


def test_the_wrapper_keeps_the_signature_and_adds_context() -> None:
    model = context.with_context(QwenLike())
    parameters = inspect.signature(cast(Any, model).generate).parameters
    original = inspect.signature(QwenLike().generate).parameters
    assert [name for name in parameters if name != "context"] == list(original)
    assert parameters["context"].kind is inspect.Parameter.KEYWORD_ONLY
    assert list(parameters)[-1] == "kwargs"  # context sits before **kwargs


def test_the_real_qwen3_signature_takes_the_wrapper() -> None:
    assert context.applies_to(QWEN3_GENERATE)
    assert serve._self_test(QWEN3_GENERATE) == []


def test_context_becomes_system_prompt() -> None:
    model = QwenLike()
    context.with_context(model)
    cast(Any, model).generate("a.wav", context="NovaNet, eSIM", language="Chinese")
    assert model.calls[0]["system_prompt"] == "NovaNet, eSIM"
    assert model.calls[0]["language"] == "Chinese"


def test_a_system_prompt_the_call_sets_wins() -> None:
    model = QwenLike()
    context.with_context(model)
    cast(Any, model).generate("a.wav", context="NovaNet", system_prompt="mine")
    assert model.calls[0]["system_prompt"] == "mine"


def test_no_context_changes_nothing() -> None:
    model = QwenLike()
    context.with_context(model)
    cast(Any, model).generate("a.wav", context=None)
    cast(Any, model).generate("a.wav")
    assert [call["system_prompt"] for call in model.calls] == [None, None]


def test_wrapping_twice_wraps_once() -> None:
    model = QwenLike()
    context.with_context(context.with_context(model))
    generate = cast(Any, model).generate
    assert generate.__wrapped__.__func__ is QwenLike.generate


@pytest.mark.parametrize("model", [WithContext(), Plain()])
def test_other_models_are_left_alone(model: object) -> None:
    before = cast(Any, model).generate
    context.with_context(model)
    assert cast(Any, model).generate == before


def run_serial(model: object, **fields: Any) -> list[str]:
    """One transcription request through the server's own STT adapter, with ``model`` loaded."""
    emitted: list[str] = []
    request = SimpleNamespace(
        request_id="t",
        model_name="stand-in",
        payload=server.TranscriptionTaskPayload(
            request=server.TranscriptionRequest(model="stand-in", **fields),
            filename="a.wav",
            audio=np.zeros(1600, dtype=np.float32),
            sample_rate=16000,
        ),
        cancel_event=threading.Event(),
        emit_data=emitted.append,
        emit_done=lambda: None,
    )
    original = server.model_provider.load_model
    server.model_provider.load_model = lambda name: model  # type: ignore[assignment]
    try:
        server.STTExecutionAdapter().run_serial(cast(Any, request))
    finally:
        server.model_provider.load_model = original  # type: ignore[assignment]
    return emitted


def test_the_server_passes_context_to_qwen3_as_system_prompt(patched: None) -> None:
    model = QwenLike()
    run_serial(model, context="NovaNet", language="Chinese")
    assert model.calls[0]["system_prompt"] == "NovaNet"
    assert model.calls[0]["language"] == "Chinese"
    assert model.calls[0]["max_tokens"] == 8192  # the server's own defaults still apply


def test_without_the_patch_the_server_drops_context() -> None:
    model = QwenLike()
    run_serial(model, context="NovaNet")
    assert model.calls[0]["system_prompt"] is None


def test_the_server_without_context_sends_no_system_prompt(patched: None) -> None:
    model = QwenLike()
    run_serial(model)
    assert model.calls[0]["system_prompt"] is None


def test_models_with_context_get_it_unchanged(patched: None) -> None:
    emitted = run_serial(WithContext(), context="NovaNet")
    assert '"NovaNet"' in emitted[0]


def test_install_twice_wraps_the_loader_once(patched: None) -> None:
    loader = server._load_model_for_inference
    serve.install()
    assert server._load_model_for_inference is loader


def test_a_different_version_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import mlx_audio.version

    monkeypatch.setattr(mlx_audio.version, "__version__", "0.7.0")
    with pytest.raises(serve.PatchError, match=r"mlx-audio is 0\.7\.0"):
        serve.install()
    monkeypatch.setattr("sys.argv", ["serve"])
    with pytest.raises(SystemExit) as exit_info:
        serve.main()
    assert exit_info.value.code == 2
    assert "refusing to start" in capsys.readouterr().err


def test_a_changed_hook_site_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve, "_ADAPTER_FILTERS", "if key in something_else")
    with pytest.raises(serve.PatchError, match="passes a request field only if generate names it"):
        serve.install()


def test_the_server_honours_max_tokens_with_a_context(patched: None) -> None:
    """The gateway caps the tokens by the audio's length (providers.asr.token_cap); the request's
    ``max_tokens`` has to reach Qwen3-ASR's ``generate`` next to the context."""
    model = QwenLike()
    run_serial(model, context="NovaNet", language="Chinese", max_tokens=46)
    assert model.calls[0]["max_tokens"] == 46
    assert model.calls[0]["system_prompt"] == "NovaNet"
