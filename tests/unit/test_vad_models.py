"""Silero VAD and Smart Turn: the numpy wrappers (state, context, framing, resets, the 8 s
window, features) and the real models on the speech fixtures."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from mlx_realtime.session.turn import (
    EvaluateTurn,
    Framer,
    TurnEnd,
    TurnManager,
    TurnParams,
    TurnStart,
)
from mlx_realtime.vad.base import FRAME_SAMPLES
from mlx_realtime.vad.executors import VadRuntime
from mlx_realtime.vad.gate import GateParams, LongSilence, SpeechEnd, SpeechGate, SpeechStart
from mlx_realtime.vad.onnx import ModelFileMissing, OnnxModel
from mlx_realtime.vad.silero import CONTEXT_SAMPLES, SileroStream
from mlx_realtime.vad.smart_turn import WINDOW_SAMPLES, SmartTurnModel, turn_features, turn_window
from mlx_realtime.vad.whisper_features import compute_whisper_log_mel_features
from tests.speech import EN, SILERO, SMART_TURN, ZH, clip, needs_models, silence

FRAME_BYTES = 2 * FRAME_SAMPLES


class FakeSilero(OnnxModel):
    """Records the inputs; answers scripted probabilities and an incremented state."""

    def __init__(self, probabilities: list[float] | None = None) -> None:
        self.inputs: list[dict[str, npt.NDArray[Any]]] = []
        self._probabilities = probabilities or []

    def run(self, inputs: Mapping[str, npt.NDArray[Any]]) -> list[npt.NDArray[np.float32]]:
        self.inputs.append({name: value.copy() for name, value in inputs.items()})
        probability = self._probabilities.pop(0) if self._probabilities else 0.0
        return [np.array([[probability]], dtype=np.float32), inputs["state"] + 1]


def ramp(start: int) -> bytes:
    return np.arange(start, start + FRAME_SAMPLES, dtype="<i2").tobytes()


def test_silero_carries_context_and_state_between_frames() -> None:
    model = FakeSilero([0.25, 0.75])
    stream = SileroStream(model)
    assert stream.probability(ramp(0), in_speech=False) == 0.25
    assert stream.probability(ramp(1000), in_speech=False) == 0.75
    first, second = model.inputs
    assert first["input"].shape == (1, CONTEXT_SAMPLES + FRAME_SAMPLES)
    assert first["input"].dtype == np.float32
    assert not first["input"][0, :CONTEXT_SAMPLES].any()  # no context before the first frame
    np.testing.assert_array_equal(  # the context is the previous frame's last 64 samples
        second["input"][0, :CONTEXT_SAMPLES], first["input"][0, -CONTEXT_SAMPLES:]
    )
    assert first["input"][0, CONTEXT_SAMPLES + 1] == np.float32(1 / 32768)
    assert first["state"].shape == (2, 1, 128)
    assert (second["state"] == 1).all()  # the state the first call returned
    assert int(second["sr"]) == 16000
    with pytest.raises(ValueError, match="512"):
        stream.probability(bytes(100), in_speech=False)


def frames_until_reset(stream: SileroStream, *, in_speech: bool) -> int | None:
    for index in range(200):
        stream.probability(ramp(0), in_speech=in_speech)
        if stream.resets:
            return index + 1
    return None


def test_the_state_resets_like_pipecat_during_clear_non_speech() -> None:
    # Due after 2 s of audio (62.5 windows of 512 samples), once the VAD is quiet and the last
    # five windows scored below 0.1.
    model = FakeSilero()
    stream = SileroStream(model, reset_interval_s=2.0)
    assert frames_until_reset(stream, in_speech=False) == 63
    stream.probability(ramp(0), in_speech=False)
    latest = model.inputs[-1]  # the reset applies from the next window: state and context
    assert not latest["state"].any()
    assert not latest["input"][0, :CONTEXT_SAMPLES].any()
    assert frames_until_reset(SileroStream(FakeSilero()), in_speech=True) is None
    assert frames_until_reset(SileroStream(FakeSilero([0.5] * 300)), in_speech=False) is None
    never = SileroStream(FakeSilero(), reset_interval_s=None)
    assert frames_until_reset(never, in_speech=False) is None


def test_a_missing_model_file_says_how_to_get_it() -> None:
    with pytest.raises(ModelFileMissing, match="mlxrt models"):
        OnnxModel(SILERO.parent / "nope.onnx")


def test_the_turn_window_is_the_last_8s_front_padded() -> None:
    short = np.full(16000, 1000, dtype="<i2").tobytes()
    window = turn_window(short)
    assert (window.shape, window.dtype) == ((WINDOW_SAMPLES,), np.float32)
    assert not window[: WINDOW_SAMPLES - 16000].any()  # zeros first: the speech sits at the end
    assert window[-1] == np.float32(1000 / 32768)
    long = np.arange(10 * 16000, dtype=np.int64) % 30000
    window = turn_window(long.astype("<i2").tobytes())
    assert window[0] == np.float32(long[2 * 16000] / 32768)  # the first 2 s are dropped
    assert turn_features(short).shape == (1, 80, 800)


@pytest.mark.filterwarnings("ignore")  # transformers warns about missing torch on import
def test_features_match_transformers_whisper_feature_extractor() -> None:
    from transformers import WhisperFeatureExtractor

    window = turn_window(clip(ZH))
    reference = WhisperFeatureExtractor(chunk_length=8)(  # exactly as Smart Turn's inference.py
        window,
        sampling_rate=16000,
        return_tensors="np",
        padding="max_length",
        max_length=8 * 16000,
        truncation=True,
        do_normalize=True,
    ).input_features[0]
    ours = compute_whisper_log_mel_features(window)
    assert ours.shape == reference.shape == (80, 800)
    assert float(np.abs(ours - reference).max()) < 1e-6  # M0 (S7.3): 1.2e-7, float32 rounding


@needs_models
def test_silero_hears_the_fixtures_speech_and_not_silence() -> None:
    stream = SileroStream(OnnxModel(SILERO))
    audio = silence(1000) + clip(ZH) + silence(1000)
    framer = Framer()
    probabilities = [
        stream.probability(frame, in_speech=False) for _, frame in framer.push(0, audio)
    ]
    per_ms = {index * 32: p for index, p in enumerate(probabilities)}
    assert max(p for ms, p in per_ms.items() if ms < 900) < 0.1  # leading silence
    assert np.mean([p for ms, p in per_ms.items() if 1500 <= ms < 4500]) > 0.7  # speech
    assert max(p for ms, p in per_ms.items() if ms >= 5300) < 0.1  # trailing silence


@needs_models
def test_silero_on_the_executor_matches_direct_calls() -> None:
    runtime = VadRuntime(
        silero_model=SILERO,
        reset_interval_s=None,
        smart_turn_model=None,
        workers=1,
        intra_op_threads=1,
    )
    direct = SileroStream(OnnxModel(SILERO), reset_interval_s=None)
    detector = runtime.speech_detector()
    frames = [frame for _, frame in Framer().push(0, clip(EN))]

    async def through_executor() -> list[float]:
        return [
            await detector.speech_probability(frame, start_ms=0, in_speech=False)
            for frame in frames
        ]

    assert asyncio.run(through_executor()) == [
        direct.probability(frame, in_speech=False) for frame in frames
    ]
    assert runtime.turn_end_detector() is None
    asyncio.run(runtime.aclose())


@needs_models
def test_smart_turn_scores_a_finished_sentence_on_every_worker() -> None:
    runtime = VadRuntime(
        silero_model=SILERO,
        reset_interval_s=2.0,
        smart_turn_model=SMART_TURN,
        workers=2,
        intra_op_threads=2,
    )
    detector = runtime.turn_end_detector()
    assert detector is not None
    turn = clip(ZH) + silence(200)
    direct = SmartTurnModel(SMART_TURN, intra_op_threads=1).probability(turn)

    async def evaluate() -> list[float]:
        await runtime.warmup()
        return list(
            await asyncio.gather(*(detector.end_of_turn_probability(turn) for _ in range(4)))
        )

    scores = asyncio.run(evaluate())
    asyncio.run(runtime.aclose())
    assert 0.0 <= direct <= 1.0
    assert all(abs(score - direct) < 1e-5 for score in scores)
    assert direct > 0.5  # a complete read sentence with its trailing silence


def turns_on_real_speech(audio: bytes, *, smart_turn: bool, verdicts: list[float]) -> list[object]:
    """Real Silero frames → speech gate → TurnManager, evaluations answered from ``verdicts``."""
    stream = SileroStream(OnnxModel(SILERO))
    gate = SpeechGate(GateParams(0.5, 150, 200, 3000 if smart_turn else None))
    turns = TurnManager(TurnParams(smart_turn, 0.5, 8000, 300))
    actions: list[object] = []
    for start, frame in Framer().push(0, audio):
        start_ms = start // 16
        probability = stream.probability(frame, in_speech=gate.in_speech)
        for event in gate.on_frame(start_ms, start_ms + 32, probability):
            match event:
                case SpeechStart():
                    pending = turns.on_speech_start(event.onset_ms)
                case SpeechEnd():
                    pending = turns.on_speech_end(event.end_ms, event.detected_ms)
                case LongSilence():
                    pending = turns.on_long_silence()
            while pending:
                action = pending.pop(0)
                actions.append(action)
                if isinstance(action, EvaluateTurn):
                    pending += turns.on_verdict(action.evaluation, verdicts.pop(0))
    return actions


@needs_models
def test_turns_on_real_speech_without_smart_turn() -> None:
    # Speech: 0.82-4.16 s (Chinese), then 5.42-9.10 s (English); a 1.26 s pause between.
    audio = silence(500) + clip(ZH) + silence(700) + clip(EN) + silence(1000)
    actions = turns_on_real_speech(audio, smart_turn=False, verdicts=[])
    assert [type(action).__name__ for action in actions] == ["TurnStart", "TurnEnd"] * 2
    first_start, first_end, second_start, second_end = actions
    assert isinstance(first_start, TurnStart)
    assert 780 <= first_start.onset_ms <= 880
    assert isinstance(first_end, TurnEnd)
    assert 4100 <= first_end.audio_end_ms <= 4220
    assert isinstance(second_start, TurnStart)
    assert 5350 <= second_start.onset_ms <= 5480
    assert isinstance(second_end, TurnEnd)
    assert 9050 <= second_end.audio_end_ms <= 9180


@needs_models
def test_an_incomplete_verdict_joins_both_sentences_into_one_turn() -> None:
    audio = silence(500) + clip(ZH) + silence(700) + clip(EN) + silence(1000)
    actions = turns_on_real_speech(audio, smart_turn=True, verdicts=[0.2, 0.8])
    assert [type(action).__name__ for action in actions] == [
        "TurnStart",
        "EvaluateTurn",  # after the Chinese sentence: incomplete, held
        "EvaluateTurn",  # after the English one: complete
        "TurnEnd",
    ]
    end = actions[-1]
    assert isinstance(end, TurnEnd)
    assert end.audio_end_ms > 9000
