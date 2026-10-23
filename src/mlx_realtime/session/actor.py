"""The session actor: the only code that mutates session state (ARCHITECTURE §4.2, §6).

The actor takes commands and signals from one FIFO inbox and handles each in a synchronous
step: no handler awaits. Barge-in, cancellation and truncation are therefore atomic with
respect to everything else the session does. Work that has to wait (speech detection,
end-of-turn evaluations, ASR requests, response pipelines) runs in tasks the actor owns, which
report back through the inbox.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

from mlx_realtime import logs
from mlx_realtime.audio.pcm import CANONICAL_RATE, SAMPLE_WIDTH, ms_to_samples
from mlx_realtime.config import (
    AecConfig,
    AudioReserveConfig,
    BargeInConfig,
    G711Config,
    LimitsConfig,
    SegmenterConfig,
    SilenceTrimConfig,
    TurnConfig,
)
from mlx_realtime.providers.base import (
    ASRProvider,
    LLMProvider,
    Message,
    PrefixWarmer,
    ProviderError,
    ProviderErrorKind,
    ToolCall,
    ToolChoice,
    ToolDef,
    TTSProvider,
)
from mlx_realtime.providers.voices import VoiceBook
from mlx_realtime.session.clock import Clock
from mlx_realtime.session.commands import (
    AppendAudio,
    CancelResponse,
    ClearAudio,
    Command,
    CommitAudio,
    CreateItem,
    CreateResponse,
    DeleteItem,
    NewMessage,
    RejectEvent,
    ResponseParams,
    TruncateItem,
    UpdateSession,
)
from mlx_realtime.session.conversation import (
    ROOT,
    AssistantMessage,
    AudioMessage,
    Conversation,
    FunctionCall,
    FunctionCallOutput,
    SpokenSegment,
    TextMessage,
)
from mlx_realtime.session.echo import (
    EchoParams,
    EchoReference,
    EchoRuntime,
    EchoStage,
    log_echo_stats,
)
from mlx_realtime.session.errors import ErrorInfo
from mlx_realtime.session.facts import (
    AudioCleared,
    AudioCommitted,
    ErrorRaised,
    Fact,
    FunctionCallDelivered,
    ItemCreated,
    ItemDeleted,
    ItemTruncated,
    MessageClosed,
    MessageOpened,
    OutputAudioDelta,
    OutputTextDelta,
    OutputTranscriptDelta,
    ResponseFinished,
    ResponseStarted,
    SessionStarted,
    SessionUpdated,
    SpeechStarted,
    SpeechStopped,
    TranscriptCompleted,
    TranscriptionFailed,
)
from mlx_realtime.session.feeder import VadFeeder
from mlx_realtime.session.history import build_messages
from mlx_realtime.session.inbox import Inbox
from mlx_realtime.session.input_buffer import CommittedAudio, InputBuffer
from mlx_realtime.session.observer import NULL_OBSERVER, SessionObserver, TurnReport
from mlx_realtime.session.output import Delivery, OutputLedger
from mlx_realtime.session.pipeline import PipelineContext, PipelineRequest, run_pipeline
from mlx_realtime.session.response import Response, ResponseStatus, StatusDetails
from mlx_realtime.session.settings import UNSET, ServerVad, SessionSettings
from mlx_realtime.session.signals import (
    LlmFinished,
    LlmText,
    LlmToolCall,
    PipelineFailed,
    PipelineMark,
    PipelineSignal,
    SegmentAudio,
    SegmentStarted,
    Signal,
    SpeechFinished,
    TranscriptFailed,
    TranscriptReady,
    TurnVerdict,
    VadUpdate,
)
from mlx_realtime.session.timeline import AudioArrivals, TurnTimeline
from mlx_realtime.session.truncation import clamp_audio_end, heard_text
from mlx_realtime.session.turn import (
    EvaluateTurn,
    TurnAction,
    TurnEnd,
    TurnManager,
    TurnParams,
    TurnStart,
    TurnState,
)
from mlx_realtime.text.normalize import NormalizerOptions
from mlx_realtime.text.similarity import overlap, units
from mlx_realtime.vad.base import SpeechDetector, TurnEndDetector
from mlx_realtime.vad.gate import GateEvent, GateParams, LongSilence, SpeechEnd, SpeechStart

logger = logging.getLogger(__name__)

_BACKEND_CODES: dict[ProviderErrorKind, str] = {
    "unavailable": "backend_unavailable",
    "timeout": "backend_timeout",
    "overloaded": "backend_busy",
    "bad_response": "provider_error",
    "internal": "provider_error",
}

_VOICE_LOCKED = "Cannot update a conversation's voice if assistant audio is present."


class IdSource(Protocol):
    """Mints the ids the session hands out; their format is the protocol layer's."""

    def item(self) -> str: ...
    def response(self) -> str: ...
    def call(self) -> str: ...
    def turn(self) -> str: ...


@dataclass(frozen=True, slots=True)
class SessionDeps:
    """What a session uses but does not own."""

    llm: LLMProvider
    asr: ASRProvider
    tts: TTSProvider
    speech: SpeechDetector  # this session's own
    turn_end: TurnEndDetector | None  # None: silence alone ends a turn
    ids: IdSource
    limits: LimitsConfig
    turn: TurnConfig
    segmenter: SegmenterConfig
    normalizer: NormalizerOptions
    max_delta_ms: int
    max_lead_ms: int
    voices: VoiceBook  # D19: unknown names resolve to the default voice
    verbatim_prefixes: Sequence[str]  # D16
    clock: Clock
    aec: AecConfig
    barge_in: BargeInConfig
    echo: EchoRuntime | None  # the canceller's executor and engine (D17), when enabled
    trim_silence: SilenceTrimConfig  # each TTS segment's edge silence (M0 S3)
    reserve: AudioReserveConfig | None = None  # the audio reserve (M6); None: off
    first_segment_grace_s: float = 0.0  # TTS queue order: see PipelineContext (M6)
    observer: SessionObserver = NULL_OBSERVER  # metrics and the debug endpoint (M6)
    g711: G711Config = field(default_factory=G711Config)  # G.711 resampling, both ways (M7)


class SessionActor:
    def __init__(
        self,
        *,
        session_id: str,
        conversation_id: str,
        settings: SessionSettings,
        defaults: SessionSettings,
        deps: SessionDeps,
        inbox: Inbox,
        ledger: OutputLedger,
        emit: Callable[[Fact], None],
        wait_for_room: Callable[[], Awaitable[None]],
    ) -> None:
        self._session_id = session_id
        self._conversation_id = conversation_id
        self._settings = settings
        self._defaults = defaults
        self._deps = deps
        self._ids = deps.ids
        self._inbox = inbox
        self._ledger = ledger
        self._emit = emit
        self._clock = deps.clock
        self._conversation = Conversation()
        # D17: the client's modelled playout of our audio, for the cancellers' reference and
        # the barge-in rules; the cancellers themselves when enabled (G7 dual path: AEC3 for
        # detection, the ASR path's per aec.asr_input).
        self._echo: EchoReference | None = None
        stage: EchoStage | None = None
        if deps.echo is not None or deps.barge_in.active:
            self._echo = EchoReference(
                EchoParams.of(deps.aec, deps.barge_in), deps.clock, deps.g711.input_quality
            )
            ledger.listener = self._echo
            if deps.echo is not None:
                stage = EchoStage(
                    deps.echo.engine(), self._echo, deps.echo.executor, deps.echo.asr_path()
                )
        self._input = InputBuffer(
            deps.limits.input_audio_buffer_max_ms,
            processed=stage is not None,
            asr_input=deps.aec.asr_input,
            input_format=settings.input_format,
            g711_quality=deps.g711.input_quality,
        )
        # Audio the detector has not seen is never trimmed from the input buffer, so the
        # buffer's bound is the backlog's too: a client may send faster than real time.
        self._feeder = VadFeeder(
            deps.speech,
            inbox.put_signal,
            max_queued_ms=deps.limits.input_audio_buffer_max_ms,
            reference=self._echo,
            echo=stage,
            clock=deps.clock,
        )
        self._barge_ins: set[str] = set()  # turns that interrupted a response
        self._detected: set[str] = set()  # turns the VAD found, until their transcript
        self._spoken: dict[str, AssistantMessage] = {}  # recent audio answers, for echo turns
        self._echo_counts = {"barge_ins": 0, "barge_ins_empty_or_echo": 0, "echo_turns": 0}
        self._vad_epoch = 0  # bumped by every commit, clear or turn-detection change
        self._turns = self._new_turn_manager()
        self._feeder.configure(self._vad_epoch, self._gate_params())
        self._turn_item_id = ""  # the item a detected speech turn will commit
        self._turn_start_ms = 0
        self._speech_stopped_at = 0.0  # when the gate last reported the end of speech
        self._arrivals = AudioArrivals()
        self._timelines: dict[str, TurnTimeline] = {}  # by user audio item, until logged
        self._response: Response | None = None
        self._generation = 0
        self._voice_locked = False  # GA: no voice change once the session has spoken
        self._tasks: asyncio.TaskGroup | None = None
        self._warmup: asyncio.Task[None] | None = None  # the LLM prefix warm-up (M0 S2)
        self._warm_pending = False  # the instructions changed while this session's LLM ran
        self._pipeline_context = PipelineContext(
            llm=deps.llm,
            tts=deps.tts,
            segmenter=deps.segmenter,
            normalizer=deps.normalizer,
            max_delta_ms=deps.max_delta_ms,
            max_lead_ms=deps.max_lead_ms,
            max_audio_ms=deps.limits.max_response_audio_ms,
            post=inbox.put_signal,
            wait_for_room=wait_for_room,
            clock=deps.clock,
            trim_silence=deps.trim_silence,
            reserve=deps.reserve,
            first_segment_grace_s=deps.first_segment_grace_s,
            observer=deps.observer,
            g711_quality=deps.g711.output_quality,
        )

    async def run(self) -> None:
        try:
            async with asyncio.TaskGroup() as tasks:
                self._tasks = tasks
                tasks.create_task(self._feeder.run())
                self._emit(SessionStarted(self._session_id, self._conversation_id, self._settings))
                while True:
                    self._handle(await self._inbox.get())
        finally:
            if self._echo is not None:
                log_echo_stats(self._echo, self._echo_counts)

    def _handle(self, message: Command | Signal) -> None:
        match message:
            case AppendAudio():
                self._append(message)
            case CommitAudio(event_id=event_id):
                self._commit_buffer(event_id)
            case ClearAudio():
                self._clear_buffer()
            case UpdateSession():
                self._update_session(message)
            case CreateItem():
                self._create_item(message)
            case TruncateItem():
                self._truncate(message)
            case DeleteItem():
                self._delete(message)
            case CreateResponse():
                self._create_response(message)
            case CancelResponse():
                self._cancel_response(message)
            case RejectEvent(error=error):
                self._emit(ErrorRaised(error))
            case VadUpdate():
                self._vad_update(message)
            case TurnVerdict():
                self._turn_verdict(message)
            case TranscriptReady():
                self._transcript_ready(message)
            case TranscriptFailed():
                self._transcript_failed(message)
            case _:
                response = self._response
                if response is not None and message.generation == response.generation:
                    self._pipeline_signal(response, message)

    # Input audio and turn detection

    def _append(self, command: AppendAudio) -> None:
        audio_format = self._input.input_format
        if len(command.audio) % audio_format.sample_width:
            message = (
                "Invalid value for 'audio': audio/pcm needs whole 16-bit samples "
                "(an even byte count)."
            )
            self._reject("invalid_value", message, command.event_id, "audio")
            return
        self._input_arrived(*self._input.append(command.audio))

    def _input_arrived(self, start: int, canonical: bytes) -> None:
        """Canonical audio from ``start`` on the input timeline: to the detectors."""
        self._arrivals.record(self._input.end_ms, self._now())
        if self._echo is not None:
            self._echo.capture_arrived(start + len(canonical) // SAMPLE_WIDTH)
        if self._turns is not None and canonical:
            self._feeder.push(start, canonical)

    def _vad_update(self, update: VadUpdate) -> None:
        if update.processed:
            self._input.apply_processed(update.processed_start, update.processed)
        if update.processed_asr:
            self._input.apply_asr(update.processed_start, update.processed_asr)
        vad, turns = self._settings.turn_detection, self._turns
        if update.epoch != self._vad_epoch or vad is None or turns is None:
            return
        for event in update.events:
            for action in self._gate_event(turns, event):
                self._turn_action(action, vad)
        if turns.state is TurnState.IDLE:
            self._input.trim_before(update.keep_from_ms - vad.prefix_padding_ms)

    def _gate_event(self, turns: TurnManager, event: GateEvent) -> list[TurnAction]:
        match event:
            case SpeechStart(onset_ms=onset_ms):
                return turns.on_speech_start(onset_ms)
            case SpeechEnd(end_ms=end_ms, detected_ms=detected_ms):
                self._speech_stopped_at = self._now()
                return turns.on_speech_end(end_ms, detected_ms)
            case LongSilence():
                return turns.on_long_silence()

    def _turn_verdict(self, verdict: TurnVerdict) -> None:
        vad, turns = self._settings.turn_detection, self._turns
        if verdict.epoch != self._vad_epoch or vad is None or turns is None:
            return
        for action in turns.on_verdict(verdict.evaluation, verdict.probability):
            self._turn_action(action, vad)

    def _turn_action(self, action: TurnAction, vad: ServerVad) -> None:
        match action:
            case TurnStart(onset_ms=onset_ms):
                self._turn_item_id = self._ids.item()
                self._turn_start_ms = max(self._input.start_ms, onset_ms - vad.prefix_padding_ms)
                self._emit(SpeechStarted(self._turn_item_id, self._turn_start_ms))
                if self._response is not None and vad.interrupt_response:
                    self._barge_ins.add(self._turn_item_id)
                    self._echo_counts["barge_ins"] += 1
                    self._deps.observer.barge_in()
                    self._stop_response(self._response, StatusDetails("cancelled", "turn_detected"))
            case EvaluateTurn(evaluation=evaluation, start_ms=start_ms, end_ms=end_ms):
                audio = self._input.slice(start_ms, end_ms)
                self._spawn(self._evaluate_turn(self._vad_epoch, evaluation, audio))
            case TurnEnd(audio_end_ms=audio_end_ms):
                self._emit(SpeechStopped(self._turn_item_id, audio_end_ms))
                audio = self._input.commit_span(self._turn_start_ms, audio_end_ms)
                marks = {"speech_stopped": self._speech_stopped_at, "turn_confirmed": self._now()}
                if (speech_end := self._arrivals.time_of(audio_end_ms)) is not None:
                    marks["speech_end"] = speech_end
                self._commit(
                    self._turn_item_id,
                    audio,
                    create_response=vad.create_response,
                    marks=marks,
                    detected=True,
                )

    async def _evaluate_turn(self, epoch: int, evaluation: int, pcm: bytes) -> None:
        probability = 1.0
        detector = self._deps.turn_end
        if detector is not None:
            try:
                probability = await detector.end_of_turn_probability(pcm)
            except Exception:
                logger.exception("end-of-turn evaluation failed; the turn ends")
        self._inbox.put_signal(TurnVerdict(epoch, evaluation, probability))

    def _commit_buffer(self, event_id: str | None) -> None:
        audio = self._input.commit()
        if audio is None:
            self._reject(
                "invalid_event",
                "Error committing input audio buffer: the buffer is empty.",
                event_id,
            )
            return
        self._restart_turn_detection()  # a speech turn in progress ends with this commit
        self._commit(
            self._ids.item(), audio, create_response=False, marks={"committed": self._now()}
        )

    def _commit(
        self,
        item_id: str,
        audio: CommittedAudio,
        *,
        create_response: bool,
        marks: dict[str, float],
        detected: bool = False,
    ) -> None:
        """``detected``: a speech turn the VAD found (only those can be echo turns)."""
        transcription = self._settings.transcription
        item = AudioMessage(
            item_id,
            audio.audio_ms,
            show_transcript=transcription is not None,
            start_ms=audio.start_ms,
            end_ms=audio.end_ms,
        )
        if detected:
            self._detected.add(item_id)
        previous = self._last_item_id()
        self._conversation.append(item)
        self._emit(AudioCommitted(item, previous))
        timeline = TurnTimeline(self._ids.turn(), item_id)
        for name, at_s in marks.items():
            timeline.mark(name, at_s)
        self._timelines[item_id] = timeline
        language = transcription.language if transcription is not None else None
        with logs.bound(turn=timeline.turn_id, item=item_id):
            self._spawn(self._transcribe(item_id, audio.pcm, language))
        if create_response and self._response is None:
            self._start_response(ResponseParams(), turn_item_id=item_id, voice=self._settings.voice)

    def _clear_buffer(self) -> None:
        self._input.clear()
        self._restart_turn_detection()
        self._emit(AudioCleared())

    def _restart_turn_detection(self) -> None:
        """A new VAD epoch: the turn in progress is dropped, late detector results ignored."""
        self._vad_epoch += 1
        if self._turns is not None:
            self._turns.reset()
        self._feeder.configure(self._vad_epoch, self._gate_params())

    async def _transcribe(self, item_id: str, pcm: bytes, language: str | None) -> None:
        try:
            transcript = await self._deps.asr.transcribe(pcm, language=language)
        except ProviderError as error:
            self._inbox.put_signal(TranscriptFailed(item_id, error))
        except Exception:
            logger.exception("transcription failed")
            self._inbox.put_signal(
                TranscriptFailed(item_id, ProviderError("internal", "internal error"))
            )
        else:
            self._inbox.put_signal(TranscriptReady(item_id, transcript))

    def _transcript_ready(self, signal: TranscriptReady) -> None:
        self._mark(signal.item_id, "asr_done")
        detected = signal.item_id in self._detected
        self._detected.discard(signal.item_id)
        item = self._conversation.get(signal.item_id)
        if not isinstance(item, AudioMessage):
            self._timelines.pop(signal.item_id, None)
            self._barge_ins.discard(signal.item_id)
            return  # deleted while it was being transcribed
        item.transcript = signal.transcript.text
        item.status = "completed"
        logger.debug("transcript", extra={"item": item.id, "transcript": item.transcript})
        if detected and self._is_echo(item):
            item.echo = True
            self._echo_counts["echo_turns"] += 1
            self._deps.observer.echo_turn()
            logger.info("a turn matched our own speech: treated as noise", extra={"item": item.id})
        if item.id in self._barge_ins:
            self._barge_ins.discard(item.id)
            if item.echo or not item.transcript.strip():
                self._echo_counts["barge_ins_empty_or_echo"] += 1
                self._deps.observer.barge_in_empty_or_echo()
                logger.info("a barge-in turned out empty or echo", extra={"item": item.id})
        self._emit(TranscriptCompleted(item, self._conversation.previous_id(item.id)))
        self._log_timeline_unless_answered(item.id)
        if self._response is not None and self._response.awaiting_transcript:
            self._advance(self._response)

    def _is_echo(self, item: AudioMessage) -> bool:
        """Echo-turn suppression (D17): the transcript closely matches assistant speech the
        client was playing around the turn."""
        echo, config = self._echo, self._deps.barge_in
        transcript = item.transcript or ""
        if (
            echo is None
            or not config.echo_turn_suppression
            or len(units(transcript)) < config.echo_match_min_units
        ):
            return False
        spans = echo.played_spans(
            ms_to_samples(item.start_ms, CANONICAL_RATE),
            ms_to_samples(item.end_ms, CANONICAL_RATE),
            ms_to_samples(config.echo_match_window_ms, CANONICAL_RATE),
        )
        played = ""
        for span in spans:
            message = self._spoken.get(span.item_id)
            if message is None:
                continue
            played += "".join(
                segment.text
                for segment in message.segments
                if segment.start_ms < span.end_ms
                and segment.start_ms + segment.audio_ms > span.start_ms
            )
        return bool(played) and overlap(transcript, played) >= config.echo_match_threshold

    def _transcript_failed(self, signal: TranscriptFailed) -> None:
        self._detected.discard(signal.item_id)
        self._barge_ins.discard(signal.item_id)
        item = self._conversation.get(signal.item_id)
        if not isinstance(item, AudioMessage):
            self._timelines.pop(signal.item_id, None)
            return
        item.status = "completed"
        self._deps.observer.provider_error("asr", signal.error.kind)
        logger.warning(
            "transcription failed",
            extra={"item": item.id, "kind": signal.error.kind, "error": str(signal.error)},
        )
        error = _backend_error("asr", signal.error)
        self._emit(TranscriptionFailed(item, self._conversation.previous_id(item.id), error))
        self._log_timeline_unless_answered(item.id)
        response = self._response
        if response is None or not response.awaiting_transcript:
            return
        if response.turn_item_id == item.id:
            self._stop_response(response, StatusDetails("failed", error=error), error)
        else:
            self._advance(response)

    def _new_turn_manager(self) -> TurnManager | None:
        vad = self._settings.turn_detection
        if vad is None:
            return None
        config = self._deps.turn
        return TurnManager(
            TurnParams(
                smart_turn=self._deps.turn_end is not None,
                smart_turn_threshold=config.smart_turn.threshold,
                window_ms=config.smart_turn.window_s * 1000,
                lead_ms=vad.prefix_padding_ms,
            )
        )

    def _gate_params(self) -> GateParams | None:
        vad = self._settings.turn_detection
        if vad is None:
            return None
        config = self._deps.turn
        smart_turn = self._deps.turn_end is not None
        return GateParams(
            threshold=vad.threshold,
            min_speech_ms=config.vad.min_speech_ms,
            silence_duration_ms=vad.silence_duration_ms,
            max_silence_ms=config.smart_turn.max_silence_ms if smart_turn else None,
            min_speech_ms_while_playing=self._deps.barge_in.min_speech_ms_while_playing,
        )

    # Turn timeline

    def _mark(self, item_id: str | None, name: str) -> None:
        timeline = None if item_id is None else self._timelines.get(item_id)
        if timeline is not None:
            timeline.mark(name, self._now())

    def _log_timeline(self, item_id: str | None) -> None:
        timeline = None if item_id is None else self._timelines.pop(item_id, None)
        if timeline is not None:
            timeline.log()
            self._deps.observer.turn_finished(TurnReport(self._session_id, timeline))

    def _log_timeline_unless_answered(self, item_id: str) -> None:
        """A turn no response answers ends with its transcript; the others with their response."""
        if self._response is None or self._response.turn_item_id != item_id:
            self._log_timeline(item_id)

    # Session and conversation

    def _update_session(self, command: UpdateSession) -> None:
        patch = command.patch
        if patch.voice is not UNSET:
            voice = self._effective_voice(patch.voice)
            if self._voice_locked and voice != self._settings.voice:
                param = "session.audio.output.voice"
                self._reject("cannot_update_voice", _VOICE_LOCKED, command.event_id, param)
                if patch.input_format is not UNSET:
                    self._inbox.input_format_settled(self._settings.input_format)
                return
            patch = replace(patch, voice=voice)
        previous = self._settings
        previous_vad = self._settings.turn_detection
        self._settings = self._settings.apply(patch, self._defaults)
        if (previous.instructions, previous.tools) != (
            self._settings.instructions,
            self._settings.tools,
        ):
            self._warm_prefix()
        if patch.input_format is not UNSET:
            # Later appends decode in the new format (GA allows the change at any time); what
            # the old decoder still holds goes onto the timeline first.
            start, tail = self._input.set_format(self._settings.input_format)
            if tail:
                self._input_arrived(start, tail)
            self._inbox.input_format_settled(self._settings.input_format)
        if self._settings.turn_detection != previous_vad:
            self._turns = self._new_turn_manager()
            self._restart_turn_detection()
        self._emit(SessionUpdated(self._session_id, self._settings))

    def _warm_prefix(self) -> None:
        """The instructions or tools changed (M0 S2): put the next LLM request's prefix in the
        backend's prompt cache now, so the response that needs it does not pay the prefill.
        A response created meanwhile waits for the warm-up in the provider. While this
        session's LLM is generating, the warm-up starts when it finishes."""
        llm, settings, response = self._deps.llm, self._settings, self._response
        if not isinstance(llm, PrefixWarmer) or not (settings.instructions or settings.tools):
            self._warm_pending = False
            return
        if response is not None and response.task is not None and not response.llm_done:
            self._warm_pending = True
            return
        self._warm_pending = False
        if self._warmup is not None:
            self._warmup.cancel()  # an older prefix nobody will ask for
        messages = build_messages(self._conversation, settings.instructions)
        warm = _warm(llm, messages, settings.tools, settings.tool_choice)
        self._warmup = self._spawn(warm)

    def _create_item(self, command: CreateItem) -> None:
        spec = command.item
        if spec.id is not None and spec.id in self._conversation:
            self._reject(
                "invalid_value", f"Item '{spec.id}' already exists.", command.event_id, "item.id"
            )
            return
        after = command.previous_item_id
        if after is not None and after != ROOT and after not in self._conversation:
            message = f"Item '{after}' does not exist."
            self._reject("invalid_value", message, command.event_id, "previous_item_id")
            return
        item_id = spec.id or self._ids.item()
        if isinstance(spec, NewMessage):
            item: TextMessage | FunctionCallOutput = TextMessage(
                item_id, spec.role, list(spec.texts)
            )
        else:
            if self._conversation.find_call(spec.call_id) is None:
                message = f"No function call with call_id '{spec.call_id}' exists."
                self._reject("invalid_value", message, command.event_id, "item.call_id")
                return
            item = FunctionCallOutput(item_id, spec.call_id, spec.output)
        if after is None:
            previous = self._last_item_id()
            self._conversation.append(item)
        else:
            self._conversation.insert_after(item, after)
            previous = None if after == ROOT else after
        self._emit(ItemCreated(item, previous))

    def _truncate(self, command: TruncateItem) -> None:
        item = self._conversation.get(command.item_id)
        if item is None:
            self._item_not_found(command.item_id, command.event_id)
            return
        if not isinstance(item, AssistantMessage) or item.modality != "audio":
            message = "Only assistant messages with audio can be truncated."
            self._reject("invalid_value", message, command.event_id, "item_id")
            return
        if command.content_index != 0:
            message = "Invalid content_index: assistant messages have one content part."
            self._reject("invalid_value", message, command.event_id, "content_index")
            return
        sent_ms = self._ledger.sent_ms(item.id)
        audio_end_ms = clamp_audio_end(command.audio_end_ms, sent_ms)
        if audio_end_ms < command.audio_end_ms:
            logger.warning(
                "truncate beyond the audio sent; clamped",
                extra={"item": item.id, "audio_end_ms": command.audio_end_ms, "sent_ms": sent_ms},
            )
        item.truncated_text = heard_text(item.segments, audio_end_ms)
        if self._echo is not None:
            self._echo.truncated(item.id, audio_end_ms)  # the client stopped playing it there
        self._emit(ItemTruncated(item.id, audio_end_ms))

    def _delete(self, command: DeleteItem) -> None:
        if self._conversation.remove(command.item_id) is None:
            self._item_not_found(command.item_id, command.event_id)
            return
        self._emit(ItemDeleted(command.item_id))
        if self._response is not None and self._response.awaiting_transcript:
            self._advance(self._response)

    # Responses

    def _create_response(self, command: CreateResponse) -> None:
        active = self._response
        if active is not None:
            message = (
                f"Conversation already has an active response in progress: {active.id}. "
                "Wait until the response is finished before creating a new one."
            )
            self._reject("conversation_already_has_active_response", message, command.event_id)
            return
        voice = self._settings.voice
        if command.params.voice is not None:
            voice = self._effective_voice(command.params.voice)
            if self._voice_locked and voice != self._settings.voice:
                param = "response.audio.output.voice"
                self._reject("cannot_update_voice", _VOICE_LOCKED, command.event_id, param)
                return
        last = self._conversation.last()
        turn_item_id = last.id if isinstance(last, AudioMessage) else None
        self._start_response(command.params, turn_item_id=turn_item_id, voice=voice)

    def _cancel_response(self, command: CancelResponse) -> None:
        response = self._response
        if response is None or command.response_id not in (None, response.id):
            param = None if command.response_id is None else "response_id"
            message = "Cancellation failed: no active response found."
            self._reject("response_cancel_not_active", message, command.event_id, param)
            return
        self._stop_response(response, StatusDetails("cancelled", "client_cancelled"))

    def _start_response(
        self, params: ResponseParams, *, turn_item_id: str | None, voice: str
    ) -> None:
        settings = self._settings
        self._generation += 1
        modality = params.output_modality or settings.output_modality
        verbatim = self._verbatim_line(params.instructions)
        if verbatim is not None:
            turn_item_id = None  # a line answers no turn: no transcript to wait for (D16)
        timeline = None if turn_item_id is None else self._timelines.get(turn_item_id)
        if timeline is None:
            timeline = TurnTimeline(self._ids.turn(), None)
        timeline.mark("response_created", self._now())
        if verbatim is not None:
            timeline.note("verbatim", True)
        response = Response(
            id=self._ids.response(),
            generation=self._generation,
            conversation_id=self._conversation_id,
            modality=modality,
            voice=voice,
            instructions=settings.instructions
            if params.instructions is None
            else params.instructions,
            max_output_tokens=(
                settings.max_output_tokens
                if params.max_output_tokens is UNSET
                else params.max_output_tokens
            ),
            tool_choice=params.tool_choice or settings.tool_choice,
            metadata=params.metadata,
            turn_item_id=turn_item_id,
            verbatim=verbatim,
            output_format=params.output_format or settings.output_format,
            timeline=timeline,
            tts_done=modality == "text",
        )
        self._response = response
        if self._echo is not None:
            self._echo.response_started()
        self._emit(ResponseStarted(response))
        self._advance(response)

    def _verbatim_line(self, instructions: str | None) -> str | None:
        """The line to say as written, when the response's own instructions start with a
        configured direction (D16)."""
        if instructions is None:
            return None
        for prefix in self._deps.verbatim_prefixes:
            if instructions.startswith(prefix):
                return instructions[len(prefix) :].strip()
        return None

    def _advance(self, response: Response) -> None:
        """Start the pipeline once every transcript is final (or finish a noise turn, D7). A
        verbatim line starts at once: it needs neither the conversation nor the LLM."""
        if response.task is not None:
            return
        if response.verbatim is None:
            response.awaiting_transcript = self._conversation.pending_transcripts()
            if response.awaiting_transcript:
                return
            turn = (
                None
                if response.turn_item_id is None
                else self._conversation.get(response.turn_item_id)
            )
            noise = isinstance(turn, AudioMessage) and not (turn.transcript or "").strip()
            if noise or (isinstance(turn, AudioMessage) and turn.echo):
                self._close_response(response, "completed", None)  # noise or echo (D7, D17)
                return
        messages = (
            []
            if response.verbatim is not None
            else build_messages(self._conversation, response.instructions)
        )
        request = PipelineRequest(
            generation=response.generation,
            messages=messages,
            tools=self._settings.tools,
            tool_choice=response.tool_choice,
            max_output_tokens=response.max_output_tokens,
            modality=response.modality,
            voice=response.voice,
            verbatim=response.verbatim,
            output_format=response.output_format,
        )
        with logs.bound(response=response.id):
            response.task = self._spawn(run_pipeline(request, self._pipeline_context))

    def _pipeline_signal(self, response: Response, signal: PipelineSignal) -> None:
        match signal:
            case LlmText(text=text):
                if response.verbatim is None:
                    self._mark_response(response, "llm_first_token")
                message = response.message or self._open_message(response)
                if response.modality == "text":
                    message.text += text
                    self._emit(OutputTextDelta(response, message, text))
            case LlmToolCall(call=call):
                self._mark_response(response, "llm_first_token")
                self._deliver_call(response, call)
            case LlmFinished(done=done):
                response.llm_done = True
                response.usage = done.usage
                response.finish_reason = done.finish_reason
                if self._warm_pending:
                    self._warm_prefix()
                self._finish_if_done(response)
            case SegmentStarted():
                self._segment_started(response, signal)
            case SegmentAudio():
                self._segment_audio(response, signal)
            case SpeechFinished():
                response.tts_done = True
                self._finish_if_done(response)
            case PipelineFailed(stage=stage, error=error):
                self._deps.observer.provider_error(stage, error.kind)
                logger.warning(
                    "response failed",
                    extra={"stage": stage, "kind": error.kind, "error": str(error)},
                )
                info = _backend_error(stage, error)
                self._stop_response(response, StatusDetails("failed", error=info), info)
            case PipelineMark(name=name, at_s=at_s, values=values):
                if response.timeline is not None:
                    response.timeline.mark(name, at_s)
                    for key, value in values:
                        response.timeline.note(key, value)

    def _open_message(self, response: Response) -> AssistantMessage:
        message = AssistantMessage(
            self._ids.item(), response.modality, audio_format=response.output_format
        )
        previous = self._last_item_id()
        self._conversation.append(message)
        if message.modality == "audio" and self._echo is not None:
            self._spoken[message.id] = message  # kept after a delete: its echo may still come
            while len(self._spoken) > 16:
                del self._spoken[next(iter(self._spoken))]
        response.output.append(message)
        response.message = message
        self._emit(MessageOpened(response, message, previous))
        return message

    def _deliver_call(self, response: Response, call: ToolCall) -> None:
        item = FunctionCall(
            self._ids.item(), self._ids.call(), call.call_id, call.name, call.arguments
        )
        previous = self._last_item_id()
        self._conversation.append(item)
        response.output.append(item)
        self._emit(FunctionCallDelivered(response, item, previous))

    def _segment_started(self, response: Response, signal: SegmentStarted) -> None:
        message = response.message or self._open_message(response)
        segment = SpokenSegment(
            signal.text,
            signal.spoken,
            start_ms=message.audio_ms,
            source=signal.source,
            audio_format=message.audio_format,
        )
        message.segments.append(segment)
        message.text += signal.text
        delivery = Delivery(response.generation, message.id, signal.index, "transcript")
        self._emit(OutputTranscriptDelta(response, message, signal.text, delivery))

    def _segment_audio(self, response: Response, signal: SegmentAudio) -> None:
        message = response.message
        if message is None:
            return
        segment = message.segments[signal.index]
        segment.audio_bytes += len(signal.pcm)
        if signal.silence:
            segment.silence_bytes += len(signal.pcm)
        delivery = Delivery(
            response.generation,
            message.id,
            signal.index,
            "audio",
            len(signal.pcm),
            signal.pcm,
            message.audio_format,
        )
        self._emit(OutputAudioDelta(response, message, signal.pcm, delivery))
        if self._echo is not None:
            self._echo.audio_queued(response.generation)
        if not response.audio_started:
            response.audio_started = True
            self._voice_locked = True
            self._mark_response(response, "first_audio_sent")
        if message.audio_ms > self._deps.limits.max_response_audio_ms:
            self._stop_response(response, StatusDetails("incomplete", "max_output_tokens"))

    def _finish_if_done(self, response: Response) -> None:
        if not (response.llm_done and response.tts_done):
            return
        match response.finish_reason:
            case "length":
                self._close_response(
                    response, "incomplete", StatusDetails("incomplete", "max_output_tokens")
                )
            case "content_filter":
                self._close_response(
                    response, "incomplete", StatusDetails("incomplete", "content_filter")
                )
            case _:
                self._close_response(response, "completed", None)

    def _stop_response(
        self, response: Response, details: StatusDetails, error: ErrorInfo | None = None
    ) -> None:
        """The cancellation handler of ARCHITECTURE §6.5, in one synchronous step.

        The response scope is cancelled, its queued audio and transcript deltas are discarded by
        the writer, and the transcript becomes the text whose deltas reached the client.
        """
        if error is not None:
            self._emit(ErrorRaised(error))
        if response.task is not None:
            response.task.cancel()
        self._ledger.discard(response.generation)
        message = response.message
        if message is not None and message.modality == "audio":
            message.text = "".join(
                segment.text
                for index, segment in enumerate(message.segments)
                if self._ledger.transcript_sent(message.id, index)
            )
            if details.type == "cancelled":
                message.fallback_text = heard_text(message.segments, self._fallback_cut(message))
        if details.type == "cancelled" and self._echo is not None:
            self._echo.flush(response.generation)  # the client drops what it has not played
        self._close_response(response, details.type, details)

    def _fallback_cut(self, message: AssistantMessage) -> int:
        """D5: without a client truncate, the client can have heard at most the audio sent,
        and at most as much as has played since the first audio delta was sent."""
        sent_ms = self._ledger.sent_ms(message.id)
        first = self._ledger.first_audio_at(message.id)
        elapsed_ms = 0 if first is None else max(round((self._now() - first) * 1000), 0)
        return min(sent_ms, elapsed_ms)

    def _close_response(
        self, response: Response, status: ResponseStatus, details: StatusDetails | None
    ) -> None:
        message = response.message
        if message is not None:
            message.status = "completed" if status == "completed" else "incomplete"
            in_conversation = message.id in self._conversation
            previous = self._conversation.previous_id(message.id) if in_conversation else None
            self._emit(MessageClosed(response, message, previous, in_conversation))
        response.status = status
        response.status_details = details
        self._response = None
        if self._warm_pending:
            self._warm_prefix()  # the response ended before its LLM did (cancelled, failed)
        if self._echo is not None:
            self._echo.response_finished(response.generation)
        self._emit(ResponseFinished(response))
        self._deps.observer.response_finished(status)
        if response.turn_item_id is not None:
            self._timelines.pop(response.turn_item_id, None)
        if response.timeline is not None:
            response.timeline.log(response_id=response.id)
            self._deps.observer.turn_finished(
                TurnReport(self._session_id, response.timeline, response.id, status)
            )

    # Helpers

    def _spawn(self, coroutine: Coroutine[object, object, None]) -> asyncio.Task[None]:
        if self._tasks is None:
            raise RuntimeError("the session actor is not running")
        return self._tasks.create_task(coroutine)

    def _now(self) -> float:
        return self._clock.now()

    def _mark_response(self, response: Response, name: str) -> None:
        if response.timeline is not None:
            response.timeline.mark(name, self._now())

    def _effective_voice(self, name: str) -> str:
        """D19: a voice that is not configured is accepted and spoken by the default voice."""
        voice = self._deps.voices.effective(name)
        if voice != name:
            logger.warning(
                "unknown voice; the default voice speaks", extra={"voice": name, "effective": voice}
            )
        return voice

    def _last_item_id(self) -> str | None:
        last = self._conversation.last()
        return None if last is None else last.id

    def _reject(
        self, code: str, message: str, event_id: str | None, param: str | None = None
    ) -> None:
        self._emit(ErrorRaised(ErrorInfo("invalid_request_error", code, message, param, event_id)))

    def _item_not_found(self, item_id: str, event_id: str | None) -> None:
        message = f"Item with item_id '{item_id}' not found."
        self._reject("item_not_found", message, event_id, "item_id")


def _backend_error(stage: str, error: ProviderError) -> ErrorInfo:
    return ErrorInfo("server_error", _BACKEND_CODES[error.kind], f"{stage} failed: {error}")


async def _warm(
    llm: PrefixWarmer,
    messages: Sequence[Message],
    tools: Sequence[ToolDef],
    tool_choice: ToolChoice,
) -> None:
    """A warm-up never fails its session: whatever goes wrong is logged here."""
    try:
        await llm.warm_prefix(messages, tools, tool_choice=tool_choice)
    except Exception:
        logger.exception("LLM prefix warm-up failed")
