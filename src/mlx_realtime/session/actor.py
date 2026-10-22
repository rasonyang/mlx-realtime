"""The session actor: the only code that mutates session state (ARCHITECTURE §4.2, §6).

The actor takes commands and signals from one FIFO inbox and handles each in a synchronous
step: no handler awaits. Barge-in, cancellation and truncation are therefore atomic with
respect to everything else the session does. Work that has to wait (ASR requests, response
pipelines) runs in tasks the actor owns, which report back through the inbox.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Protocol

from mlx_realtime import logs
from mlx_realtime.audio.pcm import CANONICAL_RATE, samples_to_ms
from mlx_realtime.config import LimitsConfig, SegmenterConfig, TurnConfig
from mlx_realtime.providers.base import (
    ASRProvider,
    LLMProvider,
    ProviderError,
    ProviderErrorKind,
    ToolCall,
    TTSProvider,
)
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
from mlx_realtime.session.history import build_messages
from mlx_realtime.session.inbox import Inbox
from mlx_realtime.session.input_buffer import CommittedAudio, InputBuffer
from mlx_realtime.session.output import Delivery, OutputLedger
from mlx_realtime.session.pipeline import PipelineContext, PipelineRequest, run_pipeline
from mlx_realtime.session.response import Response, ResponseStatus, StatusDetails
from mlx_realtime.session.settings import UNSET, ServerVad, SessionSettings
from mlx_realtime.session.signals import (
    LlmFinished,
    LlmText,
    LlmToolCall,
    PipelineFailed,
    PipelineSignal,
    SegmentAudio,
    SegmentStarted,
    Signal,
    SpeechFinished,
    TranscriptFailed,
    TranscriptReady,
)
from mlx_realtime.session.truncation import clamp_audio_end, heard_text
from mlx_realtime.session.turn import (
    EvaluateTurn,
    Framer,
    SpeechStart,
    TurnAction,
    TurnEnd,
    TurnManager,
    TurnParams,
    TurnState,
)
from mlx_realtime.vad.base import FRAME_MS, SpeechDetector, TurnEndDetector

logger = logging.getLogger(__name__)

_BACKEND_CODES: dict[ProviderErrorKind, str] = {
    "unavailable": "backend_unavailable",
    "timeout": "backend_timeout",
    "overloaded": "backend_busy",
    "bad_response": "provider_error",
    "internal": "provider_error",
}


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
    speech: SpeechDetector
    turn_end: TurnEndDetector | None  # None: silence alone ends a turn
    ids: IdSource
    limits: LimitsConfig
    turn: TurnConfig
    segmenter: SegmenterConfig
    max_delta_ms: int


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
        self._conversation = Conversation()
        self._input = InputBuffer(deps.limits.input_audio_buffer_max_ms)
        self._framer = Framer()
        self._turns = self._new_turn_manager()
        self._turn_item_id = ""  # the item a detected speech turn will commit
        self._turn_start_ms = 0
        self._response: Response | None = None
        self._generation = 0
        self._tasks: asyncio.TaskGroup | None = None
        self._pipeline_context = PipelineContext(
            llm=deps.llm,
            tts=deps.tts,
            segmenter=deps.segmenter,
            max_delta_ms=deps.max_delta_ms,
            post=inbox.put_signal,
            wait_for_room=wait_for_room,
            clock_ms=lambda: asyncio.get_running_loop().time() * 1000,
        )

    async def run(self) -> None:
        async with asyncio.TaskGroup() as tasks:
            self._tasks = tasks
            self._emit(SessionStarted(self._session_id, self._conversation_id, self._settings))
            while True:
                self._handle(await self._inbox.get())

    def _handle(self, message: Command | Signal) -> None:
        match message:
            case AppendAudio(pcm=pcm):
                self._append(pcm)
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
            case TranscriptReady():
                self._transcript_ready(message)
            case TranscriptFailed():
                self._transcript_failed(message)
            case _:
                response = self._response
                if response is not None and message.generation == response.generation:
                    self._pipeline_signal(response, message)

    # Input audio and turn detection

    def _append(self, pcm: bytes) -> None:
        start, canonical = self._input.append(pcm)
        vad = self._settings.turn_detection
        if self._turns is None or vad is None:
            return
        for frame_start, frame in self._framer.push(start, canonical):
            start_ms = samples_to_ms(frame_start, CANONICAL_RATE)
            probability = self._deps.speech.speech_probability(frame, start_ms=start_ms)
            for action in self._turns.on_frame(start_ms, start_ms + FRAME_MS, probability):
                self._turn_action(action, vad)
        if self._turns.state is TurnState.IDLE:
            self._input.trim_before(self._turns.keep_from_ms(vad.prefix_padding_ms))

    def _turn_action(self, action: TurnAction, vad: ServerVad) -> None:
        match action:
            case SpeechStart(onset_ms=onset_ms):
                self._turn_item_id = self._ids.item()
                self._turn_start_ms = max(self._input.start_ms, onset_ms - vad.prefix_padding_ms)
                self._emit(SpeechStarted(self._turn_item_id, self._turn_start_ms))
                if self._response is not None and vad.interrupt_response:
                    self._stop_response(self._response, StatusDetails("cancelled", "turn_detected"))
            case EvaluateTurn(evaluation=evaluation, start_ms=start_ms, end_ms=end_ms):
                detector = self._deps.turn_end
                audio = self._input.slice(start_ms, end_ms)
                probability = 1.0 if detector is None else detector.end_of_turn_probability(audio)
                if self._turns is not None:
                    for follow_up in self._turns.on_verdict(evaluation, probability):
                        self._turn_action(follow_up, vad)
            case TurnEnd(audio_end_ms=audio_end_ms):
                self._emit(SpeechStopped(self._turn_item_id, audio_end_ms))
                audio = self._input.commit_span(self._turn_start_ms, audio_end_ms)
                self._commit(self._turn_item_id, audio, create_response=vad.create_response)

    def _commit_buffer(self, event_id: str | None) -> None:
        audio = self._input.commit()
        if audio is None:
            self._reject(
                "invalid_event",
                "Error committing input audio buffer: the buffer is empty.",
                event_id,
            )
            return
        if self._turns is not None:
            self._turns.reset()  # a speech turn in progress ends with this commit
        self._commit(self._ids.item(), audio, create_response=False)

    def _commit(self, item_id: str, audio: CommittedAudio, *, create_response: bool) -> None:
        transcription = self._settings.transcription
        item = AudioMessage(item_id, audio.audio_ms, show_transcript=transcription is not None)
        previous = self._last_item_id()
        self._conversation.append(item)
        self._emit(AudioCommitted(item, previous))
        language = transcription.language if transcription is not None else None
        with logs.bound(turn=self._ids.turn(), item=item_id):
            self._spawn(self._transcribe(item_id, audio.pcm, language))
        if create_response and self._response is None:
            self._start_response(ResponseParams(), turn_item_id=item_id)

    def _clear_buffer(self) -> None:
        self._input.clear()
        if self._turns is not None:
            self._turns.reset()
        self._emit(AudioCleared())

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
        item = self._conversation.get(signal.item_id)
        if not isinstance(item, AudioMessage):
            return  # deleted while it was being transcribed
        item.transcript = signal.transcript.text
        item.status = "completed"
        logger.debug("transcript", extra={"item": item.id, "transcript": item.transcript})
        self._emit(TranscriptCompleted(item, self._conversation.previous_id(item.id)))
        if self._response is not None and self._response.awaiting_transcript:
            self._advance(self._response)

    def _transcript_failed(self, signal: TranscriptFailed) -> None:
        item = self._conversation.get(signal.item_id)
        if not isinstance(item, AudioMessage):
            return
        item.status = "completed"
        error = _backend_error("asr", signal.error)
        self._emit(TranscriptionFailed(item, self._conversation.previous_id(item.id), error))
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
                threshold=vad.threshold,
                min_speech_ms=config.vad.min_speech_ms,
                silence_duration_ms=vad.silence_duration_ms,
                smart_turn=self._deps.turn_end is not None,
                smart_turn_threshold=config.smart_turn.threshold,
                max_silence_ms=config.smart_turn.max_silence_ms,
                window_ms=config.smart_turn.window_s * 1000,
            )
        )

    # Session and conversation

    def _update_session(self, command: UpdateSession) -> None:
        previous_vad = self._settings.turn_detection
        self._settings = self._settings.apply(command.patch, self._defaults)
        if self._settings.turn_detection != previous_vad:
            self._turns = self._new_turn_manager()
        self._emit(SessionUpdated(self._session_id, self._settings))

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
        last = self._conversation.last()
        turn_item_id = last.id if isinstance(last, AudioMessage) else None
        self._start_response(command.params, turn_item_id=turn_item_id)

    def _cancel_response(self, command: CancelResponse) -> None:
        response = self._response
        if response is None or command.response_id not in (None, response.id):
            param = None if command.response_id is None else "response_id"
            message = "Cancellation failed: no active response found."
            self._reject("response_cancel_not_active", message, command.event_id, param)
            return
        self._stop_response(response, StatusDetails("cancelled", "client_cancelled"))

    def _start_response(self, params: ResponseParams, *, turn_item_id: str | None) -> None:
        settings = self._settings
        self._generation += 1
        modality = params.output_modality or settings.output_modality
        response = Response(
            id=self._ids.response(),
            generation=self._generation,
            conversation_id=self._conversation_id,
            modality=modality,
            voice=params.voice or settings.voice,
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
            tts_done=modality == "text",
        )
        self._response = response
        self._emit(ResponseStarted(response))
        self._advance(response)

    def _advance(self, response: Response) -> None:
        """Start the pipeline once every transcript is final (or finish a noise turn, D7)."""
        if response.task is not None:
            return
        response.awaiting_transcript = self._conversation.pending_transcripts()
        if response.awaiting_transcript:
            return
        turn = (
            None if response.turn_item_id is None else self._conversation.get(response.turn_item_id)
        )
        if isinstance(turn, AudioMessage) and not (turn.transcript or "").strip():
            self._close_response(response, "completed", None)
            return
        request = PipelineRequest(
            generation=response.generation,
            messages=build_messages(self._conversation, response.instructions),
            tools=self._settings.tools,
            tool_choice=response.tool_choice,
            max_output_tokens=response.max_output_tokens,
            modality=response.modality,
            voice=response.voice,
        )
        with logs.bound(response=response.id):
            response.task = self._spawn(run_pipeline(request, self._pipeline_context))

    def _pipeline_signal(self, response: Response, signal: PipelineSignal) -> None:
        match signal:
            case LlmText(text=text):
                message = response.message or self._open_message(response)
                if response.modality == "text":
                    message.text += text
                    self._emit(OutputTextDelta(response, message, text))
            case LlmToolCall(call=call):
                self._deliver_call(response, call)
            case LlmFinished(done=done):
                response.llm_done = True
                response.usage = done.usage
                response.finish_reason = done.finish_reason
                self._finish_if_done(response)
            case SegmentStarted():
                self._segment_started(response, signal)
            case SegmentAudio():
                self._segment_audio(response, signal)
            case SpeechFinished():
                response.tts_done = True
                self._finish_if_done(response)
            case PipelineFailed(stage=stage, error=error):
                info = _backend_error(stage, error)
                self._stop_response(response, StatusDetails("failed", error=info), info)

    def _open_message(self, response: Response) -> AssistantMessage:
        message = AssistantMessage(self._ids.item(), response.modality)
        previous = self._last_item_id()
        self._conversation.append(message)
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
        segment = SpokenSegment(signal.text, signal.spoken, start_ms=message.audio_ms)
        message.segments.append(segment)
        message.text += signal.text
        delivery = Delivery(response.generation, message.id, signal.index, "transcript")
        self._emit(OutputTranscriptDelta(response, message, signal.text, delivery))

    def _segment_audio(self, response: Response, signal: SegmentAudio) -> None:
        message = response.message
        if message is None:
            return
        message.segments[signal.index].audio_bytes += len(signal.pcm)
        delivery = Delivery(response.generation, message.id, signal.index, "audio", len(signal.pcm))
        self._emit(OutputAudioDelta(response, message, signal.pcm, delivery))
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
        self._close_response(response, details.type, details)

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
        self._emit(ResponseFinished(response))

    # Helpers

    def _spawn(self, coroutine: Coroutine[object, object, None]) -> asyncio.Task[None]:
        if self._tasks is None:
            raise RuntimeError("the session actor is not running")
        return self._tasks.create_task(coroutine)

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
