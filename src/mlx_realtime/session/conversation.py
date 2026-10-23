"""The conversation: an ordered list of items (ARCHITECTURE §6.4).

Item shapes follow the GA item types: messages of every role, and the function call and
function call output items of the tool round trip (§6.7).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

from mlx_realtime.audio.formats import PCM, AudioFormat
from mlx_realtime.session.settings import Modality

type ItemStatus = Literal["in_progress", "completed", "incomplete"]

ROOT = "root"
"""``previous_item_id`` that inserts at the start of the conversation."""


@dataclass(slots=True)
class TextMessage:
    """A message created by the client with ``conversation.item.create``."""

    id: str
    role: Literal["system", "user", "assistant"]
    texts: list[str]  # one per content part
    status: ItemStatus = "completed"

    @property
    def text(self) -> str:
        return "\n".join(self.texts)


@dataclass(slots=True)
class AudioMessage:
    """A user message committed from the input audio buffer."""

    id: str
    audio_ms: int
    show_transcript: bool  # transcription was enabled when the audio was committed
    transcript: str | None = None  # set once the ASR result arrives
    status: ItemStatus = "in_progress"
    start_ms: int = 0  # where the committed audio lies on the input timeline
    end_ms: int = 0
    echo: bool = False  # the transcript was our own speech coming back (D17): not a turn

    @property
    def transcript_pending(self) -> bool:
        return self.status == "in_progress"


@dataclass(slots=True)
class SpokenSegment:
    """One synthesized segment of an assistant audio message (the segment map of §6.6)."""

    text: str  # original text, as sent in transcript deltas
    spoken: str  # normalized text sent to the TTS
    start_ms: int  # on the item's output-audio timeline
    audio_bytes: int = 0  # wire audio synthesized so far
    # For each character of ``spoken``, its index in ``text`` (and len(text) at the end);
    # None when ``spoken`` is ``text``.
    source: tuple[int, ...] | None = None
    audio_format: AudioFormat = PCM  # the wire format ``audio_bytes`` counts

    @property
    def audio_ms(self) -> int:
        return self.audio_format.bytes_to_ms(self.audio_bytes)


@dataclass(slots=True)
class AssistantMessage:
    """A message produced by a response. In audio modality ``text`` is the transcript."""

    id: str
    modality: Modality
    text: str = ""
    segments: list[SpokenSegment] = field(default_factory=list[SpokenSegment])
    truncated_text: str | None = None  # what the client heard, after conversation.item.truncate
    fallback_text: str | None = None  # D5: what it can have heard, when cut without a truncate
    status: ItemStatus = "in_progress"
    audio_format: AudioFormat = PCM  # its response's output format

    @property
    def audio_ms(self) -> int:
        return self.audio_format.bytes_to_ms(sum(segment.audio_bytes for segment in self.segments))

    @property
    def history_text(self) -> str:
        """The text later LLM requests see: a client truncate wins over the D5 fallback cut."""
        if self.truncated_text is not None:
            return self.truncated_text
        return self.text if self.fallback_text is None else self.fallback_text


@dataclass(slots=True)
class FunctionCall:
    id: str
    call_id: str  # minted for the wire
    provider_call_id: str  # the LLM provider's own id, used when history is replayed
    name: str
    arguments: str
    status: ItemStatus = "completed"


@dataclass(slots=True)
class FunctionCallOutput:
    id: str
    call_id: str
    output: str
    status: ItemStatus = "completed"


type Item = TextMessage | AudioMessage | AssistantMessage | FunctionCall | FunctionCallOutput


class Conversation:
    def __init__(self) -> None:
        self._items: list[Item] = []

    def __iter__(self) -> Iterator[Item]:
        return iter(self._items)

    def __contains__(self, item_id: str) -> bool:
        return self._index(item_id) is not None

    def get(self, item_id: str) -> Item | None:
        index = self._index(item_id)
        return None if index is None else self._items[index]

    def last(self) -> Item | None:
        return self._items[-1] if self._items else None

    def previous_id(self, item_id: str) -> str | None:
        index = self._index(item_id)
        return None if not index else self._items[index - 1].id

    def append(self, item: Item) -> None:
        self._items.append(item)

    def insert_after(self, item: Item, previous_item_id: str) -> None:
        """Insert after ``previous_item_id`` (``ROOT`` for the start); the id must exist."""
        if previous_item_id == ROOT:
            self._items.insert(0, item)
            return
        index = self._index(previous_item_id)
        if index is None:
            raise KeyError(previous_item_id)
        self._items.insert(index + 1, item)

    def remove(self, item_id: str) -> Item | None:
        index = self._index(item_id)
        return None if index is None else self._items.pop(index)

    def find_call(self, call_id: str) -> FunctionCall | None:
        return next(
            (
                item
                for item in self._items
                if isinstance(item, FunctionCall) and item.call_id == call_id
            ),
            None,
        )

    def pending_transcripts(self) -> bool:
        return any(
            isinstance(item, AudioMessage) and item.transcript_pending for item in self._items
        )

    def _index(self, item_id: str) -> int | None:
        return next((index for index, item in enumerate(self._items) if item.id == item_id), None)
