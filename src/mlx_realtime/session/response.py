"""The response state (ARCHITECTURE §6.3): five protocol states, progress tracked by flags."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from mlx_realtime.providers.base import FinishReason, ToolChoice, Usage
from mlx_realtime.session.conversation import AssistantMessage, FunctionCall
from mlx_realtime.session.errors import ErrorInfo
from mlx_realtime.session.settings import Modality

type ResponseStatus = Literal["in_progress", "completed", "cancelled", "incomplete", "failed"]
type StatusReason = Literal[
    "turn_detected", "client_cancelled", "max_output_tokens", "content_filter"
]


@dataclass(frozen=True, slots=True)
class StatusDetails:
    type: Literal["cancelled", "incomplete", "failed"]
    reason: StatusReason | None = None
    error: ErrorInfo | None = None


@dataclass(slots=True)
class Response:
    id: str
    generation: int
    conversation_id: str
    modality: Modality
    voice: str
    instructions: str
    max_output_tokens: int | None
    tool_choice: ToolChoice
    metadata: Mapping[str, str] | None
    turn_item_id: str | None  # the user audio item this response answers (D7)
    status: ResponseStatus = "in_progress"
    status_details: StatusDetails | None = None
    output: list[AssistantMessage | FunctionCall] = field(
        default_factory=list[AssistantMessage | FunctionCall]
    )
    message: AssistantMessage | None = None  # created lazily, on the first text delta
    usage: Usage | None = None
    finish_reason: FinishReason | None = None
    awaiting_transcript: bool = False
    llm_done: bool = False
    tts_done: bool = False
    task: asyncio.Task[None] | None = None  # the response scope, once the pipeline runs

    def output_index(self, item: AssistantMessage | FunctionCall) -> int:
        return self.output.index(item)
