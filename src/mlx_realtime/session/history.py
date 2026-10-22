"""The conversation as a provider-neutral message list for the next LLM request.

The list is generic: system items stay system messages wherever they occur, and a function
call is its own assistant message. Adapting this to what a backend accepts (one leading
system message, at least one user message, merged tool calls) is the provider's job
(ARCHITECTURE §5.2).
"""

from __future__ import annotations

from mlx_realtime.providers.base import Message, ToolCall
from mlx_realtime.session.conversation import (
    AssistantMessage,
    AudioMessage,
    Conversation,
    FunctionCall,
    FunctionCallOutput,
    TextMessage,
)


def build_messages(conversation: Conversation, instructions: str) -> list[Message]:
    messages = [Message("system", instructions)] if instructions else []
    for item in conversation:
        match item:
            case TextMessage(role=role):
                messages.append(Message(role, item.text))
            case AudioMessage(transcript=transcript) if transcript and transcript.strip():
                messages.append(Message("user", transcript))
            case AssistantMessage() if item.history_text:
                messages.append(Message("assistant", item.history_text))
            case FunctionCall(provider_call_id=call_id, name=name, arguments=arguments):
                messages.append(
                    Message("assistant", "", tool_calls=(ToolCall(call_id, name, arguments),))
                )
            case FunctionCallOutput(call_id=call_id, output=output):
                call = conversation.find_call(call_id)
                provider_id = call.provider_call_id if call is not None else call_id
                messages.append(Message("tool", output, tool_call_id=provider_id))
            case _:
                pass  # audio without speech, or an assistant message with nothing heard
    return messages
