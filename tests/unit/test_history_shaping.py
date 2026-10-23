"""History shaping for the Qwen template (M5): every conversation shape AICC produces around a
tool call, from the conversation items through ``build_messages`` (generic) and
``chat_messages`` (the mlx_lm provider's adaptation), rendered with the real Qwen3.6 template.

The rule: a tool result follows its call's assistant message at once. Text spoken between the
call and its result is merged into the call's message (``merge``, default) or placed after the
results (``after_output``); user notes and caller speech come after the results; an unanswered
call gets a placeholder result (``pending``, default), none (``keep``), or is left out
(``drop``); the output of a deleted call is dropped.
"""

from __future__ import annotations

from typing import Any

import pytest

from mlx_realtime.config import PENDING_TOOL_OUTPUT
from mlx_realtime.providers.base import ToolDef
from mlx_realtime.providers.chat_wire import Interlude, Shaping, Unanswered, chat_messages
from mlx_realtime.session.conversation import (
    AssistantMessage,
    AudioMessage,
    Conversation,
    FunctionCall,
    FunctionCallOutput,
    Item,
    TextMessage,
)
from mlx_realtime.session.history import build_messages
from tests.qwen_template import render as render_template

CUE = "(The conversation has just started.)"
NOTE = "(System note: {text})"
INSTRUCTIONS = "You answer the phone for NovaNet."
TOOLS = [ToolDef("lookup_balance", "Read the caller's balance.", {"type": "object"})]
RESULT = '{"balance":"12.50","hint":"Tell the caller the balance, then offer to close."}'
WAIT = "请稍等，我查一下。"

type Json = dict[str, Any]


def shaping(interlude: Interlude = "merge", unanswered: Unanswered = "pending") -> Shaping:
    return Shaping(CUE, NOTE, interlude, unanswered, PENDING_TOOL_OUTPUT)


def user(item_id: str, text: str) -> TextMessage:
    return TextMessage(item_id, "user", [text])


def said(item_id: str, text: str) -> AssistantMessage:
    return AssistantMessage(item_id, "audio", text=text, status="completed")


def call(number: int) -> FunctionCall:
    return FunctionCall(
        f"item_call{number}", f"call_{number}", f"p{number}", "lookup_balance", '{"accountId":"42"}'
    )


def output(number: int, text: str = RESULT) -> FunctionCallOutput:
    return FunctionCallOutput(f"item_out{number}", f"call_{number}", text)


def conversation(*items: Item) -> Conversation:
    conv = Conversation()
    for item in items:
        conv.append(item)
    return conv


def adapt(conv: Conversation, shape: Shaping | None = None) -> list[Json]:
    messages = chat_messages(build_messages(conv, INSTRUCTIONS), shape or shaping())
    render(messages)  # every shape must be one the real template accepts
    return messages[1:]  # without the system message


def assistant(content: str, *calls: int) -> Json:
    entry: Json = {"role": "assistant", "content": content}
    if calls:
        entry["tool_calls"] = [
            {
                "id": f"p{number}",
                "type": "function",
                "function": {"name": "lookup_balance", "arguments": '{"accountId":"42"}'},
            }
            for number in calls
        ]
    return entry


def tool(number: int, content: str = RESULT) -> Json:
    return {"role": "tool", "tool_call_id": f"p{number}", "content": content}


def render(messages: list[Json]) -> str:
    tools = [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in TOOLS
    ]
    return render_template(messages, tools)


def test_the_round_trip_replays_call_then_result() -> None:
    conv = conversation(user("u1", "查一下余额。"), call(1), output(1))
    assert adapt(conv) == [{"role": "user", "content": "查一下余额。"}, assistant("", 1), tool(1)]


def test_text_before_the_call_shares_its_message() -> None:
    conv = conversation(user("u1", "查一下余额。"), said("a1", "好的。"), call(1), output(1))
    assert adapt(conv)[1:] == [assistant("好的。", 1), tool(1)]


def test_the_hint_reaches_the_model_verbatim() -> None:
    prompt = render(
        chat_messages(
            build_messages(conversation(user("u1", "查"), call(1), output(1)), INSTRUCTIONS),
            shaping(),
        )
    )
    assert "<tool_call>\n<function=lookup_balance>\n<parameter=accountId>\n42" in prompt
    assert f"<tool_response>\n{RESULT}\n</tool_response>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


class TestWaitLine:
    """function_call → the assistant's wait line (a verbatim line) → function_call_output."""

    ITEMS = (user("u1", "查一下余额。"), call(1), said("line", WAIT), output(1))

    def test_merged_into_the_call_message(self) -> None:
        assert adapt(conversation(*self.ITEMS))[1:] == [assistant(WAIT, 1), tool(1)]

    def test_or_after_the_result(self) -> None:
        shaped = adapt(conversation(*self.ITEMS), shaping("after_output"))
        assert shaped[1:] == [assistant("", 1), tool(1), assistant(WAIT)]

    def test_with_text_before_the_call(self) -> None:
        items = (user("u1", "查一下余额。"), said("a1", "好的。"), call(1), said("line", WAIT))
        shaped = adapt(conversation(*items, output(1)))
        assert shaped[1:] == [assistant(f"好的。\n{WAIT}", 1), tool(1)]

    def test_the_result_answered_by_the_next_response(self) -> None:
        items = (*self.ITEMS, said("a2", "您的余额是十二元五角。"))
        shaped = adapt(conversation(*items, user("u2", "谢谢。")))
        assert shaped[1:] == [
            assistant(WAIT, 1),
            tool(1),
            assistant("您的余额是十二元五角。"),
            {"role": "user", "content": "谢谢。"},
        ]


class TestDeletedCall:
    def test_the_output_of_a_deleted_call_is_dropped(self) -> None:
        conv = conversation(user("u1", "查一下余额。"), call(1), output(1))
        conv.remove("item_call1")
        assert adapt(conv) == [{"role": "user", "content": "查一下余额。"}]

    def test_a_deleted_call_made_in_a_line_turn_leaves_the_rest(self) -> None:
        # AICC deletes a call made in a line's turn, after answering the call it ran.
        conv = conversation(user("u1", "查一下余额。"), call(1), said("line", WAIT), call(2))
        conv.append(output(1))
        conv.remove("item_call2")
        assert adapt(conv)[1:] == [assistant(WAIT, 1), tool(1)]


def test_several_outputs_then_one_response() -> None:
    """Two calls in one response; both outputs (in either order) before one response.create."""
    conv = conversation(user("u1", "查两个账户。"), call(1), call(2), output(2, "{}"), output(1))
    assert adapt(conv)[1:] == [assistant("", 1, 2), tool(1), tool(2, "{}")]


class TestStageNotes:
    def test_a_keypress_note_before_the_output_moves_after_it(self) -> None:
        note = "(The caller pressed 2 on their keypad.)"
        conv = conversation(user("u1", "查一下余额。"), call(1), user("n1", note), output(1))
        assert adapt(conv)[1:] == [assistant("", 1), tool(1), {"role": "user", "content": note}]

    def test_a_system_note_item(self) -> None:
        conv = conversation(user("u1", "查一下余额。"), call(1), output(1))
        conv.append(TextMessage("s1", "system", ["the caller is waiting"]))
        assert adapt(conv)[-1] == {
            "role": "user",
            "content": "(System note: the caller is waiting)",
        }

    def test_aiccs_unheard_result_note(self) -> None:
        note = "(System note: the caller has not heard the result of the lookup yet.)"
        conv = conversation(user("u1", "查一下余额。"), call(1), output(1), user("n1", note))
        assert adapt(conv)[1:] == [assistant("", 1), tool(1), {"role": "user", "content": note}]

    def test_speech_then_text_in_the_interlude_keep_their_order(self) -> None:
        """Assistant text after the caller spoke in the interlude answers the caller: it is not
        merged into the call."""
        speech = AudioMessage("u2", 800, True, "喂？", "completed")
        items = (user("u1", "查一下余额。"), call(1), speech, said("a2", "我还在查。"), output(1))
        assert adapt(conversation(*items))[1:] == [
            assistant("", 1),
            tool(1),
            {"role": "user", "content": "喂？"},
            assistant("我还在查。"),
        ]


class TestUnansweredCall:
    """A call still unanswered while a line plays and the caller speaks: the server's own turn
    builds a request before AICC returns the output."""

    ITEMS = (
        user("u1", "查一下余额。"),
        call(1),
        said("line", WAIT),
        AudioMessage("u2", 800, True, "喂？还在吗？", "completed"),
    )

    def test_pending_placeholder(self) -> None:
        assert adapt(conversation(*self.ITEMS))[1:] == [
            assistant(WAIT, 1),
            tool(1, PENDING_TOOL_OUTPUT),
            {"role": "user", "content": "喂？还在吗？"},
        ]

    def test_keep(self) -> None:
        shaped = adapt(conversation(*self.ITEMS), shaping(unanswered="keep"))
        assert shaped[1:] == [assistant(WAIT, 1), {"role": "user", "content": "喂？还在吗？"}]

    def test_drop(self) -> None:
        shaped = adapt(conversation(*self.ITEMS), shaping(unanswered="drop"))
        assert shaped[1:] == [assistant(WAIT), {"role": "user", "content": "喂？还在吗？"}]

    def test_a_call_with_nothing_after_it(self) -> None:
        conv = conversation(user("u1", "查一下余额。"), call(1))
        assert adapt(conv)[1:] == [assistant("", 1), tool(1, PENDING_TOOL_OUTPUT)]

    @pytest.mark.parametrize("interlude", ["merge", "after_output"])
    @pytest.mark.parametrize("unanswered", ["pending", "keep", "drop"])
    def test_every_setting_renders(self, interlude: Interlude, unanswered: Unanswered) -> None:
        adapt(conversation(*self.ITEMS), shaping(interlude, unanswered))


def test_a_second_call_after_the_caller_spoke_gets_its_own_block() -> None:
    conv = conversation(
        user("u1", "查一下余额。"),
        call(1),
        user("u2", "顺便查一下另一个。"),
        call(2),
        output(2, "{}"),
        output(1),
    )
    assert adapt(conv)[1:] == [
        assistant("", 1),
        tool(1),
        {"role": "user", "content": "顺便查一下另一个。"},
        assistant("", 2),
        tool(2, "{}"),
    ]


def test_a_greeting_then_a_call_without_a_user_message_gets_the_cue() -> None:
    conv = conversation(said("g", "您好。"), call(1), output(1))
    shaped = adapt(conv)
    assert shaped == [{"role": "user", "content": CUE}, assistant("您好。", 1), tool(1)]
