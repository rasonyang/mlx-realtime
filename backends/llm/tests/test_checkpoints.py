"""Prompt-cache checkpoints inside the system message (``checkpoints.py``) on mlx-lm 0.32.2,
with the real Qwen3.6 tokenizer and chat template (skipped when the snapshot is absent) and
AICC's instructions and tools (``tests/data``)."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mlx_lm import server, tokenizer_utils
from mlx_lm.models.cache import LRUPromptCache
from mlx_lm.tool_parsers import qwen3_coder

from mlxrt_llm import checkpoints, serve
from mlxrt_llm.checkpoints import MIN_SEGMENT_TOKENS, common_prefix, split_system

DATA = json.loads((Path(__file__).parent / "data" / "s1_toolparse_qwen3.6.json").read_text())
TOOLS: list[dict[str, Any]] = DATA["tools"]  # AICC's six tools
HEAD = DATA["instructions"].split("当前环节：")[0] + "当前环节：\n"  # what AICC's phases keep
ORIGINAL = server.ResponseGenerator._tokenize
FETCH = LRUPromptCache.fetch_nearest_cache
PARSER = qwen3_coder.parse_tool_call
QWEN36_SNAPSHOT = Path(
    os.environ.get(
        "MLXRT_QWEN36_SNAPSHOT",
        Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit"
        "/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46",
    )
)
GENERATOR = SimpleNamespace(
    model_provider=SimpleNamespace(
        processor=None, cli_args=SimpleNamespace(chat_template_args={"enable_thinking": False})
    )
)
ARGS = SimpleNamespace(chat_template_kwargs={"enable_thinking": False})
TURN = {"role": "user", "content": "转人工"}


@pytest.fixture
def restore() -> Iterator[None]:
    yield
    server.ResponseGenerator._tokenize = ORIGINAL
    LRUPromptCache.fetch_nearest_cache = FETCH
    qwen3_coder.parse_tool_call = PARSER


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    if not QWEN36_SNAPSHOT.is_dir():
        pytest.skip("the Qwen3.6 snapshot is not downloaded")
    return tokenizer_utils.load(QWEN36_SNAPSHOT)


def tokenize(tokenizer: Any, messages: list[dict[str, Any]], tools: Any) -> tuple[Any, ...]:
    request = server.CompletionRequest("chat", "", json.loads(json.dumps(messages)), tools, None)
    tokenize: Any = server.ResponseGenerator._tokenize
    return tokenize(GENERATOR, tokenizer, request, ARGS)


def call(phase: str, *rest: dict[str, Any]) -> list[dict[str, Any]]:
    """A call in one of AICC's phases: its instructions with ``phase`` after 当前环节：."""
    return [
        {"role": "system", "content": HEAD + phase},
        {"role": "assistant", "content": DATA["announce"]},
        *rest,
    ]


def test_common_prefix() -> None:
    assert common_prefix([1, 2, 3, 9], [1, 2, 3, 4, 5]) == 3
    assert common_prefix([1, 2], [1, 2, 3]) == 2
    assert common_prefix([7], [1, 2]) == 0


def test_split_system() -> None:
    system, user, tail = list(range(300)), [300, 301], [302]
    types = ["system", "user", "assistant"]
    assert split_system([system, user, tail], types, [295, 80]) == (
        [system[:80], system[80:295], system[295:], user, tail],
        ["system", "system", "system", "user", "assistant"],
    )
    near = MIN_SEGMENT_TOKENS - 1
    cases = {  # ends -> the system pieces' lengths
        (near,): [300],  # too close to the start
        (300,): [300],  # the end itself
        (400,): [300],  # outside
        (80, 80 + near): [80, 220],  # too close to the previous point
    }
    for ends, lengths in cases.items():
        split, _ = split_system([system, user, tail], types, ends)
        assert [len(piece) for piece in split[:-2]] == lengths, ends
    assert split_system([user, tail], ["user", "assistant"], [1]) == ([user, tail], types[1:])


def test_the_tool_definitions_get_a_checkpoint_of_their_own(tokenizer: Any, restore: None) -> None:
    stock = tokenize(tokenizer, call("开场。", TURN), TOOLS)
    checkpoints.install()
    prompt, segments, types, state, vision = tokenize(tokenizer, call("开场。", TURN), TOOLS)
    assert (prompt, state, vision) == (stock[0], stock[3], stock[4])
    assert types == ["system", "system", "user", "assistant"]
    assert [token for segment in segments for token in segment] == prompt
    assert len(segments[0]) == 827  # AICC's six tools, as M0 counted them
    assert segments[0] + segments[1] == stock[1][0]  # the stock system segment, split
    assert segments[2:] == stock[1][1:]
    assert tokenizer.decode(segments[0]).endswith("</IMPORTANT>")
    changed = tokenize(tokenizer, call("结束通话。", TURN), TOOLS)
    assert changed[1][0] == segments[0]  # new instructions, the same first checkpoint
    assert changed[1][1] != segments[1]


def test_a_phase_change_saves_the_instructions_it_keeps(tokenizer: Any, restore: None) -> None:
    """The first phase change splits where the instructions start to differ (after HEAD); the
    next one, in the same or another call, finds that checkpoint as a prefix of its prompt."""
    checkpoints.install(tools=True, shared_prefix=True)
    first = tokenize(tokenizer, call("开场：问用户需要什么帮助。", TURN), TOOLS)
    assert first[2] == ["system", "system", "user", "assistant"]  # nothing to compare with yet
    second = tokenize(tokenizer, call("结束通话：确认用户没有其它问题。", TURN), TOOLS)
    assert second[2] == ["system", "system", "system", "user", "assistant"]
    tools_part, head_part = second[1][0], second[1][1]
    assert len(tools_part) == 827
    assert tokenizer.decode(head_part).endswith("当前环节：\n")
    kept = len(tools_part) + len(head_part)
    third = tokenize(tokenizer, call("记录留言：问清回电号码。", TURN), TOOLS)
    assert third[0][:kept] == second[0][:kept]  # the head checkpoint serves the third phase
    assert [token for segment in third[1] for token in segment] == third[0]


@pytest.mark.parametrize(
    ("messages", "tools"),
    [
        pytest.param(call("开场。", TURN), None, id="no-tools"),
        pytest.param([TURN], TOOLS, id="no-system-message"),
        pytest.param(
            call(
                "开场。",
                TURN,
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "hangup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "{}"},
            ),
            TOOLS,
            id="ending-with-a-result",
        ),
    ],
)
def test_other_requests_keep_the_stock_segments(
    tokenizer: Any, restore: None, messages: list[dict[str, Any]], tools: Any
) -> None:
    stock = tokenize(tokenizer, messages, tools)
    checkpoints.install()
    assert tokenize(tokenizer, messages, tools) == stock


def test_install_is_idempotent(restore: None) -> None:
    checkpoints.install()
    patched = server.ResponseGenerator._tokenize
    checkpoints.install(shared_prefix=True)
    assert server.ResponseGenerator._tokenize is patched  # the first options stay
    assert getattr(patched, "__wrapped__", None) is ORIGINAL
    assert getattr(patched, "__mlxrt_checkpoints__", None) == {
        "tools": True,
        "shared_prefix": False,
    }


def test_install_refuses_a_changed_tokenize(monkeypatch: pytest.MonkeyPatch, restore: None) -> None:
    def _tokenize(self: Any, tokenizer: Any, request: Any, args: Any) -> tuple[Any, ...]:
        """Segments chosen some other way."""
        return ()

    monkeypatch.setattr(server.ResponseGenerator, "_tokenize", _tokenize)
    with pytest.raises(checkpoints.CheckpointPatchError, match="no longer has"):
        checkpoints.install()
    assert server.ResponseGenerator._tokenize is _tokenize


@pytest.mark.parametrize(
    ("flags", "options"),
    [
        (["--tools-checkpoint"], {"tools": True, "shared_prefix": False}),
        (["--shared-prefix-checkpoint"], {"tools": False, "shared_prefix": True}),
        (
            ["--tools-checkpoint", "--shared-prefix-checkpoint"],
            {"tools": True, "shared_prefix": True},
        ),
    ],
)
def test_the_serve_flags_install_it_and_are_not_passed_on(
    monkeypatch: pytest.MonkeyPatch, restore: None, flags: list[str], options: dict[str, bool]
) -> None:
    seen: list[list[str]] = []
    monkeypatch.setattr(sys, "argv", ["serve", "--port", "8101", *flags])
    monkeypatch.setattr(server, "main", lambda: seen.append(list(sys.argv)))
    serve.main()
    assert seen == [["serve", "--port", "8101"]]
    assert getattr(server.ResponseGenerator._tokenize, "__mlxrt_checkpoints__", None) == options


def test_without_the_flags_the_segments_are_stock(
    monkeypatch: pytest.MonkeyPatch, restore: None
) -> None:
    monkeypatch.setattr(sys, "argv", ["serve", "--port", "8101"])
    monkeypatch.setattr(server, "main", lambda: None)
    serve.main()
    assert server.ResponseGenerator._tokenize is ORIGINAL


class Layer:
    """A cache layer of a hybrid model: it cannot be trimmed back."""

    nbytes = 10

    def is_trimmable(self) -> bool:
        return False


def one_offs_after_a_checkpoint() -> tuple[LRUPromptCache, list[int], int]:
    """A 3-entry cache: the tool-definitions checkpoint, then five requests that would start
    from it and each save a system message of their own (a phase seen once). Also: how many of
    them found the checkpoint."""
    cache = LRUPromptCache(max_size=3)
    tools = list(range(100))
    cache.insert_cache("model", tools, [Layer()], cache_type="system")
    hits = 0
    for phase in range(5):
        prompt = [*tools, *([1000 + phase] * 10)]
        _, rest = cache.fetch_nearest_cache("model", [*prompt, 7])
        hits += len(rest) == 11
        cache.insert_cache("model", prompt, [Layer()], cache_type="system")
    return cache, tools, hits


def test_stock_eviction_drops_the_checkpoint_every_request_uses(restore: None) -> None:
    cache, tools, hits = one_offs_after_a_checkpoint()
    assert hits == 3  # then it was evicted, the oldest system entry, though the most used
    assert cache.fetch_nearest_cache("model", [*tools, 5]) == (None, [*tools, 5])


def test_a_hit_keeps_a_checkpoint_from_being_evicted(restore: None) -> None:
    checkpoints.install()
    cache, tools, hits = one_offs_after_a_checkpoint()
    assert hits == 5
    hit, rest = cache.fetch_nearest_cache("model", [*tools, 5])
    assert hit is not None
    assert rest == [5]
    assert len(cache) == 3  # nothing was duplicated in the eviction order
