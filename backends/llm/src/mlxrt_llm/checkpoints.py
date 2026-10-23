"""Prompt-cache checkpoints inside the system message (LLM hardening, 2026-10-23).

Qwen3.6 is a hybrid model (Gated DeltaNet layers with attention layers): its cache cannot be
trimmed back to an earlier token, so mlx_lm.server reuses only a cached prefix it saved whole.
It saves those at the ends of a chat prompt's segments: the system message (with the next
message's role token), the last user message, and the whole sequence, and only for a request
that ends with a user message. The Qwen chat template renders the tool definitions first,
inside the system message, then the instructions. So a request whose instructions are new (a
new call, an AICC phase change, a per-response override) prefills the whole system message
again, tool definitions included.

``install()`` wraps ``ResponseGenerator._tokenize`` to split the system segment at more points,
so the server also saves the cache there (it saves one at every segment end):

- ``tools``: where the tool definitions end and the instructions begin, for a request with
  tools. Any later request with the same tools starts from there: new instructions, a new call,
  a per-response override. With AICC's six tools that is 827 of 1519 system tokens.
- ``shared_prefix``: where the system message stops matching the most similar of the
  ``RECENT_SYSTEMS`` system messages seen last: the part that stays when instructions change,
  such as AICC's instructions up to ``当前环节：``, which its phase changes keep (another 575
  tokens). The first change saves it; later changes, in any call with that head, start there.

The prompt, the sampling and the output are unchanged: only where the cache is saved changes.
A split point within ``MIN_SEGMENT_TOKENS`` of the start or of another split point is skipped.

The prompt cache evicts by type and then oldest first, and mlx-lm 0.32.2 counts only insertions
as use: a checkpoint every request starts from is never inserted again, so it ages out first
once enough one-off entries (each phase's own system message, say) come after it. With the
checkpoints on, a hit also makes its entry the newest of its type (a true LRU).

Where the instructions begin is found as the server finds where the system message ends: the
first token at which the request's prompt and the rendering of an empty system message with the
same tools (and template arguments) differ.
"""

from __future__ import annotations

import inspect
import json
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from itertools import pairwise
from typing import Any, cast

# What the patch relies on, as it reads in mlx-lm 0.32.2's ResponseGenerator._tokenize.
_TOKENIZE_PARAMETERS = ["self", "tokenizer", "request", "args"]
_TOKENIZE_MARKERS = (
    "chat_template_args = self.model_provider.cli_args.chat_template_args",
    "chat_template_args.update(args.chat_template_kwargs)",
    'segment_types.append("system")',
    "return prompt, segments, segment_types, initial_state, None",
)
# What the LRU refresh relies on, in mlx-lm 0.32.2's LRUPromptCache.
_FETCH_MARKERS = (
    "result = self._trie.search(model, tokens)",
    "return copy.deepcopy(cache_entry.prompt_cache), []",
    "return copy.deepcopy(cache_entry.prompt_cache), tokens[short_length:]",
)
_ORDER_MARKERS = (
    "self._lrus = {k: deque() for k in self._ordering}",
    "self._lrus[cache_type].append((model, tokens))",
)
MIN_SEGMENT_TOKENS = 64  # a shorter stretch is not worth a cache entry of its own
RECENT_SYSTEMS = 16  # system messages remembered for the shared prefix
_RENDERS_KEPT = 32  # tool blocks whose empty-system rendering is remembered

type Tokenize = Callable[[Any, Any, Any, Any], tuple[Any, ...]]


class CheckpointPatchError(RuntimeError):
    """mlx-lm's ``_tokenize`` is not what the patch was written against."""


def common_prefix(ours: Sequence[int], theirs: Sequence[int]) -> int:
    """How many leading tokens the two sequences share."""
    count = 0
    for a, b in zip(ours, theirs, strict=False):
        if a != b:
            break
        count += 1
    return count


def split_system(
    segments: list[list[int]], segment_types: list[str], ends: Sequence[int]
) -> tuple[list[list[int]], list[str]]:
    """The segments with the system segment also split after each of ``ends`` tokens that falls
    inside it, skipping a point within ``MIN_SEGMENT_TOKENS`` of the start or of a point already
    taken. (The stock checkpoint at the system segment's end holds this exact system message;
    the new ones are what other system messages share with it, however close to that end.)"""
    if segment_types[:1] != ["system"]:
        return segments, segment_types
    system = segments[0]
    points: list[int] = []
    for end in sorted(set(ends)):
        previous = points[-1] if points else 0
        if end - previous >= MIN_SEGMENT_TOKENS and end < len(system):
            points.append(end)
    if not points:
        return segments, segment_types
    bounds = [0, *points, len(system)]
    pieces = [system[start:end] for start, end in pairwise(bounds)]
    return [*pieces, *segments[1:]], ["system"] * len(points) + segment_types


class _RecentSystems:
    """The system segments of the last requests, newest last, for the shared prefix."""

    def __init__(self, kept: int) -> None:
        self._kept = kept
        self._systems: deque[list[int]] = deque()

    def shared(self, system: list[int]) -> int:
        """The longest prefix ``system`` shares with a different remembered one."""
        return max(
            (common_prefix(system, other) for other in self._systems if other != system),
            default=0,
        )

    def remember(self, system: list[int]) -> None:
        if system in self._systems:
            self._systems.remove(system)
        self._systems.append(system)
        while len(self._systems) > self._kept:
            self._systems.popleft()


def install(*, tools: bool = True, shared_prefix: bool = False) -> None:
    """Wrap ``ResponseGenerator._tokenize`` and ``LRUPromptCache.fetch_nearest_cache``; refuse
    on any other mlx-lm. Installing again changes nothing (the first options stay)."""
    from mlx_lm import server
    from mlx_lm.models import cache as prompt_cache

    generator = cast(Any, server.ResponseGenerator)
    original = cast(Tokenize, generator._tokenize)
    if getattr(original, "__mlxrt_checkpoints__", None) is not None:
        return
    problems: list[str] = []
    parameters = list(inspect.signature(original).parameters)
    if parameters != _TOKENIZE_PARAMETERS:
        problems.append(f"ResponseGenerator._tokenize takes {parameters}")
    source = inspect.getsource(original)
    problems += [
        f"ResponseGenerator._tokenize no longer has: {marker}"
        for marker in _TOKENIZE_MARKERS
        if marker not in source
    ]
    lru = cast(Any, prompt_cache.LRUPromptCache)
    fetch_source = inspect.getsource(lru.fetch_nearest_cache)
    order_source = inspect.getsource(lru.CacheOrder)
    problems += [
        f"LRUPromptCache no longer has: {marker}"
        for marker in _FETCH_MARKERS
        if marker not in fetch_source
    ]
    problems += [
        f"LRUPromptCache.CacheOrder no longer has: {marker}"
        for marker in _ORDER_MARKERS
        if marker not in order_source
    ]
    if problems:
        raise CheckpointPatchError(
            "mlxrt_llm: refusing to patch mlx_lm.server's prompt segments:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )
    renders: OrderedDict[str, list[int]] = OrderedDict()
    recent = _RecentSystems(RECENT_SYSTEMS)

    def tools_end(
        tokenizer: Any, tools_spec: Any, template_args: dict[str, Any], prompt: list[int]
    ) -> int:
        key = json.dumps([tools_spec, template_args], ensure_ascii=False, sort_keys=True)
        rendered = renders.get(key)
        if rendered is None:
            empty = [{"role": "system", "content": ""}, {"role": "user", "content": ""}]
            rendered = list(
                tokenizer.apply_chat_template(
                    empty,
                    add_generation_prompt=False,
                    tools=tools_spec,
                    tokenize=True,
                    **template_args,
                )
            )
            renders[key] = rendered
            while len(renders) > _RENDERS_KEPT:
                renders.popitem(last=False)
        else:
            renders.move_to_end(key)
        return common_prefix(rendered, prompt)

    def _tokenize(self: Any, tokenizer: Any, request: Any, args: Any) -> tuple[Any, ...]:
        prompt, segments, segment_types, initial_state, vision = original(
            self, tokenizer, request, args
        )
        if vision is not None or request.request_type != "chat" or segment_types[:1] != ["system"]:
            return prompt, segments, segment_types, initial_state, vision
        ends: list[int] = []
        if tools and request.tools:
            template_args = dict(self.model_provider.cli_args.chat_template_args)
            if args.chat_template_kwargs:
                template_args.update(args.chat_template_kwargs)
            ends.append(tools_end(tokenizer, request.tools, template_args, prompt))
        if shared_prefix:
            system = segments[0]
            ends.append(recent.shared(system))
            recent.remember(system)
        segments, segment_types = split_system(segments, segment_types, ends)
        return prompt, segments, segment_types, initial_state, vision

    _tokenize.__wrapped__ = original  # pyright: ignore[reportFunctionMemberAccess]
    _tokenize.__mlxrt_checkpoints__ = {"tools": tools, "shared_prefix": shared_prefix}  # pyright: ignore[reportFunctionMemberAccess]
    generator._tokenize = _tokenize
    lru.fetch_nearest_cache = refreshing(lru.fetch_nearest_cache)


def refreshing(fetch: Callable[[Any, Any, list[int]], Any]) -> Callable[..., Any]:
    """``LRUPromptCache.fetch_nearest_cache`` that also makes the entry it used the newest of
    its type. (A copy trimmed from a longer entry, which a hybrid model never uses, refreshes
    nothing.)"""

    def fetch_nearest_cache(self: Any, model: Any, tokens: list[int]) -> Any:
        cache, rest = fetch(self, model, tokens)
        hit = len(tokens) - len(rest)
        if cache is None or hit <= 0:
            return cache, rest
        key = list(tokens[:hit])
        try:
            entry = self._trie.get(model, key)
        except KeyError:
            return cache, rest
        order = self._lru._lrus[entry.cache_type]
        try:
            order.remove((model, key))
        except ValueError:
            return cache, rest
        order.append((model, key))
        return cache, rest

    fetch_nearest_cache.__wrapped__ = fetch  # pyright: ignore[reportFunctionMemberAccess]
    return fetch_nearest_cache
