"""Identifiers: ``sess_``, ``conv_``, ``item_``, ``resp_``, ``call_`` and ``turn_`` (logs only),
each followed by 16 random lowercase alphanumerics. Server event ids are ``event_<n>``, minted in
order by each session's encoder.

GA example prefixes such as ``fc_`` are not copied: every conversation item, function calls
included, gets an ``item_`` id, and tool linkage uses a ``call_`` id minted here.
"""

from __future__ import annotations

import secrets
import string

_ALPHABET = string.ascii_lowercase + string.digits
_SUFFIX_LENGTH = 16


def _new(prefix: str) -> str:
    return prefix + "".join(secrets.choice(_ALPHABET) for _ in range(_SUFFIX_LENGTH))


class IdFactory:
    def session(self) -> str:
        return _new("sess_")

    def conversation(self) -> str:
        return _new("conv_")

    def item(self) -> str:
        return _new("item_")

    def response(self) -> str:
        return _new("resp_")

    def call(self) -> str:
        return _new("call_")

    def turn(self) -> str:
        return _new("turn_")
