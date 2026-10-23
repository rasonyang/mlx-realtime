"""E-mail addresses, read out so that "@" and "." are heard (G5).

Rules:

- An address is what ``protected`` recognizes as one (``first.last@example.co.uk``), never one
  inside a URL. No other category reads into it.
- Letter runs stay as written (the TTS reads words and names itself); digit runs are read digit
  by digit (zh 1 is 一, as in IDs); the symbols are spoken.
- zh: ``@`` 艾特, ``.`` 点, ``_`` 下划线, ``-`` 杠, ``+`` 加, written together:
  ``service123@163.com`` → ``service一二三艾特一六三点com``.
- en: at, dot, underscore, dash, plus, every piece separated by a space:
  ``service123@163.com`` → ``service one two three at one six three dot com``.
"""

from __future__ import annotations

import re
from functools import partial

from mlx_realtime.text.normalize import protected
from mlx_realtime.text.normalize.common import Lang, Match
from mlx_realtime.text.normalize.readers import en_digits, zh_digits

_PIECE = re.compile(r"[A-Za-z]+|[0-9]+|.")
_ZH_SYMBOLS = {"@": "艾特", ".": "点", "_": "下划线", "-": "杠", "+": "加"}
_EN_SYMBOLS = {"@": "at", ".": "dot", "_": "underscore", "-": "dash", "+": "plus"}


def find(text: str) -> list[Match]:
    """Every e-mail address in ``text``."""
    return [
        Match(start, end, partial(_read, text[start:end]))
        for start, end in protected.find_emails(text)
    ]


def _read(address: str, lang: Lang) -> str:
    pieces = [found.group() for found in _PIECE.finditer(address)]
    if lang == "zh":
        return "".join(
            zh_digits(piece) if piece.isdigit() else _ZH_SYMBOLS.get(piece, piece)
            for piece in pieces
        )
    return " ".join(
        en_digits(piece) if piece.isdigit() else _EN_SYMBOLS.get(piece, piece) for piece in pieces
    )
