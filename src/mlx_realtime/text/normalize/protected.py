"""Spans no category may read into: URLs, e-mail addresses, domains and file names.

Rules:

- ``http(s)://`` and ``ftp://`` URLs and ``www.`` addresses run to the first character that cannot
  be part of a URL; trailing punctuation (``.,;:!?'")]``) is not part of them.
- E-mail addresses: ``first.last@example.co.uk``. They are read by ``emails`` alone (when on);
  an address inside a URL is part of the URL.
- Bare domains and file names with a well-known suffix: ``163.com``, ``novanet.cn/help``,
  ``report2024.pdf``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

_URL_CHAR = r"[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]"
_SUFFIXES = (
    "com|net|org|edu|gov|info|biz|io|ai|app|dev|co|me|tv|cc|top|xyz|cn|hk|tw|jp|uk|us|de|fr"
    "|pdf|docx?|xlsx?|pptx?|txt|csv|json|md|html?|jpe?g|png|gif|zip|mp3|mp4|wav"
)
_PROTECTED = re.compile(
    rf"(?:https?|ftp)://{_URL_CHAR}+"
    rf"|www\.{_URL_CHAR}+"
    r"|(?P<email>(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)"
    rf"|(?<![A-Za-z0-9_.@/-])(?:[A-Za-z0-9-]+\.)+(?i:{_SUFFIXES})(?![A-Za-z0-9_-])"
    rf"(?:/{_URL_CHAR}*)?"
)
_TRAILING = ".,;:!?'\")]"


def find(text: str) -> Iterator[tuple[int, int]]:
    """The ``(start, end)`` of every protected span, e-mail addresses included."""
    for found in _PROTECTED.finditer(text):
        end = found.end()
        while end > found.start() and text[end - 1] in _TRAILING:
            end -= 1
        yield found.start(), end


def find_emails(text: str) -> Iterator[tuple[int, int]]:
    """The ``(start, end)`` of every e-mail address that is not part of a URL."""
    for found in _PROTECTED.finditer(text):
        if found["email"] is not None:
            yield found.start(), found.end()
