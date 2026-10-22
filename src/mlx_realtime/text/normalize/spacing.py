"""zh/en spacing, applied last to the normalized text (ARCHITECTURE §7).

Rules:

- Exactly one space between a CJK character and an ASCII letter or digit, in either order:
  ``我用iPhone手机`` → ``我用 iPhone 手机``.
- Spaces between two CJK characters are removed: ``你 好`` → ``你好``.
- Any other run of spaces becomes one space. Only the ASCII space counts; line breaks and tabs
  are kept.
"""

from __future__ import annotations

from mlx_realtime.text.lang import is_cjk


def space(text: str) -> str:
    """Apply the spacing rules to ``text``."""
    out: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == " ":
            end = index
            while end < len(text) and text[end] == " ":
                end += 1
            between_cjk = bool(out) and end < len(text) and is_cjk(out[-1]) and is_cjk(text[end])
            if not between_cjk:
                out.append(" ")
            index = end
            continue
        if out and _needs_space(out[-1], char):
            out.append(" ")
        out.append(char)
        index += 1
    return "".join(out)


def _needs_space(before: str, after: str) -> bool:
    return (is_cjk(before) and _is_latin(after)) or (_is_latin(before) and is_cjk(after))


def _is_latin(char: str) -> bool:
    return char.isascii() and char.isalnum()
