"""Helpers to pull structured content out of free-form model output."""

from __future__ import annotations

import re

_FENCE = re.compile(
    r"```[ \t]*([\w+-]*)[ \t]*\r?\n(.*?)```",
    re.DOTALL,
)


def code_blocks(text: str, lang: str | None = None) -> list[str]:
    """Return fenced code blocks, optionally filtered to a language tag.

    Language matching is lenient: a block tagged `python` matches lang="py"
    and vice versa. If no fenced block matches, returns an empty list.
    """
    out: list[str] = []
    aliases = _lang_aliases(lang) if lang else None
    for tag, body in _FENCE.findall(text):
        if aliases is None or tag.lower() in aliases:
            out.append(body.strip("\n"))
    return out


def best_code(text: str, lang: str | None = None) -> str:
    """Best-effort single code block: the first matching fenced block, else
    the whole text stripped (some models omit fences)."""
    blocks = code_blocks(text, lang)
    if blocks:
        return blocks[0]
    # Fall back to any fenced block regardless of language.
    if lang:
        any_block = code_blocks(text, None)
        if any_block:
            return any_block[0]
    return text.strip()


def _lang_aliases(lang: str) -> set[str]:
    lang = lang.lower()
    groups = [
        {"python", "py", "python3"},
        {"sql", "sqlite", "postgresql", "mysql"},
        {"json", "json5"},
    ]
    for g in groups:
        if lang in g:
            return g | {""}       # allow untagged blocks too
    return {lang, ""}
