"""
Sentence boundary detection for legal (especially SCOTUS) opinion text.

Heuristic rules (no extra dependencies):
- ``.`` ``!`` ``?`` can end a sentence only when followed by content that looks like a new
  sentence (uppercase, quote, ``(``, ``[``, section sign) after optional space;
  ``!`` / ``?`` almost always end sentences.
- Periods after abbreviation-style tokens (``U.S.``, ``F.2d``, ``v.``, single initials, …)
  do **not** end a sentence.
- ``3.14``-style decimals do not split.
- Ellipsis runs (``...``) are skipped as boundaries.
"""

from __future__ import annotations

import re

# Normalized token keys (punctuation stripped, lower) for dotted abbreviations / reporters.
_LEGAL_ABBREV_KEYS: frozenset[str] = frozenset(
    {
        "a",
        "a2d",
        "a3d",
        "am",
        "app",
        "appx",
        "b",
        "c",
        "civ",
        "cf",
        "ch",
        "ci",
        "co",
        "cong",
        "d",
        "dept",
        "dr",
        "e",
        "eg",
        "etal",
        "etc",
        "f",
        "f2d",
        "f3d",
        "f4th",
        "fed",
        "ff",
        "fig",
        "fla",
        "fn",
        "id",
        "ie",
        "inc",
        "jr",
        "l",
        "led",
        "led2d",
        "ll",
        "llp",
        "ltd",
        "m",
        "mr",
        "mrs",
        "ms",
        "n",
        "nn",
        "no",
        "ns",
        "p",
        "pat",
        "pct",
        "pet",
        "pm",
        "pp",
        "r",
        "rep",
        "resp",
        "rev",
        "s",
        "sct",
        "sr",
        "st",
        "sur",
        "t",
        "temp",
        "u",
        "usc",
        "uss",
        "us",
        "v",
        "vol",
        "vs",
        "w",
        "wash",
        "so2d",
        "sw2d",
        "sw3d",
        "ne2d",
        "nw2d",
        "se2d",
        "seq",
        "xx",
    }
)


def _abbrev_normal_form(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.lower())


def _token_immediately_before_period(text: str, dot_idx: int) -> str:
    if dot_idx < 0 or dot_idx >= len(text):
        return ""
    i = dot_idx - 1
    while i >= 0 and text[i] not in " \t\n\r\f\v":
        i -= 1
    return text[i + 1 : dot_idx + 1]


def _is_decimal_number(text: str, dot_idx: int) -> bool:
    if dot_idx <= 0 or text[dot_idx] != ".":
        return False
    if text[dot_idx - 1].isdigit() and dot_idx + 1 < len(text) and text[dot_idx + 1].isdigit():
        return True
    return False


def _starts_new_sentence_unit(text: str, after_punct: int) -> bool:
    j = after_punct
    while j < len(text) and text[j] in " \t":
        j += 1
    if j >= len(text):
        return True
    if text[j] == "\n":
        k = j + 1
        while k < len(text) and text[k] in " \t":
            k += 1
        if k >= len(text):
            return True
        return text[k].isupper() or text[k] in '"\'('
    c = text[j]
    return c.isupper() or c in '"\'([§'


def _period_closes_known_abbrev(text: str, dot_idx: int) -> bool:
    if text[dot_idx] != ".":
        return False
    tok = _token_immediately_before_period(text, dot_idx)
    if not tok.endswith("."):
        return False
    core = tok[:-1]
    if re.fullmatch(r"\d+", core or ""):
        return False
    if re.fullmatch(r"[\d,]+", core or ""):
        return False
    nf = _abbrev_normal_form(tok)
    if not nf:
        return False
    if nf in _LEGAL_ABBREV_KEYS:
        return True
    if len(nf) == 1 and nf.isalpha():
        return True
    return False


def _is_sentence_break_at(text: str, punct_idx: int) -> bool:
    ch = text[punct_idx]
    if ch not in ".!?":
        return False
    if ch in "!?":
        return _starts_new_sentence_unit(text, punct_idx + 1)
    if _is_decimal_number(text, punct_idx):
        return False
    if _period_closes_known_abbrev(text, punct_idx):
        return False
    return _starts_new_sentence_unit(text, punct_idx + 1)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """
    Non-overlapping ``[start, end)`` spans that partition ``text`` (sentence includes closing ``.!?``).
    """
    if not text:
        return []

    n = len(text)
    spans: list[tuple[int, int]] = []
    start = 0
    i = 0
    while i < n:
        c = text[i]
        if c == "." and i + 1 < n and text[i + 1] == ".":
            while i < n and text[i] == ".":
                i += 1
            continue
        if c in ".!?" and _is_sentence_break_at(text, i):
            hi = i + 1
            spans.append((start, hi))
            i = hi
            while i < n and text[i] in " \t\n\r":
                i += 1
            start = i
            continue
        i += 1

    if start < n:
        spans.append((start, n))

    return spans if spans else [(0, n)]
