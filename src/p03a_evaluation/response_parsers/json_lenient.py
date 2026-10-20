"""Recover JSON objects/arrays from typical LLM output.

Parsing strategy (per candidate text span):
  1. ``json.loads`` on well-formed JSON
  2. Heuristic repair of unescaped quotes inside string values
  3. ``partial_json_parser`` for truncated objects/arrays/strings
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from typing import Any

from partial_json_parser import Allow, loads as partial_json_loads

_FENCE_BLOCK = re.compile(r"```(?:json)?\s*\r?\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_OPENING_FENCE = re.compile(r"^```(?:json)?\s*\r?\n?", re.IGNORECASE)
_CLOSING_FENCE = re.compile(r"\s*```\s*$")
# R1-style / Llama-distill reasoning wrappers (closed or truncated mid-generation).
_THINK_BLOCK = re.compile(r"<think>\s*.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)
_MAX_INNER_QUOTE_REPAIRS = 200
_PARTIAL_ALLOW = Allow.ALL

_ParseFn = Callable[[str], Any | None]


def load_json_lenient(text: str) -> Any:
    """Parse the best JSON value recoverable from *text* (raises on total failure)."""
    s = _normalize_input(text)
    if not s:
        return {}

    values = list(_iter_parsed_values(s))
    if values:
        return _select_best_value(values)

    raise json.JSONDecodeError("No JSON object or array found", s, 0)


def _select_best_value(values: list[Any]) -> Any:
    """Prefer the richest structure; break ties by later document position."""
    return max(enumerate(values), key=lambda item: (_value_rank(item[1]), item[0]))[1]


def _value_rank(value: Any) -> tuple[int, int]:
    """Sort key for choosing the best parse (higher is better)."""
    if isinstance(value, dict):
        nested = sum(_value_rank(v)[1] for v in value.values())
        return (3, len(value) + nested)
    if isinstance(value, list):
        nested = sum(_value_rank(item)[1] for item in value)
        return (2, len(value) + nested)
    return (1, 1)


def split_think_content(text: str) -> tuple[str, str | None]:
    """Split R1-style ``<think>…</think>`` from the remainder.

    Returns ``(answer_without_think, reasoning_or_none)``. Truncated open
    ``<think>`` blocks (no closing tag) are treated as reasoning with an empty answer.
    """
    s = (text or "").strip()
    if not s:
        return "", None
    matches = list(_THINK_BLOCK.finditer(s))
    if not matches:
        return s, None
    reasoning_parts: list[str] = []
    for m in matches:
        inner = m.group(0)
        # Strip the opening tag and optional closing tag.
        inner = re.sub(r"^<think>\s*", "", inner, count=1, flags=re.IGNORECASE)
        inner = re.sub(r"\s*</think>\s*$", "", inner, count=1, flags=re.IGNORECASE)
        if inner.strip():
            reasoning_parts.append(inner.strip())
    answer = _THINK_BLOCK.sub("", s).strip()
    reasoning = "\n".join(reasoning_parts).strip() or None
    return answer, reasoning


def _strip_think_blocks(text: str) -> str:
    """Remove ``<think>…</think>`` (and truncated open ``<think>…``) so JSON parsing
    is not confused by braces inside the reasoning channel."""
    answer, _ = split_think_content(text)
    return answer


def _normalize_input(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("\ufeff"):
        s = s.lstrip("\ufeff").strip()
    return _strip_think_blocks(s)


def _iter_parsed_values(text: str) -> Iterator[Any]:
    for candidate in _iter_json_text_candidates(text):
        value = _parse_json_candidate(candidate)
        if value is not None:
            yield value


def _iter_json_text_candidates(text: str) -> Iterator[str]:
    """Yield text spans that may contain JSON, in document order."""
    stripped = text.strip()
    if not stripped:
        return

    fenced_blocks = _FENCE_BLOCK.findall(stripped)
    if fenced_blocks:
        yield from (block.strip() for block in fenced_blocks if block.strip())
        return

    seen: set[str] = set()

    def offer(span: str) -> Iterator[str]:
        s = span.strip()
        if not s or s in seen:
            return
        seen.add(s)
        yield s

    yield from offer(_strip_markdown_fences(stripped))
    yield from offer(stripped)

    for snippet in _iter_balanced_snippets(stripped):
        yield from offer(snippet)

    for index, char in enumerate(stripped):
        if char in "{[":
            yield from offer(stripped[index:])
            break


def _strip_markdown_fences(text: str) -> str:
    """Remove a leading `` ```json `` fence and an optional trailing `` ``` ``."""
    if not text.startswith("```"):
        return text
    without_open = _OPENING_FENCE.sub("", text, count=1)
    return _CLOSING_FENCE.sub("", without_open).strip()


def _parse_json_candidate(text: str) -> Any | None:
    for parser in _PARSERS:
        try:
            value = parser(text)
        except json.JSONDecodeError:
            continue
        if value is not None:
            return value
    return None


def _parse_strict(text: str) -> Any:
    return json.loads(text)


def _parse_with_inner_quote_repair(text: str) -> Any:
    current = text
    last_error: json.JSONDecodeError | None = None
    for _ in range(_MAX_INNER_QUOTE_REPAIRS):
        try:
            return json.loads(current)
        except json.JSONDecodeError as exc:
            last_error = exc
            quote_pos = _find_repairable_inner_quote(current, exc.pos or 0)
            if quote_pos is None:
                raise
            current = current[:quote_pos] + "\\" + current[quote_pos:]
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("JSON quote repair failed", text, 0)


def _parse_partial(text: str) -> Any | None:
    """Parse truncated JSON; returns ``None`` when the span is not JSON-like."""
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return None
    try:
        return partial_json_loads(stripped, _PARTIAL_ALLOW)
    except Exception:
        return None


def _iter_balanced_snippets(text: str) -> Iterator[str]:
    """Yield complete balanced ``{...}`` / ``[...]`` substrings."""
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char in "{[":
            end = _match_bracket(text, index)
            if end is not None:
                yield text[index : end + 1]
                index = end + 1
                continue
        index += 1


def _match_bracket(text: str, start: int) -> int | None:
    """Return the index of the bracket that closes the opener at *start*."""
    open_char = text[start]
    close_char = "}" if open_char == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index
    return None


def _find_repairable_inner_quote(text: str, error_pos: int) -> int | None:
    """Locate an unescaped ``"`` before *error_pos* that likely sits inside a string value."""
    search_end = min(max(error_pos, 0), len(text))
    for index in range(search_end - 1, -1, -1):
        if text[index] != '"':
            continue
        if index > 0 and text[index - 1] == "\\":
            continue
        next_pos = index + 1
        while next_pos < len(text) and text[next_pos] in " \t\r\n":
            next_pos += 1
        if next_pos < len(text) and text[next_pos] in ",}]:":
            continue
        prev_pos = index - 1
        while prev_pos >= 0 and text[prev_pos] in " \t\r\n":
            prev_pos -= 1
        if prev_pos >= 0 and text[prev_pos] not in "{[,:":
            return index
    return None


_PARSERS: tuple[_ParseFn, ...] = (
    _parse_strict,
    _parse_with_inner_quote_repair,
    _parse_partial,
)
