from __future__ import annotations

import re
from dataclasses import dataclass

from collections.abc import Sequence

from src.p01_preparation._02_prompt_creation.chunk_prompts.citation_chunking import CitationSpan
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_strategies.base import ChunkSegment


def _char_span_words_left(text: str, start: int, n: int) -> int:
    if n <= 0:
        return start
    pre = text[:start]
    words = list(re.finditer(r"\S+", pre))
    if not words:
        return 0
    if len(words) < n:
        return words[0].start()
    return words[-n].start()


def _char_span_words_right(text: str, end: int, n: int) -> int:
    if n <= 0:
        return end
    post = text[end:]
    words = list(re.finditer(r"\S+", post))
    if not words:
        return end
    sel = words[:n]
    return end + sel[-1].end()


@dataclass(frozen=True)
class WordWindowStrategy:
    """A fixed number of whitespace-delimited words before and after the citation span."""

    before: int
    after: int

    def segments(self, text: str, spans: Sequence[CitationSpan]) -> list[ChunkSegment]:
        ordered = sorted(spans, key=lambda s: s.start)
        out: list[ChunkSegment] = []
        for span in ordered:
            lo = _char_span_words_left(text, span.start, self.before)
            hi = _char_span_words_right(text, span.end, self.after)
            out.append(ChunkSegment(plain_text=text[lo:hi].strip(), surfaces=(span.surface,)))
        return out
