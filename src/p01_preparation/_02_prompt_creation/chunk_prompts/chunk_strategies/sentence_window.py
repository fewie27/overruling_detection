from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.p01_preparation._02_prompt_creation.chunk_prompts.citation_chunking import CitationSpan
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_strategies.base import ChunkSegment
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_strategies.sentence_spans import (
    sentence_spans,
)


def _sentence_index_for_pos(sentence_bounds: list[tuple[int, int]], pos: int) -> int:
    for i, (lo, hi) in enumerate(sentence_bounds):
        if lo <= pos < hi:
            return i
    if not sentence_bounds:
        return 0
    if pos >= sentence_bounds[-1][1]:
        return len(sentence_bounds) - 1
    return 0


@dataclass(frozen=True)
class SentenceWindowStrategy:
    """Full sentences: ``before`` sentences before the cite block, ``after`` sentences after."""

    before: int
    after: int

    def segments(self, text: str, spans: Sequence[CitationSpan]) -> list[ChunkSegment]:
        bounds = sentence_spans(text)
        ordered = sorted(spans, key=lambda s: s.start)
        out: list[ChunkSegment] = []
        for span in ordered:
            i0 = _sentence_index_for_pos(bounds, span.start)
            i1 = _sentence_index_for_pos(bounds, max(span.start, span.end - 1))
            lo_i = max(0, min(i0, i1) - self.before)
            hi_i = min(len(bounds) - 1, max(i0, i1) + self.after)
            lo = bounds[lo_i][0]
            hi = bounds[hi_i][1]
            out.append(ChunkSegment(plain_text=text[lo:hi].strip(), surfaces=(span.surface,)))
        return out
