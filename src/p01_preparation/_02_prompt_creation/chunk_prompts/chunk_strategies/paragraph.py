from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from src.p01_preparation._02_prompt_creation.chunk_prompts.citation_chunking import CitationSpan
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_strategies.base import ChunkSegment

_PARA_BREAK = re.compile(r"(?:\r?\n\s*){2,}")


def paragraph_bounds_containing(text: str, start: int, end: int) -> tuple[int, int]:
    """Character span of the paragraph (blank-line delimited) that contains ``[start, end)``."""
    a = min(max(0, start), len(text))
    b = min(max(0, end), len(text))
    if a > b:
        a, b = b, a
    last = 0
    for m in _PARA_BREAK.finditer(text):
        seg_lo, seg_hi = last, m.start()
        if seg_lo <= a < seg_hi:
            return seg_lo, seg_hi
        last = m.end()
    return last, len(text)


@dataclass(frozen=True)
class ParagraphStrategy:
    """Full paragraph per cite; multiple cites in the same paragraph merge into one segment."""

    def segments(self, text: str, spans: Sequence[CitationSpan]) -> list[ChunkSegment]:
        ordered = sorted(spans, key=lambda s: s.start)
        by_para: dict[tuple[int, int], list[str]] = {}
        order_keys: list[tuple[int, int]] = []
        for span in ordered:
            lo, hi = paragraph_bounds_containing(text, span.start, span.end)
            key = (lo, hi)
            if key not in by_para:
                by_para[key] = []
                order_keys.append(key)
            surf_list = by_para[key]
            if span.surface not in surf_list:
                surf_list.append(span.surface)
        out: list[ChunkSegment] = []
        for key in order_keys:
            lo, hi = key
            chunk = text[lo:hi].strip()
            out.append(ChunkSegment(plain_text=chunk, surfaces=tuple(by_para[key])))
        return out
