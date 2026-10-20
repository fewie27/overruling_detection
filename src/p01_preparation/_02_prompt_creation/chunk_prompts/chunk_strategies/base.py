from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from src.p01_preparation._02_prompt_creation.chunk_prompts.citation_chunking import CitationSpan


@dataclass(frozen=True)
class ChunkSegment:
    """One excerpt shown to the model: body text and cite surface(s) for the header."""

    plain_text: str
    surfaces: tuple[str, ...]


@runtime_checkable
class CitationChunkStrategy(Protocol):
    """Turns citation spans into ordered chunk segments (document order)."""

    def segments(self, text: str, spans: Sequence[CitationSpan]) -> list[ChunkSegment]:
        ...


def render_prompt_segments(segments: Sequence[ChunkSegment]) -> tuple[list[str], list[str]]:
    """Build labeled blocks and parallel plain-text list for logging / JSON."""
    blocks: list[str] = []
    plain: list[str] = []
    for i, seg in enumerate(segments, start=1):
        body = seg.plain_text
        plain.append(body)
        surf_line = ", ".join(repr(s) for s in seg.surfaces)
        blocks.append(f"--- Chunk {i} (cite as in opinion: {surf_line}) ---\n{body}")
    return blocks, plain


def build_chunk_blocks(
    text: str,
    spans: Sequence[CitationSpan],
    strategy: CitationChunkStrategy,
) -> tuple[list[str], list[str]]:
    """High-level entry: strategy produces segments, then shared formatting."""
    return render_prompt_segments(strategy.segments(text, spans))
