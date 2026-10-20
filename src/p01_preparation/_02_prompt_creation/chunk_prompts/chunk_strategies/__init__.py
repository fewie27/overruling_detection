"""
Pluggable citation excerpt strategies (word / sentence / paragraph windows).

Use :func:`build_chunk_blocks` with a strategy instance from
:class:`WordWindowStrategy`, :class:`SentenceWindowStrategy`, or :class:`ParagraphStrategy`.
"""

from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_strategies.base import (
    ChunkSegment,
    CitationChunkStrategy,
    build_chunk_blocks,
    render_prompt_segments,
)
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_strategies.factory import (
    ChunkStrategyKind,
    make_chunk_strategy,
)
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_strategies.paragraph import (
    ParagraphStrategy,
    paragraph_bounds_containing,
)
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_strategies.sentence_spans import (
    sentence_spans,
)
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_strategies.sentence_window import (
    SentenceWindowStrategy,
)
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_strategies.word_window import (
    WordWindowStrategy,
)

__all__ = [
    "ChunkSegment",
    "ChunkStrategyKind",
    "CitationChunkStrategy",
    "ParagraphStrategy",
    "SentenceWindowStrategy",
    "WordWindowStrategy",
    "build_chunk_blocks",
    "make_chunk_strategy",
    "paragraph_bounds_containing",
    "render_prompt_segments",
    "sentence_spans",
]
