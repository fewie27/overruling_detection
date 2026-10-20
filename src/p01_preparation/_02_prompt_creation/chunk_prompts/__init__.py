"""Citation-window / multi-request chunk prompt variants."""

from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_a import (
    CHUNK_A_CITATION_PATTERN,
    CHUNK_A_MAX_DISTINCT_CITATIONS_PER_ROW,
    CHUNK_STRATEGY,
    CHUNK_STRATEGY_KIND,
    PromptChunkA,
)
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_b import PromptChunkB
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_c import PromptChunkC
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_d import PromptChunkD
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_base import (
    CHUNK_CITATION_PATTERN,
    CHUNK_MAX_DISTINCT_CITATIONS_PER_ROW,
    PromptChunkBase,
)
from src.p01_preparation._02_prompt_creation.chunk_prompts.citation_chunking import (
    CitationSpan,
    group_us_reporter_citation_spans,
    is_us_reporter_cite,
    normalize_reporter_token,
    parse_volume_reporter_page,
)

__all__ = [
    "CHUNK_A_CITATION_PATTERN",
    "CHUNK_A_MAX_DISTINCT_CITATIONS_PER_ROW",
    "CHUNK_CITATION_PATTERN",
    "CHUNK_MAX_DISTINCT_CITATIONS_PER_ROW",
    "CHUNK_STRATEGY",
    "CHUNK_STRATEGY_KIND",
    "CitationSpan",
    "PromptChunkA",
    "PromptChunkB",
    "PromptChunkC",
    "PromptChunkD",
    "PromptChunkBase",
    "group_us_reporter_citation_spans",
    "is_us_reporter_cite",
    "normalize_reporter_token",
    "parse_volume_reporter_page",
]
