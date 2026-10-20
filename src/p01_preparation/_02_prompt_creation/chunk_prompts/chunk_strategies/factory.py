from __future__ import annotations

from typing import Literal

from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_strategies.base import CitationChunkStrategy
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_strategies.paragraph import ParagraphStrategy
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_strategies.sentence_window import SentenceWindowStrategy
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_strategies.word_window import WordWindowStrategy

ChunkStrategyKind = Literal["words", "sentences", "paragraph"]


def make_chunk_strategy(
    kind: ChunkStrategyKind,
    *,
    words_before: int = 100,
    words_after: int = 100,
    sentences_before: int = 2,
    sentences_after: int = 2,
) -> CitationChunkStrategy:
    """
    Single switch for chunk prompts: word window, sentence window, or full paragraph (per cite group).
    """
    if kind == "paragraph":
        return ParagraphStrategy()
    if kind == "sentences":
        return SentenceWindowStrategy(before=sentences_before, after=sentences_after)
    return WordWindowStrategy(before=words_before, after=words_after)
