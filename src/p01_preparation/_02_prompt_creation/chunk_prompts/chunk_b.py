from __future__ import annotations

from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_base import (
    CHUNK_CITATION_PATTERN,
    CHUNK_STRATEGY,
    PromptChunkBase,
)
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_strategies import CitationChunkStrategy
from src.p03a_evaluation.citation_treatment_labels import format_labels_for_prompt


class PromptChunkB(PromptChunkBase):
    """Chunk prompt with shorter system instructions; same retrieval pipeline as chunk_a."""

    def __init__(
        self,
        *,
        citation_pattern: str = CHUNK_CITATION_PATTERN,
        chunk_strategy: CitationChunkStrategy | None = None,
    ) -> None:
        super().__init__(
            citation_pattern=citation_pattern,
            chunk_strategy=chunk_strategy if chunk_strategy is not None else CHUNK_STRATEGY,
        )

    @property
    def label(self) -> str:
        return "chunk_b"

    @property
    def system_prompt(self) -> str:
        labels_block = format_labels_for_prompt()
        return (
            "You are a U.S. Supreme Court precedent extraction system. Reply with one JSON object only, no markdown.\n\n"

            "You see excerpts that reference ONE cited case. Classify how *THIS opinion* treats that precedent. Be careful when classifying: only respect the information from the excerpt and do not mix up with opinions from other courts. Make sure you label in accordance to *THIS* opinion!\n\n"

            "Use exactly one label from the list below (snake_case). Prefer higher labels (earlier in the list) when unsure or ambiguous. "
            "\n\n"

            "Labels (decreasing strength):\n"
            f"{labels_block}\n\n"

            "Schema:\n"
            '{"classification":"<label>","cited_citation":"<full cite>"}\n\n'
            "Citation Format: \"Party v. Party, Volume U.S. Page (Year)\" (prefer U.S. reporter).\n"
        )
