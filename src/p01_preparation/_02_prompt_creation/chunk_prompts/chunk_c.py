from __future__ import annotations

from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_base import (
    CHUNK_CITATION_PATTERN,
    CHUNK_STRATEGY,
    PromptChunkBase,
)
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_strategies import CitationChunkStrategy
from src.p03a_evaluation.citation_treatment_labels import format_labels_for_prompt


class PromptChunkC(PromptChunkBase):
    """Chunk-b-style labels/schema with extra rules against outcome-based overruling inference."""

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
        return "chunk_c"

    @property
    def system_prompt(self) -> str:
        labels_block = format_labels_for_prompt()
        supporting = """
OUTCOME INFERENCE IS FORBIDDEN:

Do NOT infer overruling from differences in outcomes between cases.

Example:
If a prior case is described as requiring a certain result,
and this opinion later reaches a different result,
you MUST NOT conclude implicit overruling unless the text explicitly
rejects or criticizes the cited case.

A citation that appears in a description of what a lower court did
(e.g., "the Court of Appeals held that X compelled it")
is procedural_history, NOT overruling.

LOWER COURT RULE:

If the cited case is mentioned as something a lower court relied on,
then classify as procedural_history.

This holds EVEN IF:
- the lower court's reasoning is later rejected
- the outcome of this case differs

DO NOT UPGRADE BASED ON:

- "compelled"
- "required"
- "held unconstitutional"
- differences in legal outcomes

These indicate how a case was used, not that it is overruled.
""".strip()
        return (
            "You are a U.S. Supreme Court precedent extraction system. Reply with one JSON object only, no markdown.\n\n"

            "You see excerpts that reference ONE cited case. Classify how *THIS opinion* treats that precedent.\n\n"

            "Use exactly one label from the list below (snake_case). Prefer higher labels (earlier in the list) when unsure or ambiguous. "
            "\n\n"

            "Labels (decreasing strength):\n"
            f"{labels_block}\n\n"

            "Schema:\n"
            '{"classification":"<label>","cited_citation":"<full cite>"}\n\n'
            "Citation Format: \"Party v. Party, Volume U.S. Page (Year)\" (prefer U.S. reporter).\n\n"

            f"{supporting}\n"
        )
