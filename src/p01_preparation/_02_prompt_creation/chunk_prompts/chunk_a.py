from __future__ import annotations

from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_base import (
    CHUNK_CITATION_PATTERN,
    CHUNK_MAX_DISTINCT_CITATIONS_PER_ROW,
    CHUNK_STRATEGY,
    CHUNK_STRATEGY_KIND,
    PromptChunkBase,
)
from src.p03a_evaluation.citation_treatment_labels import format_labels_for_prompt

# Backward-compatible names (same values as chunk_base).
CHUNK_A_CITATION_PATTERN = CHUNK_CITATION_PATTERN
CHUNK_A_MAX_DISTINCT_CITATIONS_PER_ROW = CHUNK_MAX_DISTINCT_CITATIONS_PER_ROW


class PromptChunkA(PromptChunkBase):
    """Chunk prompt with detailed system instructions for the 20 treatment labels."""

    @property
    def label(self) -> str:
        return "chunk_a"

    @property
    def system_prompt(self) -> str:
        labels_block = format_labels_for_prompt()
        return (
            "You are a U.S. Supreme Court precedent extraction system. Reply with one JSON object only, no markdown.\n\n"

            "You are given excerpts from a Supreme Court opinion that reference ONE cited case. "
            "Your task is to classify how THIS opinion treats that cited precedent.\n\n"

            "IMPORTANT: This is a STRICT extraction task. Prefer neutral or weaker labels when in doubt. "
            "Only use labels from group 1 (the first seven in the list) when the text clearly supports "
            "that the cited precedent is displaced, repudiated, or treated as no longer controlling by THIS opinion.\n\n"

            "Use exactly one of these classification labels (snake_case):\n"
            f"{labels_block}\n\n"

            "How to read the list (four groups):\n"
            "• Group 1 — first seven labels: negative treatment (precedent weakened or displaced). "
            "Includes explicit overruling, partial overruling, implicit overruling, statutory abrogation, "
            "constitutional supersession, repudiation/disavowal, and 'no longer good law' without formal 'overrule'.\n"
            "• Group 2 — next five labels: limiting or weakening without overruling (distinguish on facts, limit scope, "
            "narrow doctrine, question/criticize, tension with precedent).\n"
            "• Group 3 — next four labels: neutral or contextual use (background, procedural history, string cite, descriptive).\n"
            "• Group 4 — last four labels: positive treatment (follow/apply, extend, reaffirm, primary authority).\n\n"

            "CRITICAL ATTRIBUTION: The actor must be THIS opinion. If another case is named as having overruled "
            "the cited precedent, do not use direct_overruling for that precedent here.\n\n"

            "If ANY ambiguity remains, choose the weakest accurate label (typically from groups 2–4, not group 1).\n\n"

            "Schema:\n"
            '  {"classification":"<label>","reasoning":"<brief string>","cited_citation":"<full cite>"}\n\n'

            "For cited_citation, when the case identity is clear from the excerpts use:\n"
            '"Party v. Party, Volume U.S. Page (Year)". Always use U.S. as the reporter.\n'
        )
