from __future__ import annotations

from src.models import ExtractionResult, dedupe_classified


def merge_single(extractions: list[ExtractionResult]) -> ExtractionResult:
    if len(extractions) != 1:
        raise ValueError("merge_single expects exactly one extraction")
    return extractions[0]


def merge_citation_chunk_row(extractions: list[ExtractionResult]) -> ExtractionResult:
    """Merge per-citation classification responses for one benchmark row."""
    if not extractions:
        return ExtractionResult(classified_citations=[], raw_content="{}")
    classified: list = []
    explanations: list[str] = []
    raw_parts: list[str] = []

    for ext in extractions:
        raw_parts.append(ext.raw_content)
        if ext.explanation and ext.explanation.strip():
            explanations.append(ext.explanation.strip())
        classified.extend(ext.classified_citations)

    return ExtractionResult(
        classified_citations=dedupe_classified(classified),
        raw_content="\n---\n".join(raw_parts),
        explanation="\n".join(explanations) if explanations else None,
    )


def merger_for_prompt_label(prompt_label: str):
    if prompt_label in ("chunk_a", "chunk_b", "chunk_c", "chunk_d"):
        return merge_citation_chunk_row
    return merge_single
