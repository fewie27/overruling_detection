from __future__ import annotations

import re
from abc import ABC, abstractmethod

from src.models import CaseDocument, PreparedLLMRequest, PromptContext
from src.p01_preparation._01_retrieval.benchmark import DatasetInputRow
from src.p01_preparation._02_prompt_creation.base import PromptBuilder
from src.p01_preparation._02_prompt_creation.chunk_prompts.citation_chunking import (
    group_us_reporter_citation_spans,
    parse_volume_reporter_page,
)
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_strategies import (
    ChunkStrategyKind,
    CitationChunkStrategy,
    ParagraphStrategy,
    build_chunk_blocks,
    make_chunk_strategy,
)

CHUNK_CITATION_PATTERN = (
    r"\b\d+\s+[A-Za-z][A-Za-z0-9.\s]{1,40}?\s+(?:\([A-Za-z0-9.\s]+\)\s+)?\d+\b"
)

CHUNK_STRATEGY_KIND: ChunkStrategyKind = "words"
CHUNK_STRATEGY = make_chunk_strategy(
    CHUNK_STRATEGY_KIND,
    words_before=100,
    words_after=100,
    sentences_before=5,
    sentences_after=5,
)

# Each distinct group = one LLM call. None = no cap.
CHUNK_MAX_DISTINCT_CITATIONS_PER_ROW: int | None = 120


def canonical_citation_label(surface: str) -> str:
    p = parse_volume_reporter_page(surface)
    if p is None:
        return surface.strip()
    vol, _rep, page = p
    return f"{vol} U.S. {page}"


class PromptChunkBase(PromptBuilder, ABC):
    """
    Shared pipeline: regexp U.S. cites, grouped by volume/reporter/page, one LLM call per group.
    Subclasses differ only in ``label`` and ``system_prompt``.
    """

    def __init__(
        self,
        *,
        citation_pattern: str,
        chunk_strategy: CitationChunkStrategy | None = None,
    ) -> None:
        if not (citation_pattern or "").strip():
            raise ValueError("citation_pattern must be a non-empty string")
        self._pattern = re.compile(citation_pattern)
        self._strategy = chunk_strategy if chunk_strategy is not None else CHUNK_STRATEGY

    @property
    @abstractmethod
    def label(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        raise NotImplementedError

    @property
    def response_parser_id(self) -> str:
        return "P3_chunk_classification_parser"

    def build_user_prompt(self, ctx: PromptContext) -> str:
        return (
            f"[Template] Chunk-based citation classification for {ctx.case_name}, {ctx.citation}. "
            "Actual runs send grouped windows per matched citation; see expand_prepared_requests."
        )

    def expand_prepared_requests(
        self,
        *,
        row: DatasetInputRow,
        case_doc: CaseDocument,
        provider_label: str,
        case_citation_for_output: str,
    ) -> list[PreparedLLMRequest] | None:
        text = case_doc.majority_opinion
        groups = group_us_reporter_citation_spans(text, self._pattern)
        if not groups:
            return []

        sorted_items = sorted(groups.items(), key=lambda kv: kv[0])
        total_distinct = len(sorted_items)
        if (
            CHUNK_MAX_DISTINCT_CITATIONS_PER_ROW is not None
            and total_distinct > CHUNK_MAX_DISTINCT_CITATIONS_PER_ROW
        ):
            print(
                f"[{self.label}] row {row.row_no}: capping at {CHUNK_MAX_DISTINCT_CITATIONS_PER_ROW} of "
                f"{total_distinct} distinct U.S.-reporter groups (raise CHUNK_MAX_DISTINCT_CITATIONS_PER_ROW "
                f"or set to None in chunk_base.py).",
                flush=True,
            )
            sorted_items = sorted_items[:CHUNK_MAX_DISTINCT_CITATIONS_PER_ROW]

        ctx_name = case_doc.case_name
        ctx_cite = case_doc.citation
        out: list[PreparedLLMRequest] = []
        for i, (_gkey, spans) in enumerate(sorted_items):
            spans_sorted = sorted(spans, key=lambda s: s.start)
            blocks, chunk_texts = build_chunk_blocks(text, spans_sorted, self._strategy)
            body = "\n\n".join(blocks)
            unique_surfaces = list(dict.fromkeys(sp.surface for sp in spans_sorted))
            surfaces_line = "; ".join(unique_surfaces)
            primary = unique_surfaces[0] if unique_surfaces else ""
            cite_to_check = canonical_citation_label(primary)
            if isinstance(self._strategy, ParagraphStrategy):
                excerpt_blurb = (
                    "This single request covers every excerpt below. Each excerpt is one paragraph from the opinion "
                    "that contains a reference to this same cited precedent (grouped by volume, U.S. reporter, and page). "
                    "Substrings as they appear in the opinion:\n"
                    f"{surfaces_line}\n\n"
                    "Paragraph excerpts (headers repeat the cite substring(s) for that paragraph):\n\n"
                )
            else:
                excerpt_blurb = (
                    "This single request covers every excerpt below. Each excerpt is a window of text around one "
                    "appearance of this same cited precedent (grouped by volume, U.S. reporter, and page). "
                    "Substrings as they appear in the opinion:\n"
                    f"{surfaces_line}\n\n"
                    "Excerpts (each chunk shows the cite as quoted in the header):\n\n"
                )
            user = (
                f"Opinion: {ctx_name}, {ctx_cite}.\n\n"
                f"Citation to check: {cite_to_check}.\n"
                f"{excerpt_blurb}"
                f"{body}\n\n"
                "Classify how the current opinion treats this cited precedent using the JSON schema in your instructions."
            )
            rid = f"row-{row.row_no}-cite-{i}"
            out.append(
                PreparedLLMRequest(
                    request_id=rid,
                    row_no=row.row_no,
                    display_case_name=row.display_case_name,
                    provider_label=provider_label,
                    expected_classifications=list(row.expected_classifications),
                    case_citation_for_output=case_citation_for_output,
                    system_prompt=self.system_prompt,
                    user_prompt=user,
                    lookup_citation=row.lookup_citation,
                    finetune_system_prompt=self.system_prompt,
                    finetune_user_prompt=user,
                    merge_row_responses=True,
                    chunk_texts=chunk_texts,
                    chunk_citation_surfaces=unique_surfaces,
                    response_parser_id=self.response_parser_id,
                    trace=row.trace,
                )
            )
        return out
