from __future__ import annotations

from src.models import CaseDocument, PreparedLLMRequest
from src.p01_preparation._01_retrieval.benchmark import DatasetInputRow
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_base import (
    CHUNK_MAX_DISTINCT_CITATIONS_PER_ROW,
    canonical_citation_label,
)
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_b import PromptChunkB
from src.p01_preparation._02_prompt_creation.chunk_prompts.citation_chunking import (
    group_us_reporter_citation_spans,
)
from src.p03a_evaluation.citation_treatment_labels import format_labels_for_prompt


class PromptChunkD(PromptChunkB):
    """
    Same classification task and instructions as chunk_b, but each LLM call receives the
    full majority opinion (one call per distinct U.S.-reporter citation group).
    """

    @property
    def label(self) -> str:
        return "chunk_d"

    @property
    def system_prompt(self) -> str:
        labels_block = format_labels_for_prompt()
        return (
            "You are a U.S. Supreme Court precedent extraction system. Reply with one JSON object only, no markdown.\n\n"

            "You are given the full text of the majority opinion. It includes reference(s) to ONE cited case "
            "(identified in the user message). Classify how *THIS opinion* treats that precedent. Be careful when "
            "classifying: only respect information from this opinion and do not mix up with opinions from other "
            "courts. Make sure you label in accordance to *THIS* opinion!\n\n"

            "Use exactly one label from the list below (snake_case). Prefer higher labels (earlier in the list) when unsure or ambiguous. "
            "\n\n"

            "Labels (decreasing strength):\n"
            f"{labels_block}\n\n"

            "Schema:\n"
            '{"classification":"<label>","cited_citation":"<full cite>"}\n\n'
            "Citation Format: \"Party v. Party, Volume U.S. Page (Year)\" (prefer U.S. reporter).\n"
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
            unique_surfaces = list(dict.fromkeys(sp.surface for sp in spans_sorted))
            surfaces_line = "; ".join(unique_surfaces)
            primary = unique_surfaces[0] if unique_surfaces else ""
            cite_to_check = canonical_citation_label(primary)
            opinion_blurb = (
                "This request includes the complete majority opinion. "
                "Substrings as they appear in the opinion for this cited precedent "
                "(grouped by volume, U.S. reporter, and page):\n"
                f"{surfaces_line}\n\n"
                "Full majority opinion:\n\n"
            )
            user = (
                f"Opinion: {ctx_name}, {ctx_cite}.\n\n"
                f"Citation to check: {cite_to_check}.\n"
                f"{opinion_blurb}"
                f"{text}\n\n"
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
                    chunk_texts=[text],
                    chunk_citation_surfaces=unique_surfaces,
                    response_parser_id=self.response_parser_id,
                )
            )
        return out
