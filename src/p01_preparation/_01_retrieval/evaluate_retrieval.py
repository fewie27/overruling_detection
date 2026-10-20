from __future__ import annotations

from typing import Any

from src.models import CaseDocument
from src.models.case_trace import PipelineInputRow
from src.p01_preparation._01_retrieval.benchmark import case_matches_retrieved_caption
from src.p01_preparation._01_retrieval.case_providers.base import CaseProvider
from src.p01_preparation._01_retrieval.case_providers.benchmark_cleaned_text import BenchmarkCleanedTextProvider
from src.p02_execution.pipeline_core import PreparedRun
from src.p03a_evaluation.result_output import EvaluationRow


def _benchmark_entry_for_fallback(row: PipelineInputRow) -> dict[str, Any] | None:
    if not row.opinion_text_override:
        return None
    return {
        "case_name": row.case_name_override or row.display_case_name,
        "citation": row.lookup_citation,
        "cleaned_text_opinion": row.opinion_text_override,
    }


def _fetch_case_matching_dataset(
    provider: CaseProvider,
    case_citation: str,
    lookup_citation: str,
    *,
    require_full_casebody: bool = False,
) -> CaseDocument:
    def _caption_ok(name: str) -> bool:
        return case_matches_retrieved_caption(case_citation, name)

    case_doc = provider.get_case_by_citation(
        lookup_citation,
        case_law_caption_matcher=_caption_ok,
        require_full_casebody=require_full_casebody,
    )
    if case_matches_retrieved_caption(case_citation, case_doc.case_name):
        return case_doc
    if lookup_citation != case_citation:
        try:
            alt = provider.get_case_by_citation(
                case_citation,
                case_law_caption_matcher=_caption_ok,
                require_full_casebody=require_full_casebody,
            )
        except Exception:
            raise RuntimeError(
                f"Citation {lookup_citation!r} retrieved {case_doc.case_name!r}, "
                f"which does not match dataset case {case_citation!r}; "
                f"fallback citation also failed."
            ) from None
        if case_matches_retrieved_caption(case_citation, alt.case_name):
            return alt
    raise RuntimeError(
        f"Citation {lookup_citation!r} retrieved {case_doc.case_name!r}, "
        f"which does not match dataset case {case_citation!r}. "
        f"Fix benchmark JSONL 'lookup_citation'/'case_name' for this row."
    )


def retrieve_benchmark_case(
    row: PipelineInputRow,
    provider: CaseProvider,
    *,
    require_full_casebody: bool = False,
) -> CaseDocument:
    """Retrieve one benchmark opinion via the configured provider chain."""
    try:
        return _fetch_case_matching_dataset(
            provider,
            row.case_citation,
            row.lookup_citation,
            require_full_casebody=require_full_casebody,
        )
    except Exception as primary_exc:
        fallback_entry = _benchmark_entry_for_fallback(row)
        if fallback_entry is None:
            raise primary_exc
        backup_provider = BenchmarkCleanedTextProvider(fallback_entry)
        return backup_provider.get_case_by_citation(row.lookup_citation)


def prepare_benchmark_pairs(
    rows: list[PipelineInputRow],
    provider: CaseProvider,
    *,
    print_errors: bool,
    require_full_casebody: bool = False,
) -> tuple[list[tuple[PipelineInputRow, CaseDocument]], PreparedRun]:
    """Retrieve benchmark opinions; failed rows become precomputed ERROR entries."""
    pairs: list[tuple[PipelineInputRow, CaseDocument]] = []
    precomputed_results: list[EvaluationRow] = []
    precomputed_llm_responses: list[dict[str, object]] = []

    for row in rows:
        row_no = row.row_no
        provider_label = provider.label
        case_citation_for_output = row.case_citation
        try:
            case_doc = retrieve_benchmark_case(
                row, provider, require_full_casebody=require_full_casebody
            )
            provider_label = case_doc.retrieval_provider_label or provider.label
            pairs.append((row, case_doc))
        except Exception as exc:
            if print_errors:
                cite = f" lookup={row.lookup_citation!r}" if row.lookup_citation else ""
                print(f"[ERROR] row={row_no} case='{row.display_case_name}'{cite} -> {exc}")
            precomputed_results.append(
                EvaluationRow(
                    index_no=row_no,
                    case_name=row.display_case_name,
                    provider_label=provider_label,
                    expected=list(row.expected_classifications),
                    predicted=[],
                    evaluation="ERROR",
                    error_message=str(exc),
                )
            )
            precomputed_llm_responses.append(
                {
                    "index": row_no,
                    "case_citation": case_citation_for_output,
                    "trace": row.trace.to_dict(),
                    "state": "ERROR",
                    "response": {},
                    "chunk_texts": None,
                    "citation_classification_state": None,
                }
            )

    precomputed = PreparedRun([], precomputed_results, precomputed_llm_responses)
    return pairs, precomputed
