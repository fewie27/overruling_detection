from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Callable

from src.api_connectors.base import ConnectorResponse, LLMConnector
from src.api_usage.base import UsageAggregate
from src.models import (
    CaseDocument,
    ClassifiedCitation,
    ExtractionResult,
    PreparedLLMRequest,
    PromptContext,
    normalize_class_label,
)
from src.models.case_trace import CaseTraceMetadata, PipelineInputRow
from src.p01_preparation._02_prompt_creation.base import PromptBuilder
from src.p03a_evaluation.citation_matching import chunk_citation_state, evaluate_classified_citations
from src.p03a_evaluation.llm_response_parser import parse_llm_json_content, payload_from_raw_content
from src.p03a_evaluation.response_merge import merger_for_prompt_label
from src.p03a_evaluation.result_output import EvaluationRow


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_falsy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("0", "false", "no", "off")


def _should_print_llm_explanation() -> bool:
    if _env_truthy("SILENCE_LLM_EXPLANATION"):
        return False
    if _env_falsy("PRINT_LLM_EXPLANATION"):
        return False
    return True


def _print_llm_explanation_row(row_no: int, case_label: str, explanation: str | None) -> None:
    if not _should_print_llm_explanation() or not explanation:
        return
    print(f"[LLM explanation] row={row_no} case={case_label!r}\n{explanation}\n{'-' * 60}")


def _print_raw_llm_response_row(
    *,
    row_no: int,
    request_id: str,
    case_label: str,
    raw_content: str | None,
) -> None:
    print(
        f"[LLM raw] row={row_no} request_id={request_id} case={case_label!r}",
        flush=True,
    )
    if raw_content is None:
        print("(no response)", flush=True)
    else:
        print(raw_content, flush=True)
    print("-" * 60, flush=True)


def _debug_retrieval_enabled() -> bool:
    return _env_truthy("DEBUG_RETRIEVAL")


def _debug_retrieval(msg: str) -> None:
    if _debug_retrieval_enabled():
        print(f"[DEBUG retrieval] {msg}", flush=True)


def _response_payload(
    *,
    req: PreparedLLMRequest,
    response: dict[str, Any] | None,
    state: str,
    chunk_state: str | None = None,
    raw_llm_content: str | None = None,
    reasoning_content: str | None = None,
) -> dict[str, object]:
    return {
        "index": req.row_no,
        "request_id": req.request_id,
        "case_citation": req.case_citation_for_output,
        "trace": req.trace.to_dict(),
        "response": response or {},
        "chunk_texts": req.chunk_texts,
        "citation_classification_state": chunk_state,
        "state": state,
        "system_prompt": req.system_prompt,
        "user_prompt": req.user_prompt,
        "finetune_system_prompt": req.finetune_system_prompt,
        "finetune_user_prompt": req.finetune_user_prompt,
        "raw_llm_content": raw_llm_content,
        "reasoning_content": reasoning_content,
    }


@dataclass
class PreparedRun:
    requests: list[PreparedLLMRequest]
    precomputed_results: list[EvaluationRow]
    precomputed_llm_responses: list[dict[str, object]]


@dataclass
class FinalizedRun:
    results: list[EvaluationRow]
    llm_responses: list[dict[str, object]]
    usage_aggregate: UsageAggregate


def build_single_request(
    *,
    row_input: PipelineInputRow,
    case_doc: CaseDocument,
    provider_label: str,
    case_citation_for_output: str,
    prompt_builder: PromptBuilder,
    use_full_opinion_text: bool = False,
) -> PreparedLLMRequest:
    prompt_ctx = PromptContext(
        case_name=case_doc.case_name,
        citation=case_doc.citation,
        opinion_text=case_doc.prompt_opinion_text(full=use_full_opinion_text),
    )
    user_prompt = prompt_builder.build_user_prompt(prompt_ctx)
    return PreparedLLMRequest(
        request_id=f"row-{row_input.row_no}",
        row_no=row_input.row_no,
        display_case_name=row_input.display_case_name,
        provider_label=provider_label,
        expected_classifications=list(row_input.expected_classifications),
        case_citation_for_output=case_citation_for_output,
        system_prompt=prompt_builder.system_prompt,
        user_prompt=user_prompt,
        lookup_citation=row_input.lookup_citation,
        finetune_system_prompt=prompt_builder.finetune_system_prompt,
        finetune_user_prompt=prompt_builder.build_finetune_user_prompt(prompt_ctx),
        merge_row_responses=False,
        response_parser_id=prompt_builder.response_parser_id,
        trace=row_input.trace,
    )


def prepare_requests_from_pairs(
    *,
    pairs: Iterable[tuple[PipelineInputRow, CaseDocument]],
    prompt_builder: PromptBuilder,
    print_errors: bool,
    error_state: str = "ERROR",
    opinion_trigger_filter: bool = False,
    use_full_opinion_text: bool = False,
) -> tuple[PreparedRun, list[int]]:
    from dataclasses import replace

    from src.p01_preparation.opinion_trigger_filter import filter_pairs_by_opinion_triggers

    filter_result = filter_pairs_by_opinion_triggers(pairs, enabled=opinion_trigger_filter)
    if opinion_trigger_filter and print_errors and filter_result.skipped_row_nos:
        skipped_csv = ",".join(str(n) for n in filter_result.skipped_row_nos)
        print(
            f"[opinion filter] skipped {filter_result.skipped_count} case(s) "
            f"without overruling trigger words: {skipped_csv}",
            flush=True,
        )
        print(
            f"[opinion filter] processing {len(filter_result.kept)} case(s)",
            flush=True,
        )

    prepared_requests: list[PreparedLLMRequest] = []
    precomputed_results: list[EvaluationRow] = []
    precomputed_llm_responses: list[dict[str, object]] = []

    for row_input, case_doc in filter_result.kept:
        row_no = row_input.row_no
        provider_label = case_doc.retrieval_provider_label or "unknown"
        case_citation_for_output = row_input.case_citation
        try:
            if row_input.case_name_override is not None or row_input.opinion_text_override is not None:
                case_doc = CaseDocument(
                    citation=case_doc.citation,
                    source_url=case_doc.source_url,
                    raw_data=case_doc.raw_data,
                    case_name_override=row_input.case_name_override,
                    opinion_text_override=row_input.opinion_text_override,
                    retrieval_provider_label=case_doc.retrieval_provider_label,
                    opinion_scope=case_doc.opinion_scope,
                )
            if use_full_opinion_text:
                # Force all consumers of majority_opinion / prompt text to see the full casebody.
                case_doc = replace(
                    case_doc,
                    opinion_text_override=case_doc.prompt_opinion_text(full=True),
                )
            case_citation_for_output = case_doc.citation

            expanded = prompt_builder.expand_prepared_requests(
                row=row_input,
                case_doc=case_doc,
                provider_label=provider_label,
                case_citation_for_output=case_citation_for_output,
            )
            if expanded is not None:
                if not expanded:
                    raise RuntimeError(
                        "Chunk prompt found no citation matches in the opinion text "
                        "(edit CHUNK_CITATION_PATTERN in chunk_prompts/chunk_base.py — shared by chunk_a–chunk_d — "
                        "or check the opinion source)."
                    )
                for req in expanded:
                    if not req.trace.case_law_id and row_input.trace.case_law_id:
                        req.trace = row_input.trace
                prepared_requests.extend(expanded)
            else:
                prepared_requests.append(
                    build_single_request(
                        row_input=row_input,
                        case_doc=case_doc,
                        provider_label=provider_label,
                        case_citation_for_output=case_citation_for_output,
                        prompt_builder=prompt_builder,
                        use_full_opinion_text=use_full_opinion_text,
                    )
                )
        except Exception as exc:
            if print_errors:
                cite = f" lookup={row_input.lookup_citation!r}" if row_input.lookup_citation else ""
                print(f"[ERROR] row={row_no} case='{row_input.display_case_name}'{cite} -> {exc}")
            precomputed_results.append(
                EvaluationRow(
                    index_no=row_no,
                    case_name=row_input.display_case_name,
                    provider_label=provider_label,
                    expected=list(row_input.expected_classifications),
                    predicted=[],
                    evaluation=error_state,
                    error_message=str(exc),
                )
            )
            precomputed_llm_responses.append(
                {
                    "index": row_no,
                    "case_citation": case_citation_for_output,
                    "trace": row_input.trace.to_dict(),
                    "state": error_state,
                    "response": {},
                    "chunk_texts": None,
                    "citation_classification_state": None,
                }
            )

    return (
        PreparedRun(
            requests=prepared_requests,
            precomputed_results=precomputed_results,
            precomputed_llm_responses=precomputed_llm_responses,
        ),
        list(filter_result.skipped_row_nos),
    )


def _chunk_state_for_request(
    req: PreparedLLMRequest,
    extraction: ExtractionResult,
    strict_classes: set[str],
) -> str | None:
    if not req.chunk_citation_surfaces:
        return None
    return chunk_citation_state(
        req.expected_classifications,
        req.chunk_citation_surfaces,
        extraction.classified_citations,
        strict_classes=strict_classes,
    )


def finalize_evaluate_responses(
    *,
    prepared: PreparedRun,
    response_map: dict[str, ConnectorResponse],
    error_map: dict[str, str] | None,
    usage_adapter,
    print_errors: bool,
    print_raw: bool = False,
    prompt_builder_label: str,
    strict_classes: set[str],
) -> FinalizedRun:
    merger = merger_for_prompt_label(prompt_builder_label)

    usage_aggregate = UsageAggregate()
    results = list(prepared.precomputed_results)
    llm_responses = list(prepared.precomputed_llm_responses)

    by_row: dict[int, list[PreparedLLMRequest]] = defaultdict(list)
    for req in prepared.requests:
        by_row[req.row_no].append(req)
    for lst in by_row.values():
        lst.sort(key=lambda r: r.request_id)

    for row_no in sorted(by_row):
        reqs = by_row[row_no]
        extractions: list[ExtractionResult] = []
        sub_payloads: list[dict[str, object]] = []
        row_failed = False

        for req in reqs:
            response = response_map.get(req.request_id)
            if response is None:
                msg = (error_map or {}).get(req.request_id, "missing response")
                if print_errors:
                    print(f"[ERROR] row={req.row_no} case='{req.display_case_name}' -> {msg}")
                row_failed = True
                sub_payloads.append(
                    _response_payload(req=req, response={}, state="ERROR", raw_llm_content=None)
                )
                continue

            if print_raw:
                _print_raw_llm_response_row(
                    row_no=req.row_no,
                    request_id=req.request_id,
                    case_label=req.display_case_name,
                    raw_content=response.raw_content,
                )
            usage_aggregate.add(usage_adapter.extract_usage(response.raw_response))
            extraction = parse_llm_json_content(
                response.raw_content, response_parser_id=req.response_parser_id
            )
            _print_llm_explanation_row(req.row_no, req.display_case_name, extraction.explanation)
            extractions.append(extraction)
            sub_payloads.append(
                _response_payload(
                    req=req,
                    response=payload_from_raw_content(extraction.raw_content),
                    state="OK",
                    chunk_state=_chunk_state_for_request(req, extraction, strict_classes),
                    raw_llm_content=extraction.raw_content,
                    reasoning_content=response.reasoning_content,
                )
            )

        first = reqs[0]
        if row_failed or not extractions:
            evaluation = "ERROR"
            predicted: list[ClassifiedCitation] = []
            counts = None
            error_msg = "one or more sub-requests failed"
            for sp in sub_payloads:
                sp["state"] = evaluation
        else:
            merged = merger(extractions)
            predicted = merged.classified_citations
            evaluation, counts = evaluate_classified_citations(
                first.expected_classifications,
                predicted,
                strict_classes=strict_classes,
            )
            error_msg = None
            for sp in sub_payloads:
                sp["state"] = evaluation

        results.append(
            EvaluationRow(
                index_no=first.row_no,
                case_name=first.display_case_name,
                provider_label=first.provider_label,
                expected=list(first.expected_classifications),
                predicted=predicted,
                evaluation=evaluation,
                counts=counts,
                error_message=error_msg,
            )
        )
        llm_responses.extend(sub_payloads)

    results.sort(key=lambda x: x.index_no)
    llm_responses.sort(key=lambda x: (int(x["index"]), str(x.get("request_id", ""))))
    return FinalizedRun(results=results, llm_responses=llm_responses, usage_aggregate=usage_aggregate)


def finalize_production_responses(
    *,
    prepared: PreparedRun,
    response_map: dict[str, ConnectorResponse],
    error_map: dict[str, str] | None,
    usage_adapter,
    print_errors: bool,
    print_raw: bool = False,
    prompt_builder_label: str,
) -> FinalizedRun:
    merger = merger_for_prompt_label(prompt_builder_label)
    usage_aggregate = UsageAggregate()
    results = list(prepared.precomputed_results)
    llm_responses = list(prepared.precomputed_llm_responses)

    by_row: dict[int, list[PreparedLLMRequest]] = defaultdict(list)
    for req in prepared.requests:
        by_row[req.row_no].append(req)
    for lst in by_row.values():
        lst.sort(key=lambda r: r.request_id)

    for row_no in sorted(by_row):
        reqs = by_row[row_no]
        extractions: list[ExtractionResult] = []
        sub_payloads: list[dict[str, object]] = []
        row_failed = False

        for req in reqs:
            response = response_map.get(req.request_id)
            if response is None:
                msg = (error_map or {}).get(req.request_id, "missing response")
                if print_errors:
                    print(f"[ERROR] row={req.row_no} case='{req.display_case_name}' -> {msg}")
                row_failed = True
                sub_payloads.append(
                    _response_payload(req=req, response={}, state="ERROR", raw_llm_content=None)
                )
                continue

            if print_raw:
                _print_raw_llm_response_row(
                    row_no=req.row_no,
                    request_id=req.request_id,
                    case_label=req.display_case_name,
                    raw_content=response.raw_content,
                )
            usage_aggregate.add(usage_adapter.extract_usage(response.raw_response))
            extraction = parse_llm_json_content(
                response.raw_content, response_parser_id=req.response_parser_id
            )
            _print_llm_explanation_row(req.row_no, req.display_case_name, extraction.explanation)
            extractions.append(extraction)
            sub_payloads.append(
                _response_payload(
                    req=req,
                    response=payload_from_raw_content(extraction.raw_content),
                    state="OK",
                    raw_llm_content=extraction.raw_content,
                    reasoning_content=response.reasoning_content,
                )
            )

        first = reqs[0]
        state = "ERROR" if row_failed or not extractions else "OK"
        for sp in sub_payloads:
            sp["state"] = state

        if row_failed or not extractions:
            predicted: list[ClassifiedCitation] = []
            error_msg = "one or more sub-requests failed"
        else:
            merged = merger(extractions)
            predicted = merged.classified_citations
            error_msg = None

        results.append(
            EvaluationRow(
                index_no=first.row_no,
                case_name=first.display_case_name,
                provider_label=first.provider_label,
                expected=[],
                predicted=predicted,
                evaluation=state,
                error_message=error_msg,
            )
        )
        llm_responses.extend(sub_payloads)

    results.sort(key=lambda x: x.index_no)
    llm_responses.sort(key=lambda x: (int(x["index"]), str(x.get("request_id", ""))))
    return FinalizedRun(results=results, llm_responses=llm_responses, usage_aggregate=usage_aggregate)


def run_iterative_requests(
    *,
    llm_connector: LLMConnector,
    usage_adapter,
    prepared: PreparedRun,
    print_errors: bool,
    print_raw: bool = False,
    finalize_fn: Callable[..., FinalizedRun],
    finalize_kwargs: dict[str, Any],
) -> FinalizedRun:
    response_map: dict[str, ConnectorResponse] = {}
    error_map: dict[str, str] = {}
    total_req = len(prepared.requests)
    for idx, req in enumerate(prepared.requests, start=1):
        try:
            if total_req > 1:
                print(
                    f"[LLM {idx}/{total_req}] {req.request_id} row={req.row_no} "
                    f"case={req.display_case_name!r}",
                    flush=True,
                )
            response_map[req.request_id] = llm_connector.extract_overruled_cases(
                system_prompt=req.system_prompt,
                user_prompt=req.user_prompt,
            )
        except Exception as exc:
            error_map[req.request_id] = str(exc)
            if print_errors:
                cite = f" lookup={req.lookup_citation!r}" if req.lookup_citation else ""
                print(f"[ERROR] row={req.row_no} case='{req.display_case_name}'{cite} -> {exc}")
    return finalize_fn(
        prepared=prepared,
        response_map=response_map,
        error_map=error_map,
        usage_adapter=usage_adapter,
        print_errors=print_errors,
        print_raw=print_raw,
        **finalize_kwargs,
    )


def serialize_request(req: PreparedLLMRequest) -> dict[str, Any]:
    data = asdict(req)
    data["trace"] = req.trace.to_dict()
    return data


def deserialize_request(data: dict[str, Any]) -> PreparedLLMRequest:
    d = dict(data)
    d.setdefault("merge_row_responses", False)
    d.setdefault("chunk_texts", None)
    d.setdefault("chunk_citation_surfaces", None)
    d.setdefault("response_parser_id", "P1_overrule_parser")
    d["expected_classifications"] = [
        ClassifiedCitation.make(c["cite"], c["cls"]) for c in (d.get("expected_classifications") or [])
    ]
    trace_raw = d.pop("trace", None)
    d["trace"] = CaseTraceMetadata.from_dict(trace_raw if isinstance(trace_raw, dict) else None)
    return PreparedLLMRequest(**d)


def serialize_evaluation_row(row: EvaluationRow) -> dict[str, Any]:
    return {
        "index_no": row.index_no,
        "case_name": row.case_name,
        "provider_label": row.provider_label,
        "expected": [asdict(c) for c in row.expected],
        "predicted": [asdict(c) for c in row.predicted],
        "evaluation": row.evaluation,
        "error_message": row.error_message,
    }


def deserialize_evaluation_row(data: dict[str, Any]) -> EvaluationRow:
    return EvaluationRow(
        index_no=int(data["index_no"]),
        case_name=str(data["case_name"]),
        provider_label=str(data["provider_label"]),
        expected=[
            ClassifiedCitation.make(c["cite"], c["cls"]) for c in (data.get("expected") or [])
        ],
        predicted=[
            ClassifiedCitation.make(c["cite"], c["cls"]) for c in (data.get("predicted") or [])
        ],
        evaluation=str(data["evaluation"]),
        counts=None,
        error_message=data.get("error_message"),
    )
