from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from src.api_connectors.base import BatchStatus, LLMConnector
from src.api_connectors.factory import build_llm_connector
from src.api_connectors.jsonl_connector import BATCH_JOB_CONTEXT_BASENAME
from src.api_usage.factory import build_usage_adapter
from src.batch_job_store import delete_batch_job_context, load_batch_job_context, save_batch_job_context
from src.models import DEFAULT_STRICT_CLASSES, normalize_class_label, parse_strict_classes_csv
from src.p01_preparation._02_prompt_creation.factory import build_prompt_builder
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_strategies import ChunkStrategyKind
from src.p01_preparation._01_retrieval.benchmark import load_dataset_inputs
from src.p01_preparation._01_retrieval.case_providers import OpinionRetrievalChainProvider, opinion_retrieval_metadata
from src.p01_preparation._01_retrieval.evaluate_retrieval import prepare_benchmark_pairs
from src.p01_preparation._01_retrieval.volume_loader import load_reporter_volume_cases
from src.p02_execution.pipeline_core import (
    PreparedRun,
    deserialize_evaluation_row,
    deserialize_request,
    finalize_evaluate_responses,
    finalize_production_responses,
    prepare_requests_from_pairs,
    run_iterative_requests,
    serialize_evaluation_row,
    serialize_request,
)
from src.p03a_evaluation.finetune_dataset import (
    FinetuneDatasetFormat,
    strip_finetune_fields_from_llm_responses,
    write_finetune_dataset_unfiltered,
    write_finetune_datasets,
)
from src.p03a_evaluation.result_output import build_metadata, write_outputs
from src.p03b_extraction.production_output import build_production_metadata, write_production_outputs


def batch_job_model_from_context(job_context: dict[str, Any]) -> str | None:
    return job_context.get("resolved_llm_model") or job_context.get("llm_model")


def _is_jsonl_connector(connector_name: str) -> bool:
    return connector_name.strip().lower() in {"jsonl", "jsonl_file", "file_jsonl"}


def _build_connector(
    connector_name: str,
    *,
    model: str | None,
    reasoning_effort: str | None = None,
    jsonl_output_dir: str | None = None,
) -> LLMConnector:
    return build_llm_connector(
        connector_name,
        model=model,
        reasoning_effort=reasoning_effort,
        jsonl_output_dir=jsonl_output_dir,
    )


def _normalize_strict_classes(strict_classes: Iterable[str] | None) -> set[str]:
    if strict_classes is None:
        return set(DEFAULT_STRICT_CLASSES)
    return {normalize_class_label(c) for c in strict_classes if normalize_class_label(c)}


def _prepare_benchmark_requests(
    *,
    dataset_path: str,
    from_idx: Optional[int],
    to_idx: Optional[int],
    indices: Optional[list[int]],
    print_errors: bool,
    prompt_builder,
    opinion_trigger_filter: bool = False,
    use_full_opinion_text: bool = False,
) -> PreparedRun:
    provider = OpinionRetrievalChainProvider()
    input_rows = load_dataset_inputs(
        dataset_path, from_idx=from_idx, to_idx=to_idx, indices=indices
    )
    pairs, precomputed = prepare_benchmark_pairs(
        input_rows,
        provider,
        print_errors=print_errors,
        require_full_casebody=use_full_opinion_text,
    )
    prepared, _skipped_row_nos = prepare_requests_from_pairs(
        pairs=pairs,
        prompt_builder=prompt_builder,
        print_errors=print_errors,
        opinion_trigger_filter=opinion_trigger_filter,
        use_full_opinion_text=use_full_opinion_text,
    )
    prepared.precomputed_results.extend(precomputed.precomputed_results)
    prepared.precomputed_llm_responses.extend(precomputed.precomputed_llm_responses)
    return prepared


def run_benchmark(
    dataset_path: str,
    output_dir: str,
    *,
    llm_connector_name: str = "openai",
    prompt_style: str = "a",
    llm_model: str | None = None,
    reasoning_effort: str | None = None,
    from_idx: Optional[int] = None,
    to_idx: Optional[int] = None,
    indices: Optional[list[int]] = None,
    strict_classes: Optional[Iterable[str]] = None,
    print_errors: bool = True,
    print_raw: bool = False,
    finetune_dataset_format: FinetuneDatasetFormat | None = None,
    chunk_strategy_kind: ChunkStrategyKind | None = None,
    use_full_opinion_text: bool = False,
    opinion_trigger_filter: bool = False,
) -> tuple[str, str, str, tuple[str, str] | None]:
    strict_set = _normalize_strict_classes(strict_classes)
    llm_connector: LLMConnector = _build_connector(
        llm_connector_name, model=llm_model, reasoning_effort=reasoning_effort
    )
    usage_adapter = build_usage_adapter(llm_connector.label, llm_connector.model)
    prompt_builder = build_prompt_builder(prompt_style, chunk_strategy_kind=chunk_strategy_kind)
    prepared = _prepare_benchmark_requests(
        dataset_path=dataset_path,
        from_idx=from_idx,
        to_idx=to_idx,
        indices=indices,
        print_errors=print_errors,
        prompt_builder=prompt_builder,
        opinion_trigger_filter=opinion_trigger_filter,
        use_full_opinion_text=use_full_opinion_text,
    )
    finalized = run_iterative_requests(
        llm_connector=llm_connector,
        usage_adapter=usage_adapter,
        prepared=prepared,
        print_errors=print_errors,
        print_raw=print_raw,
        finalize_fn=finalize_evaluate_responses,
        finalize_kwargs={
            "prompt_builder_label": prompt_builder.label,
            "strict_classes": strict_set,
        },
    )
    meta_from, meta_to, meta_indices = (
        (None, None, indices) if indices else (from_idx, to_idx, None)
    )
    metadata = build_metadata(
        dataset_path=dataset_path,
        from_idx=meta_from,
        to_idx=meta_to,
        indices=meta_indices,
        connector_label=llm_connector.label,
        model=llm_connector.model,
        prompt_builder=prompt_builder,
        usage_adapter=usage_adapter,
        usage_aggregate=finalized.usage_aggregate,
        results=finalized.results,
        strict_classes=strict_set,
        use_batch_pricing=False,
        opinion_retrieval_info=opinion_retrieval_metadata(),
        reasoning_level=getattr(llm_connector, "reasoning_effort", None),
    )
    metadata.setdefault("execution", {})
    metadata["execution"]["use_full_opinion_text"] = use_full_opinion_text
    metadata["execution"]["opinion_trigger_filter"] = opinion_trigger_filter
    if chunk_strategy_kind is not None:
        metadata["execution"]["chunk_strategy"] = chunk_strategy_kind
    finetune_paths: tuple[str, str] | None = None
    if finetune_dataset_format:
        p_match, p_wrong = write_finetune_datasets(
            output_dir,
            finalized.llm_responses,
            dataset_format=finetune_dataset_format,
        )
        finetune_paths = (p_match, p_wrong)
        metadata.setdefault("execution", {})
        metadata["execution"]["finetune_dataset_jsonl_path"] = p_match
        metadata["execution"]["finetune_dataset_wrong_jsonl_path"] = p_wrong
        metadata["execution"]["finetune_dataset_format"] = finetune_dataset_format
    strip_finetune_fields_from_llm_responses(finalized.llm_responses)
    evaluation_path, metadata_path, llm_arrays_path = write_outputs(
        output_dir=output_dir,
        results=finalized.results,
        llm_responses=finalized.llm_responses,
        metadata=metadata,
    )
    usage_adapter.print_summary(finalized.usage_aggregate, use_batch_pricing=False)
    return evaluation_path, metadata_path, llm_arrays_path, finetune_paths


def run_benchmark_batch_submit(
    dataset_path: str,
    output_dir: str,
    *,
    llm_connector_name: str = "openai",
    prompt_style: str = "a",
    llm_model: str | None = None,
    reasoning_effort: str | None = None,
    from_idx: Optional[int] = None,
    to_idx: Optional[int] = None,
    indices: Optional[list[int]] = None,
    strict_classes: Optional[Iterable[str]] = None,
    print_errors: bool = True,
    print_raw: bool = False,
    finetune_dataset_format: FinetuneDatasetFormat | None = None,
    chunk_strategy_kind: ChunkStrategyKind | None = None,
    use_full_opinion_text: bool = False,
    opinion_trigger_filter: bool = False,
) -> str:
    strict_set = _normalize_strict_classes(strict_classes)
    resolved_output_dir = str(Path(output_dir).resolve())
    Path(resolved_output_dir).mkdir(parents=True, exist_ok=True)
    llm_connector: LLMConnector = _build_connector(
        llm_connector_name,
        model=llm_model,
        reasoning_effort=reasoning_effort,
        jsonl_output_dir=resolved_output_dir if _is_jsonl_connector(llm_connector_name) else None,
    )
    prompt_builder = build_prompt_builder(prompt_style, chunk_strategy_kind=chunk_strategy_kind)
    prepared = _prepare_benchmark_requests(
        dataset_path=dataset_path,
        from_idx=from_idx,
        to_idx=to_idx,
        indices=indices,
        print_errors=print_errors,
        prompt_builder=prompt_builder,
        opinion_trigger_filter=opinion_trigger_filter,
        use_full_opinion_text=use_full_opinion_text,
    )
    batch_submit = llm_connector.submit_batch(prepared.requests)
    context_path = (
        str(Path(resolved_output_dir) / BATCH_JOB_CONTEXT_BASENAME)
        if _is_jsonl_connector(llm_connector_name)
        else None
    )
    save_batch_job_context(
        batch_id=batch_submit.batch_id,
        payload={
            "pipeline": "evaluate",
            "dataset_path": dataset_path,
            "output_dir": resolved_output_dir,
            "connector_name": llm_connector_name,
            "prompt_style": prompt_style,
            "llm_model": llm_model,
            "reasoning_effort": reasoning_effort,
            "resolved_llm_model": llm_connector.model,
            "from_idx": from_idx,
            "to_idx": to_idx,
            "indices": indices,
            "strict_classes": sorted(strict_set),
            "print_errors": print_errors,
            "print_raw": print_raw,
            "prepared_requests": [serialize_request(r) for r in prepared.requests],
            "precomputed_results": [serialize_evaluation_row(r) for r in prepared.precomputed_results],
            "precomputed_llm_responses": prepared.precomputed_llm_responses,
            "finetune_dataset_format": finetune_dataset_format,
            "opinion_retrieval_info": opinion_retrieval_metadata(),
            "chunk_strategy": chunk_strategy_kind,
            "use_full_opinion_text": use_full_opinion_text,
            "opinion_trigger_filter": opinion_trigger_filter,
        },
        context_path=context_path,
    )
    return batch_submit.batch_id


def run_production(
    *,
    reporter: str,
    volumes: list[str],
    output_dir: str,
    min_page: int | None = None,
    max_page: int | None = None,
    llm_connector_name: str = "openai",
    prompt_style: str = "a",
    llm_model: str | None = None,
    reasoning_effort: str | None = None,
    print_errors: bool = True,
    print_raw: bool = False,
    relations_db_path: str | None = None,
    opinion_trigger_filter: bool = True,
    finetune_dataset_format: FinetuneDatasetFormat | None = None,
    chunk_strategy_kind: ChunkStrategyKind | None = None,
    use_full_opinion_text: bool = False,
) -> tuple[str, str, str, str | None]:
    llm_connector: LLMConnector = _build_connector(
        llm_connector_name, model=llm_model, reasoning_effort=reasoning_effort
    )
    usage_adapter = build_usage_adapter(llm_connector.label, llm_connector.model)
    prompt_builder = build_prompt_builder(prompt_style, chunk_strategy_kind=chunk_strategy_kind)
    page_min = min_page if len(volumes) == 1 else None
    page_max = max_page if len(volumes) == 1 else None
    pairs = load_reporter_volume_cases(
        reporter=reporter,
        volumes=volumes,
        min_page=page_min,
        max_page=page_max,
    )
    if not pairs:
        raise RuntimeError(
            f"No cases found for reporter={reporter!r} volumes={volumes!r} "
            f"with min_page={page_min} max_page={page_max}."
        )
    retrieved_count = len(pairs)
    prepared, filter_skipped = prepare_requests_from_pairs(
        pairs=pairs,
        prompt_builder=prompt_builder,
        print_errors=print_errors,
        error_state="ERROR",
        opinion_trigger_filter=opinion_trigger_filter,
        use_full_opinion_text=use_full_opinion_text,
    )
    if not prepared.requests:
        raise RuntimeError(
            f"No cases to process after retrieval "
            f"(retrieved={retrieved_count}, opinion_trigger_filter={opinion_trigger_filter}, "
            f"skipped_by_filter={len(filter_skipped)}). "
            f"Use --no-opinion-trigger-filter to send every opinion to the LLM."
        )
    finalized = run_iterative_requests(
        llm_connector=llm_connector,
        usage_adapter=usage_adapter,
        prepared=prepared,
        print_errors=print_errors,
        print_raw=print_raw,
        finalize_fn=finalize_production_responses,
        finalize_kwargs={"prompt_builder_label": prompt_builder.label},
    )
    run_id = output_dir.rsplit("/", 1)[-1]
    metadata = build_production_metadata(
        reporter=reporter,
        volumes=volumes,
        min_page=page_min,
        max_page=page_max,
        connector_label=llm_connector.label,
        model=llm_connector.model,
        prompt_builder=prompt_builder,
        usage_adapter=usage_adapter,
        usage_aggregate=finalized.usage_aggregate,
        llm_responses=finalized.llm_responses,
        use_batch_pricing=False,
        reasoning_level=getattr(llm_connector, "reasoning_effort", None),
        relations_db_path=relations_db_path,
        opinion_trigger_filter=opinion_trigger_filter,
        cases_retrieved=retrieved_count,
        cases_skipped_by_opinion_filter=len(filter_skipped),
    )
    metadata.setdefault("execution", {})
    metadata["execution"]["use_full_opinion_text"] = use_full_opinion_text
    if chunk_strategy_kind is not None:
        metadata["execution"]["chunk_strategy"] = chunk_strategy_kind
    finetune_path: str | None = None
    if finetune_dataset_format:
        finetune_path = write_finetune_dataset_unfiltered(
            output_dir,
            finalized.llm_responses,
            dataset_format=finetune_dataset_format,
        )
        metadata.setdefault("execution", {})
        metadata["execution"]["finetune_dataset_jsonl_path"] = finetune_path
        metadata["execution"]["finetune_dataset_format"] = finetune_dataset_format
    strip_finetune_fields_from_llm_responses(finalized.llm_responses)
    llm_path, metadata_path, db_path = write_production_outputs(
        output_dir=output_dir,
        llm_responses=finalized.llm_responses,
        metadata=metadata,
        run_id=run_id,
        relations_db_path=relations_db_path,
    )
    usage_adapter.print_summary(finalized.usage_aggregate, use_batch_pricing=False)
    return llm_path, metadata_path, db_path, finetune_path


def run_production_batch_submit(
    *,
    reporter: str,
    volumes: list[str],
    output_dir: str,
    min_page: int | None = None,
    max_page: int | None = None,
    llm_connector_name: str = "openai",
    prompt_style: str = "a",
    llm_model: str | None = None,
    reasoning_effort: str | None = None,
    print_errors: bool = True,
    print_raw: bool = False,
    relations_db_path: str | None = None,
    opinion_trigger_filter: bool = True,
    finetune_dataset_format: FinetuneDatasetFormat | None = None,
    chunk_strategy_kind: ChunkStrategyKind | None = None,
    use_full_opinion_text: bool = False,
) -> str:
    resolved_output_dir = str(Path(output_dir).resolve())
    Path(resolved_output_dir).mkdir(parents=True, exist_ok=True)
    llm_connector: LLMConnector = _build_connector(
        llm_connector_name,
        model=llm_model,
        reasoning_effort=reasoning_effort,
        jsonl_output_dir=resolved_output_dir if _is_jsonl_connector(llm_connector_name) else None,
    )
    prompt_builder = build_prompt_builder(prompt_style, chunk_strategy_kind=chunk_strategy_kind)
    page_min = min_page if len(volumes) == 1 else None
    page_max = max_page if len(volumes) == 1 else None
    pairs = load_reporter_volume_cases(
        reporter=reporter,
        volumes=volumes,
        min_page=page_min,
        max_page=page_max,
    )
    if not pairs:
        raise RuntimeError(
            f"No cases found for reporter={reporter!r} volumes={volumes!r} "
            f"with min_page={page_min} max_page={page_max}."
        )
    retrieved_count = len(pairs)
    prepared, filter_skipped = prepare_requests_from_pairs(
        pairs=pairs,
        prompt_builder=prompt_builder,
        print_errors=print_errors,
        error_state="ERROR",
        opinion_trigger_filter=opinion_trigger_filter,
        use_full_opinion_text=use_full_opinion_text,
    )
    if not prepared.requests:
        raise RuntimeError(
            f"No cases to submit after retrieval "
            f"(retrieved={retrieved_count}, opinion_trigger_filter={opinion_trigger_filter}, "
            f"skipped_by_filter={len(filter_skipped)}). "
            f"Use --no-opinion-trigger-filter to send every opinion to the LLM."
        )
    batch_submit = llm_connector.submit_batch(prepared.requests)
    context_path = (
        str(Path(resolved_output_dir) / BATCH_JOB_CONTEXT_BASENAME)
        if _is_jsonl_connector(llm_connector_name)
        else None
    )
    save_batch_job_context(
        batch_id=batch_submit.batch_id,
        payload={
            "pipeline": "production",
            "reporter": reporter,
            "volumes": volumes,
            "min_page": page_min,
            "max_page": page_max,
            "output_dir": resolved_output_dir,
            "connector_name": llm_connector_name,
            "prompt_style": prompt_style,
            "llm_model": llm_model,
            "reasoning_effort": reasoning_effort,
            "resolved_llm_model": llm_connector.model,
            "print_errors": print_errors,
            "print_raw": print_raw,
            "relations_db_path": relations_db_path,
            "opinion_trigger_filter": opinion_trigger_filter,
            "cases_retrieved": retrieved_count,
            "cases_skipped_by_opinion_filter": len(filter_skipped),
            "finetune_dataset_format": finetune_dataset_format,
            "chunk_strategy": chunk_strategy_kind,
            "use_full_opinion_text": use_full_opinion_text,
            "prepared_requests": [serialize_request(r) for r in prepared.requests],
            "precomputed_results": [serialize_evaluation_row(r) for r in prepared.precomputed_results],
            "precomputed_llm_responses": prepared.precomputed_llm_responses,
        },
        context_path=context_path,
    )
    return batch_submit.batch_id


def finalize_batch_job(
    batch_id: str,
    *,
    print_errors: bool = True,
    print_raw: bool | None = None,
) -> tuple[str, str, str, tuple[str, str] | str | None]:
    job_context = load_batch_job_context(batch_id)
    if print_raw is None:
        print_raw = bool(job_context.get("print_raw", False))
    pipeline = job_context.get("pipeline", "evaluate")
    if pipeline in ("production", "extract"):
        return _finalize_production_batch_job(
            batch_id, job_context, print_errors=print_errors, print_raw=print_raw
        )
    return _finalize_evaluate_batch_job(
        batch_id, job_context, print_errors=print_errors, print_raw=print_raw
    )


def _finalize_evaluate_batch_job(
    batch_id: str,
    job_context: dict[str, Any],
    *,
    print_errors: bool,
    print_raw: bool,
) -> tuple[str, str, str, tuple[str, str] | None]:
    llm_connector: LLMConnector = _build_connector(
        job_context["connector_name"],
        model=batch_job_model_from_context(job_context),
        reasoning_effort=job_context.get("reasoning_effort"),
    )
    usage_adapter = build_usage_adapter(llm_connector.label, llm_connector.model)
    prompt_builder = build_prompt_builder(job_context["prompt_style"])
    strict_set = _normalize_strict_classes(job_context.get("strict_classes"))

    prepared = PreparedRun(
        requests=[deserialize_request(item) for item in job_context["prepared_requests"]],
        precomputed_results=[
            deserialize_evaluation_row(item) for item in job_context["precomputed_results"]
        ],
        precomputed_llm_responses=list(job_context["precomputed_llm_responses"]),
    )
    response_map = llm_connector.collect_batch_results(batch_id)
    finalized = finalize_evaluate_responses(
        prepared=prepared,
        response_map=response_map,
        error_map=None,
        usage_adapter=usage_adapter,
        print_errors=print_errors,
        print_raw=print_raw,
        prompt_builder_label=prompt_builder.label,
        strict_classes=strict_set,
    )
    job_indices = job_context.get("indices")
    meta_from = None if job_indices else job_context.get("from_idx")
    meta_to = None if job_indices else job_context.get("to_idx")
    metadata = build_metadata(
        dataset_path=job_context["dataset_path"],
        from_idx=meta_from,
        to_idx=meta_to,
        indices=job_indices,
        connector_label=llm_connector.label,
        model=llm_connector.model,
        prompt_builder=prompt_builder,
        usage_adapter=usage_adapter,
        usage_aggregate=finalized.usage_aggregate,
        results=finalized.results,
        strict_classes=strict_set,
        use_batch_pricing=True,
        opinion_retrieval_info=job_context.get("opinion_retrieval_info"),
        reasoning_level=getattr(llm_connector, "reasoning_effort", None),
    )
    metadata.setdefault("execution", {})
    metadata["execution"]["use_full_opinion_text"] = bool(
        job_context.get("use_full_opinion_text", False)
    )
    metadata["execution"]["opinion_trigger_filter"] = bool(
        job_context.get("opinion_trigger_filter", False)
    )
    if job_context.get("chunk_strategy") is not None:
        metadata["execution"]["chunk_strategy"] = job_context["chunk_strategy"]
    metadata.setdefault("batch", {})
    metadata["batch"]["batch_id"] = batch_id
    status = llm_connector.get_batch_status(batch_id)
    metadata["batch"]["status"] = status.status
    finetune_dataset_format = job_context.get("finetune_dataset_format")
    if finetune_dataset_format not in {"prompt_completion", "openai_messages"}:
        finetune_dataset_format = None
    finetune_paths: tuple[str, str] | None = None
    if finetune_dataset_format:
        p_match, p_wrong = write_finetune_datasets(
            job_context["output_dir"],
            finalized.llm_responses,
            dataset_format=finetune_dataset_format,
        )
        finetune_paths = (p_match, p_wrong)
        metadata.setdefault("execution", {})
        metadata["execution"]["finetune_dataset_jsonl_path"] = p_match
        metadata["execution"]["finetune_dataset_wrong_jsonl_path"] = p_wrong
        metadata["execution"]["finetune_dataset_format"] = finetune_dataset_format
    strip_finetune_fields_from_llm_responses(finalized.llm_responses)
    evaluation_path, metadata_path, llm_arrays_path = write_outputs(
        output_dir=job_context["output_dir"],
        results=finalized.results,
        llm_responses=finalized.llm_responses,
        metadata=metadata,
    )
    usage_adapter.print_summary(finalized.usage_aggregate, use_batch_pricing=True)
    delete_batch_job_context(batch_id)
    return evaluation_path, metadata_path, llm_arrays_path, finetune_paths


def _finalize_production_batch_job(
    batch_id: str,
    job_context: dict[str, Any],
    *,
    print_errors: bool,
    print_raw: bool,
) -> tuple[str, str, str, str | None]:
    llm_connector: LLMConnector = _build_connector(
        job_context["connector_name"],
        model=batch_job_model_from_context(job_context),
        reasoning_effort=job_context.get("reasoning_effort"),
    )
    usage_adapter = build_usage_adapter(llm_connector.label, llm_connector.model)
    prompt_builder = build_prompt_builder(job_context["prompt_style"])

    prepared = PreparedRun(
        requests=[deserialize_request(item) for item in job_context["prepared_requests"]],
        precomputed_results=[
            deserialize_evaluation_row(item) for item in job_context["precomputed_results"]
        ],
        precomputed_llm_responses=list(job_context["precomputed_llm_responses"]),
    )
    response_map = llm_connector.collect_batch_results(batch_id)
    finalized = finalize_production_responses(
        prepared=prepared,
        response_map=response_map,
        error_map=None,
        usage_adapter=usage_adapter,
        print_errors=print_errors,
        print_raw=print_raw,
        prompt_builder_label=prompt_builder.label,
    )
    run_id = str(job_context["output_dir"]).rsplit("/", 1)[-1]
    relations_db_path = job_context.get("relations_db_path")
    job_volumes: list[str] = job_context.get("volumes") or [str(job_context["volume"])]
    metadata = build_production_metadata(
        reporter=job_context["reporter"],
        volumes=job_volumes,
        min_page=job_context.get("min_page"),
        max_page=job_context.get("max_page"),
        connector_label=llm_connector.label,
        model=llm_connector.model,
        prompt_builder=prompt_builder,
        usage_adapter=usage_adapter,
        usage_aggregate=finalized.usage_aggregate,
        llm_responses=finalized.llm_responses,
        use_batch_pricing=True,
        reasoning_level=getattr(llm_connector, "reasoning_effort", None),
        relations_db_path=relations_db_path,
        opinion_trigger_filter=bool(job_context.get("opinion_trigger_filter", True)),
        cases_retrieved=job_context.get("cases_retrieved"),
        cases_skipped_by_opinion_filter=int(
            job_context.get("cases_skipped_by_opinion_filter") or 0
        ),
    )
    metadata.setdefault("execution", {})
    metadata["execution"]["use_full_opinion_text"] = bool(
        job_context.get("use_full_opinion_text", False)
    )
    if job_context.get("chunk_strategy") is not None:
        metadata["execution"]["chunk_strategy"] = job_context["chunk_strategy"]
    metadata.setdefault("batch", {})
    metadata["batch"]["batch_id"] = batch_id
    status = llm_connector.get_batch_status(batch_id)
    metadata["batch"]["status"] = status.status
    finetune_dataset_format = job_context.get("finetune_dataset_format")
    if finetune_dataset_format not in {"prompt_completion", "openai_messages"}:
        finetune_dataset_format = None
    finetune_path: str | None = None
    if finetune_dataset_format:
        finetune_path = write_finetune_dataset_unfiltered(
            job_context["output_dir"],
            finalized.llm_responses,
            dataset_format=finetune_dataset_format,
        )
        metadata.setdefault("execution", {})
        metadata["execution"]["finetune_dataset_jsonl_path"] = finetune_path
        metadata["execution"]["finetune_dataset_format"] = finetune_dataset_format
    strip_finetune_fields_from_llm_responses(finalized.llm_responses)
    llm_path, metadata_path, db_path = write_production_outputs(
        output_dir=job_context["output_dir"],
        llm_responses=finalized.llm_responses,
        metadata=metadata,
        run_id=run_id,
        relations_db_path=relations_db_path,
    )
    usage_adapter.print_summary(finalized.usage_aggregate, use_batch_pricing=True)
    delete_batch_job_context(batch_id)
    return llm_path, metadata_path, db_path, finetune_path


def get_batch_status(
    batch_id: str,
    *,
    llm_connector_name: str,
    llm_model: str | None = None,
    reasoning_effort: str | None = None,
) -> BatchStatus:
    llm_connector: LLMConnector = _build_connector(
        llm_connector_name, model=llm_model, reasoning_effort=reasoning_effort
    )
    return llm_connector.get_batch_status(batch_id)
