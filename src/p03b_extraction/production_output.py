"""Production pipeline output writers (JSON artifacts + SQLite relations)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.api_usage.base import ApiUsageAdapter, UsageAggregate
from src.p01_preparation._02_prompt_creation.base import PromptBuilder
from src.p03b_extraction.relation_store import (
    collect_relations_from_llm_responses,
    write_relations_to_sqlite,
)


def build_production_metadata(
    *,
    reporter: str,
    volumes: list[str],
    min_page: int | None,
    max_page: int | None,
    connector_label: str,
    model: str,
    prompt_builder: PromptBuilder,
    usage_adapter: ApiUsageAdapter,
    usage_aggregate: UsageAggregate,
    llm_responses: list[dict[str, Any]],
    use_batch_pricing: bool = False,
    reasoning_level: str | None = None,
    relations_db_path: str | None = None,
    opinion_trigger_filter: bool = False,
    cases_retrieved: int | None = None,
    cases_skipped_by_opinion_filter: int = 0,
) -> dict[str, Any]:
    from src.api_usage.openai import estimated_cost_usd, resolve_estimation_rates

    status_counts = {"OK": 0, "ERROR": 0}
    for entry in llm_responses:
        state = str(entry.get("state", "ERROR")).upper()
        if state == "OK":
            status_counts["OK"] += 1
        else:
            status_counts["ERROR"] += 1

    in_rate, out_rate, pricing_mode = resolve_estimation_rates(
        usage_adapter, use_batch_pricing=use_batch_pricing
    )
    relations = collect_relations_from_llm_responses(llm_responses)
    return {
        "pipeline": "production",
        "execution": {
            "reporter": reporter,
            "volumes": volumes,
            "volume": volumes[0] if len(volumes) == 1 else None,
            "min_page": min_page,
            "max_page": max_page,
            "opinion_trigger_filter": opinion_trigger_filter,
            "cases_retrieved": cases_retrieved,
            "cases_skipped_by_opinion_filter": cases_skipped_by_opinion_filter,
            "case_count": len({entry.get("index") for entry in llm_responses}),
            "relations_extracted": len(relations),
            "relations_db_path": relations_db_path,
        },
        "llm": {
            "connector": connector_label,
            "model": model,
            "reasoning_level": reasoning_level,
            "prompt_style": prompt_builder.label,
            "calls_with_usage": usage_aggregate.llm_calls_with_usage,
        },
        "usage": {
            "input_tokens": usage_aggregate.input_tokens,
            "cached_input_tokens": usage_aggregate.cached_input_tokens,
            "output_tokens": usage_aggregate.output_tokens,
            "reasoning_tokens": usage_aggregate.reasoning_tokens,
            "total_tokens": usage_aggregate.total_tokens,
        },
        "costs": {
            "api_cost_usd": usage_aggregate.cost_usd if usage_aggregate.any_cost_in_api else None,
            "estimated_cost_usd": estimated_cost_usd(
                usage_aggregate.input_tokens,
                usage_aggregate.output_tokens,
                in_rate,
                out_rate,
            ),
            "estimation_inputs": {
                "pricing_mode": pricing_mode,
                "parsed_input_usd_per_1m_tokens": in_rate,
                "parsed_output_usd_per_1m_tokens": out_rate,
            },
        },
        "status_counts": status_counts,
    }


def write_production_outputs(
    *,
    output_dir: str,
    llm_responses: list[dict[str, Any]],
    metadata: dict[str, Any],
    run_id: str,
    relations_db_path: str | None = None,
) -> tuple[str, str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    llm_arrays_path = out_dir / "llm_result_arrays.json"
    with open(llm_arrays_path, "w", encoding="utf-8") as f:
        json.dump({"responses": llm_responses}, f, ensure_ascii=True, indent=2)

    relations = collect_relations_from_llm_responses(llm_responses)
    db_path = write_relations_to_sqlite(
        relations,
        run_id=run_id,
        db_path=relations_db_path,
    )

    metadata = dict(metadata)
    metadata.setdefault("execution", {})
    metadata["execution"]["output_dir"] = str(out_dir)
    metadata["execution"]["llm_result_arrays_path"] = str(llm_arrays_path)
    metadata["execution"]["relations_db_path"] = db_path
    metadata["execution"]["relations_extracted"] = len(relations)

    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)

    return str(llm_arrays_path), str(metadata_path), db_path
