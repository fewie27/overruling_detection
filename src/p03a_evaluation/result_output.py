from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.api_usage.base import ApiUsageAdapter, UsageAggregate
from src.api_usage.openai import estimated_cost_usd, resolve_estimation_rates
from src.models import ClassifiedCitation, PromptContext
from src.p01_preparation._02_prompt_creation.base import PromptBuilder
from src.p03a_evaluation.citation_matching import (
    EvaluationCounts,
    compute_class_metrics_and_confusion,
    format_classified_citations,
)
from src.p03a_evaluation.stats_helpers import enrich_evaluation_metadata


@dataclass
class EvaluationRow:
    index_no: int
    case_name: str
    provider_label: str
    expected: list[ClassifiedCitation]
    predicted: list[ClassifiedCitation]
    evaluation: str
    counts: EvaluationCounts | None = None
    error_message: str | None = None

    @property
    def expected_text(self) -> str:
        return format_classified_citations(self.expected)

    @property
    def predicted_text(self) -> str:
        if self.error_message:
            return f"ERROR: {self.error_message}"
        return format_classified_citations(self.predicted)


def _citation_name_from_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if "," in text:
        return text.split(",", 1)[0].strip()
    return text


def _per_citation_rows(results: list[EvaluationRow]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in results:
        for c in r.predicted:
            out.append(
                {
                    "index_no": r.index_no,
                    "retrieval_provider": r.provider_label,
                    "observed_case_citation_name": _citation_name_from_text(c.cite),
                    "observed_citation": c.cite,
                    "classification_label": c.cls,
                }
            )
    out.sort(
        key=lambda r: (
            int(r["index_no"]),
            str(r["observed_citation"]).lower(),
            str(r["classification_label"]).lower(),
        )
    )
    return out


def write_outputs(
    *,
    output_dir: str,
    results: list[EvaluationRow],
    llm_responses: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[str, str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluation_path = out_dir / "results.csv"
    with open(evaluation_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index_no",
                "case_name",
                "provider_label",
                "expected_citations",
                "llm_citations",
                "evaluation",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "index_no": r.index_no,
                    "case_name": r.case_name,
                    "provider_label": r.provider_label,
                    "expected_citations": r.expected_text,
                    "llm_citations": r.predicted_text,
                    "evaluation": r.evaluation,
                }
            )

    llm_arrays_path = out_dir / "llm_result_arrays.json"
    with open(llm_arrays_path, "w", encoding="utf-8") as f:
        json.dump({"responses": llm_responses}, f, ensure_ascii=True, indent=2)

    classification_csv_path = out_dir / "citation_classification_results.csv"
    classification_rows = _per_citation_rows(results)
    with open(classification_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index_no",
                "retrieval_provider",
                "observed_case_citation_name",
                "observed_citation",
                "classification_label",
            ],
        )
        writer.writeheader()
        for row in classification_rows:
            writer.writerow(row)

    metadata = dict(metadata)
    strict_classes = {
        str(c) for c in (metadata.get("execution") or {}).get("strict_classes") or []
    }
    _, _, confusion_matrix = compute_class_metrics_and_confusion(
        ((r.expected, r.predicted) for r in results),
        strict_classes=strict_classes,
    )
    confusion_pdf_path = out_dir / "confusion_matrix.pdf"
    _write_confusion_matrix_pdf(confusion_pdf_path, confusion_matrix)

    metadata.setdefault("execution", {})
    metadata["execution"]["output_dir"] = str(out_dir)
    metadata["execution"]["evaluation_csv_path"] = str(evaluation_path)
    metadata["execution"]["llm_result_arrays_path"] = str(llm_arrays_path)
    metadata["execution"]["citation_classification_csv_path"] = str(classification_csv_path)
    metadata["execution"]["citation_classification_rows"] = len(classification_rows)
    metadata["execution"]["confusion_matrix_pdf_path"] = str(confusion_pdf_path)

    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)

    return str(evaluation_path), str(metadata_path), str(llm_arrays_path)


def _write_confusion_matrix_pdf(path: Path, confusion_matrix: dict[str, Any]) -> None:
    """Render a compact, readable confusion-matrix table as PDF."""
    import matplotlib.pyplot as plt

    rows = list(confusion_matrix.get("row_labels") or [])
    cols = list(confusion_matrix.get("col_labels") or [])
    counts = [list(map(str, row)) for row in (confusion_matrix.get("counts") or [])]
    fig_w = max(6.0, 1.15 * len(cols) + 2.5)
    fig_h = max(3.0, 0.42 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title("Confusion matrix — rows: expected, columns: predicted", fontsize=11, pad=10)
    table = ax.table(
        cellText=counts,
        rowLabels=rows,
        colLabels=cols,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.4)
    for (r, c), cell in table.get_celld().items():
        if r == 0 or c == -1:
            cell.set_facecolor("#ececec")
            cell.get_text().set_fontweight("bold")
        elif c >= 0 and r > 0 and counts[r - 1][c] == "0":
            cell.get_text().set_color("#999999")
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
def build_metadata(
    *,
    dataset_path: str,
    from_idx: int | None,
    to_idx: int | None,
    indices: list[int] | None = None,
    connector_label: str,
    model: str,
    prompt_builder: PromptBuilder,
    usage_adapter: ApiUsageAdapter,
    usage_aggregate: UsageAggregate,
    results: list[EvaluationRow],
    strict_classes: set[str],
    use_batch_pricing: bool = False,
    opinion_retrieval_info: dict[str, Any] | None = None,
    reasoning_level: str | None = None,
) -> dict[str, Any]:
    status_counts = {"ERROR": 0, "MATCH": 0, "PARTIAL": 0, "WRONG": 0}
    provider_counts: dict[str, int] = {}
    citation_level_counts = {"matched": 0, "missing": 0, "unexpected": 0}

    for r in results:
        status_counts[r.evaluation] = status_counts.get(r.evaluation, 0) + 1
        provider_counts[r.provider_label] = provider_counts.get(r.provider_label, 0) + 1
        if r.counts is not None:
            citation_level_counts["matched"] += r.counts.matched
            citation_level_counts["missing"] += r.counts.missing
            citation_level_counts["unexpected"] += r.counts.unexpected

    class_level_metrics, aggregate_class_metrics, _confusion_matrix = (
        compute_class_metrics_and_confusion(
            ((r.expected, r.predicted) for r in results),
            strict_classes=strict_classes,
        )
    )

    in_rate, out_rate, pricing_mode = resolve_estimation_rates(
        usage_adapter, use_batch_pricing=use_batch_pricing
    )
    exec_block: dict[str, Any] = {
        "dataset_path": dataset_path,
        "from_idx": from_idx,
        "to_idx": to_idx,
        "indices": indices,
        "strict_classes": sorted(strict_classes),
    }
    if opinion_retrieval_info:
        exec_block["opinion_retrieval"] = opinion_retrieval_info
    metadata: dict[str, Any] = {
        "execution": exec_block,
        "llm": {
            "connector": connector_label,
            "model": model,
            "reasoning_level": reasoning_level,
            "prompt_style": prompt_builder.label,
            "calls_with_usage": usage_aggregate.llm_calls_with_usage,
            "system_prompt": prompt_builder.system_prompt,
            "user_prompt": prompt_builder.build_user_prompt(
                PromptContext(case_name="[CASE NAME]", citation="[CITATION]", opinion_text="[OPINION TEXT]")
            ),
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
        "observed_indices": [r.index_no for r in results],
        "status_counts": status_counts,
        "citation_level_counts": citation_level_counts,
        "class_level_metrics": class_level_metrics,
        "aggregate_class_metrics": aggregate_class_metrics,
        "provider_counts": provider_counts,
    }
    enrich_evaluation_metadata(metadata)
    return metadata
