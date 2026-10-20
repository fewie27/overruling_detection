from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.models.case_trace import CaseTraceMetadata


def _normalize_relation_label(relation: str) -> str:
    return relation.strip().casefold().replace(" ", "_").replace("-", "_")


def _is_overruling_recognized(relation: str) -> bool:
    return _normalize_relation_label(relation) == "overruling_recognized"


def _recognized_by_from_item(item: dict[str, Any], *, relation: str) -> str | None:
    if not _is_overruling_recognized(relation):
        return None
    raw = item.get("recognized_by")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip()


@dataclass(frozen=True)
class CitationRelation:
    src_reporter: str | None
    src_volume: str | None
    src_page: str | None
    relation: str
    dst: str
    reasoning: str | None
    quotes: list[str]
    case_law_id: int | None = None
    decision_date: str | None = None
    recognized_by: str | None = None


def _quotes_to_json(quotes: list[str]) -> str:
    return json.dumps(quotes, ensure_ascii=True)


def _normalize_quotes(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def extract_relations_from_response(
    response: dict[str, Any],
    *,
    trace: CaseTraceMetadata,
) -> list[CitationRelation]:
    """Map parsed LLM JSON payloads to traceable citation relations."""
    if not isinstance(response, dict):
        return []

    base = CitationRelation(
        src_reporter=trace.reporter,
        src_volume=trace.volume,
        src_page=trace.page,
        relation="",
        dst="",
        reasoning=None,
        quotes=[],
        case_law_id=trace.case_law_id,
        decision_date=trace.decision_date,
    )
    out: list[CitationRelation] = []

    overruled_cases = response.get("overruled_cases")
    if isinstance(overruled_cases, list):
        for item in overruled_cases:
            if not isinstance(item, dict):
                continue
            case = item.get("case")
            relation = item.get("classification")
            if not isinstance(case, str) or not case.strip():
                continue
            if not isinstance(relation, str) or not relation.strip():
                continue
            reasoning = item.get("reasoning")
            relation_label = relation.strip()
            out.append(
                CitationRelation(
                    src_reporter=base.src_reporter,
                    src_volume=base.src_volume,
                    src_page=base.src_page,
                    relation=relation_label,
                    dst=case.strip(),
                    reasoning=str(reasoning).strip() if isinstance(reasoning, str) and reasoning.strip() else None,
                    quotes=_normalize_quotes(item.get("quotes")),
                    case_law_id=base.case_law_id,
                    decision_date=base.decision_date,
                    recognized_by=_recognized_by_from_item(item, relation=relation_label),
                )
            )
        if out:
            return out

    directly_overruled = response.get("directly_overruled")
    top_reasoning = response.get("reasoning")
    if isinstance(directly_overruled, list):
        shared_reasoning = (
            str(top_reasoning).strip()
            if isinstance(top_reasoning, str) and top_reasoning.strip()
            else None
        )
        for cite in directly_overruled:
            if isinstance(cite, str) and cite.strip():
                out.append(
                    CitationRelation(
                        src_reporter=base.src_reporter,
                        src_volume=base.src_volume,
                        src_page=base.src_page,
                        relation="overruled",
                        dst=cite.strip(),
                        reasoning=shared_reasoning,
                        quotes=[],
                        case_law_id=base.case_law_id,
                        decision_date=base.decision_date,
                    )
                )
        if out:
            return out

    cited_citation = response.get("cited_citation")
    classification = response.get("classification")
    if isinstance(cited_citation, str) and cited_citation.strip() and isinstance(classification, str):
        chunk_reasoning = response.get("reasoning")
        out.append(
            CitationRelation(
                src_reporter=base.src_reporter,
                src_volume=base.src_volume,
                src_page=base.src_page,
                relation=classification.strip(),
                dst=cited_citation.strip(),
                reasoning=(
                    str(chunk_reasoning).strip()
                    if isinstance(chunk_reasoning, str) and chunk_reasoning.strip()
                    else None
                ),
                quotes=[],
                case_law_id=base.case_law_id,
                decision_date=base.decision_date,
            )
        )
        return out

    if isinstance(response, list):
        for item in response:
            if not isinstance(item, dict):
                continue
            cited = item.get("cited_decision")
            overruled = item.get("overruled")
            if not isinstance(cited, str) or not cited.strip():
                continue
            if overruled is not True:
                continue
            rationale = item.get("rationale")
            quote = item.get("quote")
            quotes = [str(quote).strip()] if isinstance(quote, str) and quote.strip() else []
            out.append(
                CitationRelation(
                    src_reporter=base.src_reporter,
                    src_volume=base.src_volume,
                    src_page=base.src_page,
                    relation="overruled",
                    dst=cited.strip(),
                    reasoning=(
                        str(rationale).strip()
                        if isinstance(rationale, str) and rationale.strip()
                        else None
                    ),
                    quotes=quotes,
                    case_law_id=base.case_law_id,
                    decision_date=base.decision_date,
                )
            )
    return out


def collect_relations_from_llm_responses(
    llm_responses: Iterable[dict[str, Any]],
) -> list[CitationRelation]:
    relations: list[CitationRelation] = []
    for entry in llm_responses:
        if str(entry.get("state", "")).upper() == "ERROR":
            continue
        trace_raw = entry.get("trace") or {}
        trace = CaseTraceMetadata.from_dict(trace_raw if isinstance(trace_raw, dict) else None)
        response = entry.get("response")
        if not isinstance(response, dict):
            continue
        relations.extend(extract_relations_from_response(response, trace=trace))
    return relations


def default_relations_db_path() -> str:
    override = os.getenv("EXTRACT_RELATIONS_DB", "").strip()
    if override:
        return override
    return str(Path("output") / "citation_relations.sqlite")


def _ensure_recognized_by_column(conn: sqlite3.Connection) -> None:
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(citation_relations)").fetchall()
    }
    if "recognized_by" not in columns:
        conn.execute("ALTER TABLE citation_relations ADD COLUMN recognized_by TEXT")


def init_relations_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS citation_relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            case_law_id INTEGER,
            decision_date TEXT,
            src_reporter TEXT,
            src_volume TEXT,
            src_page TEXT,
            relation TEXT NOT NULL,
            dst TEXT NOT NULL,
            recognized_by TEXT,
            reasoning TEXT,
            quotes TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    _ensure_recognized_by_column(conn)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_citation_relations_run ON citation_relations(run_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_citation_relations_src "
        "ON citation_relations(src_reporter, src_volume, src_page)"
    )
    conn.commit()


def write_relations_to_sqlite(
    relations: list[CitationRelation],
    *,
    run_id: str,
    db_path: str | None = None,
) -> str:
    path = db_path or default_relations_db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        init_relations_db(conn)
        conn.executemany(
            """
            INSERT INTO citation_relations (
                run_id, case_law_id, decision_date,
                src_reporter, src_volume, src_page,
                relation, dst, recognized_by, reasoning, quotes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    rel.case_law_id,
                    rel.decision_date,
                    rel.src_reporter,
                    rel.src_volume,
                    rel.src_page,
                    rel.relation,
                    rel.dst,
                    rel.recognized_by,
                    rel.reasoning,
                    _quotes_to_json(rel.quotes),
                )
                for rel in relations
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return path
