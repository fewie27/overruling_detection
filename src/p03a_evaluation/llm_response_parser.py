"""Select a parser by ``response_parser_id`` and normalize model JSON into :class:`ExtractionResult`."""

from __future__ import annotations

import json
from typing import Any

from src.models import ExtractionResult
from src.p03a_evaluation.response_parsers import get_parser
from src.p03a_evaluation.response_parsers.json_lenient import load_json_lenient


def parse_llm_json_content(
    raw_content: str, *, response_parser_id: str = "P1_overrule_parser"
) -> ExtractionResult:
    return get_parser(response_parser_id).parse(raw_content or "")


def payload_from_raw_content(raw_content: str) -> dict[str, Any]:
    """Structured JSON for `llm_result_arrays.json` (best-effort parse)."""
    try:
        data = load_json_lenient(raw_content or "")
    except (json.JSONDecodeError, ValueError):
        return {}
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {"citations": data}
    return {}
