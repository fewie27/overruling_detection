"""LLM output parsers; select via :func:`get_parser` using ``PromptBuilder.response_parser_id``."""

from __future__ import annotations

from typing import Final

from src.p03a_evaluation.response_parsers.P1_overrule_parser import P1OverruleParser
from src.p03a_evaluation.response_parsers.P2_classification_parser import P2ClassificationParser
from src.p03a_evaluation.response_parsers.P3_chunk_classification_parser import P3ChunkClassificationParser
from src.p03a_evaluation.response_parsers.P4_detailed_parser import P4DetailedParser
from src.p03a_evaluation.response_parsers.base import LlmResponseParser

PARSER_ID_P1_OVERRULE: Final[str] = "P1_overrule_parser"
PARSER_ID_P2_CLASSIFICATION: Final[str] = "P2_classification_parser"
PARSER_ID_P3_CHUNK_CLASSIFICATION: Final[str] = "P3_chunk_classification_parser"
PARSER_ID_P4_DETAILED: Final[str] = "P4_detailed_parser"

_PARSERS: dict[str, LlmResponseParser] = {
    PARSER_ID_P1_OVERRULE: P1OverruleParser(),
    PARSER_ID_P2_CLASSIFICATION: P2ClassificationParser(),
    PARSER_ID_P3_CHUNK_CLASSIFICATION: P3ChunkClassificationParser(),
    PARSER_ID_P4_DETAILED: P4DetailedParser(),
}


def get_parser(response_parser_id: str) -> LlmResponseParser:
    p = _PARSERS.get(response_parser_id)
    if p is not None:
        return p
    return _PARSERS[PARSER_ID_P1_OVERRULE]
