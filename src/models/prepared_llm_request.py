from dataclasses import dataclass, field

from src.models.case_trace import CaseTraceMetadata
from src.models.classified_citation import ClassifiedCitation


@dataclass
class PreparedLLMRequest:
    """Single normalized LLM request prepared by the pipeline."""

    request_id: str
    row_no: int
    display_case_name: str
    provider_label: str
    expected_classifications: list[ClassifiedCitation]
    case_citation_for_output: str
    system_prompt: str
    user_prompt: str
    lookup_citation: str | None = None
    finetune_system_prompt: str | None = None
    finetune_user_prompt: str | None = None
    # When True, finalize merges extractions for the same row_no (multi-prompt strategies).
    merge_row_responses: bool = False
    # Chunk prompts: plain texts (e.g. paragraphs) included in the user message; None otherwise.
    chunk_texts: list[str] | None = None
    # Chunk prompts: cite substrings for this request (for per-citation evaluation); None otherwise.
    chunk_citation_surfaces: list[str] | None = None
    # Which :mod:`src.p03a_evaluation.response_parsers` to use when parsing model JSON.
    response_parser_id: str = "P1_overrule_parser"
    # Trace metadata (case law id, reporter/volume/page, decision date) for production output.
    trace: CaseTraceMetadata = field(default_factory=CaseTraceMetadata)
