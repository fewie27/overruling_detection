from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.models import CaseDocument, PreparedLLMRequest, PromptContext

if TYPE_CHECKING:
    from src.models.case_trace import PipelineInputRow


class PromptBuilder(ABC):
    @property
    @abstractmethod
    def label(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        raise NotImplementedError

    @property
    def response_parser_id(self) -> str:
        """Id for :mod:`src.p03a_evaluation.response_parsers` (see ``get_parser``)."""
        return "P1_overrule_parser"

    @abstractmethod
    def build_user_prompt(self, ctx: PromptContext) -> str:
        raise NotImplementedError

    @property
    def finetune_system_prompt(self) -> str | None:
        return None

    def build_finetune_user_prompt(self, ctx: PromptContext) -> str | None:
        return None

    def expand_prepared_requests(
        self,
        *,
        row: PipelineInputRow,
        case_doc: CaseDocument,
        provider_label: str,
        case_citation_for_output: str,
    ) -> list[PreparedLLMRequest] | None:
        """
        If this returns non-None, these requests replace the default single-request build
        (e.g. one row -> many LLM calls). If None, the runner builds one PreparedLLMRequest from build_*.
        """
        return None
