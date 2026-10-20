from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.models import PreparedLLMRequest


@dataclass
class ConnectorResponse:
    """Raw model output from a connector. Parsing is intentionally out of scope here;
    connectors only handle the connection/transport. The raw text is parsed downstream
    (see :func:`src.p03a_evaluation.llm_response_parser.parse_llm_json_content`).

    ``reasoning_content`` holds the model's native reasoning/thinking text (Claude
    ``thinking`` blocks, DeepSeek ``reasoning_content``) when the provider exposes it.
    It is ``None`` when the provider returns no separate reasoning channel."""

    raw_content: str
    raw_response: Any = None
    reasoning_content: str | None = None


@dataclass(frozen=True)
class BatchSubmitResult:
    batch_id: str


@dataclass(frozen=True)
class BatchStatus:
    batch_id: str
    status: str
    is_terminal: bool
    is_success: bool


class LLMConnector(ABC):
    @property
    @abstractmethod
    def label(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def model(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def extract_overruled_cases(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ConnectorResponse:
        raise NotImplementedError

    def submit_batch(self, requests: list[PreparedLLMRequest]) -> BatchSubmitResult:
        raise NotImplementedError(f"Batch mode is not implemented for connector {self.label!r}")

    def get_batch_status(self, batch_id: str) -> BatchStatus:
        raise NotImplementedError(f"Batch mode is not implemented for connector {self.label!r}")

    def collect_batch_results(self, batch_id: str) -> dict[str, ConnectorResponse]:
        raise NotImplementedError(f"Batch mode is not implemented for connector {self.label!r}")
