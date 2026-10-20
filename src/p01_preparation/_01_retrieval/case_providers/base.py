from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from src.models import CaseDocument


class CaseProvider(ABC):
    @property
    @abstractmethod
    def label(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_case_by_citation(
        self,
        citation: str,
        *,
        case_law_caption_matcher: Callable[[str], bool] | None = None,
        require_full_casebody: bool = False,
    ) -> CaseDocument:
        raise NotImplementedError
