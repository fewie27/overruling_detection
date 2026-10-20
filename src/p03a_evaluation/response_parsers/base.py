"""Base type for normalizing model JSON into :class:`ExtractionResult`."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import ExtractionResult


class LlmResponseParser(ABC):
    """Parse raw model text into a normalized extraction (list of classified citations)."""

    @abstractmethod
    def parse(self, raw_content: str) -> ExtractionResult:
        raise NotImplementedError
