from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: Optional[float] = None


class ApiUsageAdapter(ABC):
    @property
    @abstractmethod
    def label(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def extract_usage(self, response: Any) -> TokenUsage | None:
        raise NotImplementedError

    @abstractmethod
    def read_pricing_per_1m(self) -> tuple[Optional[float], Optional[float]]:
        raise NotImplementedError

    def read_batch_pricing_per_1m(self) -> tuple[Optional[float], Optional[float]]:
        """Optional batch-API rates (USD per 1M tokens). Default: unset."""
        return None, None

    @abstractmethod
    def print_summary(self, aggregate: "UsageAggregate", *, use_batch_pricing: bool = False) -> None:
        raise NotImplementedError


@dataclass
class UsageAggregate:
    llm_calls_with_usage: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: float = 0.0
    any_cost_in_api: bool = False

    def add(self, usage: TokenUsage | None) -> None:
        if usage is None:
            return
        self.llm_calls_with_usage += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.total_tokens += usage.total_tokens
        self.reasoning_tokens += usage.reasoning_tokens
        self.cached_input_tokens += usage.cached_input_tokens
        if usage.cost_usd is not None:
            self.cost_usd += usage.cost_usd
            self.any_cost_in_api = True
