from __future__ import annotations

from typing import Any, Optional

from src.api_usage.base import ApiUsageAdapter, TokenUsage, UsageAggregate


class JsonlUsageAdapter(ApiUsageAdapter):
    def __init__(self, model: str) -> None:
        self._model = model

    @property
    def label(self) -> str:
        return "jsonl"

    def extract_usage(self, response: Any) -> TokenUsage | None:
        _ = response
        return None

    def read_pricing_per_1m(self) -> tuple[Optional[float], Optional[float]]:
        return None, None

    def print_summary(self, aggregate: UsageAggregate, *, use_batch_pricing: bool = False) -> None:
        _ = aggregate, use_batch_pricing
        print("JSONL connector: token usage not available (offline file workflow).")
