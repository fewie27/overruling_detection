from __future__ import annotations

from typing import Any, Optional

from src.api_usage.base import ApiUsageAdapter, TokenUsage, UsageAggregate
from src.config.models import lookup_pricing, models_config_path


class GeminiUsageAdapter(ApiUsageAdapter):
    def __init__(self, model: str) -> None:
        self._model = model

    @property
    def label(self) -> str:
        return "gemini"

    def extract_usage(self, response: Any) -> TokenUsage | None:
        _ = response
        return None

    def read_pricing_per_1m(self) -> tuple[Optional[float], Optional[float]]:
        return lookup_pricing("gemini", self._model, use_batch=False)

    def print_summary(self, aggregate: UsageAggregate, *, use_batch_pricing: bool = False) -> None:
        _ = use_batch_pricing
        print("\n--- Gemini API usage (this run) ---")
        print(f"  LLM calls with usage in response: {aggregate.llm_calls_with_usage}")
        print("  (Usage/cost fields are not currently parsed for Gemini responses.)")
        in_rate, out_rate = self.read_pricing_per_1m()
        if in_rate is None or out_rate is None:
            print(
                f"  Estimated cost (USD):      (no pricing for model {self._model!r} in "
                f"{models_config_path()})"
            )
