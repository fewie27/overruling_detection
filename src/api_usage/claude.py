from __future__ import annotations

from typing import Any, Optional

from src.api_usage.base import ApiUsageAdapter, TokenUsage, UsageAggregate
from src.api_usage.openai import estimated_cost_usd, resolve_estimation_rates
from src.config.models import lookup_pricing, models_config_path


class ClaudeUsageAdapter(ApiUsageAdapter):
    def __init__(self, model: str) -> None:
        self._model = model

    @property
    def label(self) -> str:
        return "claude"

    def extract_usage(self, response: Any) -> TokenUsage | None:
        if isinstance(response, dict):
            usage = response.get("usage")
            if not isinstance(usage, dict):
                return None
            inp = int(usage.get("input_tokens", 0) or 0)
            out = int(usage.get("output_tokens", 0) or 0)
            return TokenUsage(
                input_tokens=inp,
                output_tokens=out,
                total_tokens=inp + out,
                reasoning_tokens=0,
                cached_input_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
                cost_usd=None,
            )
        u = getattr(response, "usage", None)
        if u is None:
            return None
        inp = int(getattr(u, "input_tokens", 0) or 0)
        out = int(getattr(u, "output_tokens", 0) or 0)
        cached = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        return TokenUsage(
            input_tokens=inp,
            output_tokens=out,
            total_tokens=inp + out,
            reasoning_tokens=0,
            cached_input_tokens=cached,
            cost_usd=None,
        )

    def read_pricing_per_1m(self) -> tuple[Optional[float], Optional[float]]:
        return lookup_pricing("claude", self._model, use_batch=False)

    def read_batch_pricing_per_1m(self) -> tuple[Optional[float], Optional[float]]:
        return lookup_pricing("claude", self._model, use_batch=True)

    def print_summary(self, aggregate: UsageAggregate, *, use_batch_pricing: bool = False) -> None:
        print("\n--- Anthropic Claude API usage (this run) ---")
        print(f"  LLM calls with usage in response: {aggregate.llm_calls_with_usage}")
        if aggregate.llm_calls_with_usage == 0:
            print("  (No usage blocks returned — e.g. all rows failed before the API call.)")
            return
        print(f"  Sum input tokens:          {aggregate.input_tokens}")
        print(f"  Sum cached input tokens:   {aggregate.cached_input_tokens}")
        print(f"  Sum output tokens:         {aggregate.output_tokens}")
        print(f"  Sum total tokens:          {aggregate.total_tokens}")
        print("  Sum cost (USD, from API):  (not provided in standard Messages API responses)")
        in_rate, out_rate, mode = resolve_estimation_rates(self, use_batch_pricing=use_batch_pricing)
        est = estimated_cost_usd(aggregate.input_tokens, aggregate.output_tokens, in_rate, out_rate)
        if est is not None:
            src = f"config/models.yaml ({self._model}, {mode} rates)"
            print(f"  Estimated cost (USD):      ${est:.6f}  (from {src})")
        else:
            print(
                f"  Estimated cost (USD):      (no pricing for model {self._model!r} in "
                f"{models_config_path()})"
            )
