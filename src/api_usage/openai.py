from __future__ import annotations

from typing import Any, Optional

from src.api_usage.base import ApiUsageAdapter, TokenUsage, UsageAggregate
from src.config.models import lookup_pricing, models_config_path


def estimated_cost_usd(
    input_tokens: int,
    output_tokens: int,
    input_usd_per_1m: Optional[float],
    output_usd_per_1m: Optional[float],
) -> Optional[float]:
    if input_usd_per_1m is None or output_usd_per_1m is None:
        return None
    return (input_tokens / 1_000_000.0) * input_usd_per_1m + (output_tokens / 1_000_000.0) * output_usd_per_1m


def resolve_estimation_rates(
    adapter: ApiUsageAdapter,
    *,
    use_batch_pricing: bool,
) -> tuple[Optional[float], Optional[float], str]:
    """
    For batch finalization, prefer batch rates from config/models.yaml when available.
    Returns (input_rate, output_rate, mode) where mode is "batch" or "standard".
    """
    if use_batch_pricing:
        bin_rate, bout_rate = adapter.read_batch_pricing_per_1m()
        if bin_rate is not None and bout_rate is not None:
            return bin_rate, bout_rate, "batch"
    in_rate, out_rate = adapter.read_pricing_per_1m()
    return in_rate, out_rate, "standard"


class OpenAIUsageAdapter(ApiUsageAdapter):
    def __init__(self, model: str) -> None:
        self._model = model

    @property
    def label(self) -> str:
        return "openai"

    def extract_usage(self, response: Any) -> TokenUsage | None:
        if isinstance(response, dict):
            usage = response.get("usage")
            if not isinstance(usage, dict):
                return None
            output_details = usage.get("output_tokens_details") or {}
            input_details = usage.get("input_tokens_details") or {}
            return TokenUsage(
                input_tokens=int(usage.get("input_tokens", 0) or 0),
                output_tokens=int(usage.get("output_tokens", 0) or 0),
                total_tokens=int(usage.get("total_tokens", 0) or 0),
                reasoning_tokens=int(output_details.get("reasoning_tokens", 0) or 0),
                cached_input_tokens=int(input_details.get("cached_tokens", 0) or 0),
                cost_usd=_usd_cost_from_response(response),
            )
        u = getattr(response, "usage", None)
        if u is None:
            return None
        try:
            reasoning = int(getattr(u.output_tokens_details, "reasoning_tokens", 0) or 0)
        except Exception:
            reasoning = 0
        try:
            cached = int(getattr(u.input_tokens_details, "cached_tokens", 0) or 0)
        except Exception:
            cached = 0
        return TokenUsage(
            input_tokens=int(getattr(u, "input_tokens", 0) or 0),
            output_tokens=int(getattr(u, "output_tokens", 0) or 0),
            total_tokens=int(getattr(u, "total_tokens", 0) or 0),
            reasoning_tokens=reasoning,
            cached_input_tokens=cached,
            cost_usd=_usd_cost_from_response(response),
        )

    def read_pricing_per_1m(self) -> tuple[Optional[float], Optional[float]]:
        return lookup_pricing("openai", self._model, use_batch=False)

    def read_batch_pricing_per_1m(self) -> tuple[Optional[float], Optional[float]]:
        return lookup_pricing("openai", self._model, use_batch=True)

    def print_summary(self, aggregate: UsageAggregate, *, use_batch_pricing: bool = False) -> None:
        print("\n--- OpenAI API usage (this run) ---")
        print(f"  LLM calls with usage in response: {aggregate.llm_calls_with_usage}")
        if aggregate.llm_calls_with_usage == 0:
            print("  (No usage blocks returned — e.g. all rows failed before the API call.)")
            return
        print(f"  Sum input tokens:          {aggregate.input_tokens}")
        print(f"  Sum cached input tokens:   {aggregate.cached_input_tokens}")
        print(f"  Sum output tokens:         {aggregate.output_tokens}")
        print(f"  Sum reasoning tokens:      {aggregate.reasoning_tokens}")
        print(f"  Sum total tokens:          {aggregate.total_tokens}")
        if aggregate.any_cost_in_api:
            print(f"  Sum cost (USD, from API):  ${aggregate.cost_usd:.6f}")
        else:
            print("  Sum cost (USD, from API):  (not provided in API responses)")
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


def _usd_cost_from_response(response: Any) -> Optional[float]:
    if isinstance(response, dict):
        d = response
    else:
        try:
            d = response.model_dump(mode="json")
        except Exception:
            return None
    for key in ("cost", "total_cost", "usage_cost", "estimated_cost"):
        val = d.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    usage = d.get("usage")
    if isinstance(usage, dict):
        for key in ("cost", "total_cost"):
            val = usage.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
    return None
