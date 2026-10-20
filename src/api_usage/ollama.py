from __future__ import annotations

from typing import Any, Optional

from src.api_usage.base import ApiUsageAdapter, TokenUsage, UsageAggregate
from src.api_usage.openai import estimated_cost_usd
from src.config.models import lookup_pricing, models_config_path


class OllamaUsageAdapter(ApiUsageAdapter):
    def __init__(self, model: str) -> None:
        self._model = model

    @property
    def label(self) -> str:
        return "ollama"

    def extract_usage(self, response: Any) -> TokenUsage | None:
        if not isinstance(response, dict):
            return None
        inp = int(response.get("prompt_eval_count") or 0)
        out = int(response.get("eval_count") or 0)
        if inp == 0 and out == 0:
            return None
        return TokenUsage(
            input_tokens=inp,
            output_tokens=out,
            total_tokens=inp + out,
            reasoning_tokens=0,
            cached_input_tokens=0,
            cost_usd=None,
        )

    def read_pricing_per_1m(self) -> tuple[Optional[float], Optional[float]]:
        return lookup_pricing("ollama", self._model, use_batch=False)

    def print_summary(self, aggregate: UsageAggregate, *, use_batch_pricing: bool = False) -> None:
        _ = use_batch_pricing
        print("\n--- Ollama API usage (this run) ---")
        print(f"  LLM calls with usage in response: {aggregate.llm_calls_with_usage}")
        if aggregate.llm_calls_with_usage == 0:
            print("  (No token counts in API responses — e.g. failed calls or older Ollama builds.)")
            return
        print(f"  Sum prompt tokens (prompt_eval_count): {aggregate.input_tokens}")
        print(f"  Sum completion tokens (eval_count):   {aggregate.output_tokens}")
        print(f"  Sum total (prompt + completion):     {aggregate.total_tokens}")
        in_rate, out_rate = self.read_pricing_per_1m()
        est = estimated_cost_usd(aggregate.input_tokens, aggregate.output_tokens, in_rate, out_rate)
        if est is not None:
            print(
                f"  Estimated cost (USD):          ${est:.6f}  "
                f"(from config/models.yaml, model {self._model!r})"
            )
        else:
            print(
                f"  Estimated cost (USD):          (no pricing for local model {self._model!r}; "
                f"add rates under ollama in {models_config_path()} if needed)"
            )
