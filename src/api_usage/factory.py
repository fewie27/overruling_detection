from __future__ import annotations

from src.api_usage.base import ApiUsageAdapter
from src.api_usage.claude import ClaudeUsageAdapter
from src.api_usage.deepseek import DeepSeekUsageAdapter
from src.api_usage.gemini import GeminiUsageAdapter
from src.api_usage.jsonl_usage import JsonlUsageAdapter
from src.api_usage.ollama import OllamaUsageAdapter
from src.api_usage.openai import OpenAIUsageAdapter


def build_usage_adapter(connector_label: str, model: str) -> ApiUsageAdapter:
    normalized = connector_label.strip().lower()
    if normalized == "openai":
        return OpenAIUsageAdapter(model=model)
    if normalized == "gemini":
        return GeminiUsageAdapter(model=model)
    if normalized == "claude":
        return ClaudeUsageAdapter(model=model)
    if normalized == "deepseek":
        return DeepSeekUsageAdapter(model=model)
    if normalized == "ollama":
        return OllamaUsageAdapter(model=model)
    if normalized == "jsonl":
        return JsonlUsageAdapter(model=model)
    raise ValueError(f"Unsupported usage adapter for connector {connector_label!r}")
