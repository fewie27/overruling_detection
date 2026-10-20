from __future__ import annotations

from src.api_connectors.base import LLMConnector
from src.api_connectors.claude_connector import ClaudeConnector
from src.api_connectors.deepseek_connector import DeepSeekConnector
from src.api_connectors.gemini_connector import GeminiConnector
from src.api_connectors.jsonl_connector import JsonlFileConnector
from src.api_connectors.ollama_connector import OllamaConnector
from src.api_connectors.openai_connector import OpenAIConnector


def build_llm_connector(
    connector_name: str,
    model: str | None = None,
    *,
    reasoning_effort: str | None = None,
    jsonl_output_dir: str | None = None,
) -> LLMConnector:
    normalized = connector_name.strip().lower()
    if normalized == "openai":
        return OpenAIConnector(model=model, reasoning_effort=reasoning_effort)
    if normalized == "gemini":
        return GeminiConnector(model=model)
    if normalized == "claude":
        return ClaudeConnector(model=model, reasoning_effort=reasoning_effort)
    if normalized == "deepseek":
        return DeepSeekConnector(model=model, reasoning_effort=reasoning_effort)
    if normalized == "ollama":
        return OllamaConnector(model=model)
    if normalized in {"jsonl", "jsonl_file", "file_jsonl"}:
        return JsonlFileConnector(model=model, output_dir=jsonl_output_dir)
    raise ValueError(
        f"Unsupported connector {connector_name!r}. Allowed: openai, gemini, claude, deepseek, ollama, jsonl."
    )
