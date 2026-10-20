from __future__ import annotations

from src.models import PromptContext
from src.p01_preparation._02_prompt_creation.base import PromptBuilder
from src.p01_preparation._02_prompt_creation.single_prompts.prompt_i import (
    PromptI,
)


class PromptIV2(PromptBuilder):
    """Prompt I without opinion text in the user prompt (citation/name only)."""

    @property
    def label(self) -> str:
        return "prompt_i_v2"

    @property
    def response_parser_id(self) -> str:
        return "P4_detailed_parser"

    @property
    def system_prompt(self) -> str:
        return PromptI().system_prompt

    def build_user_prompt(self, ctx: PromptContext) -> str:
        cite = (ctx.citation or "").strip()
        name = (ctx.case_name or "").strip()
        decision = f"{name}, {cite}".strip(", ") if cite else name
        return (
            "Extract the overruling cases as described in the system prompt. Return JSON only.\n"
            f"Overruling decision: {decision}\n"
            "No opinion text is provided; rely on your knowledge of this decision."
        )
