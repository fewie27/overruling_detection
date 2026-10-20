from __future__ import annotations

from src.models import PromptContext
from src.p01_preparation._02_prompt_creation.base import PromptBuilder


class PromptA(PromptBuilder):
    @property
    def label(self) -> str:
        return "prompt_a"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a very experienced lawyer that is an expert for overrulling cases in the "
            "United States. You solve problems analytically IMPORTANT: Only answer in "
            "well-formatted JSON objects and nothing else. Start your answer with putting your reasoning in the JSON output field 'reasoning'."
        )

    def build_user_prompt(self, ctx: PromptContext) -> str:
        return (
            "Your entire reply must be a single valid JSON object (no markdown, no prose outside JSON).\n\n"
            f"This is the opinion of the court of {ctx.case_name}, {ctx.citation}:\n\n"
            f"{ctx.opinion_text}\n\n"
            "By looking at this text only: Which case or cases was/were overruled or not good law anymore according to *this* opinion?"
            "Task: Analyze the provided judicial opinion text and extract Surpreme Court(!) cases that are no longer 'good law' as a result of this decision."
            "Return the citation string(s) as 'X vs. Y, NNN U.S. MMM (YYYY)'. "
            'e.g. {"reasoning": "...", "directly_overruled": ["Tom v. Jerry, 123 U.S. 456 (2026)"]}'
            "Every attribute can have zero, one or multiple cases. Return empty array [] if no cases are found."
            "Always use modern and official citation format for the case citations. Even if those cases are quite old."
        )

    @property
    def finetune_system_prompt(self) -> str | None:
        return (
            'U.S. Supreme Court overruling expert. Reply with JSON only; put your reasoning in "reasoning".'
        )

    def build_finetune_user_prompt(self, ctx: PromptContext) -> str | None:
        return (
            f"Opinion: {ctx.case_name}, {ctx.citation}.\n\n{ctx.opinion_text}\n\n"
            "Which Supreme Court holdings are no longer good law in this decision? "
            'Return JSON: {"reasoning": "...", "directly_overruled": ["Citation"]}. '
            "Citations as 'Name v. Name, NNN U.S. MMM (YYYY)'; use [] if none."
        )
