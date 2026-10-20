from __future__ import annotations

from src.models import PromptContext
from src.p01_preparation._02_prompt_creation.base import PromptBuilder
from src.p03a_evaluation.citation_treatment_labels import format_labels_for_prompt


class PromptF(PromptBuilder):
    """One request per opinion: every citation, each labeled (full-text classification)."""

    @property
    def label(self) -> str:
        return "prompt_f"

    @property
    def response_parser_id(self) -> str:
        return "P2_classification_parser"

    @property
    def system_prompt(self) -> str:
        labels_block = format_labels_for_prompt()
        return (
            "You are a U.S. Supreme Court precedent extraction system. Reply with one JSON object only, no markdown.\n\n"
            "Classify every Supreme Court citation you can find in the opinion in the user message. "
            "The actor whose treatment matters is THIS opinion — not other courts, not legal history in general.\n\n"
            "Use exactly one label from the list below (snake_case) for every citation. "
            "Prefer higher labels (earlier in the list) when unsure or ambiguous.\n\n"
            "Labels (decreasing strength):\n"
            f"{labels_block}"
        )

    def build_user_prompt(self, ctx: PromptContext) -> str:
        return (
            "Extract the Supreme Court citations from the opinion and classify them according to the system prompt.\n\n"
            f"### Opinion ###\n"
            f"{ctx.case_name}, {ctx.citation}.\n\n"
            f"{ctx.opinion_text}\n\n\n"
            "----------------------------------------\n\n"
            "For every citation from the opinion classify according to the labels in the system prompt. The result should be a JSON object with attribute citations which holds an array with all existing citations."
            "Respond is one JSON object only, following this format:\n"
            '{"citations":[{"cite":"Party v. Party, 123 U.S. 456 (2000)","class":"<label>"},...]}\n\n'
            'Each item uses the key "class" for the label. Use "cite" for the full citation string when the case is identifiable. '
            'Omit items you cannot map to a label; use an empty "citations" array if there are no Supreme Court citations.\n'
            "Always use modern citation format: 'Party v. Party, Volume U.S. Page (Year)' (prefer U.S. reporter)."
        )


