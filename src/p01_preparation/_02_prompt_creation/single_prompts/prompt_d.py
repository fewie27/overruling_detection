from __future__ import annotations

from src.models import PromptContext
from src.p01_preparation._02_prompt_creation.base import PromptBuilder


class PromptD(PromptBuilder):
    @property
    def label(self) -> str:
        return "prompt_d"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a precision-oriented legal analyst specializing in US Supreme Court precedent. \n\n"
            "Your task is to identify cases that this specific opinion EXPLICITLY overrules.\n\n"
            "RULES:\n"
            '1. ONLY include cases where the text uses phrases like "is overruled," "no longer good law," or "we depart from the holding in."\n'
            "2. VERIFICATION: For every case identified, you must extract a direct quote from the provided text that proves the overruling.\n"
            "3. GROUNDING: Do not use your internal knowledge to suggest cases. If the text does not explicitly overrule a case, return an empty list.\n"
            "4. OUTPUT: You must respond ONLY with a JSON object."
        )

    def build_user_prompt(self, ctx: PromptContext) -> str:
        return (
            "Task: Analyze the provided Supreme Court opinion and identify cases that are overruled.\n"
            "Schema Definition:\n"
            "1. 'reasoning': Internal logic for your choice.\n"
            "2. 'directly_overruled': All Supreme Court cases this opinion displaces — explicit overruling, "
            "'no longer good law', 'no longer viable', 'superseded', or equivalent clear loss of controlling authority.\n"
            "\n"
            "STRICT RULE: Do not include the 'lower court', 'Court of Appeals', or 'the judgment below'. These are not overruled cases; they are procedural actions.\n\n"
            "### EXAMPLE ###\n"
            "Input Text: '...While the Court of Appeals ruled for the petitioner, we must vacate that judgment. Furthermore, we find our prior holding in Smith v. Jones, 100 U.S. 1 (1900) to be incompatible with the Constitution. We today expressly overrule Smith.'\n"
            "Output: {\n"
            '  "reasoning": "The court vacated the lower court (procedural) but specifically used the word \'overrule\' regarding the Supreme Court case Smith v. Jones.",\n'
            '  "directly_overruled": ["Smith v. Jones, 100 U.S. 1 (1900)"]\n'
            "}\n\n"
            "### ACTUAL TASK ###\n"
            f"Case: {ctx.case_name}, {ctx.citation}\n"
            f"Text: {ctx.opinion_text}\n\n"
            "JSON Output:"
        )
