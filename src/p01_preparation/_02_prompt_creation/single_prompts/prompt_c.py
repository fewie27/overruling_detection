from __future__ import annotations

from src.models import PromptContext
from src.p01_preparation._02_prompt_creation.base import PromptBuilder


class PromptC(PromptBuilder):
    @property
    def label(self) -> str:
        return "prompt_c"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a very experienced lawyer that is an expert for overrulling cases in the "
            "United States. You solve problems analytically IMPORTANT: Only answer in "
            "well-formatted JSON objects and nothing else. Start your answer with putting your reasoning in the JSON output field 'reasoning'."
        )

    def build_user_prompt(self, ctx: PromptContext) -> str:
        return (
            "Your entire reply must be a single valid JSON object. \n\n"
            "Context: You are a legal scholar specializing in SCOTUS jurisprudence.\n\n"
            f"Task: Analyze the provided judicial opinion of {ctx.case_name}, {ctx.citation}.\n\n"
            "Identify any Supreme Court cases that this decision explicitly or functionally overrules, vacates, or renders \"no longer good law\" (either in whole or in part).\n\n"
            "Search Strategy:\n\n"
            "1. Look for explicit language: \"overruled,\" \"disavowed,\" \"no longer followed,\" \"we depart from.\"\n"
            "2. Look for functional language: \"inconsistent with our holding today,\" \"cannot be reconciled,\" \"limited to its facts,\" or \"abandoned.\"\n"
            "3. Pay special attention to historical \"Nominative Reporters\" (e.g., Dall., Cranch, Wheat., Pet., Howard, Black, Wall.). These are official Supreme Court citations.\n\n"
            "Formatting Rules:\n"
            '- Return citations as: "Party A v. Party B, [Volume] U.S. [Page] ([Year])". \n'
            '- If a case uses a nominative reporter, include it in parentheses: "X v. Y, 74 U.S. (7 Wall.) 624 (1869)".\n'
            "JSON Structure:\n"
            "{\n"
            '  "reasoning": "A brief explanation of how the current opinion treats the cited cases.",\n'
            '  "directly_overruled": ["Albert v. Bert, 123 U.S. 456 (2026)"]\n'
            "}\n\n"
            'If no cases are overruled, return "directly_overruled": [].\n\n'
            "Opinion Text:\n"
            f"{ctx.opinion_text}\n"
        )
