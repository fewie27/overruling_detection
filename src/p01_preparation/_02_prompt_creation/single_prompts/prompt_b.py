from __future__ import annotations

from src.models import PromptContext
from src.p01_preparation._02_prompt_creation.base import PromptBuilder


class PromptB(PromptBuilder):
    @property
    def label(self) -> str:
        return "prompt_b"

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
            "Task: Analyze the provided Supreme Court opinion. Identify ONLY the Supreme Court cases where the CURRENT opinion is the formal \"cause of death\" for that case's holding. Also provide your reasoning as output in the field 'reasoning'.\n\n"
            "Definitions for Extraction:\n\n"
            "directly_overruled: List Supreme Court cases where the CURRENT opinion is the formal cause of loss of "
            "authority for that precedent — either explicit overruling language OR the Court rejects the precedent's "
            "central holding or logic so that it is no longer valid precedent from this decision onward.\n\n"
            "CRITICAL EXCLUSIONS (To prevent \"PARTIAL\" errors):\n"
            "- NO HISTORICAL CITATIONS: If the Court mentions that a case was overruled by a PREVIOUS decision (e.g., \"Case X, which we overruled in Case Y\"), do NOT include Case X. \n"
            "- NO CRITICIZED EXAMPLES: If the Court criticizes a \"line of cases\" but only formally overrules one specific case, do NOT list the other examples in the line.\n"
            "- NO \"SEE ALSO\": Do not extract cases cited with signals like \"See, e.g.,\" \"cf.,\" or \"See also.\"\n"
            "- SCOTUS ONLY: Exclude all lower court citations (F.2d, F.3d, F. Supp, etc.).\n\n"
            "CITATION RULES:\n"
            "- Use format: \"Name, Vol U.S. Page (Year)\".\n"
            "- Normalize all nominative reporters (e.g., 17 How. 477) to the U.S. reporter (58 U.S. 477).\n\n"
            "JSON Structure:\n"
            "{\n"
            '  "reasoning": "Reasoning for the extraction...",\n'
            '  "directly_overruled": ["Name, Citation"]\n'
            "}\n"
        )
