from __future__ import annotations

from src.models import PromptContext
from src.p01_preparation._02_prompt_creation.base import PromptBuilder


class PromptE(PromptBuilder):
    @property
    def label(self) -> str:
        return "prompt_e"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a very experienced lawyer and expert on overruled cases in the United States. "
            "You solve problems analytically.\n\n"
            "OUTPUT FORMAT: Always reply with a single valid JSON object and nothing else — "
            "no markdown, no prose outside JSON. The object must contain EXACTLY these two keys:\n"
            '  "reasoning": your step-by-step analysis as a string\n'
            '  "directly_overruled": array of citation strings (empty array [] if none)\n'
            "No other keys are permitted.\n\n"
            "CLASSIFICATION RULES:\n"
            "'directly_overruled': Include cases where the Court uses explicit overruling language OR explicitly "
            "addresses that case's authority as ended or no longer controlling (e.g. 'no longer persuasive', "
            "'we decline to follow X'). "
            "Do NOT include a case because you infer weakened authority without explicit treatment in the text. "
            "Strong criticism or mere limiting without clear loss of authority does NOT qualify.\n\n"
            "CITATION ACCURACY — critical:\n"
            "- Extract citation strings VERBATIM from the opinion text. Never reconstruct from memory.\n"
            "- Never guess or infer page numbers, volume numbers, or years.\n\n"
            "EXHAUSTIVENESS:\n"
            "- When the Court groups cases together with identical treatment, include ALL named cases.\n"
            "- Scan the entire opinion — do not stop after finding the first overruled case."
        )

    def build_user_prompt(self, ctx: PromptContext) -> str:
        return (
            f"This is the opinion of the court of {ctx.case_name}, {ctx.citation}:\n\n"
            f"{ctx.opinion_text}\n\n"
            "Which Supreme Court cases does this opinion overrule or declare no longer good law? "
            "Return them in 'directly_overruled' as defined in your instructions and return JSON only."
        )

    @property
    def finetune_system_prompt(self) -> str | None:
        return (
            "SCOTUS precedent expert. One JSON object only: "
            '"reasoning" (string), "directly_overruled" ([]). '
            "List cases this opinion explicitly displaces or declares no longer controlling. "
            "Citations verbatim from the opinion. Always use modern and official citation format for the case citations."
        )

    def build_finetune_user_prompt(self, ctx: PromptContext) -> str | None:
        return (
            f"{ctx.case_name}, {ctx.citation}\n\n{ctx.opinion_text}\n\n"
            "JSON only: reasoning, directly_overruled per system rules."
        )
