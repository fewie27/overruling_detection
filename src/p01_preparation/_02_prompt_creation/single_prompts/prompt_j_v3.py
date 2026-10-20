from __future__ import annotations

from src.models import PromptContext
from src.p01_preparation._02_prompt_creation.base import PromptBuilder


_SYSTEM_PROMPT = """
You are a legal citation analyst. From the judicial opinion text, identify every citation that receives a negative or procedural treatment and classify each as exactly one of: overruled, overruling_recognized, bad_law, negative_treatment. When classification is overruling_recognized, set recognized_by to the decision that already overruled or displaced the cited case; otherwise leave recognized_by empty.

Return valid JSON only. No preamble, no markdown fences, no text outside the JSON. The `case` field MUST use modern citation format: `Party v. Party, Volume Reporter Page (Year)`.

```json
{
  "overruling_decision": "Full case name and citation of the opinion being analyzed",
  "legal_question": "One sentence: the central legal question this opinion resolves",
  "overruled_cases": [
    {
      "case": "Party v. Party, Volume Reporter Page (Year)",
      "classification": "overruled | overruling_recognized | bad_law | negative_treatment",
      "recognized_by": "Party v. Party, Volume Reporter Page (Year) — required when classification is overruling_recognized; otherwise empty string",
      "classification_note": "If you resolved ambiguity between two labels, briefly state which two and why you chose this one. Otherwise empty string.",
      "prior_holding": "One sentence: what the cited case stood for, or its procedural fate, as described in THIS opinion",
      "this_opinion_holding": "One sentence: how this treatment affects the outcome or reasoning of this opinion",
      "reasoning": "1–3 sentences explaining the classification",
      "quotes": [
        "exact quote from THIS opinion, under 25 words",
        "second quote if needed"
      ]
    }
  ]
}
```
""".strip()


class PromptJV3(PromptBuilder):
    """Prompt J with a minimal system prompt (task + classes + output schema); model chooses its own method."""

    @property
    def label(self) -> str:
        return "prompt_j_v3"

    @property
    def response_parser_id(self) -> str:
        return "P4_detailed_parser"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def build_user_prompt(self, ctx: PromptContext) -> str:
        cite = (ctx.citation or "").strip()
        name = (ctx.case_name or "").strip()
        decision = f"{name}, {cite}".strip(", ") if cite else name
        return (
            "Extract citations that receive one of the four treatment labels (overruled, "
            "overruling_recognized, bad_law, negative_treatment) from the opinion text. "
            "Return JSON only.\n"
            f"Opinion under review: {decision}\n\n"
            f"Opinion text:\n{ctx.opinion_text}"
        )
