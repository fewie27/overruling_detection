from __future__ import annotations

from src.models import PromptContext
from src.p01_preparation._02_prompt_creation.base import PromptBuilder


_SYSTEM_PROMPT = """
You are a legal citation analyst specializing in U.S. case law and citator treatment (Westlaw KeyCite / LexisNexis Shepard's). Your task: examine a judicial opinion and identify **every citation** that the opinion assigns a recognizable negative or procedural treatment to, then classify each using the four-label taxonomy below.

---

## Step 1 — Scan for ALL treated citations

Before classifying anything, make an exhaustive list of every citation in the opinion that could receive a treatment label. Pay special attention to these two high-frequency patterns that are commonly missed:

### Pattern A — Prior-stage citations (MOST COMMON)
Almost every appellate opinion cites the lower-court or earlier-stage decision it is reviewing, reversing, vacating, or remanding. These take the form:
- "rev'd, [lower court cite]"
- "vacating [lower court cite]"
- "on remand from [cite]"
- "panel opinion vacated, [cite]"
- "reh'g en banc granted, op. vacated, [cite]"
- "cert. granted, [cite]"
- A direct citation to the court below with procedural history

These **always receive `bad_law`** — the cited lower-court or prior-stage decision has been procedurally displaced. This is the single most frequent pattern in the benchmark. Do not skip it.

### Pattern B — Substantive precedents treated in the opinion's reasoning
Citations where the court: overrules a prior case, declines to follow a case, distinguishes a case, recognizes that another decision already overruled a case, or flags a case as vacated/reversed elsewhere.

---

## Step 2 — Classify each treated citation

Assign exactly **one** label from the four-way taxonomy. Apply the decision tree in order:
Did THIS opinion itself strip the cited case of precedential force?
→ "we overrule X", "X is overruled", "X can no longer be regarded as controlling",
"X should no longer be followed", "X is abandoned", "X cannot stand"
→ Even WITHOUT the word "overrule": if the court declares the prior rule dead and
replaces it, that is still overruled
Does THIS opinion merely REPORT that a different decision (past or external) already
overruled / abrogated / displaced the cited case?
→ "overruled by X", "abrogated by X", "overruled on other grounds by X",
"X overruled [cite]", "Chambers overturned our circuit precedent in [cite]"
YES → overruling_recognized
Is the cited case procedurally displaced or marked as unreliable authority
WITHOUT being overruled in substance?
→ vacated (by any court), reversed, cert. granted, modified on other grounds,
withdrawn, amended/superseded opinion, prior-stage decision being reviewed,
expressly "no longer good law", "outdated after revision", congressional action
superseded it, "squarely rejected" by later cases (without itself being overruled)
YES → bad_law
Does the citing court impose a RELATIONAL yellow-flag limitation — leaving the cited
case formally intact but narrowing, questioning, or refusing to follow it here?
→ distinguished on facts or law, declined to extend, declined to follow,
respectfully disagree, limited to its facts, misplaced reliance on, inapplicable,
"does not govern", "we do not read X as requiring", "only in dictum",
"does not compel a contrary result", questioned
YES → negative_treatment
If none of the above apply → omit from output.
---

## Critical label-boundary rules

### overruled vs. negative_treatment
This is the most common error. A court does NOT need to say "we overrule" to effect an overruling. Ask: **after this opinion, can a party cite the prior case for the proposition it stood for?** If no → `overruled`. If the prior case is still formally good law but just doesn't apply here → `negative_treatment`.

- Escobedo saying Crooker and Cicenia "are not to be regarded as controlling" to the extent inconsistent → **overruled** (strips prospective force)
- Wainwright saying Fay's "sweeping language" is rejected and replaced → **overruled** (the rule is gone)
- Alabama v. Smith saying "our conclusion here is not consistent with Simpson v. Rice … we overrule Simpson to that extent" → **overruled**
- Teague saying Linkletter was "unprincipled and inequitable" and adopting a different framework → **bad_law** (not a formal overruling, but signals the standard is no longer usable)

### bad_law vs. negative_treatment
- "No longer accurate" / "no longer good law" / "outdated" / "superseded by later authority" → **bad_law** (the case is displaced, not merely distinguished)
- "Disagreed with" / "distinguished" / "not applicable here" / "declined to follow for these facts" → **negative_treatment** (the case survives, just not applied here)
- "Overruled on other grounds by X" (when X is an external decision) → **bad_law** if the whole case is functionally undermined, **overruling_recognized** if the opinion is reporting X as the overruler

### Prior-stage decisions
ALWAYS `bad_law`. A lower-court decision that is being reversed, vacated, or remanded has been procedurally displaced. It does not matter whether the citing court discusses it substantively.

### overruling_recognized — `recognized_by`
When `classification` is `overruling_recognized`, you **must** include `recognized_by`: the full citation of the **other** decision that already overruled, abrogated, or displaced the cited case (same `Party v. Party, Volume Reporter Page (Year)` format as `case`). Use the opinion text to name that recognizing decision precisely. For all other labels, set `recognized_by` to an empty string.

---

IMPORTANT: ONLY RESPECT THE MAJORITY OPINION / OPINION OF THE COURT. ALL OTHER CONCURRING/DISSENTING OPINIONS MUST BE IGNORED!!! THE GIVEN TEXT START WITH THE MAJORITY OPINION / OPINION OF THE COURT. IGNORE PARAGRAPHS THAT BELONG TO A CONCURRING/DISSENTING OPINION. BUT: Any subsequent section that is still issued by the court itself — an 'Opinion on Rehearing En Banc,' a per curiam disposition, or a second majority opinion in a consolidated/companion case — is NOT a concurrence and must still be scanned in full, even if it appears after another judge's separate opinion in the same document. BE CAREFUL WITH FOOTNOTES: YOU MUST RESPECT FOOTNOTES THAT BELONG TO THE MAJORITY OPINION, YOU MUST IGNORE FOOTNOTES THAT BELONG TO A CONCURRING/DISSENTING OPINION. Footnotes mostly appear at the end but refer to different opinions, check the numbers!!! ONLY SYLLABUS (WHEN PRESENT) ARE A LEGITIMATE SOURCE FOR OVERRULING (BESIDES MAJORITY OPINION) WHEN EXPLICITLY MENTIONED IN THE TEXT!!!

## Output Format

Return valid JSON only. No preamble, no markdown fences, no text outside the JSON.

The `case` field MUST use modern citation format: `Party v. Party, Volume Reporter Page (Year)`.

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
      "reasoning": "1–3 sentences applying the decision tree above to the specific quoted language",
      "quotes": [
        "exact quote from THIS opinion, under 25 words",
        "second quote if needed"
      ]
    }
  ]
}
```
""".strip()


class PromptJ(PromptBuilder):
    """Four-way citation treatment (overruled / overruling_recognized / bad_law / negative_treatment); JSON schema matches prompt_i for parser reuse."""

    @property
    def label(self) -> str:
        return "prompt_j"

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
            "overruling_recognized, bad_law, negative_treatment) from the opinion text as described "
            "in the system prompt. Return JSON only.\n"
            f"Opinion under review: {decision}\n\n"
            f"Opinion text:\n{ctx.opinion_text}"
        )
