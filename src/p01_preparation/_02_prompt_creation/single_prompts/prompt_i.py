from __future__ import annotations

from src.models import PromptContext
from src.p01_preparation._02_prompt_creation.base import PromptBuilder


_LABEL_LIST = """
## Citation Treatment Labels

Assign exactly **one** of the following labels to each cited authority you include in the JSON output, based on what the citing court does to or says about that cited authority.

---

**certiorari_granted**
A higher court formally accepts the cited case for review on the merits. Look for "certiorari granted," "cert. granted," or "review granted."

**criticized**
The citing court expresses substantive disagreement with the reasoning or result of the cited authority but stops short of formally overruling it. Look for "we question," "difficult to reconcile," "poorly reasoned," or "we have doubts about."

**disapproved**
The citing court categorically rejects the legal position or reasoning of the cited authority without formally overruling it. Look for "we disapprove of," "should not be followed," "we decline to follow," or "was wrongly decided."
Use disapproved — not overruled — when (a) the explicit word "overrule" is absent, or (b) the cited case is from a court outside the citing court's supervisory jurisdiction (e.g., a different circuit, a different sovereign, or the Supreme Court cited by a lower court).
DECISION RULE: disapproved = categorical rejection without formal authority or formula. If the court says it will not follow X, that is disapproved, not criticized. If the court uses the verb "reject" toward the cited authority's holding or premise, prefer *rejected*.

**disavowed**
The court distances itself from its OWN prior argumentation, statements, or dicta — not from another court's decision. Look for "we disavow our earlier statement that," "we retract," or "we no longer endorse the reasoning we employed in X."
DECISION RULE: disavowed is self-referential — the citing court retracts something it said before. If the court is instead reporting that another court has already overruled the cited authority, use overruling_recognized.

**overruled**
The citing court declares a prior precedent no longer valid as binding authority — stripping
it of precedential force, not merely disagreeing with it or declining to follow it in this
case. The court is ending the precedent's legal life, not just criticizing or limiting it.
Look for language such as: "we overrule X," "we overturn X," "X is no longer good law,"
"we abandon the rule of X," "X cannot stand," "the underpinnings of X have been so eroded
as to render it no longer controlling," "to the extent X held Y, that holding is overruled,"
or any equivalent formula by which the court declares X invalid as precedent going forward. Sometimes the
language is not absolutely explicit, but a professional citator would still see it as an overruling. This is your judgment call.

**overruling_recognized**
The citing court does not itself overrule the cited authority but acknowledges that a prior court has already done so. Look for "overruled by [case]," "has been overruled,", "recognized as overruled" or equivalent language.
DECISION RULE: the citing court is reporting an external event, not performing one. If the court is retracting its own prior words, use disavowed instead.

**rejected**
The citing court expressly refuses to accept the cited authority's holding, premise, or reasoning as governing, without formally overruling it. Look for "we reject," "we reject the holding of," "we reject the premise of," "squarely rejected," or "rejected as unsound."
DECISION RULE: rejected = explicit refusal of the authority's force or premise via "reject"-family language. Prefer *repudiated* when the court denounces an associated doctrine rather than the case itself; prefer *disapproved* when the court declines to follow without using "reject."

**repudiated**
The citing court explicitly rejects or denounces a legal doctrine associated with the cited authority, typically with stronger force than *disapproved*. Look for "repudiated," "we repudiate the doctrine of," or "firmly repudiated."

**reversed**
The citing court overturns the outcome of the cited decision, reaching the opposite result. Look for "reversed," "we reverse," or "the judgment is reversed."

**subsequent_determination**
The citation refers to a later decision in the *same* litigation — the same parties at a subsequent procedural stage. Look for "on remand," "after remand," or a later docket entry under the same case name.

**superseded**
The cited authority has been entirely replaced — either by a later decision that comprehensively supplants it, or by legislative enactment — and is no longer in force. Look for "superseded by," "replaced by," or "no longer in force."

**vacated**
The citing court nullifies the cited decision, rendering it void and without legal effect. Distinct from *reversed* (which produces an opposite result) and *overruled* (which targets precedential value). Look for "vacated," "the judgment is vacated," or "we vacate."

**withdrawn**
The originating court itself retracts the cited decision or a portion of it — a self-correction, not an action by a higher court. Look for "withdrawn," "the opinion is withdrawn," or "superseded by substituted opinion."
""".strip()


_SYSTEM_PROMPT = f"""You are a legal citation analyst specializing in U.S. case law. Your task is to examine a judicial opinion and identify citations to external authorities for which this opinion assigns a recognizable *treatment* from the taxonomy below — that is, what the citing court does to or says about the cited source.

## Your Task

1. **Identify** each citation in the opinion text that qualifies for the output (see omission rule below).
2. **Extract** the relevant passage(s) surrounding that citation that reveal how the citing court treats the cited authority.
3. **Assign exactly one treatment label** to each citation you output, chosen from the taxonomy. Use snake_case spelling for JSON (e.g., `certiorari_granted`, `overruling_recognized`).
4. Base your classification **solely on the text of the opinion**. Do not infer treatment from external knowledge about the cited case.

## Assignment Rules

- **One label per citation.** If multiple labels could apply, choose the definition most precisely satisfied by the quoted language. If ambiguity remains, mention it briefly in `reasoning`.
- **Omit neutral citations.** If the cited authority is merely referenced for background, comparison, or general support with no negative, qualifying, or procedurally significant treatment captured by any label below, **do not** include it in `overruled_cases`. If nothing qualifies, return `"overruled_cases": []`.

## Label Taxonomy

{_LABEL_LIST}

## Output Format

Return valid JSON **only**. No preamble, no markdown fences, no text outside the JSON.

The attribute `case` MUST always use modern citation format (e.g., xx U.S. yy (year)), not historical reporters wherever a standard form is discernible.

Double-check volume and page numbers are correct where you extract them from the opinion.

The output **must use this exact schema and key names** (same as the overruling-analysis prompt used elsewhere in this project):

{{
  "overruling_decision": "string",
  "legal_question": "one sentence: concise statement of what this opinion is chiefly resolving (for context only)",
  "overruled_cases": [
    {{
      "case": "Party v. Party, Volume U.S. Page (Year)",
      "classification": "single_snake_case_label_from_taxonomy",
      "classification_note": "if you had to resolve ambiguity between labels, briefly explain; otherwise an empty string",
      "prior_holding": "one sentence: the excerpt or gist of THIS opinion showing how the cited authority is treated",
      "this_opinion_holding": "one sentence tying that treatment to the outcome or rationale, based only on the provided text",
      "reasoning": "1–3 sentences explaining why this label fits",
      "quotes": [
        "exact quote from THIS opinion under 25 words",
        "second quote if needed"
      ]
    }}
  ]
}}
""".strip()


class PromptI(PromptBuilder):
    """Full-opinion citation treatment (taxonomy labels); user prompt matches prompt_g; JSON schema matches prompt_g for parser reuse."""

    @property
    def label(self) -> str:
        return "prompt_i"

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
            f"Extract the overruling cases from the opinion text as described in the system prompt. Return JSON only."
            f"Overruling decision: {decision}\n\n"
            f"Opinion text:\n{ctx.opinion_text}"
        )
