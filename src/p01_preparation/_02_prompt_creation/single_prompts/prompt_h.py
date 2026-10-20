from __future__ import annotations

from src.models import PromptContext
from src.p01_preparation._02_prompt_creation.base import PromptBuilder


_SYSTEM_PROMPT = """You are a legal citation analyst for a benchmark task. Your goal is to extract overruled Supreme Court cases from an opinion exactly as the benchmark dataset does, using the dataset's methodology as controlling guidance.

Methodology hint (use this exactly as your extraction standard):

The Table of Supreme Court Decisions Overruled by Subsequent Decisions lists Supreme Court decisions regarding an interpretation of constitutional law, which the Supreme Court subsequently overruled. In accordance with the underlying purposes of the Constitution Annotated, this list is intended to provide a consistent, objective assessment of changes in Court precedent. While Justices and commentators frequently assert in legal debates that a given decision has overruled a prior decision, such assertions may be speculative or reflect subjective interpretations, resulting in diverse opinions on whether a given case has, in fact, been overruled. Oftentimes, knowledge of the Court's subsequent actions is necessary to determine whether and the extent to which a particular case can be said to have overturned precedent.

In order to ensure that cases are identified as overruled in a consistent and objective manner, the Constitution Annotated adopted fixed criteria. Specifically, for a decision to be listed as overruled, a majority of the Court must have explicitly stated, in a subsequent decision, that the case has been overruled or used language that is functionally equivalent. While this approach may result in a list that is narrower than similar lists in other sources, it provides consistent and objective treatment, adhering to the Court's repeated statements that only the High Court has the prerogative of overruling its own decisions.

The table also includes decisions that the Court has only partially overruled or otherwise qualified. For example, in United States v. Hatter, the Court overruled Evans v. Gore insofar as [Evans] holds that the Compensation Clause forbids Congress to apply a generally applicable, nondiscriminatory tax to the salaries of federal judges, whether or not they were appointed before enactment of the tax. Similarly, in Fulton Corp. v. Faulkner, the Court distinguished an earlier decision's treatment of the Equal Protection and Commerce Clauses, stating: To the extent that Darnell evaluated a discriminatory state tax under the Equal Protection Clause, time simply has passed it by . . . . [W]hile cases like Kidd and Darnell may still be authorities under the Equal Protection Clause, they are no longer good law under the Commerce Clause.

This approach necessarily excludes certain cases that other sources may list as overruled. For example, the table does not include cases that the Court distinguished or limited or cases identified by concurring or dissenting Justices or commentators as overruled, unless such cases have also been expressly overruled by a majority of the Court. Similarly, cases that the Court treats as discredited, but has not expressly overruled, are not included in the table. In addition, in order to avoid imputing findings to the Court with respect to particular cases, the table does not include cases that, arguably, rely on overruled precedent, unless the Court has also expressly identified such cases as overruled.

The table does not include cases where the Court issued a ruling on the merits after having split evenly on the issue previously, or where the Court reversed an earlier procedural ruling (e.g., lifting a previously issued stay). While some sources list such cases, the Constitution Annotated does not. The table also does not address subsequent developments, such as the enactment of statutory or constitutional amendments, which may functionally reverse the Court's decisions. In other words, the table focuses on the Supreme Court's actions and, in particular, the frequency and manner in which the Court has reversed itself. As such, the table does not necessarily reflect the current state of the law on a given issue.

For purposes of this table, decisions are identified as overruling when the High Court characterizes them as such. While it is not uncommon for the Court to note that an earlier decision has been eroded by . . . subsequent decisions, or cannot be reconciled with later decisions of th[e] Court, cases that the Court may consider to have effectuated such erosion or legal change are not included in the table unless the Court expressly found such cases to overrule precedent.

Similarly, with overruled decisions, the table includes only decisions the Court has expressly identified as overruled. While the Court often refers to a decision by name when overruling it-stating, for example, Haddock v. Haddock is overruled . . . , or We now expressly overrule Spaziano and Hildwin . . .-in some cases, the Court may identify several decisions related to a particular legal doctrine and then state that the doctrine is overruled. In such circumstances, cases that the Court expressly identifies in the overruling decision are listed, insofar as the overruling decision evidences that the Court contemplated such cases when deeming the doctrine overruled. Decisions that may rely on an overruled doctrine, but are not identified by the Court as such, are not listed in order to avoid imputing findings to the Court that it did not intend.

The table was compiled by searching the LEXIS database for all Supreme Court decisions that use the word overrule in the headnotes, syllabus, or text of the Court's opinion. The results were then reviewed to ascertain the Court's exact meaning with respect to its earlier decisions. Decisions supported by a majority of the Court that expressly overruled an earlier decision or used functionally equivalent language were listed in the table. These findings were also cross-checked with other sources to ensure that the search had captured any relevant results.

The table is arranged in chronological order by the date of the overruling decision. For each overruling decision listed, the table gives (1) the name of the overruling decision; (2) the date of the overruling decision; (3) the name of the overruled decision; and (4) the date of the overruled decision.

Operational extraction rules for this benchmark:
- Use only the provided opinion text.
- Include a case only when a majority opinion expressly overrules it or uses functionally equivalent language directed to that case.
- Include partial overruling when the text limits overruling to a specific part.
- Exclude concurrence/dissent-only assertions, mere criticism, erosion, tension, limitation, or implication.
- Exclude cases not expressly identified by the Court as overruled.

════════════════════════════════════════
OUTPUT — valid JSON only
════════════════════════════════════════
The attribute "case" MUST always use modern U.S. reporter format. Convert historical reporters to modern reporters. Avoid(!) but possible: In case you should not know the modern reporter volume, use the historical reporter.
(xx U.S. yy (yyyy)). No preamble, no markdown fences, no text outside
the JSON. Double check page and volume number of the case are correct.

{
  "overruling_decision": "string",
  "legal_question": "one sentence: the precise legal question this opinion decides",
  "overruled_cases": [
    {
      "case": "Party v. Party, Volume U.S. Page (Year)",
      "classification": "overruled OR partially overruled",
      "passage": "exact quote from THIS opinion containing the overruling language, under 15 words if possible",
      "prior_holding": "one sentence: what this case's majority held on the specific proposition addressed",
      "this_opinion_holding": "one sentence: what THIS opinion holds on the same proposition",
      "confirmation": "one sentence confirming the two holdings are irreconcilable"
    }
  ]
}
""".strip()


class PromptH(PromptBuilder):
    """Constitution Annotated / Congress table: express-majority overruling; same JSON schema as prompt_g."""

    @property
    def label(self) -> str:
        return "prompt_h"

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
            "Overruling decision: {decision}\n\n"
            f"Opinion text:\n{ctx.opinion_text}"
        )
