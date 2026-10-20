"""
Single source of truth for citation-treatment classification labels (chunk / classification JSON).

Only group 1 (negative treatment) counts as “overruled” for benchmark evaluation against the dataset.
"""

from __future__ import annotations

from typing import Final

from src.models import normalize_class_label

# All valid `classification` string values (snake_case), fixed order: group1, group2, group3, group4.
CITATION_TREATMENT_LABELS: Final[tuple[str, ...]] = (
    # ── 1A: This court overrules ─────────────────────────────────────────────
    # The opinion itself performs the overruling act.
    # These are the ONLY labels that count as "overruled" for benchmark evaluation.
    # (Strict benchmark: direct + explicit_partial only. Soft: add implicit_overruling.)
    "direct_overruling",
    "explicit_partial_overruling",
    "implicit_overruling",        # contested; exclude from strict benchmark
    "implicit_doctrine_displacement",  # reframing renders prior rule inoperative; counts as overruled
    "expressly_disavowed",        # repudiation without formal overruling language

    # ── 1B: This court weakens (NOT overruling) ──────────────────────────────
    # This opinion constrains the precedent, but the precedent survives.
    "distinguished_on_facts",
    "limited_to_facts",
    "narrowed_doctrinally",
    "questioned_criticized",

    # ── 1C: External invalidity observed (NOT overruling by this court) ──────
    # This opinion merely notes that an external act (statute, constitutional change)
    # has displaced the cited case. The court reports; it does not act.
    "superseded_by_external_authority",

    # ── 2: Neutral / contextual ──────────────────────────────────────────────
    "background_reference",
    "procedural_history",
    "string_citation",

    # ── 3: Positive treatment ────────────────────────────────────────────────
    "followed_applied",
    "extended",
    "reaffirmed",
    "relied_on_as_primary_authority",
)

assert len(CITATION_TREATMENT_LABELS) == 17

LABEL_DESCRIPTIONS = """
You are classifying how THIS opinion — not any other court, not legal history generally —
treats the cited case. Ask: does the text of this excerpt perform a legal act toward
the cited case, or merely describe one?

Labels (use the earliest applicable when ambiguous):

1A: This court overrules
Use only when this opinion's own language performs the overruling.

"direct_overruling": This opinion explicitly overrules the cited case in whole.
"explicit_partial_overruling": This opinion explicitly overrules the cited case on a specific issue, holding, or in part only.
"implicit_overruling": This opinion's holding is irreconcilable with the cited case, effectively overruling it without stating so. Use sparingly—only when the logical conflict is unambiguous.
"implicit_doctrine_displacement": This opinion's holding reframes the controlling legal question such that the cited case's rule is rendered inoperative in its original domain, effectively overruling it without confronting it directly. Distinct from implicit_overruling in that the conflict arises from reframing rather than a direct logical contradiction between holdings.
"expressly_disavowed": This opinion repudiates or disowns the cited case's reasoning without using formal overruling language.

1B: This court weakens (not overruling)
The cited case survives, but this opinion constrains its reach.

"distinguished_on_facts": This opinion declines to apply the cited case because the facts are materially different. The cited case's rule is left intact.
"limited_to_facts": This opinion confines the cited case to its own specific facts, refusing to extend the rule further.
"narrowed_doctrinally": This opinion reinterprets or shrinks the cited case's legal rule without overruling it.
"questioned_criticized": This opinion expresses doubt or criticism about the cited case's reasoning or continued soundness, but does not overrule it.

1C: External invalidity observed (not overruling by this court)
This opinion notes that something outside this court has displaced the cited case. The court is reporting a fact, not performing a legal act.

"superseded_by_external_authority": This opinion notes the cited case has been displaced by a subsequent statute, constitutional amendment, or other external legal development. This court did not cause the displacement.

2: Neutral / contextual
No evaluative stance. The cited case is mentioned but not treated.

"background_reference": Cited for general legal context, historical background, or as illustrative authority, with no evaluative treatment.
"procedural_history": Cited as part of the procedural posture or prior litigation history of the case at hand.
"string_citation": Appears in a list of authorities or footnote with minimal or no discussion.

3: Positive treatment
This opinion approves, follows, or extends the cited case.

"followed_applied": This opinion applies the cited case's rule to reach its outcome.
"extended": This opinion applies the cited case's rule to a new context or set of facts beyond its original scope.
"reaffirmed": This opinion explicitly endorses the cited case as correctly decided.
"relied_on_as_primary_authority": The cited case is the central, load-bearing precedent for this opinion's core legal reasoning.
"""

_ALL_LABELS: Final[frozenset[str]] = frozenset(CITATION_TREATMENT_LABELS)

# Map raw prompt labels (snake_case) -> benchmark-vocabulary class (normalized).
# Labels not listed here pass through (after normalization) so generic benchmarks can
# define any vocabulary; only the negative-treatment group is collapsed onto the
# canonical "overruled" / "partially overruled" labels used by overruling benchmarks.
_LABEL_TO_BENCHMARK_CLASS: Final[dict[str, str]] = {
    "direct_overruling": "overruled",
    "implicit_overruling": "overruled",
    "implicit_doctrine_displacement": "overruled",
    "expressly_disavowed": "overruled",
    "explicit_partial_overruling": "partially overruled",
}


def format_labels_for_prompt() -> str:
    """Return the full label description block for system prompts."""
    return LABEL_DESCRIPTIONS.strip()


def is_valid_citation_treatment_label(label: str) -> bool:
    return label.strip().lower() in _ALL_LABELS


def to_benchmark_class(label: str) -> str:
    """Map a raw prompt label to a benchmark-vocabulary class string (normalized)."""
    raw = (label or "").strip().lower()
    canonical = _LABEL_TO_BENCHMARK_CLASS.get(raw)
    return normalize_class_label(canonical if canonical is not None else label)
