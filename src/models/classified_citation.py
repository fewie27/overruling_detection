"""Citation paired with a treatment-class label (benchmark vocabulary)."""

from __future__ import annotations

from dataclasses import dataclass


def normalize_class_label(label: str) -> str:
    """Normalize a class label: lowercase, snake_case/hyphens -> spaces, collapse whitespace."""
    s = (label or "").strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(s.split())


# Default benchmark labels treated as overruling-positive when not configured otherwise.
DEFAULT_STRICT_CLASSES: frozenset[str] = frozenset({"overruled", "partially overruled"})


@dataclass(frozen=True)
class ClassifiedCitation:
    """A citation string paired with its treatment-class label (benchmark vocabulary, normalized)."""

    cite: str
    cls: str

    @staticmethod
    def make(cite: str, cls: str) -> "ClassifiedCitation":
        return ClassifiedCitation(cite=(cite or "").strip(), cls=normalize_class_label(cls))

    def to_dict(self) -> dict[str, str]:
        return {"cite": self.cite, "class": self.cls}

    @staticmethod
    def from_dict(d: dict) -> "ClassifiedCitation":
        cite = str(d.get("cite") or d.get("citation") or d.get("case_citation") or "").strip()
        cls = str(d.get("class") or d.get("classification") or "")
        return ClassifiedCitation.make(cite, cls)


def parse_strict_classes_csv(text: str | None) -> set[str]:
    """Parse a CLI-style comma-separated class list; returns the default set when input is empty/None."""
    if text is None or not text.strip():
        return set(DEFAULT_STRICT_CLASSES)
    parts = [normalize_class_label(p) for p in text.split(",")]
    return {p for p in parts if p}


def dedupe_classified(items: list[ClassifiedCitation]) -> list[ClassifiedCitation]:
    """Stable de-duplication by (cite, cls)."""
    seen: set[tuple[str, str]] = set()
    out: list[ClassifiedCitation] = []
    for c in items:
        key = (c.cite, c.cls)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
