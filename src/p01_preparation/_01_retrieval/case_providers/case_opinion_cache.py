from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.models import CaseDocument

_CACHE_FORMAT_VERSION = 3

OPINION_SCOPE_MAJORITY_ONLY = "majority_only"
OPINION_SCOPE_FULL_CASEBODY = "full_casebody"

# Opinion text stored explicitly; metadata holds raw_data keys except casebody (no duplicate full JSON).


def default_case_cache_dir() -> str:
    return os.getenv("CASE_OPINION_CACHE_DIR", str(Path("output") / "case_opinion_cache"))


def _normalize_citation_key(citation: str) -> str:
    return " ".join((citation or "").strip().lower().split())


def _path_for_citation(citation: str, *, cache_dir: Path) -> Path:
    key = _normalize_citation_key(citation)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _metadata_without_casebody(raw_data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in raw_data.items() if k != "casebody"}


def _stored_provider_label(stored: dict[str, Any]) -> str | None:
    explicit = stored.get("retrieval_provider_label")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    resolved = stored.get("resolved_via")
    if isinstance(resolved, str) and resolved.strip():
        return resolved.strip()
    return None


def _infer_opinion_scope(stored: dict[str, Any]) -> str:
    explicit = stored.get("opinion_scope")
    if explicit in (OPINION_SCOPE_MAJORITY_ONLY, OPINION_SCOPE_FULL_CASEBODY):
        return str(explicit)
    # Legacy caches only stored majority text — never treat as full casebody.
    resolved = str(stored.get("resolved_via") or "").strip().lower()
    if resolved == "case_law" and isinstance(stored.get("opinions"), list) and stored["opinions"]:
        return OPINION_SCOPE_FULL_CASEBODY
    return OPINION_SCOPE_MAJORITY_ONLY


def _opinions_from_stored(stored: dict[str, Any]) -> list[dict[str, Any]]:
    raw_opinions = stored.get("opinions")
    if isinstance(raw_opinions, list) and raw_opinions:
        out: list[dict[str, Any]] = []
        for item in raw_opinions:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                out.append(
                    {
                        "type": item.get("type"),
                        "text": item["text"],
                    }
                )
            elif isinstance(item, str) and item.strip():
                out.append({"text": item})
        if out:
            return out
    text = stored.get("opinion_text")
    if isinstance(text, str) and text.strip():
        return [{"text": text}]
    return []


def _casebody_opinions_from_doc(doc: CaseDocument) -> list[dict[str, Any]]:
    opinions = doc.raw_data.get("casebody", {}).get("opinions", [])
    if not isinstance(opinions, list):
        return []
    out: list[dict[str, Any]] = []
    for item in opinions:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        out.append({"type": item.get("type"), "text": text})
    return out


def load_case_from_cache(
    citation: str,
    *,
    cache_dir: str | None = None,
    require_full_casebody: bool = False,
) -> CaseDocument | None:
    base = Path(cache_dir) if cache_dir is not None else Path(default_case_cache_dir())
    path = _path_for_citation(citation, cache_dir=base)
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            stored = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    try:
        scope = _infer_opinion_scope(stored)
        if require_full_casebody and scope != OPINION_SCOPE_FULL_CASEBODY:
            # Majority-only / legacy entries must not be treated as full opinion text.
            return None

        if "opinion_text" in stored or "opinions" in stored:
            case_name = str(stored.get("case_name") or stored.get("citation") or "").strip()
            opinions = _opinions_from_stored(stored)
            raw_data: dict[str, Any] = {
                "name": case_name or str(stored["citation"]),
                "name_abbreviation": case_name or str(stored["citation"]),
                "casebody": {"opinions": opinions},
            }
            for k, v in (stored.get("metadata") or {}).items():
                if k != "casebody":
                    raw_data[k] = v
            return CaseDocument(
                citation=str(stored["citation"]),
                source_url=str(stored["source_url"]),
                raw_data=raw_data,
                retrieval_provider_label=_stored_provider_label(stored),
                opinion_scope=scope,
            )
        return CaseDocument(
            citation=str(stored["citation"]),
            source_url=str(stored["source_url"]),
            raw_data=dict(stored["raw_data"]),
            retrieval_provider_label=_stored_provider_label(stored),
            opinion_scope=scope,
        )
    except (KeyError, TypeError, ValueError):
        return None


def save_case_to_cache(
    doc: CaseDocument,
    *,
    resolved_via: str,
    cache_key_citation: str | None = None,
    cache_dir: str | None = None,
    opinion_scope: str | None = None,
) -> None:
    """Persist LLM-facing opinion text(s) plus small metadata (no full case.law JSON)."""
    base = Path(cache_dir) if cache_dir is not None else Path(default_case_cache_dir())
    key = cache_key_citation if cache_key_citation is not None else doc.citation
    path = _path_for_citation(key, cache_dir=base)
    path.parent.mkdir(parents=True, exist_ok=True)

    opinions = _casebody_opinions_from_doc(doc)
    majority = doc.majority_opinion
    if not opinions and majority.strip():
        opinions = [{"text": majority}]

    if opinion_scope in (OPINION_SCOPE_MAJORITY_ONLY, OPINION_SCOPE_FULL_CASEBODY):
        scope = opinion_scope
    elif resolved_via == "court_listener":
        scope = OPINION_SCOPE_MAJORITY_ONLY
    elif resolved_via == "case_law":
        scope = OPINION_SCOPE_FULL_CASEBODY
    else:
        scope = doc.opinion_scope or OPINION_SCOPE_MAJORITY_ONLY

    case_name = doc.case_name
    payload: dict[str, Any] = {
        "cache_format_version": _CACHE_FORMAT_VERSION,
        "citation": doc.citation,
        "source_url": doc.source_url,
        "resolved_via": resolved_via,
        "case_name": case_name,
        "opinion_text": majority,
        "opinions": opinions,
        "opinion_scope": scope,
        "metadata": _metadata_without_casebody(doc.raw_data),
    }
    if doc.retrieval_provider_label:
        payload["retrieval_provider_label"] = doc.retrieval_provider_label
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
