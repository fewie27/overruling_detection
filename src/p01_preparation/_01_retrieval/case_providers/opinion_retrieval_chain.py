from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from src.p01_preparation._01_retrieval.case_providers.base import CaseProvider
from src.p01_preparation._01_retrieval.case_providers.case_law import CaseLawStaticProvider
from src.p01_preparation._01_retrieval.case_providers.case_opinion_cache import (
    OPINION_SCOPE_FULL_CASEBODY,
    OPINION_SCOPE_MAJORITY_ONLY,
    default_case_cache_dir,
    load_case_from_cache,
    save_case_to_cache,
)
from src.p01_preparation._01_retrieval.case_providers.court_listener import CourtListenerProvider
from src.models import CaseDocument

# Fixed resolution order; benchmark JSON text is a separate fallback in the pipeline runner.
OPINION_RETRIEVAL_ORDER: tuple[str, ...] = (
    "local",
    "court_listener",
    "case_law",
    "benchmark_opinion",
)


def _debug_chain_enabled() -> bool:
    return os.getenv("DEBUG_RETRIEVAL", "").strip().lower() in ("1", "true", "yes", "on")


def _debug_chain(msg: str) -> None:
    if _debug_chain_enabled():
        print(f"[DEBUG chain] {msg}", flush=True)


def opinion_retrieval_metadata() -> dict[str, Any]:
    return {
        "opinion_retrieval_order": list(OPINION_RETRIEVAL_ORDER),
        "case_opinion_cache_dir": default_case_cache_dir(),
    }


class OpinionRetrievalChainProvider(CaseProvider):
    """
    Tries, in order: on-disk cache, CourtListener (if API token is set), then static.case.law JSON.

    Successful network fetches are written to the local cache. Per-row result source is stored
    on CaseDocument.retrieval_provider_label (local, court_listener, or case_law).

    When ``require_full_casebody`` is True, majority-only cache hits are skipped; CourtListener
    is asked for all cluster sub-opinions, then CAP/case.law as fallback.
    """

    def __init__(self) -> None:
        self._case_law = CaseLawStaticProvider()
        self._cl: CourtListenerProvider | None
        if os.getenv("COURTLISTENER_API_TOKEN", "").strip():
            self._cl = CourtListenerProvider()
        else:
            self._cl = None

    @property
    def label(self) -> str:
        return "opinion_retrieval_chain"

    def get_case_by_citation(
        self,
        citation: str,
        *,
        case_law_caption_matcher: Callable[[str], bool] | None = None,
        require_full_casebody: bool = False,
    ) -> CaseDocument:
        from_cache = load_case_from_cache(
            citation,
            require_full_casebody=require_full_casebody,
        )
        if from_cache is not None:
            if case_law_caption_matcher is None or case_law_caption_matcher(from_cache.case_name):
                _debug_chain(
                    f"citation={citation!r} source=local cache_hit "
                    f"opinion_len={len(from_cache.majority_opinion)} "
                    f"scope={from_cache.opinion_scope!r} "
                    f"require_full={require_full_casebody}"
                )
                return from_cache
            _debug_chain(
                f"citation={citation!r} cache_hit caption_mismatch vs dataset; skipping cache for re-resolve"
            )
        elif require_full_casebody:
            _debug_chain(
                f"citation={citation!r} cache_miss_or_majority_only (require_full_casebody=True)"
            )
        else:
            _debug_chain(f"citation={citation!r} cache_miss")

        errors: list[str] = []

        if self._cl is not None:
            try:
                doc = self._cl.get_case_by_citation(
                    citation,
                    case_law_caption_matcher=case_law_caption_matcher,
                    require_full_casebody=require_full_casebody,
                )
                scope = (
                    OPINION_SCOPE_FULL_CASEBODY
                    if require_full_casebody
                    else OPINION_SCOPE_MAJORITY_ONLY
                )
                out = replace(
                    doc,
                    retrieval_provider_label="court_listener",
                    opinion_scope=scope,
                )
                # Never downgrade a full-casebody cache entry to majority-only.
                if scope == OPINION_SCOPE_MAJORITY_ONLY:
                    existing = load_case_from_cache(citation, require_full_casebody=False)
                    if existing is not None and existing.opinion_scope == OPINION_SCOPE_FULL_CASEBODY:
                        _debug_chain(
                            f"citation={citation!r} source=court_listener success "
                            f"(majority); keeping existing full_casebody cache"
                        )
                        return out
                save_case_to_cache(
                    out,
                    resolved_via="court_listener",
                    cache_key_citation=citation,
                    opinion_scope=scope,
                )
                _debug_chain(
                    f"citation={citation!r} source=court_listener success "
                    f"scope={scope} opinion_len={len(out.majority_opinion)} "
                    f"full_len={len(out.full_opinion_text)} "
                    f"n_opinions={len(out._casebody_opinion_texts())}"
                )
                return out
            except Exception as exc:
                _debug_chain(f"citation={citation!r} source=court_listener failed: {exc!r}")
                errors.append(f"court_listener: {exc}")

        case_law_citation = citation
        if self._cl is not None:
            alt_us = self._cl.best_us_reporter_citation_for_case_law(citation)
            if alt_us and alt_us != citation:
                case_law_citation = alt_us
                _debug_chain(
                    f"citation={citation!r} case_law_retry_with_us_reporter={case_law_citation!r}"
                )

        try:
            doc = self._case_law.get_case_by_citation(
                case_law_citation,
                case_law_caption_matcher=case_law_caption_matcher,
            )
        except Exception as exc:
            _debug_chain(
                f"citation={citation!r} source=case_law failed using={case_law_citation!r}: {exc!r}"
            )
            errors.append(f"case_law[{case_law_citation}]: {exc}")
            msg = f"All opinion retrieval steps failed for {citation!r}. " + "; ".join(errors)
            raise RuntimeError(msg) from exc

        out = replace(
            doc,
            retrieval_provider_label="case_law",
            opinion_scope=OPINION_SCOPE_FULL_CASEBODY,
        )
        save_case_to_cache(
            out,
            resolved_via="case_law",
            cache_key_citation=citation,
            opinion_scope=OPINION_SCOPE_FULL_CASEBODY,
        )
        _debug_chain(
            f"citation={citation!r} source=case_law success using={case_law_citation!r} "
            f"opinion_len={len(out.majority_opinion)} full_len={len(out.full_opinion_text)}"
        )
        return out
