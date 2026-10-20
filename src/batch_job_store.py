from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.api_connectors.jsonl_connector import BATCH_JOB_CONTEXT_BASENAME


def _batch_jobs_dir() -> Path:
    path = Path("output") / "batch_jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _context_candidates(batch_id: str) -> list[Path]:
    """Search paths for persisted batch job context (newest layout first)."""
    run_dir = Path(batch_id)
    candidates = [
        run_dir / BATCH_JOB_CONTEXT_BASENAME,
        Path(f"{batch_id}.json"),  # legacy jsonl: context beside absolute JSONL path
        _batch_jobs_dir() / f"{batch_id}.json",
    ]
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve() if path.is_absolute() else path
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _resolve_context_path(batch_id: str, *, context_path: str | None = None) -> Path:
    if context_path:
        return Path(context_path)
    for candidate in _context_candidates(batch_id):
        if candidate.is_file():
            return candidate
    return _batch_jobs_dir() / f"{batch_id}.json"


def save_batch_job_context(
    *,
    batch_id: str,
    payload: dict[str, Any],
    context_path: str | None = None,
) -> str:
    path = Path(context_path) if context_path else _batch_jobs_dir() / f"{batch_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    return str(path)


def load_batch_job_context(batch_id: str) -> dict[str, Any]:
    path = _resolve_context_path(batch_id)
    if not path.is_file():
        tried = ", ".join(str(p) for p in _context_candidates(batch_id))
        raise FileNotFoundError(
            f"No local context found for batch id {batch_id!r}. Tried: {tried}"
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid context JSON in {path}")
    return payload


def delete_batch_job_context(batch_id: str) -> None:
    """Remove persisted context after a batch run was finalized successfully."""
    path = _resolve_context_path(batch_id)
    path.unlink(missing_ok=True)
