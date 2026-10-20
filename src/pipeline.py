"""Backward-compatible entry: orchestration lives in `src.p02_execution.runner`."""

from src.p02_execution.runner import (
    batch_job_model_from_context,
    finalize_batch_job,
    get_batch_status,
    run_benchmark,
    run_benchmark_batch_submit,
    run_production,
    run_production_batch_submit,
)

__all__ = [
    "batch_job_model_from_context",
    "finalize_batch_job",
    "get_batch_status",
    "run_benchmark",
    "run_benchmark_batch_submit",
    "run_production",
    "run_production_batch_submit",
]
