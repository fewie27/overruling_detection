from __future__ import annotations

import argparse
import time

from dotenv import load_dotenv

from src.batch_job_store import load_batch_job_context
from src.p02_execution.runner import batch_job_model_from_context, finalize_batch_job, get_batch_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Poll a submitted batch job and finalize pipeline outputs when it succeeds."
    )
    parser.add_argument("--batch_id", required=True, help="Batch job id printed during --batch submission.")
    parser.add_argument(
        "--poll_seconds",
        type=int,
        default=30,
        help="Polling interval in seconds while waiting for completion (default: 30).",
    )
    parser.add_argument(
        "--print_raw",
        action="store_true",
        help="Print each LLM response body to stdout before parsing (overrides batch submit setting).",
    )
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    context = load_batch_job_context(args.batch_id)
    connector_name = context["connector_name"]
    llm_model = batch_job_model_from_context(context)

    if connector_name.strip().lower() != "jsonl":
        while True:
            status = get_batch_status(
                args.batch_id,
                llm_connector_name=connector_name,
                llm_model=llm_model,
                reasoning_effort=context.get("reasoning_effort"),
            )
            print(f"Batch {args.batch_id}: {status.status}")
            if status.is_terminal:
                break
            time.sleep(max(1, args.poll_seconds))
    else:
        status = get_batch_status(
            args.batch_id,
            llm_connector_name=connector_name,
            llm_model=llm_model,
            reasoning_effort=context.get("reasoning_effort"),
        )
        print(f"Batch {args.batch_id} (jsonl): {status.status} — skipping poll; finalize immediately.")

    if not status.is_success:
        raise RuntimeError(f"Batch {args.batch_id} finished in non-success state: {status.status}")

    pipeline = context.get("pipeline", "evaluate")
    result = finalize_batch_job(
        args.batch_id,
        print_errors=True,
        print_raw=args.print_raw if args.print_raw else None,
    )
    if pipeline in ("production", "extract"):
        llm_path, metadata_path, db_path, finetune_path = result
        print(f"Done. LLM arrays written to: {llm_path}")
        print(f"Metadata written to: {metadata_path}")
        print(f"Relations written to SQLite: {db_path}")
        if finetune_path:
            print(f"Finetune dataset written to: {finetune_path}")
        return

    evaluation_path, metadata_path, llm_arrays_path, finetune_paths = result
    print(f"Done. Results written to: {evaluation_path}")
    print(f"Metadata written to: {metadata_path}")
    print(f"LLM arrays written to: {llm_arrays_path}")
    if finetune_paths:
        p_match, p_wrong = finetune_paths
        print(f"Finetune dataset (MATCH) written to: {p_match}")
        print(f"Finetune dataset (non-MATCH) written to: {p_wrong}")
