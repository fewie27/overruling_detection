import argparse
import datetime
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.p01_preparation._01_retrieval.volume_loader import parse_volume_spec
from src.api_connectors.jsonl_connector import format_jsonl_batch_submit_message
from src.p02_execution.runner import run_production, run_production_batch_submit


def resolve_output_dir(user_path: str | None) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if user_path is None:
        return str(Path("output") / f"production_results_{ts}")
    p = Path(user_path)
    parent = p.parent if str(p.parent) else Path(".")
    stem = p.stem
    return str(parent / f"{stem}_{ts}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production run: extract citation treatment relations from a full case-law reporter volume."
    )
    parser.add_argument(
        "--reporter",
        required=True,
        help='Reporter label (e.g. "U.S.", "F. Appx."). Maps to the CAP static slug for volume download.',
    )
    parser.add_argument(
        "--volume",
        required=True,
        help=(
            "Volume number(s): single (542), comma list (540,541,542), inclusive range "
            "(540-545), or mixed (540-542,550). min_page/max_page apply only to a single volume."
        ),
    )
    parser.add_argument(
        "--min_page",
        type=int,
        default=None,
        help="Optional minimum case start page (inclusive); only used with exactly one volume.",
    )
    parser.add_argument(
        "--max_page",
        type=int,
        default=None,
        help="Optional maximum case start page (inclusive); only used with exactly one volume.",
    )
    parser.add_argument(
        "--api_connector",
        default="openai",
        choices=["openai", "gemini", "claude", "deepseek", "ollama", "jsonl"],
        help="LLM provider backend to use.",
    )
    parser.add_argument(
        "--prompt_style",
        default="j",
        help=(
            "Prompt variant: a, b, c, d, e, f, g, h, i, i_v2, i_v3, j, j_v1, j_v2, j_v3, j_v4, "
            "chunk_a–chunk_d."
        ),
    )
    parser.add_argument(
        "--chunk_strategy",
        default=None,
        choices=["words", "sentences", "paragraph"],
        help=(
            "Chunking strategy for chunk_a–chunk_c (ignored for other prompts). "
            "Default: words (see CHUNK_STRATEGY_KIND in chunk_base.py)."
        ),
    )
    parser.add_argument(
        "--full-opinion-text",
        action="store_true",
        help=(
            "Pass the full casebody (all opinions) to the LLM instead of majority/lead only. "
            "Default keeps the majority-only filter."
        ),
    )
    parser.add_argument(
        "--llm_model",
        default=None,
        help="LLM model name (API id). Defaults to provider default in config/models.yaml.",
    )
    parser.add_argument(
        "--reasoning_effort",
        default=None,
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort for OpenAI or DeepSeek (--api_connector openai|deepseek).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Base path for output folder name; timestamp is appended. "
            "Each run folder contains llm_result_arrays.json, metadata.json "
            "(default: output/production_results_<timestamp>/)."
        ),
    )
    parser.add_argument(
        "--relations_db",
        default=None,
        help="SQLite path for citation relations (default: output/citation_relations.sqlite).",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Submit requests via provider batch API and print the created batch id.",
    )
    parser.add_argument(
        "--no-opinion-trigger-filter",
        action="store_true",
        help=(
            "Send every retrieved opinion to the LLM. Default: skip opinions that lack "
            "hint words for overruling (overrule, cannot stand, no longer be followed, …)."
        ),
    )
    parser.add_argument(
        "--finetune_dataset_format",
        default=None,
        choices=["prompt_completion", "openai_messages"],
        help=(
            "Optional: write finetune_dataset.jsonl with all production responses (no MATCH filter). "
            "'prompt_completion' keeps legacy {prompt, completion}; 'openai_messages' writes chat format. "
            "When the model returns reasoning/thinking, it is prepended to the answer as a "
            "<think>...</think> block in both formats."
        ),
    )
    parser.add_argument(
        "--print_raw",
        action="store_true",
        help="Print each LLM response body to stdout before parsing.",
    )
    return parser


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    try:
        volumes = parse_volume_spec(args.volume)
    except ValueError as exc:
        parser.error(str(exc))

    if len(volumes) > 1 and (args.min_page is not None or args.max_page is not None):
        print(
            "[production] min_page/max_page ignored when multiple volumes are specified.",
            file=sys.stderr,
            flush=True,
        )

    output_dir = resolve_output_dir(args.output_dir)
    opinion_trigger_filter = not args.no_opinion_trigger_filter

    if args.batch:
        batch_id = run_production_batch_submit(
            reporter=args.reporter,
            volumes=volumes,
            output_dir=output_dir,
            min_page=args.min_page,
            max_page=args.max_page,
            llm_connector_name=args.api_connector,
            prompt_style=args.prompt_style,
            llm_model=args.llm_model,
            reasoning_effort=args.reasoning_effort,
            relations_db_path=args.relations_db,
            opinion_trigger_filter=opinion_trigger_filter,
            print_raw=args.print_raw,
            finetune_dataset_format=args.finetune_dataset_format,
            chunk_strategy_kind=args.chunk_strategy,
            use_full_opinion_text=args.full_opinion_text,
        )
        if args.api_connector.strip().lower() in {"jsonl", "jsonl_file", "file_jsonl"}:
            print(format_jsonl_batch_submit_message(run_dir=batch_id))
        else:
            print(f"Batch submitted successfully. Batch ID: {batch_id}")
            print(f"Output directory reserved for retrieval: {output_dir}")
            print("Run batch_retrieve.py with this batch id when the job completes.")
            if args.finetune_dataset_format:
                print("Finetune dataset will be written when you run batch_retrieve.py for this batch.")
        return

    llm_path, metadata_path, db_path, finetune_path = run_production(
        reporter=args.reporter,
        volumes=volumes,
        output_dir=output_dir,
        min_page=args.min_page,
        max_page=args.max_page,
        llm_connector_name=args.api_connector,
        prompt_style=args.prompt_style,
        llm_model=args.llm_model,
        reasoning_effort=args.reasoning_effort,
        relations_db_path=args.relations_db,
        opinion_trigger_filter=opinion_trigger_filter,
        print_raw=args.print_raw,
        finetune_dataset_format=args.finetune_dataset_format,
        chunk_strategy_kind=args.chunk_strategy,
        use_full_opinion_text=args.full_opinion_text,
    )
    print(f"Done. LLM arrays written to: {llm_path}")
    print(f"Metadata written to: {metadata_path}")
    print(f"Relations written to SQLite: {db_path}")
    if finetune_path:
        print(f"Finetune dataset written to: {finetune_path}")
