import argparse
import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.models import DEFAULT_STRICT_CLASSES, parse_strict_classes_csv
from src.p01_preparation._01_retrieval.benchmark import parse_indices_csv
from src.api_connectors.jsonl_connector import format_jsonl_batch_submit_message
from src.p02_execution.runner import run_benchmark, run_benchmark_batch_submit


def resolve_output_dir(user_path: str | None) -> str:
    """Build output folder path with an embedded timestamp so each run gets a unique folder."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if user_path is None:
        return str(Path("output") / f"evaluation_results_{ts}")
    p = Path(user_path)
    parent = p.parent if str(p.parent) else Path(".")
    stem = p.stem
    return str(parent / f"{stem}_{ts}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation: retrieve opinions, classify citations, score against ground truth."
    )
    parser.add_argument(
        "--path_to_dataset",
        required=True,
        help="Path to benchmark JSONL (one entry per line: id, case_citation, citations[]).",
    )
    parser.add_argument(
        "--api_connector",
        default="openai",
        choices=["openai", "gemini", "claude", "deepseek", "ollama", "jsonl"],
        help="LLM provider backend to use.",
    )
    parser.add_argument(
        "--prompt_style",
        default="a",
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
            "Default keeps the majority-only filter. Majority-only cache hits are skipped so "
            "full CAP casebody text is retrieved."
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
        help=(
            "Reasoning effort for OpenAI, DeepSeek or Claude "
            "(--api_connector openai|deepseek|claude). For Claude this enables extended "
            "thinking with an increasing budget (minimal=1024 .. xhigh=16384 tokens); "
            "'none' disables thinking."
        ),
    )
    parser.add_argument("--from_idx", type=int, default=None, help="Start id (1-based, inclusive).")
    parser.add_argument("--to_idx", type=int, default=None, help="End id (1-based, inclusive).")
    parser.add_argument(
        "--indices",
        default=None,
        metavar="N[,N|A-B...]",
        help=(
            "Comma-separated benchmark ids and inclusive ranges to run "
            "(e.g. 3,21,37 or 1-79,81-100). Overrides --from_idx/--to_idx."
        ),
    )
    parser.add_argument(
        "--strict_classes",
        default=None,
        metavar="LABEL[,LABEL...]",
        help=(
            "Comma-separated class labels for which extra (not-in-benchmark) citations also count "
            "as errors. Default: " + ",".join(sorted(DEFAULT_STRICT_CLASSES))
        ),
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help=(
            "Base path for output folder name; timestamp is appended to folder name. "
            "Each run folder contains results.csv, llm_result_arrays.json, metadata.json, "
            "citation_classification_results.csv (default: output/evaluation_results_<timestamp>/)."
        ),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Submit requests via provider batch API and print the created batch id.",
    )
    parser.add_argument(
        "--opinion-trigger-filter",
        action="store_true",
        help=(
            "Skip opinions that lack overruling hint words (overrule, cannot stand, "
            "no longer be followed, …). Default: off. Skipped case ids are printed at submit/run time."
        ),
    )
    parser.add_argument(
        "--finetune_dataset_format",
        default=None,
        choices=["prompt_completion", "openai_messages"],
        help=(
            "Optional: write finetune_dataset.jsonl (MATCH) and finetune_dataset_wrong.jsonl (non-MATCH). "
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
    indices_list: list[int] | None = None
    from_idx = args.from_idx
    to_idx = args.to_idx
    if args.indices is not None:
        try:
            indices_list = parse_indices_csv(args.indices)
        except ValueError as exc:
            parser.error(str(exc))
        if not indices_list:
            parser.error("--indices requires at least one 1-based id.")
        from_idx = None
        to_idx = None
    strict_classes = parse_strict_classes_csv(args.strict_classes)
    output_dir = resolve_output_dir(args.output_csv)
    if args.batch:
        batch_id = run_benchmark_batch_submit(
            dataset_path=args.path_to_dataset,
            output_dir=output_dir,
            llm_connector_name=args.api_connector,
            prompt_style=args.prompt_style,
            llm_model=args.llm_model,
            reasoning_effort=args.reasoning_effort,
            from_idx=from_idx,
            to_idx=to_idx,
            indices=indices_list,
            strict_classes=strict_classes,
            print_errors=True,
            print_raw=args.print_raw,
            finetune_dataset_format=args.finetune_dataset_format,
            chunk_strategy_kind=args.chunk_strategy,
            use_full_opinion_text=args.full_opinion_text,
            opinion_trigger_filter=args.opinion_trigger_filter,
        )
        if args.api_connector.strip().lower() in {"jsonl", "jsonl_file", "file_jsonl"}:
            print(format_jsonl_batch_submit_message(run_dir=batch_id))
        else:
            print(f"Batch submitted successfully. Batch ID: {batch_id}")
            print(f"Output directory reserved for retrieval: {output_dir}")
            if args.finetune_dataset_format:
                print("Finetune datasets will be written when you run batch_retrieve.py for this batch.")
        return

    evaluation_path, metadata_path, llm_arrays_path, finetune_paths = run_benchmark(
        dataset_path=args.path_to_dataset,
        output_dir=output_dir,
        llm_connector_name=args.api_connector,
        prompt_style=args.prompt_style,
        llm_model=args.llm_model,
        reasoning_effort=args.reasoning_effort,
        from_idx=from_idx,
        to_idx=to_idx,
        indices=indices_list,
        strict_classes=strict_classes,
        print_errors=True,
        print_raw=args.print_raw,
        finetune_dataset_format=args.finetune_dataset_format,
        chunk_strategy_kind=args.chunk_strategy,
        use_full_opinion_text=args.full_opinion_text,
        opinion_trigger_filter=args.opinion_trigger_filter,
    )
    print(f"Done. Results written to: {evaluation_path}")
    print(f"Metadata written to: {metadata_path}")
    print(f"LLM arrays written to: {llm_arrays_path}")
    if finetune_paths:
        p_match, p_wrong = finetune_paths
        print(f"Finetune dataset (MATCH) written to: {p_match}")
        print(f"Finetune dataset (non-MATCH) written to: {p_wrong}")
