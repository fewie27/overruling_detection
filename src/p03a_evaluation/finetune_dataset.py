from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

FinetuneDatasetFormat = Literal["prompt_completion", "openai_messages"]


def format_finetune_prompt(*, system_prompt: str, user_prompt: str) -> str:
    return f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}"


def compact_json_text(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return text
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    return json.dumps(parsed, ensure_ascii=True, separators=(",", ":"))


def _reasoning_text(row: dict[str, Any]) -> str | None:
    reasoning = row.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    return None


def _assistant_output_with_reasoning(answer: str, reasoning: str | None) -> str:
    """Prepend the model's reasoning as an R1-style ``<think>`` block when present so
    the fine-tuned model learns to emit its chain-of-thought before the JSON answer."""
    if reasoning:
        return f"<think>\n{reasoning}\n</think>\n{answer}"
    return answer


def _resolve_prompts(row: dict[str, Any]) -> tuple[str, str] | None:
    fin_sys = row.get("finetune_system_prompt")
    fin_usr = row.get("finetune_user_prompt")
    system_prompt = (
        fin_sys
        if isinstance(fin_sys, str) and fin_sys.strip()
        else row.get("system_prompt")
    )
    user_prompt = (
        fin_usr
        if isinstance(fin_usr, str) and fin_usr.strip()
        else row.get("user_prompt")
    )
    if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
        return None
    return system_prompt, user_prompt


def _finetune_jsonl_line_prompt_completion(row: dict[str, Any]) -> str | None:
    raw = row.get("raw_llm_content")
    if not isinstance(raw, str) or not raw.strip():
        return None
    prompts = _resolve_prompts(row)
    if prompts is None:
        return None
    system_prompt, user_prompt = prompts
    prompt_text = format_finetune_prompt(system_prompt=system_prompt, user_prompt=user_prompt)
    completion_line = compact_json_text(raw)
    if not completion_line.strip():
        return None
    completion_line = _assistant_output_with_reasoning(completion_line, _reasoning_text(row))
    record = {"prompt": prompt_text, "completion": completion_line}
    return json.dumps(record, ensure_ascii=True) + "\n"


def _finetune_jsonl_line_openai_messages(row: dict[str, Any]) -> str | None:
    raw = row.get("raw_llm_content")
    if not isinstance(raw, str) or not raw.strip():
        return None
    prompts = _resolve_prompts(row)
    if prompts is None:
        return None
    system_prompt, user_prompt = prompts
    assistant_content = compact_json_text(raw)
    if not assistant_content.strip():
        return None
    assistant_content = _assistant_output_with_reasoning(assistant_content, _reasoning_text(row))
    record = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content},
        ]
    }
    return json.dumps(record, ensure_ascii=True) + "\n"


def _finetune_jsonl_line_from_row(
    row: dict[str, Any], *, dataset_format: FinetuneDatasetFormat
) -> str | None:
    if dataset_format == "prompt_completion":
        return _finetune_jsonl_line_prompt_completion(row)
    if dataset_format == "openai_messages":
        return _finetune_jsonl_line_openai_messages(row)
    raise ValueError(f"Unsupported finetune dataset format: {dataset_format}")


def strip_finetune_fields_from_llm_responses(llm_responses: list[dict[str, Any]]) -> None:
    for row in llm_responses:
        row.pop("system_prompt", None)
        row.pop("user_prompt", None)
        row.pop("finetune_system_prompt", None)
        row.pop("finetune_user_prompt", None)
        row.pop("raw_llm_content", None)
        row.pop("reasoning_content", None)


def _write_finetune_jsonl_file(
    path: Path,
    llm_responses: list[dict[str, Any]],
    *,
    dataset_format: FinetuneDatasetFormat,
    match_only: bool | None = None,
) -> str:
    """Write finetune JSONL lines.

    match_only:
      True  — only rows with state == "MATCH" (evaluate pipeline)
      False — only rows with state != "MATCH" (evaluate pipeline)
      None  — all rows (production pipeline; no evaluation filter)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in llm_responses:
            if match_only is not None:
                state = row.get("state")
                if match_only:
                    if state != "MATCH":
                        continue
                else:
                    if state == "MATCH":
                        continue
            line = _finetune_jsonl_line_from_row(row, dataset_format=dataset_format)
            if line is not None:
                f.write(line)
    return str(path)


def write_finetune_datasets(
    output_dir: str,
    llm_responses: list[dict[str, Any]],
    *,
    dataset_format: FinetuneDatasetFormat = "prompt_completion",
) -> tuple[str, str]:
    base = Path(output_dir)
    p_match = _write_finetune_jsonl_file(
        base / "finetune_dataset.jsonl",
        llm_responses,
        match_only=True,
        dataset_format=dataset_format,
    )
    p_wrong = _write_finetune_jsonl_file(
        base / "finetune_dataset_wrong.jsonl",
        llm_responses,
        match_only=False,
        dataset_format=dataset_format,
    )
    return p_match, p_wrong


def write_finetune_dataset_unfiltered(
    output_dir: str,
    llm_responses: list[dict[str, Any]],
    *,
    dataset_format: FinetuneDatasetFormat = "prompt_completion",
) -> str:
    """Write all rows into a single finetune_dataset.jsonl (no MATCH/wrong split)."""
    return _write_finetune_jsonl_file(
        Path(output_dir) / "finetune_dataset.jsonl",
        llm_responses,
        match_only=None,
        dataset_format=dataset_format,
    )
