from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.api_connectors.base import BatchStatus, BatchSubmitResult, ConnectorResponse, LLMConnector
from src.models import PreparedLLMRequest

PROMPT_REQUESTS_BASENAME = "prompt_requests.jsonl"
PROMPT_RESPONSES_BASENAME = "prompt_responses.jsonl"
BATCH_JOB_CONTEXT_BASENAME = "batch_job_context.json"


def format_jsonl_batch_submit_message(*, run_dir: str) -> str:
    """Human-readable next steps after a jsonl batch export."""
    run_path = Path(run_dir).resolve()
    return "\n".join(
        [
            "JSONL batch export complete.",
            "",
            f"Run directory: {run_path}",
            f"  Prompt requests:  {run_path / PROMPT_REQUESTS_BASENAME}",
            "",
            "Next steps:",
            "  1. Run your inference script on prompt_requests.jsonl",
            "     (add output_text on each line with the model JSON response).",
            f"  2. Save the completed file as: {run_path / PROMPT_RESPONSES_BASENAME}",
            "",
            "  3. Finalize evaluation:",
            f"     python batch_retrieve.py --batch_id {run_path}",
        ]
    )


class JsonlFileConnector(LLMConnector):
    """
    Export batch prompts under a run directory, then read completed responses from the same folder.

    Layout (under the run ``output_dir`` used as ``batch_id``):

      prompt_requests.jsonl   — written on submit
      prompt_responses.jsonl  — filled by your inference script, read on finalize
      batch_job_context.json  — internal job state (written by runner)
    """

    def __init__(self, model: str | None = None, *, output_dir: str | None = None) -> None:
        self._model = model or "jsonl-file"
        self._output_dir = output_dir

    @property
    def label(self) -> str:
        return "jsonl"

    @property
    def model(self) -> str:
        return self._model

    def extract_overruled_cases(
        self, *, system_prompt: str, user_prompt: str
    ) -> ConnectorResponse:
        raise NotImplementedError(
            "Interactive chat is not supported for the jsonl connector; use batch submit/import."
        )

    def submit_batch(self, requests: list[PreparedLLMRequest]) -> BatchSubmitResult:
        if not self._output_dir:
            raise ValueError(
                "Jsonl batch submit requires an output run directory (set when building the connector)."
            )
        run_dir = Path(self._output_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        requests_path = run_dir / PROMPT_REQUESTS_BASENAME
        with open(requests_path, "w", encoding="utf-8") as f:
            for req in requests:
                line = {
                    "custom_id": req.request_id,
                    "system_prompt": req.system_prompt,
                    "user_prompt": req.user_prompt,
                    "response_parser_id": req.response_parser_id,
                }
                f.write(json.dumps(line, ensure_ascii=True) + "\n")
        return BatchSubmitResult(batch_id=str(run_dir))

    def get_batch_status(self, batch_id: str) -> BatchStatus:
        return BatchStatus(
            batch_id=batch_id,
            status="completed",
            is_terminal=True,
            is_success=True,
        )

    def _responses_path(self, batch_id: str) -> Path:
        path = Path(batch_id)
        # Legacy: batch_id was the path to a single JSONL file.
        if path.is_file() and path.suffix == ".jsonl":
            return path
        return path / PROMPT_RESPONSES_BASENAME

    def collect_batch_results(self, batch_id: str) -> dict[str, ConnectorResponse]:
        path = self._responses_path(batch_id)
        if not path.is_file():
            run_dir = Path(batch_id).resolve()
            raise FileNotFoundError(
                f"JSONL responses file not found: {path}\n"
                f"Place your completed inference output at: {run_dir / PROMPT_RESPONSES_BASENAME}"
            )
        results: dict[str, ConnectorResponse] = {}
        with open(path, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                obj: dict[str, Any] = json.loads(line)
                cid = str(obj.get("custom_id") or "")
                if not cid:
                    continue
                content = obj.get("output_text")
                if content is None:
                    content = obj.get("raw_model_output")
                if content is None and isinstance(obj.get("response"), dict):
                    content = json.dumps(obj["response"])
                text = (content if isinstance(content, str) else str(content or "")).strip()
                if not text:
                    raise RuntimeError(
                        f"JSONL line for custom_id={cid!r} has no output_text / raw_model_output; "
                        "fill model output before finalize."
                    )
                results[cid] = ConnectorResponse(raw_content=text, raw_response=obj)
        if not results:
            raise RuntimeError(f"No usable JSONL result rows in {path!r}.")
        return results
