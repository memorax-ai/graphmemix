"""Run benchmark-specific LLM scorers without changing the original QA Judge.

Scorers are plain modules exposing request, parsing and summary functions.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

from ..benchmarks.bundle import iter_jsonl, read_json, write_json
from .judge import JudgeBackend
from .prediction_status import method_failure


def get_scorer(name):
    from .native import m3exam_judge, mobilemem_omni_judge, personamem_v2_judge
    scorers = {"m3exam": m3exam_judge, "mobilemem_omni": mobilemem_omni_judge,
               "personamem_v2_open": personamem_v2_judge}
    if name not in scorers:
        raise ValueError(f"unknown dedicated Judge: {name}")
    return scorers[name]


class JudgeResponseError(RuntimeError):
    """Transport completed without a usable judgment; retain response diagnostics."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class OpenAICompatibleJudge:
    """Minimal dependency-free client for OpenAI-compatible chat endpoints."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        scoring_protocol: str,
        max_tokens: int | None = None,
    ) -> None:
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("judge max_tokens must be positive")
        self.max_tokens = max_tokens
        self.request_config = {"transport_version": 2, "max_tokens_override": max_tokens}
        self.model = model
        self.scoring_protocol = scoring_protocol
        self.protocol = get_scorer(scoring_protocol)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def judge(self, item: Mapping[str, Any]) -> Mapping[str, Any]:
        body = self.protocol.request_body(self.model, item)
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"judge HTTP {exc.code}: {detail[:1000]}") from exc
        choice = payload["choices"][0]
        content = choice["message"].get("content")
        diagnostics = {"finish_reason": choice.get("finish_reason"),
                       "usage": payload.get("usage"),
                       "request_max_tokens": body.get("max_tokens")}
        if choice.get("finish_reason") not in (None, "stop"):
            raise JudgeResponseError("judge did not finish normally", diagnostics)
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "")) if isinstance(part, Mapping) else str(part)
                for part in content
            )
        if not isinstance(content, str) or not content.strip():
            raise JudgeResponseError("judge returned empty content", diagnostics)
        try:
            result = self.protocol.parse_response(content)
        except (ValueError, TypeError) as exc:
            raise JudgeResponseError(str(exc), diagnostics) from exc
        return {**result, "judge_response_metadata": diagnostics}


def judge_predictions(
    backend: JudgeBackend,
    bundle_root: Path,
    predictions_path: Path,
    output_path: Path,
    *,
    resume: bool = True,
    max_items: int | None = None,
    question_ids_path: Path | None = None,
    question_ids: list[str] | None = None,
    concurrency: int = 1,
    scoring_protocol: str,
) -> dict[str, Any]:
    protocol = get_scorer(scoring_protocol)
    if getattr(backend, "scoring_protocol", scoring_protocol) != scoring_protocol:
        raise ValueError("backend and runner scoring protocols differ")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    manifest = read_json(bundle_root / "manifest.json")
    if scoring_protocol == "personamem_v2_open" and str(manifest.get("benchmark", "")).lower().replace("-", "_") != "personamem_v2":
        raise ValueError("personamem_v2_open requires a PersonaMem-v2 bundle")
    required_benchmark = getattr(protocol, "benchmark", None)
    if required_benchmark and str(manifest.get("benchmark", "")).lower().replace("-", "_") != required_benchmark:
        raise ValueError(f"{scoring_protocol} requires a {required_benchmark} bundle")
    question_path = bundle_root / str(
        manifest.get("tables", {}).get("questions", "questions.jsonl")
    )
    questions = {str(row["question_id"]): row for row in iter_jsonl(question_path)}
    predictions = list(iter_jsonl(predictions_path))
    reporting_ids: set[str] | None = None
    if question_ids_path is not None and question_ids is not None:
        raise ValueError("provide question_ids or question_ids_path, not both")
    if question_ids_path is not None or question_ids is not None:
        values = list(question_ids) if question_ids is not None else [
            line.strip()
            for line in question_ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(set(values)) != len(values):
            raise ValueError("question-id allowlist contains duplicates")
        reporting_ids = set(values)
        unknown_reporting_ids = reporting_ids - questions.keys()
        if unknown_reporting_ids:
            raise KeyError(
                "question-id allowlist contains unknown values: "
                + ", ".join(sorted(unknown_reporting_ids)[:5])
            )
        predictions = [
            row for row in predictions
            if str(row.get("question_id", "")) in reporting_ids
        ]
        missing_reporting_ids = reporting_ids - {
            str(row.get("question_id", "")) for row in predictions
        }
        if missing_reporting_ids:
            raise ValueError(
                "predictions are incomplete for the reporting track: "
                + ", ".join(sorted(missing_reporting_ids)[:5])
            )
    if max_items is not None:
        if max_items < 0:
            raise ValueError("max_items must be non-negative")
        predictions = predictions[:max_items]
    prediction_ids = [str(row.get("question_id", "")) for row in predictions]
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("predictions contain duplicate question_id values")
    selected_ids = set(prediction_ids)
    unknown_ids = selected_ids - questions.keys()
    if unknown_ids:
        raise KeyError(
            "predictions reference unknown question_id values: "
            + ", ".join(sorted(unknown_ids)[:5])
        )

    # Validate every selected task before any API call or output overwrite.
    items = {
        str(row["question_id"]): protocol.judge_item(
            questions[str(row["question_id"])], str(row.get("prediction", ""))
        ) for row in predictions
    }
    item_hashes = {
        qid: hashlib.sha256(json.dumps(item, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        for qid, item in items.items()
    }

    prediction_hashes = {
        str(row.get("question_id", "")): hashlib.sha256(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        for row in predictions
    }
    failures = {str(row["question_id"]): method_failure(row) for row in predictions}
    existing: dict[str, dict[str, Any]] = {}
    if resume and output_path.is_file():
        for row in iter_jsonl(output_path):
            if (
                row.get("status") == "ok"
                and str(row["question_id"]) in selected_ids
                and not failures[str(row["question_id"])]
                and row.get("judge_model") == backend.model
                and row.get("judge_request_config") == getattr(backend, "request_config", None)
                and row.get("judge_protocol") == protocol.PROTOCOL_VERSION
                and row.get("judge_rubric_sha256") == protocol.RUBRIC_SHA256
                and row.get("judge_item_sha256") == item_hashes[str(row["question_id"])]
                and row.get("prediction_row_sha256")
                == prediction_hashes[str(row["question_id"])]
            ):
                existing[str(row["question_id"])] = row
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(prediction: Mapping[str, Any]) -> dict[str, Any]:
        question_id = str(prediction.get("question_id", ""))
        question = questions[question_id]
        record: dict[str, Any] = {
            "question_id": question_id,
            "semantic_question_id": question.get("semantic_question_id", question_id),
            "context_id": question["context_id"],
            "subset": question.get("subset", "default"),
            "prediction": str(prediction.get("prediction", "")),
            "judge_model": backend.model,
            "judge_request_config": getattr(backend, "request_config", None),
            "judge_protocol": protocol.PROTOCOL_VERSION,
            "judge_rubric_sha256": protocol.RUBRIC_SHA256,
            "prediction_row_sha256": prediction_hashes[question_id],
            "judge_item_sha256": item_hashes[question_id],
            **protocol.record_fields(items[question_id]),
        }
        failure = failures[question_id]
        if failure:
            record.update({"status": "method_error", **protocol.zero(), **failure, "score": 0.0})
            if "label" in record:
                record.update(label="WRONG", correct=False)
            return record
        empty_is_zero = getattr(protocol, "empty_prediction_is_zero", True)
        skip_item = getattr(protocol, "skip_item", lambda item: False)
        if (empty_is_zero and not record["prediction"].strip()) or skip_item(items[question_id]):
            record.update({"status": "ok", **protocol.zero()})
            return record
        try:
            raw = backend.judge(items[question_id])
            normalized = protocol.normalize(raw)
            record.update(
                {
                    "status": "ok",
                    **normalized,
                    **({"judge_response_metadata": raw["judge_response_metadata"]}
                       if "judge_response_metadata" in raw else {}),
                }
            )
        except Exception as exc:  # preserve failures for resumable, auditable runs
            record.update(
                {
                    "status": "error",
                    **protocol.zero(),
                    "error": f"{type(exc).__name__}: {exc}",
                    **({"judge_response_metadata": exc.diagnostics}
                       if isinstance(exc, JudgeResponseError) else {}),
                }
            )
        return record

    pending = [
        prediction
        for prediction in predictions
        if str(prediction.get("question_id", "")) not in existing
    ]
    with output_path.open("w", encoding="utf-8") as handle:
        for row in existing.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for record in executor.map(evaluate, pending):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

    combined = list(iter_jsonl(output_path))
    combined = [row for row in combined if str(row["question_id"]) in selected_ids]
    summary = {
        "benchmark": manifest.get("benchmark"),
        "bundle": str(bundle_root),
        "predictions": str(predictions_path),
        "judgments": str(output_path),
        "judge_model": backend.model,
        "judge_protocol": protocol.PROTOCOL_VERSION,
        "judge_rubric_sha256": protocol.RUBRIC_SHA256,
        "question_ids": str(question_ids_path) if question_ids_path else None,
        "reporting_track_size": len(reporting_ids) if reporting_ids is not None else None,
        "concurrency": concurrency,
        "scoring_protocol": scoring_protocol,
        **protocol.summarize(combined),
    }
    summary_path = output_path.with_suffix(".summary.json")
    write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def backend_from_env(
    *,
    model: str,
    base_url: str,
    api_key_env: str,
    timeout_seconds: float,
    scoring_protocol: str,
    max_tokens: int | None = None,
) -> OpenAICompatibleJudge:
    return OpenAICompatibleJudge(
        model=model,
        base_url=base_url,
        api_key=os.environ.get(api_key_env),
        timeout_seconds=timeout_seconds,
        scoring_protocol=scoring_protocol,
        max_tokens=max_tokens,
    )
