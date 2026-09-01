#!/usr/bin/env python3
"""Judge ATM open-ended answers with the official prompt and bound inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from mm_memory_bench.methods.backends import OpenAICompatibleQwenVL
from mm_memory_bench.methods.base import GenerationConfig


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    sys.path.insert(0, str(args.official_repo.resolve()))
    from memqa.utils.evaluator.evaluate_qa import (  # noqa: PLC0415
        build_judge_prompt,
        parse_judge_response,
    )

    qas = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    ground_truth_sha256 = hashlib.sha256(args.ground_truth.read_bytes()).hexdigest()
    predictions: dict[str, tuple[str, str]] = {}
    for row in read_jsonl(args.predictions):
        metadata = row.get("metadata", {})
        if isinstance(metadata, dict) and metadata.get("status") == "error":
            raise ValueError(f"refusing to judge error row: {row.get('question_id')}")
        qid = str(row["question_id"]).rsplit(":", 1)[-1]
        answer = str(row["prediction"])
        if not answer.strip() or qid in predictions:
            raise ValueError(f"invalid or duplicate prediction for {qid}")
        predictions[qid] = (answer, canonical_sha256(row))
    expected = {str(qa["id"]) for qa in qas}
    if set(predictions) != expected:
        raise ValueError("prediction IDs do not exactly match ground truth")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"judge API key environment variable is unset: {args.api_key_env}")
    config = GenerationConfig(
        model=args.model,
        base_url=args.base_url,
        max_output_tokens=600,
        max_model_len=32768,
        temperature=0,
    )

    def judge(qa: dict[str, Any]) -> dict[str, Any]:
        qid = str(qa["id"])
        answer, prediction_hash = predictions[qid]
        common = {
            "question_id": qid,
            "prediction_row_sha256": prediction_hash,
            "ground_truth_sha256": ground_truth_sha256,
        }
        if str(qa.get("qtype")) != "open_end":
            return {**common, "correct": False, "status": "deterministic_metric"}
        model = OpenAICompatibleQwenVL(
            config,
            timeout_seconds=300,
            token_count_mode="auto",
            api_key=api_key,
        )
        prompt = build_judge_prompt(str(qa["question"]), str(qa["answer"]), answer)
        last_error: Exception | None = None
        for attempt in range(1, args.retries + 1):
            try:
                raw = model.complete(
                    [{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                parsed = parse_judge_response(raw)
                return {
                    **common,
                    "correct": str(parsed.get("accuracy", "false")).lower() == "true",
                    "explanation": str(parsed.get("explanation", "")),
                    "raw_response": raw,
                    "status": "ok",
                    "judge_model": args.model,
                    "judge_prompt": "ATM-Bench official LLM_JUDGE_PROMPT",
                    "attempts": attempt,
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        return {
            **common,
            "correct": False,
            "status": "error",
            "error": f"{type(last_error).__name__}: {last_error}",
            "judge_model": args.model,
        }

    rows: list[dict[str, Any] | None] = [None] * len(qas)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(judge, qa): index for index, qa in enumerate(qas)}
        for future in as_completed(futures):
            rows[futures[future]] = future.result()
    final = [row for row in rows if row is not None]
    failures = [row for row in final if row.get("status") == "error"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in final),
        encoding="utf-8",
    )
    print(json.dumps({"questions": len(final), "failures": len(failures)}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
