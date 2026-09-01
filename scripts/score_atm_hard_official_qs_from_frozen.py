#!/usr/bin/env python3
"""Score frozen ATM-hard predictions with the official ATM/QS decomposition.

Number and list-recall components call the evaluator from an official ATM-Bench
checkout. Open-ended decisions are supplied by a precomputed judge file, so the
output records the judge identity instead of claiming official GPT-5-mini QS.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
    parser.add_argument("--open-judgments", type=Path, required=True)
    parser.add_argument(
        "--open-judge-label",
        default="gpt-5.6-sol medium via Codex (not official gpt-5-mini)",
        help="Human-readable identity of the supplied open-ended judge.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-predictions-output", type=Path)
    parser.add_argument("--require-bound-judgments", action="store_true")
    parser.add_argument(
        "--allow-imputed-incorrect",
        action="store_true",
        help="Permit explicitly marked imputed_incorrect rows for a documented special analysis.",
    )
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Score a non-empty ground-truth subset instead of requiring all 31 questions.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.official_repo.resolve()))
    evaluator = importlib.import_module("memqa.utils.evaluator.evaluate_qa")

    qas = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    prediction_rows = read_jsonl(args.predictions)
    predictions: dict[str, str] = {}
    prediction_hashes: dict[str, str] = {}
    for row in prediction_rows:
        question_id = str(row["question_id"])
        metadata = row.get("metadata", {})
        status = metadata.get("status") if isinstance(metadata, dict) else None
        if status == "error":
            raise ValueError(f"refusing to score recorded error row: {question_id}")
        if status == "imputed_incorrect" and not args.allow_imputed_incorrect:
            raise ValueError(
                f"refusing to score imputed row without --allow-imputed-incorrect: "
                f"{question_id}"
            )
        prediction = str(row["prediction"])
        if not prediction.strip():
            raise ValueError(f"refusing to score empty prediction: {question_id}")
        native_id = question_id.rsplit(":", 1)[-1]
        if native_id in predictions:
            raise ValueError(f"duplicate prediction: {question_id}")
        predictions[native_id] = prediction
        prediction_hashes[native_id] = canonical_sha256(row)
    judgment_rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(args.open_judgments):
        native_id = str(row["question_id"]).rsplit(":", 1)[-1]
        if native_id in judgment_rows:
            raise ValueError(f"duplicate open judgment: {native_id}")
        if row.get("status") == "error":
            raise ValueError(f"refusing to score failed open judgment: {native_id}")
        judgment_rows[native_id] = row
    if args.require_bound_judgments:
        ground_truth_sha256 = hashlib.sha256(args.ground_truth.read_bytes()).hexdigest()
        for native_id, row in judgment_rows.items():
            if row.get("prediction_row_sha256") != prediction_hashes.get(native_id):
                raise ValueError(f"judgment/prediction hash mismatch: {native_id}")
            if row.get("ground_truth_sha256") != ground_truth_sha256:
                raise ValueError(f"judgment/ground-truth hash mismatch: {native_id}")
    judgments = {
        native_id: row["correct"]
        for native_id, row in judgment_rows.items()
    }
    if any(not isinstance(value, bool) for value in judgments.values()):
        raise ValueError("every judgment correct field must be a JSON boolean")
    expected_ids = {str(qa["id"]) for qa in qas}
    if args.allow_subset:
        if not qas or set(predictions) != expected_ids or set(judgments) != expected_ids:
            raise ValueError("subset question, prediction, and judgment IDs must match")
    elif len(qas) != 31 or len(predictions) != 31 or len(judgments) != 31:
        raise ValueError("expected 31 questions, predictions, and judgments")

    if args.official_predictions_output is not None:
        args.official_predictions_output.parent.mkdir(parents=True, exist_ok=True)
        with args.official_predictions_output.open("w", encoding="utf-8") as handle:
            for qa in qas:
                qid = str(qa["id"])
                handle.write(json.dumps({"id": qid, "answer": predictions[qid]}, ensure_ascii=False) + "\n")

    rows: list[dict[str, Any]] = []
    totals: dict[str, dict[str, float]] = {}
    for qa in qas:
        qid = str(qa["id"])
        qtype = str(qa["qtype"])
        answer = str(qa["answer"])
        prediction = predictions[qid]
        if qtype == "number":
            correct, normalized = evaluator._deterministic_accuracy_core(
                answer, prediction, question=str(qa["question"])
            )
            score = float(correct)
            detail: dict[str, Any] = {"normalized_prediction": normalized}
            metric = "official_normalized_exact_match"
        elif qtype == "list_recall":
            score, items = evaluator._list_jaccard_core(answer, prediction)
            detail = {"normalized_prediction_items": items}
            metric = "official_list_jaccard"
        elif qtype == "open_end":
            if judgment_rows[qid].get("status") != "ok":
                raise ValueError(
                    f"open-ended judgment is not successful for {qid}: "
                    f"{judgment_rows[qid].get('status')!r}"
                )
            score = float(judgments[qid])
            detail = {}
            metric = "supplied_open_judge"
        else:
            raise ValueError(f"unsupported qtype {qtype!r}")
        rows.append({
            "id": qid,
            "qtype": qtype,
            "metric": metric,
            "score": score,
            "prediction": prediction,
            **detail,
        })
        stats = totals.setdefault(qtype, {"count": 0.0, "score": 0.0})
        stats["count"] += 1
        stats["score"] += score

    total_score = sum(float(row["score"]) for row in rows)
    summary = {
        "protocol": "atm-official-qs-components-with-supplied-open-judge-v2",
        "official_repo": str(args.official_repo.resolve()),
        "count": len(rows),
        "score_sum": total_score,
        "qs": total_score / len(rows),
        "by_qtype": {
            key: {
                "count": int(value["count"]),
                "score_sum": value["score"],
                "score": value["score"] / value["count"],
            }
            for key, value in totals.items()
        },
        "open_end_judge": args.open_judge_label,
        "rows": rows,
    }
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
