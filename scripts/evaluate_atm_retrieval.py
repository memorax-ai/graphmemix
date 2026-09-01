#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def read_jsonl(path: Path, key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            value = str(row[key])
            if value in rows:
                raise ValueError(f"duplicate {key}={value!r} in {path}:{line_number}")
            rows[value] = row
    return rows


def unique_strings(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


H2H_ALIAS_RE = re.compile(r"^S(\d+)-(\d+)$")


def memory_sessions(path: Path | None) -> dict[tuple[str, str], list[str]]:
    result: dict[tuple[str, str], list[str]] = defaultdict(list)
    if path is None:
        return result
    for row in read_jsonl(path, "memory_id").values():
        session_id = str(row.get("session_id", ""))
        if session_id:
            result[(str(row["context_id"]), session_id)].append(str(row["memory_id"]))
    return result


def gold_ids(
    question: dict[str, Any],
    sessions: Mapping[tuple[str, str], list[str]] | None = None,
) -> list[str]:
    values: list[Any] = []
    for item in question.get("evidence", []):
        if item.get("memory_id"):
            values.append(item["memory_id"])
        # H2HMem evidence points at an answer session and expands it to the
        # canonical memories in that session.
        expanded = item.get("memory_ids", [])
        values.extend(expanded)
        if not expanded and sessions:
            native_id = str(item.get("native_id", ""))
            match = H2H_ALIAS_RE.fullmatch(native_id)
            dialogue = str(question.get("metadata", {}).get("native_dialogue", ""))
            dialogue_number = re.search(r"(\d+)$", dialogue)
            if match and dialogue_number and match.group(1) == dialogue_number.group(1):
                values.extend(sessions.get((
                    str(question["context_id"]),
                    f"session{int(match.group(2))}",
                ), []))
    return unique_strings(values)


def retrieved_ids(prediction: dict[str, Any], k: int) -> list[str]:
    return unique_strings(prediction.get("retrieved_memory_ids", []))[:k]


def metrics_for(
    questions: Iterable[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    k: int,
    sessions: Mapping[tuple[str, str], list[str]] | None = None,
) -> dict[str, Any]:
    recalls: list[float] = []
    any_hits = 0
    all_hits = 0
    total_gold = 0
    total_hits = 0

    for question in questions:
        question_id = str(question["question_id"])
        gold = gold_ids(question, sessions)
        if not gold:
            continue
        retrieved = set(retrieved_ids(predictions[question_id], k))
        hits = sum(memory_id in retrieved for memory_id in gold)
        recalls.append(hits / len(gold))
        any_hits += int(hits > 0)
        all_hits += int(hits == len(gold))
        total_gold += len(gold)
        total_hits += hits

    count = len(recalls)
    if not count:
        raise ValueError("no questions with gold evidence")
    return {
        "questions": count,
        f"R@{k}": sum(recalls) / count,
        f"Hit@{k}": any_hits / count,
        f"AllGT@{k}": all_hits / count,
        f"micro_R@{k}": total_hits / total_gold,
        "gold_evidence": total_gold,
        "gold_hits": total_hits,
    }


def compare_predictions(
    questions: dict[str, dict[str, Any]],
    current: dict[str, dict[str, Any]],
    reference: dict[str, dict[str, Any]],
    k: int,
) -> dict[str, Any]:
    exact_order = 0
    exact_set = 0
    changed: list[dict[str, Any]] = []
    for question_id in questions:
        current_ids = retrieved_ids(current[question_id], k)
        reference_ids = retrieved_ids(reference[question_id], k)
        exact_order += int(current_ids == reference_ids)
        exact_set += int(set(current_ids) == set(reference_ids))
        if current_ids != reference_ids:
            changed.append({
                "question_id": question_id,
                "current": current_ids,
                "reference": reference_ids,
            })
    count = len(questions)
    return {
        "questions": count,
        f"exact_top{k}_order": exact_order,
        f"exact_top{k}_order_rate": exact_order / count,
        f"exact_top{k}_set": exact_set,
        f"exact_top{k}_set_rate": exact_set / count,
        "changed_questions": changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate canonical Main Track retrieval IDs."
    )
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--memories",
        type=Path,
        help="Memory table used to resolve locked H2HMem session aliases.",
    )
    parser.add_argument(
        "--question-ids",
        type=Path,
        help="Optional newline-delimited reporting allowlist (for example H2HMem).",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.k <= 0:
        parser.error("--k must be positive")

    questions = read_jsonl(args.questions, "question_id")
    if args.question_ids:
        wanted = {
            line.strip()
            for line in args.question_ids.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        missing_questions = wanted - questions.keys()
        if missing_questions:
            raise ValueError(
                f"question allowlist contains {len(missing_questions)} unknown IDs"
            )
        questions = {
            question_id: row
            for question_id, row in questions.items()
            if question_id in wanted
        }
    predictions = read_jsonl(args.predictions, "question_id")
    sessions = memory_sessions(args.memories)
    missing = sorted(set(questions) - set(predictions))
    extra = sorted(set(predictions) - set(questions))
    if missing or extra:
        raise ValueError(
            f"prediction/question mismatch: missing={len(missing)} extra={len(extra)}"
        )

    subsets = sorted({str(row.get("subset", "default")) for row in questions.values()})
    report: dict[str, Any] = {
        "protocol": "mmmb-main-track-retrieval-eval-1.0",
        "k": args.k,
        "predictions": str(args.predictions),
        "metrics": {
            "all": metrics_for(questions.values(), predictions, args.k, sessions),
            **{
                subset: metrics_for(
                    (
                        row
                        for row in questions.values()
                        if str(row.get("subset", "default")) == subset
                    ),
                    predictions,
                    args.k,
                    sessions,
                )
                for subset in subsets
            },
        },
    }

    if args.reference:
        reference = read_jsonl(args.reference, "question_id")
        if set(reference) != set(questions):
            raise ValueError("reference/question ID sets do not match")
        report["reference"] = str(args.reference)
        report["comparison"] = compare_predictions(
            questions, predictions, reference, args.k
        )

    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
