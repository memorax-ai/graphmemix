#!/usr/bin/env python3
"""Controlled GraphMemix comparison for node relevance and ECV edge scores."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_atm_retrieval import memory_sessions, metrics_for, read_jsonl
from graphmemix_core import load_adjacency, missing_atomic_floor
from probe_graphmemix_pcsf import solve_forest_greedy


POSITIVE_ROLES = {"new_fact", "clarification", "corroboration"}


def expit(value: float) -> float:
    """Numerically stable scalar sigmoid without a SciPy runtime dependency."""
    if value >= 0:
        term = math.exp(-value)
        return 1.0 / (1.0 + term)
    term = math.exp(value)
    return term / (1.0 + term)


def conditioned_adjacency(
    row: Mapping[str, Any], *, power: float,
) -> dict[str, list[tuple[str, float, str]]]:
    result: dict[str, list[tuple[str, float, str]]] = {}
    for edge in row.get("edge_scores", {}).values():
        if str(edge.get("role", "")) not in POSITIVE_ROLES:
            continue
        incremental = max(0.0, min(5.0, float(edge.get("incremental_support", 0.0))))
        if incremental <= 0:
            continue
        similarity = 0.99 * (incremental / 5.0) ** power
        left, right = str(edge["left"]), str(edge["right"])
        result.setdefault(left, []).append((right, similarity, "ecv_explicit"))
        result.setdefault(right, []).append((left, similarity, "ecv_explicit"))
    return result


def prizes(
    ids: Sequence[str], atomic_values: Sequence[Any], verifier: Mapping[str, Any],
    *, floor: float, a: float, b: float, c: float,
) -> dict[str, float]:
    return {
        mid: float(expit(a * (floor if atomic is None else float(atomic)) + b * float(verifier.get(mid, 0.0)) + c))
        for mid, atomic in zip(ids, atomic_values)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--ecv-scores", type=Path, required=True)
    parser.add_argument(
        "--question-reference", type=Path,
        help="Optional JSONL defining the intended question set when strict ECV rows may be missing.",
    )
    parser.add_argument("--original-verifier", type=Path, required=True)
    parser.add_argument("--source-priors", type=Path, required=True)
    parser.add_argument("--relation-store", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--root-cost", type=float, default=0.2)
    parser.add_argument("--edge-power", type=float, action="append")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    powers = args.edge_power or [1.0, 2.0]
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))["coefficients"]
    a, b, c = (float(calibration[key]) for key in ("a", "b", "c"))
    ecv = read_jsonl(args.ecv_scores, "question_id")
    reference = read_jsonl(args.question_reference, "question_id") if args.question_reference else ecv
    original = read_jsonl(args.original_verifier, "question_id")
    priors = read_jsonl(args.source_priors, "question_id")
    questions = read_jsonl(args.bundle / "questions.jsonl", "question_id")
    sessions = memory_sessions(args.bundle / "memories.jsonl")
    fixed = load_adjacency(args.relation_store, "explicit", 8)
    variants = ["original_fixed", "ecv_fixed", "atomic_only_fixed"]
    variants += [f"original_ecv_edge_p{power:g}" for power in powers]
    variants += [f"ecv_ecv_edge_p{power:g}" for power in powers]
    variants += [f"atomic_only_ecv_edge_p{power:g}" for power in powers]
    predictions: dict[str, dict[str, dict[str, Any]]] = {label: {} for label in variants}
    diagnostics: dict[str, list[dict[str, Any]]] = {label: [] for label in variants}
    qids = sorted(set(reference) & set(original) & set(priors) & set(questions))
    fallback_questions = 0
    for done, qid in enumerate(qids, 1):
        orow = original[qid]
        fallback = qid not in ecv
        if fallback:
            fallback_questions += 1
            erow = {
                "question_id": qid, "context_id": orow["context_id"],
                "candidate_ids": orow["candidate_ids"], "atomic_scores": orow["atomic_scores"],
                "direct_scores": {}, "edge_scores": {},
            }
        else:
            erow = ecv[qid]
        ids = [str(value) for value in erow["candidate_ids"]]
        if ids != [str(value) for value in orow["candidate_ids"]]:
            raise ValueError(f"candidate mismatch for {qid}")
        atomic_values = erow["atomic_scores"]
        floor = missing_atomic_floor(priors[qid])
        original_prizes = prizes(ids, atomic_values, orow.get("verifier_scores", {}), floor=floor, a=a, b=b, c=c)
        ecv_prizes = prizes(ids, atomic_values, erow.get("direct_scores", {}), floor=floor, a=a, b=b, c=c)
        atomic_only_prizes = prizes(ids, atomic_values, {}, floor=floor, a=a, b=b, c=c)
        configurations: list[tuple[str, Mapping[str, float], Mapping[str, Sequence[tuple[str, float, str]]]]] = [
            ("original_fixed", original_prizes, fixed), ("ecv_fixed", ecv_prizes, fixed),
            ("atomic_only_fixed", atomic_only_prizes, fixed),
        ]
        for power in powers:
            local = fixed if fallback else conditioned_adjacency(erow, power=power)
            configurations.extend((
                (f"original_ecv_edge_p{power:g}", original_prizes, local),
                (f"ecv_ecv_edge_p{power:g}", ecv_prizes, local),
                (f"atomic_only_ecv_edge_p{power:g}", atomic_only_prizes, local),
            ))
        for label, node_prizes, adjacency in configurations:
            selected, diagnostic = solve_forest_greedy(
                ids, node_prizes, adjacency, k=args.k,
                edge_weight=args.edge_weight, root_cost=args.root_cost,
            )
            predictions[label][qid] = {
                "question_id": qid, "context_id": erow["context_id"],
                "retrieved_memory_ids": selected, "metadata": diagnostic,
            }
            diagnostics[label].append(diagnostic)
        if done % 100 == 0 or done == len(qids):
            print(json.dumps({"event": "progress", "done": done, "total": len(qids)}), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "protocol": "graphmemix-ecv-controlled-1.0", "questions": len(qids),
        "ecv_format_fallback_questions": fallback_questions, "variants": {},
    }
    for label, values in predictions.items():
        with (args.output_dir / f"{label}.jsonl").open("w", encoding="utf-8") as handle:
            for row in values.values():
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        metric = metrics_for((questions[qid] for qid in values), values, args.k, sessions)
        ds = diagnostics[label]
        metric.update({
            "mean_chosen_edges": float(np.mean([d.get("chosen_edges", 0) for d in ds])),
            "mean_components": float(np.mean([d.get("components", args.k) for d in ds])),
            "mean_solver_seconds": float(np.mean([d.get("solver_seconds", 0.0) for d in ds])),
        })
        report["variants"][label] = metric
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
