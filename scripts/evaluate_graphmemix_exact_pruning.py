#!/usr/bin/env python3
"""Exactly prune a frozen GraphMemix top-K set under a variable-cardinality utility."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_atm_retrieval import memory_sessions, metrics_for, read_jsonl
from graphmemix_core import missing_atomic_floor, source_score_map


POSITIVE_ROLES = {"new_fact", "clarification", "corroboration"}


def expit(value: float) -> float:
    """Numerically stable scalar sigmoid without a SciPy runtime dependency."""
    if value >= 0:
        term = math.exp(-value)
        return 1.0 / (1.0 + term)
    term = math.exp(value)
    return term / (1.0 + term)


def positive_edges(
    row: Mapping[str, Any], selected: set[str], *, power: float,
) -> list[tuple[float, str, str, str]]:
    """Return edge root-cost savings inputs restricted to the frozen top-K set."""
    result: list[tuple[float, str, str, str]] = []
    for edge in row.get("edge_scores", {}).values():
        role = str(edge.get("role", ""))
        if role not in POSITIVE_ROLES:
            continue
        incremental = max(0.0, min(5.0, float(edge.get("incremental_support", 0.0))))
        left, right = str(edge["left"]), str(edge["right"])
        if incremental <= 0 or left == right or left not in selected or right not in selected:
            continue
        similarity = 0.99 * (incremental / 5.0) ** power
        result.append((similarity, left, right, role))
    return result


def exact_subset(
    ordered_ids: Sequence[str],
    prizes: Mapping[str, float],
    edges: Sequence[tuple[float, str, str, str]],
    *,
    edge_weight: float,
    root_cost: float,
    node_cost: float,
) -> tuple[list[str], dict[str, Any]]:
    """Enumerate all nonempty subsets; Kruskal gives the exact forest per subset."""
    ids = list(dict.fromkeys(map(str, ordered_ids)))
    n = len(ids)
    if n > 20:
        raise ValueError("exact subset enumeration is intended for a frozen small top-K set")
    index = {memory_id: i for i, memory_id in enumerate(ids)}
    weighted_edges: list[tuple[float, int, int, str]] = []
    for similarity, left, right, role in edges:
        if left not in index or right not in index:
            continue
        # Joining two components saves one root cost but incurs edge cost.
        benefit = root_cost - edge_weight * -math.log(max(similarity, 1e-6))
        if benefit > 0:
            weighted_edges.append((benefit, index[left], index[right], role))
    weighted_edges.sort(key=lambda value: (-value[0], value[1], value[2], value[3]))

    best_value = -math.inf
    best_mask = 0
    best_edges: list[tuple[int, int, str]] = []
    for mask in range(1, 1 << n):
        parent = list(range(n))

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        chosen: list[tuple[int, int, str]] = []
        edge_benefit = 0.0
        for benefit, left, right, role in weighted_edges:
            if not (mask >> left & 1 and mask >> right & 1):
                continue
            left_root, right_root = find(left), find(right)
            if left_root == right_root:
                continue
            parent[right_root] = left_root
            edge_benefit += benefit
            chosen.append((left, right, role))
        members = [i for i in range(n) if mask >> i & 1]
        value = (
            sum(float(prizes[ids[i]]) - node_cost - root_cost for i in members)
            + edge_benefit
        )
        # Deterministic tie break: prefer fewer nodes, then the earlier frozen ranks.
        candidate_key = (value, -len(members), tuple(-i for i in members))
        best_members = [i for i in range(n) if best_mask >> i & 1]
        best_key = (best_value, -len(best_members), tuple(-i for i in best_members))
        if candidate_key > best_key:
            best_value, best_mask, best_edges = value, mask, chosen

    kept = [ids[i] for i in range(n) if best_mask >> i & 1]
    return kept, {
        "objective_value": best_value,
        "original_k": n,
        "selected_k": len(kept),
        "chosen_edges": len(best_edges),
        "components": len(kept) - len(best_edges),
        "node_cost": node_cost,
        "root_cost": root_cost,
        "edge_weight": edge_weight,
        "exact": True,
        "exact_scope": "frozen_input_topk",
        "globally_exact_over_candidate_pool": False,
        "search_space": (1 << n) - 1,
    }


def calibrated_prizes(
    score_row: Mapping[str, Any] | None,
    prior_row: Mapping[str, Any],
    requested_ids: Sequence[str],
    *,
    a: float,
    b: float,
    c: float,
    score_field: str,
) -> dict[str, float]:
    floor = missing_atomic_floor(prior_row)
    source = source_score_map(prior_row)
    if score_row is None:
        candidate_ids = list(requested_ids)
        atomic_scores = [source.get(memory_id) for memory_id in candidate_ids]
        verifier: Mapping[str, Any] = {}
    else:
        candidate_ids = [str(value) for value in score_row["candidate_ids"]]
        atomic_scores = score_row["atomic_scores"]
        verifier = score_row.get(score_field, {})
    return {
        str(memory_id): float(expit(
            a * (floor if atomic is None else float(atomic))
            + b * float(verifier.get(str(memory_id), 0.0))
            + c
        ))
        for memory_id, atomic in zip(candidate_ids, atomic_scores)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--ecv-scores", type=Path, required=True)
    parser.add_argument("--source-priors", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-field", default="verifier_scores")
    parser.add_argument("--node-cost", type=float, action="append")
    parser.add_argument("--root-cost", type=float, action="append")
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--edge-power", type=float, default=1.0)
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))["coefficients"]
    a, b, c = (float(calibration[key]) for key in ("a", "b", "c"))
    scores = read_jsonl(args.scores, "question_id")
    ecv = read_jsonl(args.ecv_scores, "question_id")
    priors = read_jsonl(args.source_priors, "question_id")
    retrieval = read_jsonl(args.retrieval, "question_id")
    questions = read_jsonl(args.bundle / "questions.jsonl", "question_id")
    sessions = memory_sessions(args.bundle / "memories.jsonl")
    qids = sorted(set(priors) & set(retrieval) & set(questions))
    node_costs = args.node_cost if args.node_cost is not None else [0.0]
    root_costs = args.root_cost if args.root_cost is not None else [0.2]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "protocol": "graphmemix-exact-topk-pruning-1.1",
        "optimization_scope": "all nonempty subsets of the supplied frozen Top-K",
        "globally_exact_over_candidate_pool": False,
        "questions": len(qids),
        "variants": {},
    }
    for root_cost, node_cost in itertools.product(root_costs, node_costs):
        label = f"root{root_cost:g}_node{node_cost:g}"
        output: dict[str, dict[str, Any]] = {}
        lengths: list[int] = []
        for qid in qids:
            ids = [str(value) for value in retrieval[qid]["retrieved_memory_ids"]]
            prize = calibrated_prizes(
                scores.get(qid), priors[qid], ids,
                a=a, b=b, c=c, score_field=args.score_field,
            )
            local_edges = positive_edges(ecv.get(qid, {}), set(ids), power=args.edge_power)
            selected, diagnostic = exact_subset(
                ids, prize, local_edges, edge_weight=args.edge_weight,
                root_cost=root_cost, node_cost=node_cost,
            )
            output[qid] = {
                "question_id": qid,
                "context_id": retrieval[qid]["context_id"],
                "retrieved_memory_ids": selected,
                "metadata": diagnostic,
            }
            lengths.append(len(selected))
        path = args.output_dir / f"{label}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for qid in qids:
                handle.write(json.dumps(output[qid], ensure_ascii=False) + "\n")
        metric = metrics_for((questions[qid] for qid in qids), output, 10, sessions)
        metric.update({
            "mean_selected": float(np.mean(lengths)),
            "median_selected": float(np.median(lengths)),
            "selected_distribution": dict(sorted(Counter(lengths).items())),
            "output": str(path),
        })
        report["variants"][label] = metric
        print(json.dumps({"variant": label, **metric}, ensure_ascii=False), flush=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
