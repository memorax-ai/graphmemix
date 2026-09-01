#!/usr/bin/env python3
"""Select ATM-Hard evidence with one qtype-only GraphMemix forest contract.

DeepSeek supplies a query-conditioned graph.  This selector deterministically
projects its unrestricted node and edge ontology onto memory nodes, then uses
the same GraphMemix 1-swap/Kruskal forest optimizer as the benchmark pipeline.
It never reads answers, gold evidence, question keywords, or scenario names.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

from probe_graphmemix_pcsf import solve_forest_greedy


PROTOCOL = "atm-hard-graphmemix-forest-v3"
NEGATIVE_STATUS = re.compile(
    r"excluded|rejected|contradict|not[_ -]?attended|irrelevant", re.IGNORECASE
)
SUPPORTED_QTYPES = {"list_recall", "number", "open_end"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def clamp_confidence(value: Any, *, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    return min(1.0, max(0.0, number))


def active(row: Mapping[str, Any]) -> bool:
    return not NEGATIVE_STATUS.search(str(row.get("status") or ""))


def evidential_role_bonus(row: Mapping[str, Any], qtype: str) -> float:
    """Reward generic graph roles without inspecting the question text."""
    kind = str(row.get("node_kind") or "").lower()
    relevance = str(row.get("question_relevance") or "").lower()
    bonus = 0.0
    if "answer" in kind:
        bonus = max(bonus, 0.40)
    elif "claim" in kind:
        bonus = max(bonus, 0.20)
    if "direct answer" in relevance:
        bonus = max(bonus, 0.30)
    elif "answer" in relevance:
        bonus = max(bonus, 0.10)
    if qtype == "list_recall" and (
        "collection" in kind or "answer_set" in kind or "answer set" in kind
    ):
        bonus = max(bonus, 0.30)
    return bonus


def load_hard_questions(path: Path) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []
    for row in read_jsonl(path):
        if str(row.get("subset") or "") != "hard":
            continue
        qtype = str((row.get("task") or {}).get("subcategory") or "")
        if qtype not in SUPPORTED_QTYPES:
            raise ValueError(f"unsupported ATM-Hard qtype {qtype!r}")
        questions.append({
            "question_id": str(row["question_id"]),
            "native_id": str(row["question_id"]).rsplit(":", 1)[-1],
            "context_id": str(row["context_id"]),
            "qtype": qtype,
        })
    return questions


class MemoryResolver:
    def __init__(self, memories: Sequence[Mapping[str, Any]]) -> None:
        self.by_memory_id = {str(row["memory_id"]): row for row in memories}
        self.by_source_id = {str(row["source_id"]): row for row in memories}

    def resolve(self, value: Any) -> str | None:
        raw = str(value or "")
        if raw in self.by_memory_id:
            return raw
        if raw in self.by_source_id:
            return str(self.by_source_id[raw]["memory_id"])
        stem = os.path.splitext(os.path.basename(raw))[0]
        if stem in self.by_source_id:
            return str(self.by_source_id[stem]["memory_id"])
        return None

    def modality(self, memory_id: str) -> str:
        row = self.by_memory_id[memory_id]
        if ":email:" in memory_id:
            return "email"
        if ":image:" in memory_id:
            return "image"
        if ":video:" in memory_id:
            return "video"
        return str(row.get("modality") or "text")


def resolved_supports(row: Mapping[str, Any], resolver: MemoryResolver) -> set[str]:
    values = row.get("support_memory_ids", [])
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return set()
    return {memory_id for value in values if (memory_id := resolver.resolve(value))}


def structural_coverage(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    resolver: MemoryResolver,
    *,
    qtype: str,
) -> dict[str, float]:
    """Measure support across graph structures without privileging hub size."""
    totals: dict[str, float] = defaultdict(float)
    structures: list[tuple[set[str], float]] = []
    for row in nodes:
        if not active(row):
            continue
        supports = resolved_supports(row, resolver)
        if row.get("is_memory"):
            memory_id = resolver.resolve(row.get("memory_id") or row.get("node_id"))
            if memory_id:
                supports.add(memory_id)
        if supports:
            structures.append((
                supports,
                clamp_confidence(row.get("confidence"))
                + evidential_role_bonus(row, qtype),
            ))
    for row in edges:
        if active(row) and (supports := resolved_supports(row, resolver)):
            structures.append((supports, clamp_confidence(row.get("confidence"))))
    for supports, confidence in structures:
        contribution = confidence / math.sqrt(len(supports))
        for memory_id in supports:
            totals[memory_id] += contribution
    maximum = max(totals.values(), default=0.0)
    if maximum <= 0:
        return dict(totals)
    denominator = math.log1p(maximum)
    return {
        memory_id: math.log1p(value) / denominator
        for memory_id, value in totals.items()
    }


def edge_incidence_scores(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    resolver: MemoryResolver,
    *,
    qtype: str,
) -> dict[str, float]:
    """Score memories used directly as endpoints of graph-structure edges.

    Relation labels are deliberately ignored: the graph builder may use any
    ontology.  Only endpoint resolution, active status, confidence, and the
    generic evidential role of the opposite graph node affect this score.
    """
    by_node = {str(row["node_id"]): row for row in nodes}
    totals: dict[str, float] = defaultdict(float)
    for edge in edges:
        if not active(edge):
            continue
        for memory_endpoint, graph_endpoint in (
            (edge.get("source"), edge.get("target")),
            (edge.get("target"), edge.get("source")),
        ):
            memory_id = resolver.resolve(memory_endpoint)
            graph_node = by_node.get(str(graph_endpoint or ""))
            if (
                memory_id is None
                or graph_node is None
                or graph_node.get("is_memory")
                or not active(graph_node)
            ):
                continue
            totals[memory_id] += (
                clamp_confidence(edge.get("confidence"))
                + evidential_role_bonus(graph_node, qtype)
            )
    maximum = max(totals.values(), default=0.0)
    if maximum <= 0:
        return dict(totals)
    return {memory_id: value / maximum for memory_id, value in totals.items()}


def graph_obligation_ledger(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    resolver: MemoryResolver,
    selected: Sequence[str],
    *,
    qtype: str,
) -> str:
    """Describe multi-component numeric scope without copying graph claims.

    A ledger is emitted only if one aggregate node fans out, through distinct
    first-hop branches, to at least two same-role temporal structures at the
    same graph distance.  This distinguishes aggregation from a single event
    chain without reading question text or relation names.
    """
    if qtype != "number":
        return ""
    by_node = {str(row["node_id"]): row for row in nodes}
    adjacency: dict[str, set[str]] = defaultdict(set)
    incident_selected: dict[str, set[str]] = defaultdict(set)
    chosen = set(selected)
    for edge in edges:
        if not active(edge):
            continue
        source, target = str(edge.get("source") or ""), str(edge.get("target") or "")
        if not source or not target:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)
        direct = {
            memory_id for endpoint in (source, target)
            if (memory_id := resolver.resolve(endpoint))
        }
        overlap = (resolved_supports(edge, resolver) | direct) & chosen
        incident_selected[source].update(overlap)
        incident_selected[target].update(overlap)

    aggregate = [
        node_id for node_id, row in by_node.items()
        if active(row) and any(
            token in str(row.get("node_kind") or "").lower()
            for token in ("answer", "claim")
        )
    ]
    best: tuple[tuple[int, int], list[tuple[Any, ...]]] | None = None
    excluded_kinds = ("answer", "claim", "question", "ambiguity", "exclusion", "hypothesis")
    for root_id in aggregate:
        distances = {root_id: 0}
        first_hops = {root_id: {root_id}}
        queue = deque([root_id])
        while queue:
            node_id = queue.popleft()
            if distances[node_id] >= 3:
                continue
            for neighbor in adjacency[node_id]:
                candidate_distance = distances[node_id] + 1
                propagated = {neighbor} if node_id == root_id else first_hops[node_id]
                if neighbor not in distances or candidate_distance < distances[neighbor]:
                    distances[neighbor] = candidate_distance
                    first_hops[neighbor] = set(propagated)
                    queue.append(neighbor)
                elif candidate_distance == distances[neighbor]:
                    first_hops[neighbor].update(propagated)

        candidates: list[tuple[Any, ...]] = []
        for node_id, distance in distances.items():
            row = by_node.get(node_id)
            if row is None or row.get("is_memory") or not active(row):
                continue
            kind = str(row.get("node_kind") or "").lower()
            if any(token in kind for token in excluded_kinds):
                continue
            confidence = clamp_confidence(row.get("confidence"))
            overlap = (
                resolved_supports(row, resolver) | incident_selected[node_id]
            ) & chosen
            if confidence < 0.7 or not overlap:
                continue
            priority = (
                distance, -confidence, str(row.get("timestamp_or_range") or ""), node_id
            )
            candidates.append((priority, row, overlap, first_hops[node_id]))

        branch_groups: dict[tuple[int, str], set[str]] = defaultdict(set)
        for priority, row, _, branches in candidates:
            if not row.get("timestamp_or_range"):
                continue
            branch_groups[(priority[0], str(row.get("node_kind") or "").lower())].update(branches)
        branch_count = max((len(value) for value in branch_groups.values()), default=0)
        score = (branch_count, len(candidates))
        if branch_count >= 2 and (best is None or score > best[0]):
            best = (score, candidates)
    if best is None:
        return ""

    lines = [
        "Graph coverage ledger (organizational only; calculate all values from raw evidence):"
    ]
    for _, row, overlap, _ in sorted(best[1])[:20]:
        line = f"- {str(row.get('label') or row.get('node_id') or 'component')}"
        if row.get("timestamp_or_range"):
            line += f" | time={row['timestamp_or_range']}"
        supports = ",".join(sorted(mid.rsplit(":", 1)[-1] for mid in overlap))
        line += f" | selected_supports={supports}"
        lines.append(line)
    return "\n".join(lines)[:3500]


def graph_timeline_context(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    resolver: MemoryResolver,
    selected: Sequence[str],
    *,
    qtype: str,
) -> str:
    """Expose a bounded timeline only when aggregate graph claims lack support.

    The gate uses qtype and graph topology, never question text.  It addresses
    aggregate numeric questions where raw endpoint memories alone cannot carry
    the graph builder's multi-interval decomposition.
    """
    if qtype != "number":
        return ""
    aggregate = [
        row for row in nodes
        if active(row) and any(
            token in str(row.get("node_kind") or "").lower()
            for token in ("answer", "claim")
        )
    ]
    temporal = [
        row for row in nodes
        if active(row)
        and not row.get("is_memory")
        and row.get("timestamp_or_range")
        and not any(
            token in str(row.get("node_kind") or "").lower()
            for token in ("answer", "claim")
        )
    ]
    if not aggregate or any(resolved_supports(row, resolver) for row in aggregate):
        return ""
    if len(temporal) < 2:
        return ""

    chosen = set(selected)
    incident: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if not active(edge):
            continue
        overlap = resolved_supports(edge, resolver) & chosen
        for endpoint in (str(edge.get("source") or ""), str(edge.get("target") or "")):
            incident[endpoint].update(overlap)
    ranked = []
    for row in temporal:
        overlap = (resolved_supports(row, resolver) | incident[str(row.get("node_id") or "")]) & chosen
        ranked.append((
            (
                clamp_confidence(row.get("confidence")),
                len(overlap),
                str(row.get("timestamp_or_range")),
                str(row.get("node_id") or ""),
            ),
            row,
            overlap,
        ))
    lines = [
        "Graph timeline (an organizational hypothesis; verify against raw memories):"
    ]
    for _, row, overlap in sorted(ranked, key=lambda value: value[0], reverse=True)[:24]:
        label = str(row.get("label") or row.get("node_id") or "event")
        description = str(row.get("description") or "")
        supports = ",".join(sorted(mid.rsplit(":", 1)[-1] for mid in overlap))
        line = f"- {row.get('timestamp_or_range')} | {label}"
        if description:
            line += f" | {description}"
        line += f" | confidence={clamp_confidence(row.get('confidence'))}"
        if supports:
            line += f" | selected_supports={supports}"
        lines.append(line)
    return "\n".join(lines)[:4000]


def project_evidence_graph(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    resolver: MemoryResolver,
    *,
    qtype: str,
) -> tuple[list[str], dict[str, float], dict[str, list[tuple[str, float, str]]], dict[str, Any]]:
    """Project an unrestricted evidence graph into a weighted memory graph."""
    by_node = {str(row["node_id"]): row for row in nodes}
    members: dict[str, set[str]] = {}
    contributions: dict[str, list[float]] = defaultdict(list)
    provenance: dict[str, set[str]] = defaultdict(set)

    for node_id, row in by_node.items():
        node_members = resolved_supports(row, resolver) if active(row) else set()
        if row.get("is_memory") and active(row):
            memory_id = resolver.resolve(row.get("memory_id") or node_id)
            if memory_id:
                node_members.add(memory_id)
                contributions[memory_id].append(
                    clamp_confidence(row.get("confidence"))
                    + evidential_role_bonus(row, qtype)
                )
                provenance[memory_id].add(f"memory:{node_id}")
        if not row.get("is_memory") and active(row):
            confidence = (
                clamp_confidence(row.get("confidence"))
                + evidential_role_bonus(row, qtype)
            )
            for memory_id in node_members:
                contributions[memory_id].append(confidence)
                provenance[memory_id].add(f"hub:{node_id}")
        members[node_id] = node_members

    active_edges: list[Mapping[str, Any]] = []
    for edge in edges:
        if not active(edge):
            continue
        confidence = clamp_confidence(edge.get("confidence"))
        edge_members = resolved_supports(edge, resolver)
        for memory_id in edge_members:
            contributions[memory_id].append(confidence)
            provenance[memory_id].add(f"edge:{edge.get('edge_id', '')}")
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source in members:
            members[source].update(edge_members)
        if target in members:
            members[target].update(edge_members)
        active_edges.append(edge)

    prizes = {
        memory_id: max(values) if values else 0.5
        for memory_id, values in contributions.items()
    }
    if not prizes:
        raise ValueError("evidence graph resolves to no active bundle memories")
    ordered = sorted(prizes, key=lambda memory_id: (-prizes[memory_id], memory_id))

    projected: dict[tuple[str, str], tuple[float, str]] = {}

    def add_edge(left: str, right: str, similarity: float, relation: str) -> None:
        if left == right or left not in prizes or right not in prizes:
            return
        key = tuple(sorted((left, right)))
        candidate = (min(0.999, max(1e-6, similarity)), relation)
        current = projected.get(key)
        if current is None or candidate[0] > current[0]:
            projected[key] = candidate

    # A hub with several supporting memories becomes a deterministic star.
    # This preserves its connectivity without manufacturing an O(n^2) clique.
    for node_id, node_members in members.items():
        usable = sorted(
            (memory_id for memory_id in node_members if memory_id in prizes),
            key=lambda memory_id: (-prizes[memory_id], memory_id),
        )
        if len(usable) < 2:
            continue
        row = by_node[node_id]
        confidence = clamp_confidence(row.get("confidence"))
        specificity = math.sqrt(max(1.0, (len(usable) - 1) / 2.0))
        confidence /= specificity
        relation = f"hub:{str(row.get('node_kind') or 'concept')}"
        anchor = usable[0]
        for memory_id in usable[1:]:
            add_edge(anchor, memory_id, confidence, relation)

    # Original graph edges connect one representative memory from each side.
    # All relation names are accepted; only negative evidential status is gated.
    for edge in active_edges:
        left_members = [value for value in members.get(str(edge.get("source")), set()) if value in prizes]
        right_members = [value for value in members.get(str(edge.get("target")), set()) if value in prizes]
        if not left_members or not right_members:
            continue
        key = lambda memory_id: (-prizes[memory_id], memory_id)
        left = min(left_members, key=key)
        right = min(right_members, key=key)
        add_edge(
            left,
            right,
            clamp_confidence(edge.get("confidence")),
            f"graph:{str(edge.get('relation') or 'related')}",
        )

    adjacency: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
    for (left, right), (similarity, relation) in sorted(projected.items()):
        adjacency[left].append((right, similarity, relation))
        adjacency[right].append((left, similarity, relation))
    for memory_id in adjacency:
        adjacency[memory_id].sort(key=lambda value: (-value[1], value[0], value[2]))
    diagnostic = {
        "active_graph_nodes": sum(active(row) for row in nodes),
        "active_graph_edges": len(active_edges),
        "resolved_memory_candidates": len(ordered),
        "projected_memory_edges": len(projected),
        "provenance_counts": {memory_id: len(provenance[memory_id]) for memory_id in ordered},
    }
    return ordered, prizes, dict(adjacency), diagnostic


def representation_actions(
    selected: Sequence[str],
    prizes: Mapping[str, float],
    resolver: MemoryResolver,
    *,
    high_media_limit: int,
) -> list[dict[str, str]]:
    media = [
        memory_id for memory_id in selected
        if resolver.modality(memory_id) in {"image", "video"}
    ]
    high = set(sorted(media, key=lambda memory_id: (-prizes[memory_id], memory_id))[:high_media_limit])
    return [
        {
            "memory_id": memory_id,
            "action": "high" if memory_id in high else "text",
        }
        for memory_id in selected
    ]


def stable_representation_actions(
    selected: Sequence[str],
    baseline_selected: Sequence[str],
    baseline_prizes: Mapping[str, float],
    resolver: MemoryResolver,
    *,
    high_media_limit: int,
) -> list[dict[str, str]]:
    """Preserve baseline presentation choices after a membership-only bonus."""
    baseline = {
        row["memory_id"]: row["action"]
        for row in representation_actions(
            baseline_selected,
            baseline_prizes,
            resolver,
            high_media_limit=high_media_limit,
        )
    }
    high_count = sum(baseline.get(memory_id) == "high" for memory_id in selected)
    actions: list[dict[str, str]] = []
    for memory_id in selected:
        action = baseline.get(memory_id)
        if action is None:
            if (
                resolver.modality(memory_id) in {"image", "video"}
                and high_count < high_media_limit
            ):
                action = "high"
                high_count += 1
            else:
                action = "text"
        actions.append({"memory_id": memory_id, "action": action})
    return actions


def select_question(
    graph_dir: Path,
    question: Mapping[str, str],
    resolver: MemoryResolver,
    *,
    budgets: Mapping[str, int],
    high_media_limits: Mapping[str, int],
    source_prior: Mapping[str, Any],
    prior_limits: Mapping[str, int],
    prior_weights: Mapping[str, float],
    coverage_weights: Mapping[str, float],
    incidence_weights: Mapping[str, float],
    edge_weight: float,
    root_cost: float,
) -> dict[str, Any]:
    nodes = read_jsonl(graph_dir / "graph_nodes.jsonl")
    edges = read_jsonl(graph_dir / "graph_edges.jsonl")
    manifest = json.loads((graph_dir / "graph_manifest.json").read_text(encoding="utf-8"))
    if str(manifest.get("question_id") or "").rsplit(":", 1)[-1] != question["native_id"]:
        raise ValueError(f"{question['native_id']}: graph manifest question mismatch")
    candidates, prizes, adjacency, projection = project_evidence_graph(
        nodes, edges, resolver, qtype=question["qtype"]
    )
    qtype = question["qtype"]
    coverage = structural_coverage(nodes, edges, resolver, qtype=qtype)
    prior_limit = int(prior_limits[qtype])
    prior_ids = source_prior.get("retrieval_ids", [])
    if not isinstance(prior_ids, Sequence) or isinstance(prior_ids, (str, bytes)):
        raise ValueError(f"{question['question_id']}: invalid source prior retrieval_ids")
    prior_rank = {
        memory_id: rank
        for rank, value in enumerate(prior_ids[:prior_limit], 1)
        if (memory_id := resolver.resolve(value))
    }
    for memory_id in prior_rank:
        if memory_id not in prizes:
            candidates.append(memory_id)
            prizes[memory_id] = 0.5
    for memory_id in candidates:
        rank_bonus = 0.0
        if memory_id in prior_rank:
            rank_bonus = 1.0 - (prior_rank[memory_id] - 1) / max(1, prior_limit)
        prizes[memory_id] += (
            float(prior_weights[qtype]) * rank_bonus
            + float(coverage_weights[qtype]) * coverage.get(memory_id, 0.0)
        )
    presentation_prizes = dict(prizes)
    incidence = edge_incidence_scores(nodes, edges, resolver, qtype=qtype)
    incidence_weight = float(incidence_weights[qtype])
    for memory_id in candidates:
        prizes[memory_id] += incidence_weight * incidence.get(memory_id, 0.0)
    k = min(int(budgets[question["qtype"]]), len(candidates))
    baseline_selected: list[str] = []
    baseline_forest: dict[str, Any] = {}
    if incidence_weight > 0:
        baseline_selected, baseline_forest = solve_forest_greedy(
            candidates,
            presentation_prizes,
            adjacency,
            k=k,
            edge_weight=edge_weight,
            root_cost=root_cost,
        )
        baseline_forest.pop("solver_seconds", None)
    selected, forest = solve_forest_greedy(
        candidates,
        prizes,
        adjacency,
        k=k,
        edge_weight=edge_weight,
        root_cost=root_cost,
    )
    # Wall-clock timing is useful while profiling but is not part of the
    # frozen retrieval contract; retaining it would make identical replays
    # differ byte-for-byte.
    forest.pop("solver_seconds", None)
    if incidence_weight > 0:
        chosen = set(selected)
        stable_selected = [memory_id for memory_id in baseline_selected if memory_id in chosen]
        already = set(stable_selected)
        stable_selected.extend(memory_id for memory_id in selected if memory_id not in already)
        selected = stable_selected
        actions = stable_representation_actions(
            selected,
            baseline_selected,
            presentation_prizes,
            resolver,
            high_media_limit=int(high_media_limits[qtype]),
        )
    else:
        actions = representation_actions(
            selected,
            prizes,
            resolver,
            high_media_limit=int(high_media_limits[qtype]),
        )
    timeline = graph_timeline_context(nodes, edges, resolver, selected, qtype=qtype)
    ledger = graph_obligation_ledger(nodes, edges, resolver, selected, qtype=qtype)
    graph_context = "\n\n".join(value for value in (timeline, ledger) if value)[:4000]
    metadata = {
        "protocol": PROTOCOL,
        "uses_gold": False,
        "routing_signal": "canonical_qtype_only",
        "qtype": question["qtype"],
        "candidate_k": len(candidates),
        "budget_k": k,
        "selected_k": len(selected),
        "selected_actions": actions,
        "selected_prizes": {memory_id: prizes[memory_id] for memory_id in selected},
        "prize_contract": {
            "source_prior_limit": prior_limit,
            "source_prior_weight": float(prior_weights[qtype]),
            "structural_coverage_weight": float(coverage_weights[qtype]),
            "edge_incidence_weight": incidence_weight,
        },
        "projection": projection,
        "forest": {
            **forest,
            "edge_weight": edge_weight,
            "root_cost": root_cost,
        },
    }
    if incidence_weight > 0:
        metadata["forest"]["presentation_baseline_chosen_edges"] = baseline_forest.get(
            "chosen_edges", []
        )
    if graph_context:
        metadata["derived_graph_evidence"] = graph_context
        metadata["graph_context_gates"] = [
            name for name, value in (
                ("unsupported_aggregate_with_temporal_structure", timeline),
                ("multi_branch_coverage_obligation", ledger),
            ) if value
        ]
    return {
        "question_id": question["question_id"],
        "context_id": question["context_id"],
        "retrieved_memory_ids": selected,
        "metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--questions-jsonl", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--source-priors", type=Path, required=True)
    parser.add_argument("--list-k", type=int, default=16)
    parser.add_argument("--number-k", type=int, default=10)
    parser.add_argument("--open-k", type=int, default=16)
    parser.add_argument("--list-high-media", type=int, default=8)
    parser.add_argument("--number-high-media", type=int, default=6)
    parser.add_argument("--open-high-media", type=int, default=6)
    parser.add_argument("--list-prior-limit", type=int, default=8)
    parser.add_argument("--number-prior-limit", type=int, default=4)
    parser.add_argument("--open-prior-limit", type=int, default=0)
    parser.add_argument("--list-prior-weight", type=float, default=0.75)
    parser.add_argument("--number-prior-weight", type=float, default=0.75)
    parser.add_argument("--open-prior-weight", type=float, default=0.0)
    parser.add_argument("--list-coverage-weight", type=float, default=0.2)
    parser.add_argument("--number-coverage-weight", type=float, default=0.2)
    parser.add_argument("--open-coverage-weight", type=float, default=0.2)
    parser.add_argument("--list-incidence-weight", type=float, default=0.25)
    parser.add_argument("--number-incidence-weight", type=float, default=0.0)
    parser.add_argument("--open-incidence-weight", type=float, default=0.0)
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--root-cost", type=float, default=0.12)
    args = parser.parse_args()
    positive_values = (
        args.list_k, args.number_k, args.open_k,
        args.list_high_media, args.number_high_media, args.open_high_media,
    )
    if any(value <= 0 for value in positive_values):
        raise ValueError("evidence and media budgets must be positive")
    non_negative_values = (
        args.list_prior_limit, args.number_prior_limit, args.open_prior_limit,
        args.list_prior_weight, args.number_prior_weight, args.open_prior_weight,
        args.list_coverage_weight, args.number_coverage_weight, args.open_coverage_weight,
        args.list_incidence_weight, args.number_incidence_weight, args.open_incidence_weight,
    )
    if any(value < 0 for value in non_negative_values):
        raise ValueError("prior, structural coverage, and incidence settings must be non-negative")
    if args.edge_weight < 0 or args.root_cost < 0:
        raise ValueError("forest costs must be non-negative")

    questions = load_hard_questions(args.questions_jsonl)
    memories = read_jsonl(args.bundle / "memories.jsonl")
    resolver = MemoryResolver(memories)
    source_priors = {
        str(row["question_id"]): row for row in read_jsonl(args.source_priors)
    }
    missing_priors = {
        question["question_id"] for question in questions
        if question["question_id"] not in source_priors
    }
    if missing_priors:
        raise ValueError(f"source priors missing {len(missing_priors)} ATM-Hard questions")
    budgets = {"list_recall": args.list_k, "number": args.number_k, "open_end": args.open_k}
    media = {
        "list_recall": args.list_high_media,
        "number": args.number_high_media,
        "open_end": args.open_high_media,
    }
    prior_limits = {
        "list_recall": args.list_prior_limit,
        "number": args.number_prior_limit,
        "open_end": args.open_prior_limit,
    }
    prior_weights = {
        "list_recall": args.list_prior_weight,
        "number": args.number_prior_weight,
        "open_end": args.open_prior_weight,
    }
    coverage_weights = {
        "list_recall": args.list_coverage_weight,
        "number": args.number_coverage_weight,
        "open_end": args.open_coverage_weight,
    }
    incidence_weights = {
        "list_recall": args.list_incidence_weight,
        "number": args.number_incidence_weight,
        "open_end": args.open_incidence_weight,
    }
    output: list[dict[str, Any]] = []
    for question in questions:
        graph_dir = args.graph_root / question["native_id"] / "workspace-output"
        row = select_question(
            graph_dir,
            question,
            resolver,
            budgets=budgets,
            high_media_limits=media,
            source_prior=source_priors[question["question_id"]],
            prior_limits=prior_limits,
            prior_weights=prior_weights,
            coverage_weights=coverage_weights,
            incidence_weights=incidence_weights,
            edge_weight=args.edge_weight,
            root_cost=args.root_cost,
        )
        output.append(row)
        print(json.dumps({
            "question_id": question["question_id"],
            "qtype": question["qtype"],
            "candidate_k": row["metadata"]["candidate_k"],
            "selected_k": row["metadata"]["selected_k"],
            "chosen_edges": row["metadata"]["forest"]["chosen_edges"],
        }, ensure_ascii=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output),
        encoding="utf-8",
    )
    audit = {
        "protocol": PROTOCOL,
        "uses_gold": False,
        "routing_signal": "canonical_qtype_only",
        "questions": len(output),
        "budgets": budgets,
        "high_media_limits": media,
        "source_priors": str(args.source_priors),
        "prior_limits": prior_limits,
        "prior_weights": prior_weights,
        "coverage_weights": coverage_weights,
        "incidence_weights": incidence_weights,
        "edge_weight": args.edge_weight,
        "root_cost": args.root_cost,
        "records": output,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"questions": len(output), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
