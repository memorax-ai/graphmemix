#!/usr/bin/env python3
"""Build a query-independent GraphMemix RelationStore from frozen GME indexes.

The first (non-atomic) embedding unit of a text/image memory is its canonical
graph view.  Video memories have one such unit per sampled frame; their graph
vector is the normalized mean of those frame vectors.  Atomic text chunks are
deliberately excluded so memories with more chunks do not acquire more graph
mass or graph edges.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def read_jsonl(path: Path, key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            value = str(row[key])
            if value in rows:
                raise ValueError(f"duplicate {key}={value!r} at {path}:{line_number}")
            rows[value] = row
    return rows


def read_id_allowlist(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    values = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not values:
        raise ValueError(f"empty memory-ID allowlist: {path}")
    return values


def normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if norm <= 0:
        raise ValueError("zero graph vector")
    return value / norm


def canonical_unit_rows(units: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    """Map each memory to canonical rows, excluding Atomic text auxiliaries."""
    result: dict[str, list[int]] = defaultdict(list)
    for row_id, unit in enumerate(units):
        if unit.get("representation"):
            continue
        result[str(unit["memory_id"])].append(row_id)
    return dict(result)


def explicit_edges(memories: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Create only schema-supported reply/round and consecutive-turn edges."""
    rows = list(memories)
    edges: dict[tuple[str, str], dict[str, Any]] = {}

    def add(left: str, right: str, relation: str) -> None:
        if left == right:
            return
        key = tuple(sorted((left, right)))
        current = edges.setdefault(key, {
            "left": key[0], "right": key[1], "explicit_relations": [],
        })
        if relation not in current["explicit_relations"]:
            current["explicit_relations"].append(relation)

    by_round: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_session: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        context = str(row["context_id"])
        memory_id = str(row["memory_id"])
        session = str(row.get("session_id") or "")
        round_id = str(row.get("metadata", {}).get("native_round_id") or "")
        if round_id:
            by_round[(context, round_id)].append(memory_id)
        if session:
            by_session[(context, session)].append(row)
    for values in by_round.values():
        for index, left in enumerate(values):
            for right in values[index + 1:]:
                add(left, right, "same_round")
    for values in by_session.values():
        ordered = sorted(values, key=lambda row: (int(row.get("sequence", 0)), str(row["memory_id"])))
        for left, right in zip(ordered, ordered[1:]):
            add(str(left["memory_id"]), str(right["memory_id"]), "consecutive_turn")
    return edges


def _timestamp(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return parsed


def _location(row: Mapping[str, Any]) -> tuple[float, float] | None:
    metadata = row.get("metadata")
    raw = metadata.get("raw_metadata") if isinstance(metadata, Mapping) else None
    value = raw.get("location") if isinstance(raw, Mapping) else None
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) < 2
        or not all(isinstance(item, (int, float)) for item in value[:2])
    ):
        return None
    latitude, longitude = float(value[0]), float(value[1])
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None
    return latitude, longitude


def haversine_meters(
    left: tuple[float, float], right: tuple[float, float],
) -> float:
    radius = 6_371_000.0
    left_lat, left_lon = map(math.radians, left)
    right_lat, right_lon = map(math.radians, right)
    delta_lat = right_lat - left_lat
    delta_lon = right_lon - left_lon
    value = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(left_lat) * math.cos(right_lat)
        * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * radius * math.asin(min(1.0, math.sqrt(value)))


def spatiotemporal_edges(
    memories: Iterable[Mapping[str, Any]],
    *,
    max_gap_seconds: float,
    max_distance_meters: float,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Link consecutive media into sparse, query-independent event chains."""
    if max_gap_seconds <= 0 or max_distance_meters <= 0:
        return {}
    by_context: dict[str, list[tuple[dt.datetime, tuple[float, float], Mapping[str, Any]]]] = defaultdict(list)
    for row in memories:
        if str(row.get("kind") or "") != "media":
            continue
        timestamp = _timestamp(row.get("timestamp"))
        location = _location(row)
        if timestamp is None or location is None:
            continue
        by_context[str(row["context_id"])].append((timestamp, location, row))

    edges: dict[tuple[str, str], dict[str, Any]] = {}
    for values in by_context.values():
        values.sort(key=lambda value: (value[0], str(value[2]["memory_id"])))
        for left, right in zip(values, values[1:]):
            gap_seconds = (right[0] - left[0]).total_seconds()
            if gap_seconds < 0 or gap_seconds > max_gap_seconds:
                continue
            distance_meters = haversine_meters(left[1], right[1])
            if distance_meters > max_distance_meters:
                continue
            key = tuple(sorted((str(left[2]["memory_id"]), str(right[2]["memory_id"]))))
            edges[key] = {
                "left": key[0],
                "right": key[1],
                "explicit_relations": ["spatiotemporal_event"],
                "spatiotemporal_gap_seconds": gap_seconds,
                "spatiotemporal_distance_meters": distance_meters,
            }
    return edges


def merge_explicit_edges(
    target: dict[tuple[str, str], dict[str, Any]],
    source: Mapping[tuple[str, str], Mapping[str, Any]],
) -> None:
    for key, incoming in source.items():
        current = target.setdefault(
            key,
            {"left": key[0], "right": key[1], "explicit_relations": []},
        )
        for relation in incoming.get("explicit_relations", []):
            if relation not in current["explicit_relations"]:
                current["explicit_relations"].append(relation)
        for name, value in incoming.items():
            if name not in {"left", "right", "explicit_relations"}:
                current[name] = value


def reconstruct_vectors(index_path: Path, state: Mapping[str, Any]) -> tuple[list[str], np.ndarray]:
    import faiss  # Imported lazily so pure unit tests do not require FAISS.

    index = faiss.read_index(str(index_path))
    units = list(state["embedding_units"])
    if int(index.ntotal) != len(units):
        raise ValueError(f"index/unit mismatch: {index.ntotal} != {len(units)}")
    rows = canonical_unit_rows(units)
    memory_ids = sorted(rows)
    vectors: list[np.ndarray] = []
    for memory_id in memory_ids:
        frame_vectors = [normalize(index.reconstruct(int(row_id))) for row_id in rows[memory_id]]
        vectors.append(normalize(np.mean(frame_vectors, axis=0)))
    return memory_ids, np.asarray(vectors, dtype=np.float32)


def mutual_neighbors(vectors: np.ndarray, max_k: int) -> list[tuple[int, int, int, int, float]]:
    """Return (i, j, rank_i_to_j, rank_j_to_i, cosine) for mutual max-k pairs."""
    import faiss

    if max_k <= 0:
        return []
    search = faiss.IndexFlatIP(int(vectors.shape[1]))
    search.add(np.ascontiguousarray(vectors, dtype=np.float32))
    scores, indices = search.search(vectors, min(max_k + 1, len(vectors)))
    ranked: list[dict[int, tuple[int, float]]] = []
    for self_id, (row_scores, row_indices) in enumerate(zip(scores, indices)):
        current: dict[int, tuple[int, float]] = {}
        rank = 0
        for score, other in zip(row_scores, row_indices):
            other = int(other)
            if other < 0 or other == self_id:
                continue
            rank += 1
            if rank > max_k:
                break
            current[other] = (rank, float(score))
        ranked.append(current)
    result: list[tuple[int, int, int, int, float]] = []
    for left, values in enumerate(ranked):
        for right, (left_rank, score) in values.items():
            if right <= left or left not in ranked[right]:
                continue
            right_rank, reverse_score = ranked[right][left]
            result.append((left, right, left_rank, right_rank, (score + reverse_score) / 2.0))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--memory-ids", type=Path,
        help="optional memory-ID allowlist for a controlled subset profile",
    )
    parser.add_argument("--max-semantic-k", type=int, default=16)
    parser.add_argument("--explicit-similarity", type=float, default=0.99)
    parser.add_argument("--spatiotemporal-max-gap-seconds", type=float, default=0)
    parser.add_argument("--spatiotemporal-max-distance-meters", type=float, default=0)
    args = parser.parse_args()
    if args.max_semantic_k <= 0:
        parser.error("--max-semantic-k must be positive")
    if not 0 < args.explicit_similarity <= 1:
        parser.error("--explicit-similarity must be in (0, 1]")
    if (args.spatiotemporal_max_gap_seconds > 0) != (
        args.spatiotemporal_max_distance_meters > 0
    ):
        parser.error("both spatiotemporal thresholds must be positive or both zero")

    started = time.perf_counter()
    memories = read_jsonl(args.bundle / "memories.jsonl", "memory_id")
    allowlist = read_id_allowlist(args.memory_ids)
    if allowlist is not None:
        unknown = allowlist - memories.keys()
        if unknown:
            raise ValueError(f"allowlist contains {len(unknown)} unknown memory IDs")
        memories = {memory_id: memories[memory_id] for memory_id in allowlist}
    explicit = explicit_edges(memories.values())
    native_explicit_count = len(explicit)
    spatiotemporal = spatiotemporal_edges(
        memories.values(),
        max_gap_seconds=args.spatiotemporal_max_gap_seconds,
        max_distance_meters=args.spatiotemporal_max_distance_meters,
    )
    merge_explicit_edges(explicit, spatiotemporal)
    all_edges: dict[tuple[str, str], dict[str, Any]] = {key: dict(value) for key, value in explicit.items()}
    context_summaries: list[dict[str, Any]] = []
    seen_memories: set[str] = set()

    state_paths = sorted(args.checkpoint_root.glob("contexts/*/state.json"))
    if not state_paths:
        raise FileNotFoundError(f"no checkpoint states below {args.checkpoint_root}")
    for state_path in state_paths:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        index_path = state_path.parent / str(state["index_file"])
        memory_ids, vectors = reconstruct_vectors(index_path, state)
        unknown = set(memory_ids) - memories.keys()
        if unknown:
            raise ValueError(f"checkpoint contains {len(unknown)} unknown canonical memories")
        seen_memories.update(memory_ids)
        semantic = mutual_neighbors(vectors, args.max_semantic_k)
        for left_i, right_i, left_rank, right_rank, similarity in semantic:
            left, right = memory_ids[left_i], memory_ids[right_i]
            key = tuple(sorted((left, right)))
            row = all_edges.setdefault(key, {
                "left": key[0], "right": key[1], "explicit_relations": [],
            })
            row.update({
                "semantic_similarity": max(-1.0, min(1.0, similarity)),
                "semantic_rank_left": left_rank if left == key[0] else right_rank,
                "semantic_rank_right": right_rank if right == key[1] else left_rank,
            })
        context_summaries.append({
            "context_id": str(state["context_id"]),
            "memories": len(memory_ids), "semantic_mutual_edges_max_k": len(semantic),
        })
        print(json.dumps({"event": "context", **context_summaries[-1]}, ensure_ascii=False), flush=True)

    missing = memories.keys() - seen_memories
    if missing:
        raise ValueError(f"{len(missing)} bundle memories lack a canonical checkpoint vector")
    rows: list[dict[str, Any]] = []
    for key in sorted(all_edges):
        row = all_edges[key]
        explicit_present = bool(row.get("explicit_relations"))
        row["explicit_similarity"] = args.explicit_similarity if explicit_present else None
        row["explicit_cost"] = -math.log(args.explicit_similarity) if explicit_present else None
        similarity = row.get("semantic_similarity")
        row["semantic_cost"] = (
            -math.log(max(float(similarity), 1e-6))
            if similarity is not None and float(similarity) > 0 else None
        )
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "protocol": "graphmemix-relation-store-1.0",
        "bundle": str(args.bundle), "checkpoint_root": str(args.checkpoint_root),
        "memory_ids": str(args.memory_ids) if args.memory_ids else None,
        "max_semantic_k": args.max_semantic_k,
        "explicit_similarity": args.explicit_similarity,
        "spatiotemporal_max_gap_seconds": args.spatiotemporal_max_gap_seconds,
        "spatiotemporal_max_distance_meters": args.spatiotemporal_max_distance_meters,
        "memories": len(memories), "edges": len(rows),
        "native_explicit_edges": native_explicit_count,
        "spatiotemporal_edges": len(spatiotemporal),
        "explicit_edges": sum(bool(row.get("explicit_relations")) for row in rows),
        "semantic_edges": sum(row.get("semantic_similarity") is not None for row in rows),
        "contexts": context_summaries, "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
