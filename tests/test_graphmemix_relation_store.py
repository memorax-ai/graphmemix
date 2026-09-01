from __future__ import annotations

import numpy as np

from scripts.build_graphmemix_relation_store import (
    canonical_unit_rows,
    explicit_edges,
    haversine_meters,
    normalize,
    read_id_allowlist,
    spatiotemporal_edges,
)


def test_atomic_text_units_do_not_change_canonical_graph_rows() -> None:
    units = [
        {"memory_id": "m1", "text": "full"},
        {"memory_id": "m1", "text": "chunk", "representation": "media_atomic_text"},
        {"memory_id": "m2", "text": "frame 0", "image": "0.jpg"},
        {"memory_id": "m2", "text": "frame 1", "image": "1.jpg"},
    ]
    assert canonical_unit_rows(units) == {"m1": [0], "m2": [2, 3]}


def test_normalize_rejects_zero_vectors() -> None:
    try:
        normalize(np.zeros(3, dtype=np.float32))
    except ValueError as exc:
        assert "zero graph vector" in str(exc)
    else:
        raise AssertionError("zero graph vector should fail")


def test_explicit_edges_are_schema_grounded_and_do_not_cross_sessions() -> None:
    rows = [
        {"memory_id": "u1", "context_id": "c", "session_id": "s1", "sequence": 1,
         "metadata": {"native_round_id": "r1"}},
        {"memory_id": "a1", "context_id": "c", "session_id": "s1", "sequence": 2,
         "metadata": {"native_round_id": "r1"}},
        {"memory_id": "u2", "context_id": "c", "session_id": "s1", "sequence": 3,
         "metadata": {"native_round_id": "r2"}},
        {"memory_id": "x", "context_id": "c", "session_id": "s2", "sequence": 4,
         "metadata": {"native_round_id": "r3"}},
    ]
    edges = explicit_edges(rows)
    assert ("a1", "u1") in edges
    assert ("a1", "u2") in edges
    assert not any("x" in edge and len(set(edge) & {"u1", "a1", "u2"}) for edge in edges)


def test_memory_id_allowlist_is_strict(tmp_path) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("m2\n\nm1\nm2\n", encoding="utf-8")
    assert read_id_allowlist(path) == {"m1", "m2"}


def test_spatiotemporal_edges_link_only_consecutive_nearby_media() -> None:
    rows = [
        {
            "memory_id": "m1", "context_id": "c", "kind": "media",
            "timestamp": "2026-01-01 10:00:00",
            "metadata": {"raw_metadata": {"location": [52.2, 0.12]}},
        },
        {
            "memory_id": "m2", "context_id": "c", "kind": "media",
            "timestamp": "2026-01-01 10:30:00",
            "metadata": {"raw_metadata": {"location": [52.21, 0.12]}},
        },
        {
            "memory_id": "m3", "context_id": "c", "kind": "media",
            "timestamp": "2026-01-01 10:45:00",
            "metadata": {"raw_metadata": {"location": [53.0, 0.12]}},
        },
        {
            "memory_id": "email", "context_id": "c", "kind": "email",
            "timestamp": "2026-01-01 10:05:00",
            "metadata": {"raw_metadata": {"location": [52.2, 0.12]}},
        },
    ]
    edges = spatiotemporal_edges(
        rows, max_gap_seconds=3600, max_distance_meters=5000,
    )
    assert set(edges) == {("m1", "m2")}
    assert edges[("m1", "m2")]["explicit_relations"] == ["spatiotemporal_event"]
    assert 1000 < haversine_meters((52.2, 0.12), (52.21, 0.12)) < 1200
