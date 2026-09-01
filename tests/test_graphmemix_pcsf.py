from __future__ import annotations

from scripts.probe_graphmemix_pcsf import solve_forest, solve_forest_greedy


def test_graphmemix_uses_a_cheap_edge_to_reduce_fragmentation() -> None:
    selected, diagnostic = solve_forest(
        ["a", "b", "c"], {"a": 0.9, "b": 0.8, "c": 0.1},
        {"a": [("c", 0.99, "explicit")], "c": [("a", 0.99, "explicit")]},
        k=2, edge_weight=0.1, root_cost=1.0,
    )
    assert selected == ["a", "c"]
    assert diagnostic["chosen_edges"] == 1
    assert diagnostic["components"] == 1


def test_graphmemix_degenerates_to_topk_without_edges() -> None:
    selected, diagnostic = solve_forest(
        ["a", "b", "c"], {"a": 0.9, "b": 0.8, "c": 0.1}, {},
        k=2, edge_weight=0.1, root_cost=1.0,
    )
    assert selected == ["a", "b"]
    assert diagnostic["chosen_edges"] == 0
    assert diagnostic["components"] == 2


def test_greedy_ablation_finds_the_same_simple_structural_swap() -> None:
    selected, diagnostic = solve_forest_greedy(
        ["a", "b", "c"], {"a": 0.9, "b": 0.8, "c": 0.1},
        {"a": [("c", 0.99, "explicit")], "c": [("a", 0.99, "explicit")]},
        k=2, edge_weight=0.1, root_cost=1.0,
    )
    assert selected == ["a", "c"]
    assert diagnostic["chosen_edges"] == 1
    assert diagnostic["exact"] is False
