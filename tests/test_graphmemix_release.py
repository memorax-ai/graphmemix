from __future__ import annotations

import json
import hashlib
import tomllib
from pathlib import Path

from scripts.audit_graphmemix_release import expand_allowlist
from scripts.select_atm_hard_graphmemix_forest import (
    MemoryResolver,
    edge_incidence_scores,
    graph_obligation_ledger,
    graph_timeline_context,
    load_hard_questions,
    project_evidence_graph,
    representation_actions,
    structural_coverage,
)
from scripts.run_memix_answers_from_retrieval import retrieval_contract_sha256


ROOT = Path(__file__).resolve().parents[1]


def test_release_package_and_single_result_manifest_are_graphmemix() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["name"] == "graphmemix"
    assert project["project"]["scripts"]["graphmemix"] == "mm_memory_bench.cli:main"
    results = json.loads(
        (ROOT / "release/benchmark_results.json").read_text(encoding="utf-8")
    )
    assert set(results["extensions"]) == {"atm_hard_graphmemix_forest_v3"}
    extension = results["extensions"]["atm_hard_graphmemix_forest_v3"]
    assert extension["primary_run"]["qs_percent"] == 57.94831617886647
    assert extension["primary_run"]["artifact_sha256"]["score"] == (
        "f7d5d6746aead4ac252ab32cc2621cd81e88588b397c09d330ea1e898f5dc565"
    )
    assert not (ROOT / "release/benchmark_reproduction_20260811.json").exists()


def test_benchmark_config_freezes_four_datasets_and_two_backbones() -> None:
    config = json.loads(
        (ROOT / "configs/release/graphmemix.json").read_text(encoding="utf-8")
    )
    suite = config["benchmark_suite"]
    assert {key: value["expected_questions"] for key, value in suite["datasets"].items()} == {
        "atm": 1044,
        "mem_gallery": 1711,
        "memeye": 1855,
        "h2hmem": 1982,
    }
    assert {value["model"] for value in suite["backbones"].values()} == {
        "Qwen/Qwen3-VL-8B-Instruct",
        "google/gemma-4-12B-it",
    }
    assert suite["source_top_l"] == 24
    assert suite["candidate_limit"] == 48
    assert suite["reader_k"] == 10
    assert {value["reasoning_effort"] for value in suite["backbones"].values()} == {"none"}
    calibration = json.loads((ROOT / suite["calibration_file"]).read_text(encoding="utf-8"))
    assert calibration["coefficients"] == {
        "a": 4.054056101800787,
        "b": 0.22537987366704051,
        "c": -3.711336391577528,
    }


def test_atm_hard_v3_config_freezes_qtype_only_selector() -> None:
    config = json.loads(
        (ROOT / "configs/release/atm_hard_graph_proof.json").read_text(encoding="utf-8")
    )
    selector = config["selector"]
    assert selector["protocol"] == "atm-hard-graphmemix-forest-v3"
    assert selector["routing_signal"] == "canonical_qtype_only"
    assert selector["uses_question_text_rules"] is False
    assert selector["uses_gold"] is False
    assert selector["structural_coverage_weights"] == {
        "list_recall": 0.2, "number": 0.2, "open_end": 0.2,
    }
    assert selector["edge_incidence_weights"] == {
        "list_recall": 0.25, "number": 0.0, "open_end": 0.0,
    }


def test_atm_hard_public_question_loader_ignores_default_and_answers(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    records = [
        {
            "question_id": "atm_bench:question:hard:q1",
            "context_id": "",
            "subset": "hard",
            "prompt": [{"type": "text", "text": "List every image."}],
            "task": {"subcategory": "list_recall"},
            "answer": {"text": "must-not-be-read"},
        },
        {
            "question_id": "atm_bench:question:hard:q2",
            "context_id": "",
            "subset": "hard",
            "prompt": [{"type": "text", "text": "How much?"}],
            "task": {"subcategory": "number"},
            "answer": {"text": "100"},
        },
        {
            "question_id": "atm_bench:question:default:q3",
            "context_id": "",
            "subset": "default",
            "prompt": [{"type": "text", "text": "List default."}],
            "task": {"subcategory": "list_recall"},
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    assert load_hard_questions(path) == [
        {
            "question_id": "atm_bench:question:hard:q1",
            "native_id": "q1",
            "context_id": "",
            "qtype": "list_recall",
        },
        {
            "question_id": "atm_bench:question:hard:q2",
            "native_id": "q2",
            "context_id": "",
            "qtype": "number",
        },
    ]


def test_atm_hard_selector_projects_graph_and_assigns_one_action_per_memory() -> None:
    memories = [
        {"memory_id": "m:image:a", "source_id": "a", "modality": "image"},
        {"memory_id": "m:video:b", "source_id": "b", "modality": "video"},
        {"memory_id": "m:email:c", "source_id": "c", "modality": "email"},
    ]
    resolver = MemoryResolver(memories)
    nodes = [
        {"node_id": "a", "is_memory": True, "memory_id": "a", "confidence": 0.9},
        {"node_id": "b", "is_memory": True, "memory_id": "b", "confidence": 0.8},
        {"node_id": "c", "is_memory": True, "memory_id": "c", "confidence": 0.7},
        {
            "node_id": "hub",
            "is_memory": False,
            "node_kind": "collection",
            "confidence": 0.85,
            "support_memory_ids": ["a", "b", "c"],
        },
    ]
    edges = [
        {
            "edge_id": "e1",
            "source": "a",
            "target": "hub",
            "relation": "any-agent-defined-relation",
            "confidence": 0.75,
        }
    ]
    candidates, prizes, adjacency, diagnostic = project_evidence_graph(
        nodes, edges, resolver, qtype="list_recall"
    )

    assert set(candidates) == {"m:image:a", "m:video:b", "m:email:c"}
    assert diagnostic["projected_memory_edges"] >= 2
    assert any(relation.startswith(("hub:", "graph:")) for values in adjacency.values() for _, _, relation in values)
    actions = representation_actions(candidates, prizes, resolver, high_media_limit=1)
    assert [row["memory_id"] for row in actions] == candidates
    assert sum(row["action"] == "high" for row in actions) == 1
    assert sum(row["action"] == "text" for row in actions) == 2


def test_atm_hard_structural_coverage_discount_large_hubs() -> None:
    resolver = MemoryResolver([
        {"memory_id": "m:a", "source_id": "a"},
        {"memory_id": "m:b", "source_id": "b"},
        {"memory_id": "m:c", "source_id": "c"},
    ])
    nodes = [
        {"node_id": "small", "confidence": 1.0, "support_memory_ids": ["a"]},
        {"node_id": "large", "confidence": 1.0, "support_memory_ids": ["b", "c"]},
    ]
    coverage = structural_coverage(nodes, [], resolver, qtype="number")
    assert coverage["m:a"] == 1.0
    assert 0.0 < coverage["m:b"] < coverage["m:a"]
    assert coverage["m:b"] == coverage["m:c"]


def test_atm_hard_edge_incidence_ignores_relation_vocabulary() -> None:
    resolver = MemoryResolver([
        {"memory_id": "m:a", "source_id": "a"},
        {"memory_id": "m:b", "source_id": "b"},
    ])
    nodes = [
        {"node_id": "a", "is_memory": True, "memory_id": "a"},
        {"node_id": "b", "is_memory": True, "memory_id": "b"},
        {"node_id": "hub", "node_kind": "collection", "confidence": 0.9},
    ]
    edges = [
        {"source": "a", "target": "hub", "relation": "invented-one", "confidence": 0.8},
        {"source": "b", "target": "hub", "relation": "another-ontology", "confidence": 0.4},
    ]
    scores = edge_incidence_scores(nodes, edges, resolver, qtype="list_recall")
    assert scores["m:a"] == 1.0
    assert 0.0 < scores["m:b"] < scores["m:a"]


def test_atm_hard_obligation_ledger_requires_distinct_temporal_branches() -> None:
    resolver = MemoryResolver([
        {"memory_id": "m:a", "source_id": "a"},
        {"memory_id": "m:b", "source_id": "b"},
    ])
    nodes = [
        {"node_id": "root", "node_kind": "answer", "confidence": 0.9},
        {"node_id": "branch-a", "node_kind": "group", "confidence": 0.9},
        {"node_id": "branch-b", "node_kind": "group", "confidence": 0.9},
        {
            "node_id": "event-a", "node_kind": "event", "confidence": 0.9,
            "timestamp_or_range": "2024-01", "support_memory_ids": ["a"],
        },
        {
            "node_id": "event-b", "node_kind": "event", "confidence": 0.9,
            "timestamp_or_range": "2024-02", "support_memory_ids": ["b"],
        },
    ]
    split_edges = [
        {"source": "root", "target": "branch-a", "confidence": 0.9},
        {"source": "root", "target": "branch-b", "confidence": 0.9},
        {"source": "branch-a", "target": "event-a", "confidence": 0.9},
        {"source": "branch-b", "target": "event-b", "confidence": 0.9},
    ]
    context = graph_obligation_ledger(
        nodes, split_edges, resolver, ["m:a", "m:b"], qtype="number"
    )
    assert "2024-01" in context and "2024-02" in context
    shared_edges = [
        {"source": "root", "target": "branch-a", "confidence": 0.9},
        {"source": "branch-a", "target": "event-a", "confidence": 0.9},
        {"source": "branch-a", "target": "event-b", "confidence": 0.9},
    ]
    assert not graph_obligation_ledger(
        nodes, shared_edges, resolver, ["m:a", "m:b"], qtype="number"
    )
    assert not graph_obligation_ledger(
        nodes, split_edges, resolver, ["m:a", "m:b"], qtype="open_end"
    )


def test_atm_hard_graph_timeline_requires_unsupported_aggregate() -> None:
    resolver = MemoryResolver([
        {"memory_id": "m:a", "source_id": "a"},
        {"memory_id": "m:b", "source_id": "b"},
    ])
    temporal = [
        {
            "node_id": "event-1", "node_kind": "event", "confidence": 0.9,
            "timestamp_or_range": "2024-01-01", "support_memory_ids": ["a"],
        },
        {
            "node_id": "event-2", "node_kind": "event", "confidence": 0.8,
            "timestamp_or_range": "2024-01-02", "support_memory_ids": ["b"],
        },
    ]
    unsupported = [
        {"node_id": "aggregate", "node_kind": "answer", "support_memory_ids": []},
        *temporal,
    ]
    context = graph_timeline_context(
        unsupported, [], resolver, ["m:a", "m:b"], qtype="number"
    )
    assert "2024-01-01" in context and "2024-01-02" in context
    supported = json.loads(json.dumps(unsupported))
    supported[0]["support_memory_ids"] = ["a"]
    assert not graph_timeline_context(
        supported, [], resolver, ["m:a", "m:b"], qtype="number"
    )
    assert not graph_timeline_context(
        unsupported, [], resolver, ["m:a", "m:b"], qtype="list_recall"
    )


def test_reader_resume_contract_binds_representation_actions() -> None:
    base = {
        "question_id": "q1",
        "retrieved_memory_ids": ["m1"],
        "metadata": {"selected_actions": [{"memory_id": "m1", "action": "text"}]},
    }
    changed = json.loads(json.dumps(base))
    changed["metadata"]["selected_actions"][0]["action"] = "high"
    assert retrieval_contract_sha256(base) != retrieval_contract_sha256(changed)


def test_release_uses_exact_approved_atm_graph_prompt() -> None:
    release_prompt = (
        ROOT / "configs/release/prompts/atm_hard_evidence_graph_v3_compact.txt"
    ).read_bytes()
    assert hashlib.sha256(release_prompt).hexdigest() == (
        "652b9491d399aea7503e939a13e96fc991a3779f4dfeeb131fb72f81a9d442fa"
    )


def test_release_allowlist_excludes_generated_python_caches() -> None:
    files, missing = expand_allowlist(ROOT / "release/repository_allowlist.txt")
    assert not missing
    assert all("__pycache__" not in path.parts for path in files)
    assert all(path.suffix not in {".pyc", ".pyo"} for path in files)
