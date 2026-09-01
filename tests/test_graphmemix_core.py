from __future__ import annotations

from scripts.graphmemix_core import (
    candidate_pool,
    edge_similarity,
    memory_location,
    memory_snippet,
    missing_atomic_floor,
)


def test_candidate_pool_preserves_sources_then_uses_deterministic_edge_priority() -> None:
    prior = {"retrieval_ids": ["s1", "s2", "tail"]}
    adjacency = {
        "s1": [("n2", 0.8, "semantic"), ("n1", 0.9, "explicit")],
        "s2": [("n3", 0.9, "explicit")],
    }
    assert candidate_pool(prior, adjacency, source_top_l=2, candidate_limit=4) == [
        "s1", "s2", "n1", "n3",
    ]


def test_edge_mode_filters_semantic_rank_but_preserves_explicit_edge() -> None:
    row = {
        "explicit_relations": ["same_round"], "explicit_similarity": 0.99,
        "semantic_similarity": 0.8, "semantic_rank_left": 9, "semantic_rank_right": 2,
    }
    assert edge_similarity(row, "explicit", 8) == 0.99
    assert edge_similarity(row, "full", 8) == 0.99
    assert edge_similarity({**row, "explicit_relations": []}, "full", 8) is None


def test_missing_atomic_floor_uses_full_source_prior_not_candidate_prefix() -> None:
    assert missing_atomic_floor({"retrieval_scores": [0.9, 0.8, 0.2]}) == 0.19


def test_memory_snippet_reads_canonical_memory_level_derived_caption() -> None:
    memory = {
        "content": [{"type": "image", "asset_id": "asset-1"}],
        "metadata": {
            "derived": {
                "caption": "A dental receipt shows a total cost of £50.00.",
                "short_caption": "A dental receipt.",
            }
        },
    }
    assert memory_snippet(memory) == (
        "A dental receipt.\nA dental receipt shows a total cost of £50.00."
    )


def test_memory_location_reads_canonical_derived_field() -> None:
    memory = {
        "metadata": {
            "derived": {
                "location_name": "Granta Place, Cambridge",
                "city": "Cambridge",
            }
        }
    }
    assert memory_location(memory) == "Granta Place, Cambridge"
    assert "Location: Granta Place, Cambridge" in memory_snippet(memory)


def test_memory_snippet_keeps_content_text_and_deduplicates_nested_derived_text() -> None:
    memory = {
        "content": [{
            "type": "text",
            "text": "Original email body",
            "annotations": {"native_image_caption": "Shared caption"},
        }],
        "metadata": {
            "derived": {
                "caption": [{"caption": "Shared caption"}, {"text": "Second caption"}],
            }
        },
    }
    assert memory_snippet(memory) == (
        "Original email body\nShared caption\nSecond caption"
    )


def test_memory_snippet_applies_limit_after_combining_all_supported_sources() -> None:
    memory = {
        "content": [],
        "metadata": {"derived": {"ocr_text": "12345", "caption": "abcdef"}},
    }
    assert memory_snippet(memory, 8) == "abcdef"


def test_memory_snippet_keeps_caption_ahead_of_repeated_ocr() -> None:
    memory = {
        "content": [],
        "metadata": {
            "derived": {
                "caption": "A restaurant receipt with the final total.",
                "ocr_text": "\n".join(["TOTAL £42.50"] * 100),
            }
        },
    }
    value = memory_snippet(
        memory, 120, query="How much was the total?", ocr_chars=40
    )
    assert value.startswith("A restaurant receipt with the final total.")
    assert value.count("TOTAL £42.50") == 1
    assert len(value) <= 120
