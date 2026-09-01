from __future__ import annotations

import pytest

from scripts.score_graphmemix_ecv import (
    direct_support_value,
    validate_candidate_rows,
    visible_question_text,
)


def test_edge_only_ecv_forces_direct_support_to_zero() -> None:
    assert direct_support_value({"direct_support": 5}, edge_only=True) == 0.0


def test_joint_ecv_clamps_direct_support() -> None:
    assert direct_support_value({"direct_support": 9}, edge_only=False) == 5.0
    assert direct_support_value({"direct_support": -2}, edge_only=False) == 0.0
    assert direct_support_value({"direct_support": "invalid"}, edge_only=False) == 0.0


def test_ecv_question_text_includes_multiple_choice_options() -> None:
    value = visible_question_text({
        "prompt": [{"type": "text", "text": "Which object was shown?"}],
        "choices": [
            {"choice_id": "A", "text": "Red book"},
            {"choice_id": "B", "text": "Blue cup"},
        ],
    })
    assert value == (
        "Which object was shown?\nChoices:\nA: Red book\nB: Blue cup"
    )


def test_ecv_structured_rows_reject_duplicates_and_ineligible_anchors() -> None:
    valid = {
        "id": "C01", "direct_support": 0, "best_anchor_id": "C00",
        "incremental_support": 3, "role": "new_fact",
    }
    assert validate_candidate_rows(
        [valid],
        expected_aliases={"C01"},
        eligible_anchors={"C01": {"C00"}},
    ) == [valid]
    with pytest.raises(ValueError, match="duplicate"):
        validate_candidate_rows(
            [valid, valid],
            expected_aliases={"C01"},
            eligible_anchors={"C01": {"C00"}},
        )
    ineligible = validate_candidate_rows(
        [{**valid, "best_anchor_id": "C09"}],
        expected_aliases={"C01"},
        eligible_anchors={"C01": {"C00"}},
    )[0]
    assert ineligible["best_anchor_id"] is None
    assert ineligible["incremental_support"] == 0
    assert ineligible["role"] == "irrelevant"
    assert ineligible["_normalizations"] == [
        "ineligible_anchor_cleared_edge_fields"
    ]
    invalid_role = validate_candidate_rows(
        [{**valid, "role": "record_grounding"}],
        expected_aliases={"C01"},
        eligible_anchors={"C01": {"C00"}},
    )[0]
    assert invalid_role["role"] == "irrelevant"
    assert invalid_role["incremental_support"] == 0
    assert invalid_role["_normalizations"] == [
        "invalid_role_replaced_with_irrelevant",
        "nonpositive_role_cleared_incremental",
    ]
    with_extra = validate_candidate_rows(
        [valid, {**valid, "id": "C09"}],
        expected_aliases={"C01"},
        eligible_anchors={"C01": {"C00"}},
    )
    assert with_extra[0]["_normalizations"] == [
        "ignored_unknown_candidate_row"
    ]
    normalized = validate_candidate_rows(
        [{**valid, "best_anchor_id": None}],
        expected_aliases={"C01"},
        eligible_anchors={"C01": {"C00"}},
    )[0]
    assert normalized["incremental_support"] == 0
    assert normalized["role"] == "irrelevant"
    assert normalized["_normalizations"] == ["null_anchor_cleared_edge_fields"]
