from __future__ import annotations

import pytest

from scripts.score_graphmemix_verifier import validate_selected_rows


def test_node_verifier_rejects_non_list_unknown_duplicate_and_invalid_scores() -> None:
    valid = {"C00", "C01"}
    assert validate_selected_rows(
        [{"id": "C00", "score": 4}], valid_aliases=valid
    ) == [("C00", 4.0)]
    with pytest.raises(ValueError, match="must be a list"):
        validate_selected_rows({}, valid_aliases=valid)
    with pytest.raises(ValueError, match="unknown"):
        validate_selected_rows(
            [{"id": "C09", "score": 4}], valid_aliases=valid
        )
    with pytest.raises(ValueError, match="duplicate"):
        validate_selected_rows(
            [{"id": "C00", "score": 4}, {"id": "C00", "score": 3}],
            valid_aliases=valid,
        )
    with pytest.raises(ValueError, match="out of range"):
        validate_selected_rows(
            [{"id": "C00", "score": 9}], valid_aliases=valid
        )
