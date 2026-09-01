from __future__ import annotations

from mm_memory_bench.text_cleaning import clean_labeled_ocr_block, clean_ocr_text


def test_clean_ocr_deduplicates_lines_and_removes_boilerplate() -> None:
    value = "\n".join([
        "The text visible in the image is:",
        "KEEP CLEAR AT ALL TIMES",
        "KEEP CLEAR AT ALL TIMES",
        "```",
    ])
    assert clean_ocr_text(value, max_chars=200) == "KEEP CLEAR AT ALL TIMES"


def test_clean_ocr_prioritizes_query_overlap_and_numeric_facts() -> None:
    value = "\n".join([
        "WELCOME TO THE RESTAURANT",
        "TOTAL £42.50",
        "THANK YOU FOR VISITING",
    ])
    cleaned = clean_ocr_text(value, query="What was the total?", max_chars=18)
    assert cleaned == "TOTAL £42.50"


def test_clean_labeled_ocr_preserves_following_metadata() -> None:
    value = (
        "Caption: a bicycle\n"
        "OCR: NOISY SIGN\nNOISY SIGN\n"
        "Location: Cambridge\nTags: bicycle"
    )
    cleaned = clean_labeled_ocr_block(value, query="sign", max_chars=100)
    assert cleaned.count("NOISY SIGN") == 1
    assert "Location: Cambridge" in cleaned
    assert "Tags: bicycle" in cleaned
