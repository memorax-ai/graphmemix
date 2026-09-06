from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Iterable


_OCR_BOILERPLATE = (
    re.compile(r"^(?:the\s+)?text\s+visible\s+in\s+(?:the|this)\s+image\s+is\s*:?\s*$", re.I),
    re.compile(r"^(?:no|none|not)\s+(?:visible\s+)?text(?:\s+(?:was|is)\s+detected)?[.!:]?\s*$", re.I),
    re.compile(r"^(?:ocr|visible\s+text)\s*:?\s*$", re.I),
    re.compile(r"^`{3,}\w*\s*$"),
)
_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "did", "do", "for", "from", "had",
    "how", "i", "in", "is", "it", "me", "my", "of", "on", "or", "that",
    "the", "this", "to", "was", "were", "what", "when", "where", "which",
    "who", "why", "with",
}


def _normalized_line(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"\s+", " ", value).strip(" \t\r\n|")
    return value


def _line_key(value: str) -> str:
    return re.sub(r"[^\w]+", " ", value.casefold(), flags=re.UNICODE).strip()


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold())
        if len(token) >= 2 and token not in _QUERY_STOPWORDS
    }


def _useful_ocr_line(value: str) -> bool:
    if not value or any(pattern.match(value) for pattern in _OCR_BOILERPLATE):
        return False
    compact = re.sub(r"\s+", "", value)
    alnum = [character for character in compact if character.isalnum()]
    if len(alnum) < 2:
        return False
    if len(alnum) / max(1, len(compact)) < 0.25:
        return False
    if len(alnum) >= 20 and len(set(alnum)) / len(alnum) < 0.08:
        return False
    return True


def _deduplicate_lines(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    keys: list[str] = []
    for raw in values:
        value = _normalized_line(raw)
        if not _useful_ocr_line(value):
            continue
        key = _line_key(value)
        if not key or key in keys:
            continue
        # OCR engines often emit the same line with one altered character.
        if any(
            min(len(key), len(previous)) >= 12
            and SequenceMatcher(None, key, previous).ratio() >= 0.94
            for previous in keys
        ):
            continue
        result.append(value)
        keys.append(key)
    return result


@lru_cache(maxsize=16384)
def _deduplicate_text(value: str) -> tuple[str, ...]:
    return tuple(_deduplicate_lines(value.splitlines()))


def clean_ocr_text(value: str, *, query: str = "", max_chars: int = 320) -> str:
    """Return a compact, query-aware OCR view without repeated/noisy lines."""
    if max_chars <= 0 or not value.strip():
        return ""
    lines = list(_deduplicate_text(value))
    if not lines:
        return ""
    query_tokens = _tokens(query)
    ranked: list[tuple[float, int, str, bool]] = []
    for index, line in enumerate(lines):
        line_tokens = _tokens(line)
        overlap = len(query_tokens & line_tokens)
        contains_numeric_fact = bool(
            re.search(r"(?:\d|[$£€¥₹]|(?:usd|gbp|eur)\b)", line, re.I)
        )
        # Query overlap dominates; numbers/currency remain useful for receipts,
        # dates and prices even when lexical overlap is absent.
        score = overlap * 100.0 + contains_numeric_fact * 12.0
        score += min(len(line), 120) / 120.0
        ranked.append((score, index, line, bool(overlap or contains_numeric_fact)))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    if query_tokens and any(item[3] for item in ranked):
        ranked = [item for item in ranked if item[3]]

    chosen: list[tuple[int, str]] = []
    remaining = max_chars
    for _, index, line, _ in ranked:
        separator = 1 if chosen else 0
        if remaining <= separator:
            break
        if chosen and len(line) > remaining - separator:
            continue
        clipped = line[: remaining - separator].rstrip()
        if clipped:
            chosen.append((index, clipped))
            remaining -= len(clipped) + separator
    chosen.sort()
    return "\n".join(line for _, line in chosen)


def clean_labeled_ocr_block(value: str, *, query: str = "", max_chars: int = 320) -> str:
    """Clean an ``OCR:`` block embedded in a Memix reader record."""
    pattern = re.compile(
        r"(?ms)^OCR:\s*(.*?)(?=^(?:Location|City|Tags|Image caption \d+):|\Z)"
    )

    def replace(match: re.Match[str]) -> str:
        cleaned = clean_ocr_text(match.group(1), query=query, max_chars=max_chars)
        return f"OCR: {cleaned}\n" if cleaned else ""

    return pattern.sub(replace, value).strip()
