"""Shared deterministic data operations for GraphMemix experiments."""

from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from mm_memory_bench.preprocessing.text import clean_ocr_text


MEMORY_SNIPPET_PROTOCOL = (
    "graphmemix-semantic-first-clean-ocr-location-full-question-strict-ecv-7"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def edge_similarity(row: Mapping[str, Any], mode: str, semantic_k: int) -> float | None:
    values: list[float] = []
    if mode in {"explicit", "full"} and row.get("explicit_relations"):
        values.append(float(row["explicit_similarity"]))
    if (
        mode == "full"
        and row.get("semantic_similarity") is not None
        and int(row.get("semantic_rank_left", semantic_k + 1)) <= semantic_k
        and int(row.get("semantic_rank_right", semantic_k + 1)) <= semantic_k
    ):
        values.append(float(row["semantic_similarity"]))
    return max(values) if values else None


def load_adjacency(
    path: Path, mode: str, semantic_k: int,
) -> dict[str, list[tuple[str, float, str]]]:
    if mode not in {"none", "explicit", "full"}:
        raise ValueError(f"unsupported graph mode: {mode}")
    if mode == "none":
        return {}
    result: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            similarity = edge_similarity(row, mode, semantic_k)
            if similarity is None:
                continue
            left, right = str(row["left"]), str(row["right"])
            relation = (
                "explicit" if row.get("explicit_relations") and
                float(row.get("explicit_similarity", 0.0)) >= similarity else "semantic"
            )
            result[left].append((right, similarity, relation))
            result[right].append((left, similarity, relation))
    for memory_id in result:
        result[memory_id].sort(key=lambda value: (-value[1], value[0]))
    return dict(result)


def candidate_pool(
    prior: Mapping[str, Any], adjacency: Mapping[str, Sequence[tuple[str, float, str]]],
    *, source_top_l: int, candidate_limit: int,
) -> list[str]:
    """Keep Atomic top-L, then fill by edge confidence and parent source rank."""
    source_ids = [str(value) for value in prior.get("retrieval_ids", [])]
    seeds = list(dict.fromkeys(source_ids[:source_top_l]))
    if candidate_limit < len(seeds):
        raise ValueError("candidate_limit must be at least source_top_l")
    selected, seen = list(seeds), set(seeds)
    proposals: list[tuple[float, int, str]] = []
    for source_rank, seed in enumerate(seeds):
        for neighbor, similarity, _ in adjacency.get(seed, []):
            if neighbor not in seen:
                proposals.append((-float(similarity), source_rank, neighbor))
    for _, _, neighbor in sorted(proposals):
        if neighbor in seen:
            continue
        seen.add(neighbor)
        selected.append(neighbor)
        if len(selected) >= candidate_limit:
            break
    return selected


def source_score_map(prior: Mapping[str, Any]) -> dict[str, float]:
    ids = [str(value) for value in prior.get("retrieval_ids", [])]
    scores = [float(value) for value in prior.get("retrieval_scores", [])]
    return dict(zip(ids, scores))


def missing_atomic_floor(prior: Mapping[str, Any]) -> float:
    """Return the frozen floor for graph neighbors absent from source retrieval."""
    scores = [float(value) for value in prior.get("retrieval_scores", [])]
    if not scores:
        raise ValueError("source prior has no retrieval scores")
    return min(scores) - 0.01


def _snippet_strings(value: Any) -> list[str]:
    """Normalize canonical derived-text fields without stringifying containers."""
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result: list[str] = []
        for item in value:
            result.extend(_snippet_strings(item))
        return result
    if isinstance(value, Mapping):
        result = []
        for key in ("final_text", "full_text", "text", "caption", "description"):
            result.extend(_snippet_strings(value.get(key)))
        return result
    return []


def memory_location(memory: Mapping[str, Any]) -> str:
    metadata = memory.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return ""
    derived = metadata.get("derived", {})
    if isinstance(derived, Mapping):
        value = derived.get("location_name") or derived.get("city")
        if value:
            return str(value)
    return str(metadata.get("location") or "")


def memory_snippet(
    memory: Mapping[str, Any],
    max_chars: int = 420,
    *,
    query: str = "",
    ocr_chars: int = 120,
) -> str:
    semantic_pieces: list[str] = []
    ocr_pieces: list[str] = []
    for part in memory.get("content", []):
        if part.get("type") == "text" and part.get("text"):
            semantic_pieces.append(str(part["text"]))
        annotations = part.get("annotations", {})
        derived = annotations.get("derived", {}) if isinstance(annotations, Mapping) else {}
        for key in ("text", "caption"):
            if isinstance(derived, Mapping) and derived.get(key):
                semantic_pieces.append(str(derived[key]))
        if isinstance(annotations, Mapping) and annotations.get("native_image_caption"):
            semantic_pieces.append(str(annotations["native_image_caption"]))
    # Canonical benchmark converters store public captions/OCR at the memory
    # level.  This is the primary representation for ATM-Bench media; omitting
    # it silently produced empty verifier snippets for otherwise valid images.
    metadata = memory.get("metadata", {})
    memory_derived = metadata.get("derived", {}) if isinstance(metadata, Mapping) else {}
    if isinstance(memory_derived, Mapping):
        # Semantic descriptions come first. Raw OCR has a separate budget so a
        # long repeated OCR dump cannot evict the caption from a 420-char view.
        for key in (
            "short_caption", "caption", "text",
            "image_caption", "image_captions", "video_caption", "video_captions",
            "blip_caption", "blip_captions",
        ):
            semantic_pieces.extend(_snippet_strings(memory_derived.get(key)))
        location = memory_location(memory)
        if location:
            semantic_pieces.append(f"Location: {location}")
        for key in ("ocr", "ocr_text"):
            ocr_pieces.extend(_snippet_strings(memory_derived.get(key)))
    semantic = "\n".join(dict.fromkeys(
        piece.strip() for piece in semantic_pieces if piece.strip()
    ))
    semantic_reserve = min(len(semantic), max_chars * 2 // 3) if semantic else 0
    ocr_payload_budget = min(
        ocr_chars, max(0, max_chars - semantic_reserve - len("OCR: ") - 1)
    )
    cleaned_ocr = clean_ocr_text(
        "\n".join(ocr_pieces), query=query, max_chars=ocr_payload_budget
    )
    if not cleaned_ocr:
        return semantic[:max_chars]
    ocr_block = f"OCR: {cleaned_ocr}"
    semantic_budget = max(0, max_chars - len(ocr_block) - 1)
    if not semantic_budget:
        return ocr_block[:max_chars]
    return f"{semantic[:semantic_budget].rstrip()}\n{ocr_block}"[:max_chars]


def question_native_captions(question: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for part in question.get("prompt", []):
        if part.get("type") != "image":
            continue
        annotations = part.get("annotations", {})
        if not isinstance(annotations, Mapping):
            continue
        for key in ("native_image_caption", "caption"):
            if annotations.get(key):
                result.append(str(annotations[key]))
                break
        else:
            derived = annotations.get("derived", {})
            if isinstance(derived, Mapping) and (derived.get("text") or derived.get("caption")):
                result.append(str(derived.get("text") or derived.get("caption")))
    return result
