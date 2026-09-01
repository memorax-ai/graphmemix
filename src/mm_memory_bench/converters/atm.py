from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

from ..core import (
    BundleWriter,
    SchemaError,
    content_asset,
    content_text,
    provenance,
    read_json,
    stable_id,
    validate_bundle,
)


BENCHMARK_KEY = "atm_bench"
BENCHMARK_NAME = "ATM-Bench"
CONTEXT_ID = stable_id(BENCHMARK_KEY, "personal_archive")

# The public Hugging Face snapshot audited for this converter.  A local git
# checkout is recorded when available, while this remains the reproducibility
# reference for fixtures or exported snapshots without .git metadata.
OFFICIAL_DATA_REVISION = "78e826dc07e97466b2f54443831ef9a83ab8b27c"

QA_INSTRUCTION = (
    "Use only the stored memories to answer. If the evidence is insufficient, answer "
    "'Unknown'. Respond with only the answer. If the question asks you to recall or "
    "list photos, emails, or videos, return their source_id values only, comma-separated, "
    "with no extra text."
)

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_NIAH_RE = re.compile(r"^atm-bench-hard-niah(?P<size>\d+)\.json$")

_DERIVED_MEDIA_FIELDS = {
    "caption",
    "short_caption",
    "ocr_text",
    "tags",
    "entities",
    "location_name",
    "city",
    "safety_content",
    "processed_at",
    "processing_version",
    "model_used",
    "num_frames_analyzed",
}


def _locate_snapshot(raw_root: Path) -> Path:
    """Accept either data/raw or the ATM-Bench snapshot directory itself."""
    candidates = (raw_root, raw_root / BENCHMARK_KEY, raw_root / "ATM-Bench")
    for candidate in candidates:
        if (candidate / "data" / "atm-bench" / "atm-bench.json").is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"ATM-Bench snapshot not found; searched: {searched}")


def _read_array(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    if isinstance(value, dict) and isinstance(value.get("qas"), list):
        value = value["qas"]
    if not isinstance(value, list):
        raise SchemaError(f"{path}: expected a JSON array")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise SchemaError(f"{path}:/{index}: expected an object")
        rows.append(row)
    return rows


def _required_string(row: Mapping[str, Any], field: str, source: Path, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{source}:/{index}/{field}: expected a non-empty string")
    return value


def _string_or_empty(row: Mapping[str, Any], field: str, source: Path, index: int) -> str:
    value = row.get(field, "")
    if not isinstance(value, str):
        raise SchemaError(f"{source}:/{index}/{field}: expected a string")
    return value


def _string_list(row: Mapping[str, Any], field: str, source: Path, index: int) -> list[str]:
    value = row.get(field)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SchemaError(f"{source}:/{index}/{field}: expected a list of strings")
    return list(value)


def _git_revision(snapshot: Path) -> str:
    """Read HEAD without invoking git; exported snapshots fall back to the pin."""
    git_dir = snapshot / ".git"
    if git_dir.is_file():
        marker = git_dir.read_text(encoding="utf-8").strip()
        if marker.startswith("gitdir:"):
            git_dir = (snapshot / marker.split(":", 1)[1].strip()).resolve()
    head_path = git_dir / "HEAD"
    if head_path.is_file():
        head = head_path.read_text(encoding="utf-8").strip()
        if _GIT_SHA_RE.fullmatch(head):
            return head
        if head.startswith("ref:"):
            ref_name = head.split(":", 1)[1].strip()
            ref_path = git_dir / ref_name
            if ref_path.is_file():
                revision = ref_path.read_text(encoding="utf-8").strip()
                if _GIT_SHA_RE.fullmatch(revision):
                    return revision
            packed_refs = git_dir / "packed-refs"
            if packed_refs.is_file():
                for line in packed_refs.read_text(encoding="utf-8").splitlines():
                    if line.startswith(("#", "^")):
                        continue
                    parts = line.split(" ", 1)
                    if len(parts) == 2 and parts[1] == ref_name and _GIT_SHA_RE.fullmatch(parts[0]):
                        return parts[0]
    return OFFICIAL_DATA_REVISION


def _bundle_relative_path(path: Path, bundle_root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), bundle_root.resolve())).as_posix()


def _media_native_id(row: Mapping[str, Any], path_field: str, source: Path, index: int) -> str:
    value = _required_string(row, path_field, source, index)
    native_id = Path(value).stem
    if not native_id:
        raise SchemaError(f"{source}:/{index}/{path_field}: cannot derive an evidence id")
    return native_id


def _register_native_id(
    lookup: dict[str, dict[str, str]],
    native_id: str,
    memory_id: str,
    source_type: str,
) -> None:
    previous = lookup.get(native_id)
    if previous is not None:
        raise SchemaError(
            f"ambiguous native evidence id {native_id!r}: "
            f"{previous['source_type']} and {source_type}"
        )
    lookup[native_id] = {"memory_id": memory_id, "source_type": source_type}


def _resolve_evidence(
    native_ids: list[str],
    lookup: Mapping[str, Mapping[str, str]],
    *,
    source: Path,
    pointer: str,
    relation: str,
) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    for native_id in native_ids:
        target = lookup.get(native_id)
        if target is None:
            raise SchemaError(f"{source}:{pointer}: unknown evidence id {native_id!r}")
        evidence.append(
            {
                "memory_id": str(target["memory_id"]),
                "native_id": native_id,
                "source_type": str(target["source_type"]),
                "relation": relation,
            }
        )
    return evidence


def _media_metadata(row: Mapping[str, Any], path_field: str) -> dict[str, Any]:
    """Losslessly separate generated semantics from raw/technical metadata."""
    derived = {
        key: value for key, value in row.items() if key in _DERIVED_MEDIA_FIELDS
    }
    raw_metadata = {
        key: value
        for key, value in row.items()
        if key not in _DERIVED_MEDIA_FIELDS and key not in {path_field, "timestamp"}
    }
    metadata: dict[str, Any] = {"derived": derived}
    if raw_metadata:
        metadata["raw_metadata"] = raw_metadata
    return metadata


def _load_niah_variants(
    niah_dir: Path,
    hard_rows: Mapping[str, Mapping[str, Any]],
    evidence_lookup: Mapping[str, Mapping[str, str]],
    snapshot: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[int]]:
    variants: dict[str, list[dict[str, Any]]] = {}
    sizes: list[int] = []
    if not niah_dir.is_dir():
        return variants, sizes

    paths: list[tuple[int, Path]] = []
    for path in niah_dir.iterdir():
        match = _NIAH_RE.fullmatch(path.name)
        if match:
            paths.append((int(match.group("size")), path))

    checked_fields = ("question", "answer", "notes", "evidence_ids", "qtype")
    for declared_size, path in sorted(paths):
        sizes.append(declared_size)
        seen: set[str] = set()
        for index, row in enumerate(_read_array(path)):
            native_question_id = _required_string(row, "id", path, index)
            if native_question_id in seen:
                raise SchemaError(f"{path}: duplicate question id {native_question_id!r}")
            seen.add(native_question_id)
            base = hard_rows.get(native_question_id)
            if base is None:
                raise SchemaError(
                    f"{path}:/{index}: NIAH row has no matching hard question "
                    f"{native_question_id!r}"
                )
            for field in checked_fields:
                if row.get(field) != base.get(field):
                    raise SchemaError(
                        f"{path}:/{index}/{field}: differs from atm-bench-hard.json"
                    )
            native_pool = _string_list(row, "niah_evidence_ids", path, index)
            if len(native_pool) != declared_size:
                raise SchemaError(
                    f"{path}:/{index}/niah_evidence_ids: expected {declared_size} "
                    f"items, found {len(native_pool)}"
                )
            variants.setdefault(native_question_id, []).append(
                {
                    "name": f"niah{declared_size}",
                    "candidate_count": declared_size,
                    "evidence": _resolve_evidence(
                        native_pool,
                        evidence_lookup,
                        source=path,
                        pointer=f"/{index}/niah_evidence_ids",
                        relation="candidate",
                    ),
                    "provenance": provenance(path, snapshot, f"/{index}"),
                }
            )

        if seen != set(hard_rows):
            missing = sorted(set(hard_rows) - seen)
            raise SchemaError(
                f"{path}: NIAH variant is missing {len(missing)} hard questions"
            )
    return variants, sizes


def _manifest(
    revision: str,
    memory_counts: Mapping[str, int],
    question_counts: Mapping[str, int],
    niah_sizes: list[int],
) -> dict[str, Any]:
    return {
        "benchmark": BENCHMARK_NAME,
        "dataset_id": "Jingbiao/ATM-Bench",
        "description": "Long-term personalized referential memory QA over one multimodal archive.",
        "subsets": ["default", "hard"],
        "modalities": ["text", "image", "video"],
        "source": {
            "dataset": "https://huggingface.co/datasets/Jingbiao/ATM-Bench",
            "repository": "https://github.com/JingbiaoMei/ATM-Bench",
            "project": "https://atmbench.github.io/",
            "paper": "https://arxiv.org/abs/2603.01990",
            "revision": revision,
        },
        "raw_snapshot": {
            "revision_type": "git",
            "git_revision": revision,
            "converter_reference_revision": OFFICIAL_DATA_REVISION,
        },
        "license": {
            "id": "CC-BY-NC-4.0",
            "name": "Creative Commons Attribution-NonCommercial 4.0 International",
            "url": "https://creativecommons.org/licenses/by-nc/4.0/",
            "commercial_use": False,
        },
        "release_counts": {
            "memories": {**dict(memory_counts), "total": sum(memory_counts.values())},
            "questions": {**dict(question_counts), "total": sum(question_counts.values())},
        },
        "official_release_counts": {
            "memories": {"email": 6742, "image": 3759, "video": 533, "total": 11034},
            "questions": {"default": 1013, "hard": 31, "total": 1044},
        },
        "evaluation": {
            "official_implementation": (
                "https://github.com/JingbiaoMei/ATM-Bench/blob/main/"
                "memqa/utils/evaluator/evaluate_qa.py"
            ),
            "official_metrics": [
                {
                    "name": "Question Type Score (QS)",
                    "semantics": {
                        "number": "official normalized deterministic exact match",
                        "list_recall": "Jaccard similarity over answer items",
                        "open_end": "LLM-as-a-judge accuracy",
                    },
                },
                {"name": "Recall@10", "scope": "memory retrieval"},
                {"name": "Joint@10", "scope": "answer correctness and retrieval"},
            ],
            "variants": {
                "niah_candidate_counts": niah_sizes,
                "storage": "questions.metadata.evaluation_variants.niah",
                "question_rows_are_not_duplicated": True,
            },
            "private_metadata": {
                "path": "questions.metadata.evaluation_private",
                "policy": "never_expose_to_agent",
                "warning": (
                    "This field contains annotator notes, reasoning hints, and sometimes "
                    "evidence identifiers. It MUST NOT enter agent prompts, memory ingestion, "
                    "retrieval indexes, or model-visible input."
                ),
            },
        },
        "media_representation": {
            "default_content": "raw asset reference only",
            "derived_path": "memories.metadata.derived",
            "warning": (
                "Captions, OCR, tags, and other processed semantics are opt-in derived data."
            ),
        },
        "paper_release_difference": {
            "paper_version": "arXiv v1 (2026-03-02)",
            "paper": {
                "emails": 6741,
                "questions": {"default": 1013, "hard": 25, "total": 1038},
            },
            "public_release": {
                "emails": 6742,
                "questions": {"default": 1013, "hard": 31, "total": 1044},
            },
            "warning": (
                "The public release expands ATM-Bench-Hard relative to the preprint. "
                "Results must report the dataset revision and hard-set size."
            ),
        },
    }


def convert(raw_root: Path, output_root: Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Convert one official ATM-Bench snapshot into an mmmb-1.0 bundle."""
    snapshot = _locate_snapshot(Path(raw_root))
    output_root = Path(output_root)

    email_path = snapshot / "data" / "raw_memory" / "email" / "emails.json"
    image_path = snapshot / "data" / "processed_memory" / "image_batch_results.json"
    video_path = snapshot / "data" / "processed_memory" / "video_batch_results.json"
    qa_dir = snapshot / "data" / "atm-bench"
    default_qa_path = qa_dir / "atm-bench.json"
    hard_qa_path = qa_dir / "atm-bench-hard.json"

    required = (email_path, image_path, video_path, default_qa_path, hard_qa_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("ATM-Bench snapshot is incomplete: " + ", ".join(missing))

    memory_rows: list[dict[str, Any]] = []
    asset_rows: list[dict[str, Any]] = []
    evidence_lookup: dict[str, dict[str, str]] = {}
    memory_counts = {"email": 0, "image": 0, "video": 0}

    for index, row in enumerate(_read_array(email_path)):
        native_id = _required_string(row, "id", email_path, index)
        timestamp = _required_string(row, "timestamp", email_path, index)
        detail = _string_or_empty(row, "detail", email_path, index)
        memory_id = stable_id(BENCHMARK_KEY, "memory", "email", native_id)
        _register_native_id(evidence_lookup, native_id, memory_id, "email")
        metadata: dict[str, Any] = {
            "native_id": native_id,
            "source_type": "email",
            "derived": {
                "short_summary": _string_or_empty(row, "short_summary", email_path, index)
            },
        }
        extra = {
            key: value
            for key, value in row.items()
            if key not in {"id", "timestamp", "short_summary", "detail"}
        }
        if extra:
            metadata["native"] = extra
        memory_rows.append(
            {
                "memory_id": memory_id,
                "context_id": CONTEXT_ID,
                "source_id": native_id,
                "timestamp": timestamp,
                "kind": "email",
                "content": [content_text(detail)],
                "provenance": provenance(email_path, snapshot, f"/{index}"),
                "metadata": metadata,
            }
        )
        memory_counts["email"] += 1

    media_specs = (
        ("image", image_path, "image_path", "image/jpeg"),
        ("video", video_path, "video_path", "video/mp4"),
    )
    for media_type, processed_path, path_field, mime_type in media_specs:
        for index, row in enumerate(_read_array(processed_path)):
            native_id = _media_native_id(row, path_field, processed_path, index)
            native_asset_path = Path(_required_string(row, path_field, processed_path, index))
            raw_asset_path = (
                native_asset_path
                if native_asset_path.is_absolute()
                else snapshot / native_asset_path
            )
            asset_id = stable_id(BENCHMARK_KEY, "asset", media_type, native_id)
            memory_id = stable_id(BENCHMARK_KEY, "memory", media_type, native_id)
            _register_native_id(evidence_lookup, native_id, memory_id, media_type)
            asset_rows.append(
                {
                    "asset_id": asset_id,
                    "media_type": media_type,
                    "path": _bundle_relative_path(raw_asset_path, output_root),
                    "mime_type": mime_type,
                    "provenance": provenance(raw_asset_path, snapshot),
                    "metadata": {"native_id": native_id, "source_type": media_type},
                }
            )
            metadata = {
                "native_id": native_id,
                "source_type": media_type,
                **_media_metadata(row, path_field),
            }
            memory: dict[str, Any] = {
                "memory_id": memory_id,
                "context_id": CONTEXT_ID,
                "source_id": native_id,
                "kind": "media",
                "content": [content_asset(media_type, asset_id)],
                "provenance": provenance(processed_path, snapshot, f"/{index}"),
                "metadata": metadata,
            }
            timestamp = row.get("timestamp")
            if timestamp is not None:
                if not isinstance(timestamp, str):
                    raise SchemaError(
                        f"{processed_path}:/{index}/timestamp: expected a string or null"
                    )
                memory["timestamp"] = timestamp
            memory_rows.append(memory)
            memory_counts[media_type] += 1

    memory_rows.sort(
        key=lambda row: (
            str(row.get("timestamp") or "9999-99-99 99:99:99")[:19],
            {"email": 0, "image": 1, "video": 2}.get(
                str(row.get("metadata", {}).get("source_type")), 9
            ),
            str(row["memory_id"]),
        )
    )
    for sequence, row in enumerate(memory_rows):
        row["sequence"] = sequence

    default_rows = _read_array(default_qa_path)
    hard_rows_list = _read_array(hard_qa_path)
    hard_rows: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(hard_rows_list):
        native_id = _required_string(row, "id", hard_qa_path, index)
        if native_id in hard_rows:
            raise SchemaError(f"{hard_qa_path}: duplicate question id {native_id!r}")
        hard_rows[native_id] = row

    niah_variants, niah_sizes = _load_niah_variants(
        qa_dir / "niah", hard_rows, evidence_lookup, snapshot
    )

    question_rows: list[dict[str, Any]] = []
    question_counts = {"default": len(default_rows), "hard": len(hard_rows_list)}
    for subset, source, rows in (
        ("default", default_qa_path, default_rows),
        ("hard", hard_qa_path, hard_rows_list),
    ):
        seen_question_ids: set[str] = set()
        for index, row in enumerate(rows):
            native_id = _required_string(row, "id", source, index)
            if native_id in seen_question_ids:
                raise SchemaError(f"{source}: duplicate question id {native_id!r}")
            seen_question_ids.add(native_id)
            question = _required_string(row, "question", source, index)
            answer = _string_or_empty(row, "answer", source, index)
            notes = _string_or_empty(row, "notes", source, index)
            qtype = _required_string(row, "qtype", source, index)
            native_evidence = _string_list(row, "evidence_ids", source, index)
            metadata: dict[str, Any] = {
                "native_id": native_id,
                "evaluation_private": {"notes": notes},
            }
            variants = niah_variants.get(native_id)
            if variants:
                metadata["evaluation_variants"] = {"niah": variants}
            extra = {
                key: value
                for key, value in row.items()
                if key
                not in {"id", "question", "answer", "notes", "qtype", "evidence_ids"}
            }
            if extra:
                metadata["native"] = extra
            question_rows.append(
                {
                    "question_id": stable_id(BENCHMARK_KEY, "question", subset, native_id),
                    "context_id": CONTEXT_ID,
                    "subset": subset,
                    "split": "test",
                    "prompt": [content_text(question)],
                    "instruction": QA_INSTRUCTION,
                    "task": {
                        "category": "personalized_referential_memory_qa",
                        "subcategory": qtype,
                        "response_type": "text",
                    },
                    "choices": [],
                    "answer": {"text": answer},
                    "memory_scope": {"mode": "all"},
                    "evidence": _resolve_evidence(
                        native_evidence,
                        evidence_lookup,
                        source=source,
                        pointer=f"/{index}/evidence_ids",
                        relation="supports",
                    ),
                    "provenance": provenance(source, snapshot, f"/{index}"),
                    "metadata": metadata,
                }
            )

    revision = _git_revision(snapshot)
    manifest = _manifest(revision, memory_counts, question_counts, niah_sizes)
    with BundleWriter(output_root, manifest, overwrite=overwrite) as writer:
        writer.add_context(
            {
                "context_id": CONTEXT_ID,
                "benchmark": BENCHMARK_NAME,
                "metadata": {
                    "scope": "one shared long-term personal archive",
                    "modalities": ["email", "image", "video"],
                },
            }
        )
        for asset in sorted(asset_rows, key=lambda row: str(row["asset_id"])):
            writer.add_asset(asset)
        for memory in memory_rows:
            writer.add_memory(memory)
        for question in question_rows:
            writer.add_question(question)

    report = validate_bundle(output_root)
    return {
        "benchmark": BENCHMARK_NAME,
        "bundle": str(output_root),
        "raw_revision": revision,
        **report,
    }


__all__ = ["convert"]
