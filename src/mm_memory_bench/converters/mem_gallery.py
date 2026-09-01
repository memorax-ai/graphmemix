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
    guess_media_type,
    provenance,
    read_json,
    stable_id,
    validate_bundle,
)


BENCHMARK_KEY = "mem_gallery"
BENCHMARK_NAME = "Mem-Gallery"
OFFICIAL_DATA_REVISION = "af912daba984e896e253016b7c7e334ef92c2a6f"

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

_TASK_DIMENSIONS = {
    "FR": "extraction_and_adaptation",
    "VS": "extraction_and_adaptation",
    "TTL": "extraction_and_adaptation",
    "TR": "reasoning",
    "VR": "reasoning",
    "MR": "reasoning",
    "KR": "knowledge_management",
    "CD": "knowledge_management",
    "AR": "knowledge_management",
}

_FORMAT_CONSTRAINTS = {
    "AR": (
        "Provide your answer based on the information in the conversation. Only if "
        "the information about the question is not present in the conversation, "
        "reply with: “Not mentioned.”"
    ),
    "CD": (
        "Please check whether this information conflicts with the conversation, and "
        "reply strictly with either “Yes.” or “No.”"
    ),
    "VS": (
        "Return the image_id of the image(s). If there are multiple images, sort them "
        "in ascending order and separate them by commas. Format example: "
        "“D2:IMG_003, D2:IMG_010, D10:IMG_002” (for format reference only)."
    ),
}


def _locate_snapshot(raw_root: Path) -> Path:
    candidates = (
        raw_root,
        raw_root / BENCHMARK_KEY,
        raw_root / "Mem-Gallery",
        raw_root / "mem-gallery",
    )
    for candidate in candidates:
        if (candidate / "data" / "dialog").is_dir() and (
            candidate / "data" / "image"
        ).is_dir():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Mem-Gallery snapshot not found; searched: {searched}")


def _git_revision(snapshot: Path) -> str:
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
                    if (
                        len(parts) == 2
                        and parts[1] == ref_name
                        and _GIT_SHA_RE.fullmatch(parts[0])
                    ):
                        return parts[0]
    return OFFICIAL_DATA_REVISION


def _bundle_relative_path(path: Path, bundle_root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), bundle_root.resolve())).as_posix()


def _read_object(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise SchemaError(f"{path}: expected a JSON object")
    return value


def _required_string(
    row: Mapping[str, Any], field: str, source: Path, pointer: str
) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{source}:{pointer}/{field}: expected a non-empty string")
    return value


def _string_or_empty(
    row: Mapping[str, Any], field: str, source: Path, pointer: str
) -> str:
    value = row.get(field, "")
    if not isinstance(value, str):
        raise SchemaError(f"{source}:{pointer}/{field}: expected a string")
    return value


def _list_value(
    row: Mapping[str, Any], field: str, source: Path, pointer: str
) -> list[Any]:
    value = row.get(field, [])
    if not isinstance(value, list):
        raise SchemaError(f"{source}:{pointer}/{field}: expected an array")
    return list(value)


def _string_list(
    row: Mapping[str, Any], field: str, source: Path, pointer: str
) -> list[str]:
    value = _list_value(row, field, source, pointer)
    if any(not isinstance(item, str) for item in value):
        raise SchemaError(f"{source}:{pointer}/{field}: expected an array of strings")
    return value


def _query_image_values(
    row: Mapping[str, Any], field: str, source: Path, pointer: str
) -> list[str]:
    value = row.get(field)
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise SchemaError(f"{source}:{pointer}/{field}: expected a string or array of strings")


def _resolve_image(snapshot: Path, native_path: str) -> Path:
    path = Path(native_path)
    if path.is_absolute():
        return path
    normalized = native_path.replace("\\", "/")
    if normalized.startswith("../image/"):
        return snapshot / "data" / "image" / normalized[len("../image/") :]
    if normalized.startswith("./../image/"):
        return snapshot / "data" / "image" / normalized[len("./../image/") :]
    if normalized.startswith("data/image/"):
        return snapshot / normalized
    if normalized.startswith("image/"):
        return snapshot / "data" / normalized
    # This matches the official runner's fallback: join against data/image.
    return snapshot / "data" / "image" / normalized.lstrip("./")


def _asset_id(snapshot: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to((snapshot / "data" / "image").resolve())
        return stable_id(BENCHMARK_KEY, "asset", *relative.parts)
    except ValueError:
        return stable_id(BENCHMARK_KEY, "asset", *path.parts[-3:])


def _register_asset(
    *,
    writer: BundleWriter,
    registered: dict[str, str],
    snapshot: Path,
    output_root: Path,
    native_path: str,
) -> tuple[str, str]:
    image_path = _resolve_image(snapshot, native_path)
    dedup_key = str(image_path.resolve())
    previous = registered.get(dedup_key)
    media_type, mime_type = guess_media_type(image_path)
    if media_type != "image":
        raise SchemaError(f"Mem-Gallery media path is not an image: {native_path!r}")
    if previous is not None:
        return previous, media_type

    asset_id = _asset_id(snapshot, image_path)
    if asset_id in registered.values():
        raise SchemaError(f"asset id collision for {native_path!r}: {asset_id}")
    asset: dict[str, Any] = {
        "asset_id": asset_id,
        "media_type": media_type,
        "path": _bundle_relative_path(image_path, output_root),
        "provenance": provenance(image_path, snapshot),
        "metadata": {"native_path": native_path},
    }
    if mime_type:
        asset["mime_type"] = mime_type
    writer.add_asset(asset)
    registered[dedup_key] = asset_id
    return asset_id, media_type


def _history_image_content(
    *,
    writer: BundleWriter,
    registered: dict[str, str],
    snapshot: Path,
    output_root: Path,
    paths: list[str],
    native_ids: list[str],
    captions: list[str],
    source: Path,
    pointer: str,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image_index, native_path in enumerate(paths):
        asset_id, media_type = _register_asset(
            writer=writer,
            registered=registered,
            snapshot=snapshot,
            output_root=output_root,
            native_path=native_path,
        )
        native_id = native_ids[image_index] if image_index < len(native_ids) else None
        caption = captions[image_index] if image_index < len(captions) else None
        annotations: dict[str, Any] = {
            # The native JSON places media at dialogue-round scope. Assignment to the
            # split user memory is the explicitly agreed conversion convention.
            "native_scope": "dialogue_round",
            "role_assignment": "inferred_from_input_image",
            "native_image_index": image_index,
            "native_path": native_path,
        }
        if native_id is not None:
            annotations["native_image_id"] = native_id
        if caption is not None:
            annotations["native_image_caption"] = caption
        content.append(content_asset(media_type, asset_id, **annotations))
    if len(native_ids) > len(paths) or len(captions) > len(paths):
        raise SchemaError(
            f"{source}:{pointer}: image_id/image_caption has more entries than input_image"
        )
    return content


def _query_image_content(
    *,
    writer: BundleWriter,
    registered: dict[str, str],
    snapshot: Path,
    output_root: Path,
    paths: list[str],
    captions: list[str],
    source: Path,
    pointer: str,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image_index, native_path in enumerate(paths):
        asset_id, media_type = _register_asset(
            writer=writer,
            registered=registered,
            snapshot=snapshot,
            output_root=output_root,
            native_path=native_path,
        )
        annotations: dict[str, Any] = {
            "native_scope": "question",
            "role_assignment": "query",
            "native_image_index": image_index,
            "native_path": native_path,
        }
        if image_index < len(captions):
            annotations["native_image_caption"] = captions[image_index]
        content.append(content_asset(media_type, asset_id, **annotations))
    if len(captions) > len(paths):
        raise SchemaError(
            f"{source}:{pointer}: image_caption has more entries than question_image"
        )
    return content


def _evidence(
    clues: list[str],
    round_lookup: Mapping[str, tuple[str, str]],
    unresolved_stats: dict[str, int],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for clue_index, native_id in enumerate(clues):
        targets = round_lookup.get(native_id)
        if targets is None:
            result.append(
                {
                    "native_id": native_id,
                    "native_clue_index": clue_index,
                    "relation": "supports_round",
                    "unresolved": True,
                }
            )
            unresolved_stats["unresolved_clue_count"] += 1
            continue
        for role, memory_id in zip(("user", "assistant"), targets):
            result.append(
                {
                    "memory_id": memory_id,
                    "native_id": native_id,
                    "native_clue_index": clue_index,
                    "relation": "supports_round",
                    "role": role,
                }
            )
    return result


def _response_type(point: str) -> str:
    # Task-specific output constraints are public ``instruction`` strings.
    # The method-facing response carrier remains the common text interface.
    return "text"


def _manifest(revision: str, conversion_stats: dict[str, int]) -> dict[str, Any]:
    return {
        "benchmark": BENCHMARK_NAME,
        "dataset_id": "Ethan-Bei/Mem-Gallery",
        "description": (
            "Multimodal long-term conversational memory benchmark over multi-session "
            "persona-driven dialogues."
        ),
        "subsets": ["default"],
        "modalities": ["text", "image"],
        "source": {
            "dataset": "https://huggingface.co/datasets/Ethan-Bei/Mem-Gallery",
            "repository": "https://github.com/YuanchenBei/Mem-Gallery",
            "paper": "https://aclanthology.org/2026.acl-long.1892/",
            "paper_pdf": "https://aclanthology.org/2026.acl-long.1892.pdf",
            "official_runner": (
                "https://github.com/YuanchenBei/Mem-Gallery/blob/main/"
                "benchmark/run/run_bench.py"
            ),
            "revision": revision,
        },
        "raw_snapshot": {
            "revision_type": "git",
            "git_revision": revision,
            "converter_reference_revision": OFFICIAL_DATA_REVISION,
        },
        "license": {
            "id": "MIT",
            "declared_id": "MIT",
            "declared_at": "Hugging Face dataset metadata and repository README",
            "url": "https://opensource.org/license/mit",
            "caveat": (
                "The data snapshot's README links a LICENSE file that is absent from the "
                "published dataset snapshot. Images derive from multiple upstream datasets "
                "whose individual licenses are not itemized per asset."
            ),
        },
        "official_release_counts": {
            "contexts": 20,
            "sessions": 240,
            "dialogue_rounds": 3962,
            "history_images": 1003,
            "query_images": 487,
            "total_images": 1490,
            "questions": 1711,
            "questions_by_point": {
                "FR": 219,
                "VS": 306,
                "TTL": 337,
                "TR": 123,
                "VR": 174,
                "MR": 206,
                "KR": 81,
                "CD": 81,
                "AR": 184,
            },
        },
        "task_dimensions": {
            "extraction_and_adaptation": ["FR", "VS", "TTL"],
            "reasoning": ["TR", "VR", "MR"],
            "knowledge_management": ["KR", "CD", "AR"],
        },
        "evaluation": {
            "answer_metrics": ["F1", "BLEU-1", "Exact Match", "LLM-as-a-Judge"],
            "retrieval_metrics": ["Recall@K", "Precision@K", "Hit@K"],
            "format_constraints": dict(_FORMAT_CONSTRAINTS),
            "memory_visibility": {
                "mode": "all",
                "reason": (
                    "The official runner stores every processed dialogue before iterating "
                    "over human-annotated QAs. session_id is an association label, not a "
                    "memory cutoff."
                ),
            },
        },
        "conversion_report": conversion_stats,
        "conversion_notes": [
            "The native format attaches input_image to a dialogue round and does not explicitly assign the media to the user or assistant role.",
            "When splitting each native round into user and assistant memories, all round input_image assets are attached to the user memory with native_scope=dialogue_round and role_assignment=inferred_from_input_image annotations.",
            "Every native clue round maps to both split role memories so evidence remains round-level without guessing which utterance contains the support.",
            "History images and question images have distinct native_scope annotations; native captions remain annotations and are not promoted to model-visible text.",
            "Unlike the official runner, which selects input_image[0] and session_id[0], this conversion preserves every history image, every query image, and every QA session id.",
            "The Hugging Face Viewer treats the snapshot as image folders and does not expose its semantic QA/dialogue structure.",
        ],
        "risks": [
            "Per-image upstream provenance and license terms are not itemized in the public snapshot.",
            "The official runner merges user and assistant text into one retrieval unit; the unified format splits roles and retains round-level clue mappings to both units.",
            "Caption-based and raw-image memory methods are both supported by the official code and should be reported separately.",
        ],
    }


def convert(raw_root: Path, output_root: Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Convert the official Mem-Gallery snapshot into an mmmb-1.0 bundle."""
    snapshot = _locate_snapshot(Path(raw_root))
    output_root = Path(output_root)
    dialog_paths = sorted((snapshot / "data" / "dialog").glob("*.json"))
    if not dialog_paths:
        raise FileNotFoundError(f"no Mem-Gallery dialogue JSON files in {snapshot}")

    revision = _git_revision(snapshot)
    conversion_stats = {"unresolved_clue_count": 0}
    registered_assets: dict[str, str] = {}

    with BundleWriter(
        output_root, _manifest(revision, conversion_stats), overwrite=overwrite
    ) as writer:
        for source in dialog_paths:
            dataset = _read_object(source)
            scenario = source.stem
            context_id = stable_id(BENCHMARK_KEY, "context", scenario)
            profile = dataset.get("character_profile")
            if not isinstance(profile, dict):
                raise SchemaError(f"{source}:/character_profile: expected an object")
            name = profile.get("name", "")
            if not isinstance(name, str):
                raise SchemaError(f"{source}:/character_profile/name: expected a string")

            top_level_extra = {
                key: value
                for key, value in dataset.items()
                if key
                not in {
                    "character_profile",
                    "multi_session_dialogues",
                    "human-annotated QAs",
                }
            }
            context_metadata: dict[str, Any] = {
                "native_scenario": scenario,
                "native_character_profile": dict(profile),
                "official_evaluator_visibility": {
                    "visible_profile_fields": ["name"],
                    "usage": "speaker label only",
                    "not_ingested_as_memory": [
                        key for key in profile if key != "name"
                    ],
                },
            }
            if top_level_extra:
                context_metadata["native"] = top_level_extra
            writer.add_context(
                {
                    "context_id": context_id,
                    "benchmark": BENCHMARK_NAME,
                    "metadata": context_metadata,
                    "provenance": provenance(source, snapshot),
                }
            )

            sequence = 0
            writer.add_memory(
                {
                    "memory_id": stable_id(BENCHMARK_KEY, "memory", scenario, "profile"),
                    "context_id": context_id,
                    "sequence": sequence,
                    "kind": "profile",
                    "content": [
                        content_text(
                            name,
                            native_field="character_profile.name",
                            official_usage="speaker_label",
                        )
                    ],
                    "provenance": provenance(source, snapshot, "/character_profile"),
                    "metadata": {
                        "native": dict(profile),
                        "official_evaluator_visibility": {
                            "name_visible_in_speaker_label": True,
                            "profile_memory_stored_by_runner": False,
                        },
                    },
                }
            )
            sequence += 1

            sessions = dataset.get("multi_session_dialogues")
            if not isinstance(sessions, list):
                raise SchemaError(f"{source}:/multi_session_dialogues: expected an array")
            round_lookup: dict[str, tuple[str, str]] = {}
            for session_index, session in enumerate(sessions):
                session_pointer = f"/multi_session_dialogues/{session_index}"
                if not isinstance(session, dict):
                    raise SchemaError(f"{source}:{session_pointer}: expected an object")
                session_id = _required_string(
                    session, "session_id", source, session_pointer
                )
                session_date = _required_string(session, "date", source, session_pointer)
                dialogues = session.get("dialogues")
                if not isinstance(dialogues, list):
                    raise SchemaError(f"{source}:{session_pointer}/dialogues: expected an array")
                session_extra = {
                    key: value
                    for key, value in session.items()
                    if key not in {"session_id", "date", "dialogues"}
                }
                for round_index, dialogue in enumerate(dialogues):
                    pointer = f"{session_pointer}/dialogues/{round_index}"
                    if not isinstance(dialogue, dict):
                        raise SchemaError(f"{source}:{pointer}: expected an object")
                    native_round_id = _required_string(
                        dialogue, "round", source, pointer
                    )
                    if native_round_id in round_lookup:
                        raise SchemaError(
                            f"{source}:{pointer}/round: duplicate id {native_round_id!r}"
                        )
                    user_text = _string_or_empty(dialogue, "user", source, pointer)
                    assistant_text = _string_or_empty(
                        dialogue, "assistant", source, pointer
                    )
                    image_paths = _string_list(
                        dialogue, "input_image", source, pointer
                    )
                    image_ids = _string_list(dialogue, "image_id", source, pointer)
                    image_captions = _string_list(
                        dialogue, "image_caption", source, pointer
                    )
                    image_content = _history_image_content(
                        writer=writer,
                        registered=registered_assets,
                        snapshot=snapshot,
                        output_root=output_root,
                        paths=image_paths,
                        native_ids=image_ids,
                        captions=image_captions,
                        source=source,
                        pointer=pointer,
                    )

                    user_memory_id = stable_id(
                        BENCHMARK_KEY, "memory", scenario, native_round_id, "user"
                    )
                    assistant_memory_id = stable_id(
                        BENCHMARK_KEY,
                        "memory",
                        scenario,
                        native_round_id,
                        "assistant",
                    )
                    round_lookup[native_round_id] = (
                        user_memory_id,
                        assistant_memory_id,
                    )
                    dialogue_extra = {
                        key: value
                        for key, value in dialogue.items()
                        if key
                        not in {
                            "round",
                            "user",
                            "assistant",
                            "input_image",
                            "image_id",
                            "image_caption",
                        }
                    }
                    common_metadata: dict[str, Any] = {
                        "native_round_id": native_round_id,
                        "native_session_id": session_id,
                        "native_session_index": session_index,
                        "native_round_index": round_index,
                    }
                    native_extra = {**session_extra, **dialogue_extra}
                    if native_extra:
                        common_metadata["native"] = native_extra

                    for role, memory_id, content in (
                        ("user", user_memory_id, [content_text(user_text), *image_content]),
                        (
                            "assistant",
                            assistant_memory_id,
                            [content_text(assistant_text)],
                        ),
                    ):
                        role_metadata = dict(common_metadata)
                        if role == "user" and image_captions:
                            role_metadata["derived"] = {
                                "image_captions": list(image_captions)
                            }
                        if role == "user" and image_paths:
                            role_metadata["agent_visible"] = {
                                "image_ids": [
                                    image_ids[index] if index < len(image_ids) else ""
                                    for index in range(len(image_paths))
                                ]
                            }
                        memory_record = {
                            "memory_id": memory_id,
                            "context_id": context_id,
                            "session_id": session_id,
                            "sequence": sequence,
                            "timestamp": session_date,
                            "kind": "dialogue_message",
                            "role": role,
                            "content": content,
                            "provenance": provenance(source, snapshot, pointer),
                            "metadata": role_metadata,
                        }
                        if role == "user" and len(image_ids) == 1:
                            memory_record["source_id"] = image_ids[0]
                        writer.add_memory(memory_record)
                        sequence += 1

            questions = dataset.get("human-annotated QAs")
            if not isinstance(questions, list):
                raise SchemaError(f"{source}:/human-annotated QAs: expected an array")
            for question_index, row in enumerate(questions):
                pointer = f"/human-annotated QAs/{question_index}"
                if not isinstance(row, dict):
                    raise SchemaError(f"{source}:{pointer}: expected an object")
                question = _required_string(row, "question", source, pointer)
                answer = _required_string(row, "answer", source, pointer)
                point = _required_string(row, "point", source, pointer).upper()
                if point not in _TASK_DIMENSIONS:
                    raise SchemaError(f"{source}:{pointer}/point: unknown task {point!r}")
                session_ids = _string_list(row, "session_id", source, pointer)
                clues = _string_list(row, "clue", source, pointer)
                query_paths = _query_image_values(
                    row, "question_image", source, pointer
                )
                query_captions = _query_image_values(
                    row, "image_caption", source, pointer
                )
                prompt = [content_text(question)]
                prompt.extend(
                    _query_image_content(
                        writer=writer,
                        registered=registered_assets,
                        snapshot=snapshot,
                        output_root=output_root,
                        paths=query_paths,
                        captions=query_captions,
                        source=source,
                        pointer=pointer,
                    )
                )
                metadata: dict[str, Any] = {
                    "native_source_index": question_index,
                    "native_point": point,
                    "native_session_ids": list(session_ids),
                    "native_clues": list(clues),
                }
                if point in _FORMAT_CONSTRAINTS:
                    metadata["official_format_constraint"] = _FORMAT_CONSTRAINTS[point]
                extra = {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "point",
                        "question",
                        "answer",
                        "session_id",
                        "clue",
                        "question_image",
                        "image_caption",
                    }
                }
                if extra:
                    metadata["native"] = extra

                question_record: dict[str, Any] = {
                    "question_id": stable_id(
                        BENCHMARK_KEY,
                        "question",
                        scenario,
                        f"{question_index:04d}",
                    ),
                    "context_id": context_id,
                    "subset": "default",
                    "split": "test",
                    "prompt": prompt,
                    "task": {
                        "category": _TASK_DIMENSIONS[point],
                        "subcategory": point,
                        "response_type": _response_type(point),
                    },
                    "choices": [],
                    "answer": {
                        "text": answer,
                        "unanswerable": point == "AR",
                    },
                    "memory_scope": {"mode": "all"},
                    "query_at": {"session_ids": session_ids},
                    "evidence": _evidence(clues, round_lookup, conversion_stats),
                    "provenance": provenance(source, snapshot, pointer),
                    "metadata": metadata,
                }
                if point in _FORMAT_CONSTRAINTS:
                    question_record["instruction"] = _FORMAT_CONSTRAINTS[point]
                writer.add_question(question_record)

    report = validate_bundle(output_root)
    return {
        "benchmark": BENCHMARK_NAME,
        "bundle": str(output_root),
        "raw_revision": revision,
        "unresolved_clue_count": conversion_stats["unresolved_clue_count"],
        **report,
    }


__all__ = ["convert"]
