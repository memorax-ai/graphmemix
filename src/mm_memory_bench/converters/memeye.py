from __future__ import annotations

import hashlib
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


BENCHMARK_KEY = "memeye"
BENCHMARK_NAME = "MemEye"
OFFICIAL_CODE_REVISION = "0358e70d714980980dfd8c87903384db474f0b16"
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return text or "unknown"


def _locate_snapshot(raw_root: Path) -> Path:
    candidates = (raw_root, raw_root / BENCHMARK_KEY, raw_root / "MemEye")
    for candidate in candidates:
        dialog_dir = candidate / "data" / "dialog"
        if dialog_dir.is_dir() and any(dialog_dir.glob("*.json")):
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"MemEye snapshot not found; searched: {searched}")


def _git_revision(snapshot: Path) -> str | None:
    """Read a normal/worktree git HEAD without shelling out to git."""
    git_dir = snapshot / ".git"
    if git_dir.is_file():
        marker = git_dir.read_text(encoding="utf-8").strip()
        if marker.startswith("gitdir:"):
            git_dir = (snapshot / marker.split(":", 1)[1].strip()).resolve()
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if _GIT_SHA_RE.fullmatch(head):
        return head
    if not head.startswith("ref:"):
        return None
    ref_name = head.split(":", 1)[1].strip()
    ref_path = git_dir / ref_name
    if ref_path.is_file():
        value = ref_path.read_text(encoding="utf-8").strip()
        if _GIT_SHA_RE.fullmatch(value):
            return value
    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line.startswith(("#", "^")):
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2 and parts[1] == ref_name and _GIT_SHA_RE.fullmatch(parts[0]):
                return parts[0]
    return None


def _object(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise SchemaError(f"{path}: expected a JSON object")
    return value


def _list(value: Any, *, source: Path, pointer: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise SchemaError(f"{source}:{pointer}: expected a list")
    return value


def _qas(payload: Mapping[str, Any], source: Path) -> list[dict[str, Any]]:
    for key in ("human-annotated QAs", "human_annotated_qas", "qas"):
        value = payload.get(key)
        if value is None:
            continue
        rows = _list(value, source=source, pointer=f"/{key}")
        if any(not isinstance(row, dict) for row in rows):
            raise SchemaError(f"{source}:/{key}: every QA must be an object")
        return list(rows)
    return []


def _bundle_relative(path: Path, bundle_root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), bundle_root.resolve())).as_posix()


def _resolve_asset(snapshot: Path, source: Path, native_path: str) -> Path:
    cleaned = native_path.replace("file://", "")
    value = Path(cleaned)
    if value.is_absolute():
        return value
    normalized = cleaned
    for prefix in ("../image/", "./image/", "image/", "data/image/"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    candidates = (
        source.parent / value,
        source.parent.parent / value,
        snapshot / value,
        snapshot / "data" / "image" / normalized,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # Keep a deterministic raw-relative reference even for an incomplete fixture.
    return snapshot / "data" / "image" / normalized


def _asset_id(path: Path, snapshot: Path) -> str:
    try:
        native = path.resolve().relative_to(snapshot.resolve()).as_posix()
    except ValueError:
        native = path.resolve().as_posix()
    digest = hashlib.sha1(native.encode("utf-8")).hexdigest()[:12]
    return stable_id(BENCHMARK_KEY, "asset", _slug(path.stem), digest)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _points(value: Any) -> tuple[list[Any], list[str]]:
    native = value if isinstance(value, list) else []
    tags: list[str] = []
    for group in native:
        if isinstance(group, list):
            tags.extend(str(item) for item in group)
        elif group is not None:
            tags.append(str(group))
    return native, tags


def _manifest(
    requested_revision: str, resolved_revision: str | None, semantic_count: int, physical_count: int
) -> dict[str, Any]:
    return {
        "benchmark": BENCHMARK_NAME,
        "dataset_id": "MemEyeBench/MemEye",
        "description": "Mirrored open and rotated-MCQ multimodal multi-session memory QA.",
        "subsets": ["mcq", "open"],
        "modalities": ["text", "image"],
        "source": {
            "dataset": "https://huggingface.co/datasets/MemEyeBench/MemEye",
            "repository": "https://github.com/MinghoKwok/MemEye",
            "paper": "https://arxiv.org/abs/2605.15128",
        },
        "raw_snapshot": {
            "requested_revision": requested_revision,
            "resolved_data_revision": resolved_revision,
            "revision_status": (
                "resolved git commit" if resolved_revision else "unresolved local snapshot"
            ),
            "official_code_revision": OFFICIAL_CODE_REVISION,
            "warning": "Pin the Hugging Face commit when publishing results.",
        },
        "license": {
            "dataset_card_declared": "Apache-2.0",
            "dataset_card_scope_warning": (
                "The card explicitly describes the code/documentation license; it does not "
                "establish clean-room rights for every bundled image."
            ),
            "upstream_asset_risk": {
                "status": "mixed or third-party",
                "examples": ["Pitt Image Ads", "stock imagery", "generated game imagery"],
                "action": "Audit original image terms before redistribution or commercial use.",
            },
        },
        "release_counts": {
            "semantic_questions": semantic_count,
            "physical_question_rows": physical_count,
            "physical_definition": "four MCQ rotations plus one open mirror when both exist",
        },
        "evaluation": {
            "official_modes": {
                "multiple_choice": "accuracy aggregated across four option rotations",
                "open": "official open-answer evaluation protocol",
            },
            "variant_storage": {
                "semantic_key": "questions.semantic_question_id",
                "variant_key": "questions.variant",
            },
            "memory_visibility": (
                "The official full-context runner ingests every session. Target-session and "
                "clue-only modes are separate oracle/ablation methods."
            ),
        },
        "normalization": {
            "ignored_files": "data/dialog/concat_*.json (derived concatenations)",
            "context_granularity": (
                "one mode-specific context per task (eight MCQ plus eight open); this preserves "
                "small source-history differences between the mirrored files"
            ),
            "media": "relative references into the raw snapshot; files are not copied",
            "input_image_role": "assigned to the user message and explicitly annotated as inferred",
        },
        "version_drift": {
            "paper_v1_image_count": 438,
            "public_snapshot_observed_image_count": 495,
            "warning": "Report the pinned data revision because the public snapshot changed after v1.",
        },
    }


def convert(
    raw_root: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
    revision: str = "main",
) -> dict[str, Any]:
    """Convert the canonical (non-concat) MemEye task files."""
    snapshot = _locate_snapshot(Path(raw_root))
    output_root = Path(output_root)
    dialog_dir = snapshot / "data" / "dialog"

    canonical = sorted(
        path
        for path in dialog_dir.glob("*.json")
        if not path.stem.lower().startswith("concat_")
    )
    if not canonical:
        raise FileNotFoundError(f"no canonical MemEye dialog JSON found under {dialog_dir}")

    pairs: dict[str, dict[str, Path]] = {}
    for path in canonical:
        if path.stem.endswith("_Open"):
            pairs.setdefault(path.stem[: -len("_Open")], {})["open"] = path
        else:
            pairs.setdefault(path.stem, {})["mcq"] = path

    contexts: list[dict[str, Any]] = []
    memories: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    assets: dict[str, dict[str, Any]] = {}
    semantic_ids: set[str] = set()

    def register_asset(native_path: str, source: Path) -> str:
        path = _resolve_asset(snapshot, source, native_path)
        asset_id = _asset_id(path, snapshot)
        if asset_id not in assets:
            media_type, mime_type = guess_media_type(path)
            assets[asset_id] = {
                "asset_id": asset_id,
                "media_type": media_type,
                "path": _bundle_relative(path, output_root),
                **({"mime_type": mime_type} if mime_type else {}),
                "provenance": provenance(path, snapshot),
                "metadata": {"native_path": native_path},
            }
        return asset_id

    for task_name in sorted(pairs):
        sources = pairs[task_name]
        task_slug = _slug(task_name)
        context_ids: dict[str, str] = {}
        round_lookups: dict[str, dict[str, list[str]]] = {}
        profiles: dict[str, dict[str, Any]] = {}
        for mode in ("mcq", "open"):
            context_source = sources.get(mode)
            if context_source is None:
                continue
            payload = _object(context_source)
            profile_value = payload.get("character_profile")
            profile = profile_value if isinstance(profile_value, dict) else {}
            raw_sessions = payload.get("multi_session_dialogues", [])
            native_sessions = [
                {key: value for key, value in session.items() if key != "dialogues"}
                for session in raw_sessions
                if isinstance(session, dict)
            ] if isinstance(raw_sessions, list) else []
            profiles[mode] = profile
            context_id = stable_id(BENCHMARK_KEY, task_slug, mode)
            context_ids[mode] = context_id
            contexts.append(
                {
                    "context_id": context_id,
                    "benchmark": BENCHMARK_NAME,
                    "profile": ({"name": profile["name"]} if "name" in profile else {}),
                    "metadata": {
                        "native_task_name": task_name,
                        "evaluation_mode": mode,
                        "source_file": provenance(context_source, snapshot)["source_file"],
                        "paired_source_file": (
                            provenance(sources["open" if mode == "mcq" else "mcq"], snapshot)[
                                "source_file"
                            ]
                            if ("open" if mode == "mcq" else "mcq") in sources
                            else None
                        ),
                        "native": {
                            "character_profile": profile,
                            "sessions": native_sessions,
                        },
                    },
                }
            )

            round_lookup: dict[str, list[str]] = {}
            round_lookups[mode] = round_lookup
            sequence = 0
            sessions = _list(
                payload.get("multi_session_dialogues", []),
                source=context_source,
                pointer="/multi_session_dialogues",
            )
            for session_index, session in enumerate(sessions):
                if not isinstance(session, dict):
                    raise SchemaError(
                        f"{context_source}:/multi_session_dialogues/{session_index}: "
                        "expected an object"
                    )
                native_session_id = str(session.get("session_id", f"session_{session_index}"))
                date = session.get("date")
                dialogues = _list(
                    session.get("dialogues", []),
                    source=context_source,
                    pointer=f"/multi_session_dialogues/{session_index}/dialogues",
                )
                for round_index, dialogue in enumerate(dialogues):
                    pointer = f"/multi_session_dialogues/{session_index}/dialogues/{round_index}"
                    if not isinstance(dialogue, dict):
                        raise SchemaError(f"{context_source}:{pointer}: expected an object")
                    round_id = str(dialogue.get("round", f"{native_session_id}:{round_index}"))
                    image_paths = _string_list(dialogue.get("input_image", []))
                    captions = _string_list(dialogue.get("image_caption", []))
                    image_ids = _string_list(dialogue.get("image_id", []))

                    user_content = [content_text(dialogue.get("user", ""))]
                    for image_index, native_path in enumerate(image_paths):
                        annotations: dict[str, Any] = {
                            "source_field": "input_image",
                            "assigned_role": "user",
                            "role_assignment_inferred": True,
                            "image_index": image_index,
                        }
                        if image_index < len(image_ids):
                            annotations["native_image_id"] = image_ids[image_index]
                        if image_index < len(captions):
                            annotations["derived"] = {
                                "kind": "image_caption",
                                "text": captions[image_index],
                                "source_field": "image_caption",
                            }
                        user_content.append(
                            content_asset(
                                "image",
                                register_asset(native_path, context_source),
                                **annotations,
                            )
                        )

                    shared_metadata: dict[str, Any] = {
                        "native_round_id": round_id,
                        "evaluation_mode": mode,
                        "session_index": session_index,
                        "round_index": round_index,
                        "native": {
                            key: value
                            for key, value in dialogue.items()
                            if key not in {"user", "assistant", "image_caption"}
                        },
                    }
                    if captions:
                        shared_metadata["derived"] = {
                            "image_captions": captions,
                            "source_field": "image_caption",
                        }

                    round_memories: list[str] = []
                    for role, content in (
                        ("user", user_content),
                        ("assistant", [content_text(dialogue.get("assistant", ""))]),
                    ):
                        memory_id = stable_id(
                            BENCHMARK_KEY,
                            task_slug,
                            mode,
                            "memory",
                            native_session_id,
                            round_id,
                            role,
                        )
                        memory: dict[str, Any] = {
                            "memory_id": memory_id,
                            "context_id": context_id,
                            "session_id": native_session_id,
                            "sequence": sequence,
                            "kind": "dialogue_message",
                            "role": role,
                            "content": content,
                            "provenance": provenance(context_source, snapshot, pointer),
                            "metadata": {**shared_metadata, "native_role": role},
                        }
                        if date is not None:
                            memory["timestamp"] = str(date)
                        memories.append(memory)
                        round_memories.append(memory_id)
                        sequence += 1
                    if round_id in round_lookup:
                        raise SchemaError(f"{context_source}: duplicate round id {round_id!r}")
                    round_lookup[round_id] = round_memories

        def task_subcategory(mode: str) -> str:
            mode_profile = profiles.get(mode, {})
            return str(mode_profile.get("task_family", task_slug))

        mcq_source = sources.get("mcq")
        open_source = sources.get("open")
        mcq_rows = _qas(_object(mcq_source), mcq_source) if mcq_source else []
        open_rows = _qas(_object(open_source), open_source) if open_source else []
        open_by_id: dict[str, tuple[int, dict[str, Any]]] = {}
        for index, row in enumerate(open_rows):
            native_id = str(row.get("question_id", "")).strip()
            if not native_id:
                raise SchemaError(f"{open_source}:/human-annotated QAs/{index}: missing question_id")
            if native_id in open_by_id:
                raise SchemaError(f"{open_source}: duplicate question_id {native_id!r}")
            open_by_id[native_id] = (index, row)

        def common_question_fields(
            row: Mapping[str, Any],
            source: Path,
            index: int,
            semantic_id: str,
            mode: str,
        ) -> dict[str, Any]:
            prompt = [content_text(row.get("question", ""))]
            question_images: list[str] = []
            single = row.get("question_image")
            if isinstance(single, str) and single:
                question_images.append(single)
            elif isinstance(single, list):
                question_images.extend(str(value) for value in single)
            plural = row.get("question_images")
            if isinstance(plural, list):
                question_images.extend(str(value) for value in plural)
            for image_index, native_path in enumerate(question_images):
                prompt.append(
                    content_asset(
                        "image",
                        register_asset(native_path, source),
                        source_field="question_image",
                        image_index=image_index,
                    )
                )

            clue = _string_list(row.get("clue", []))
            evidence: list[dict[str, Any]] = []
            for native_round in clue:
                targets = round_lookups.get(mode, {}).get(native_round)
                if targets:
                    for memory_id in targets:
                        evidence.append(
                            {
                                "memory_id": memory_id,
                                "native_id": native_round,
                                "relation": "supports",
                            }
                        )
                else:
                    evidence.append(
                        {
                            "native_id": native_round,
                            "relation": "supports",
                            "unresolved": True,
                        }
                    )
            point, tags = _points(row.get("point", []))
            session_ids = _string_list(row.get("session_id", []))
            native_extra = {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "question_id",
                    "question",
                    "answer",
                    "point",
                    "session_id",
                    "clue",
                    "options",
                    "question_image",
                    "question_images",
                }
            }
            return {
                "context_id": context_ids[mode],
                "semantic_question_id": semantic_id,
                "subset": "default",
                "split": "test",
                "prompt": prompt,
                "memory_scope": {"mode": "all"},
                "evidence": evidence,
                "query_at": {"session_ids": session_ids},
                "tags": tags,
                "provenance": provenance(source, snapshot, f"/human-annotated QAs/{index}"),
                "metadata": {
                    "native_question_id": str(row.get("question_id", "")),
                    "native_point": point,
                    "native_session_ids": session_ids,
                    "native_clue_round_ids": clue,
                    "native": native_extra,
                },
            }

        seen_mcq: set[str] = set()
        for qa_index, row in enumerate(mcq_rows):
            native_id = str(row.get("question_id", "")).strip()
            if not native_id:
                raise SchemaError(f"{mcq_source}:/human-annotated QAs/{qa_index}: missing question_id")
            if native_id in seen_mcq:
                raise SchemaError(f"{mcq_source}: duplicate question_id {native_id!r}")
            seen_mcq.add(native_id)
            semantic_id = stable_id(BENCHMARK_KEY, task_slug, "semantic", native_id)
            semantic_ids.add(semantic_id)
            rotations = row.get("options", [])
            if not isinstance(rotations, list):
                raise SchemaError(
                    f"{mcq_source}:/human-annotated QAs/{qa_index}/options: expected a list"
                )
            if len(rotations) != 4:
                raise SchemaError(
                    f"{mcq_source}:/human-annotated QAs/{qa_index}/options: "
                    f"expected four official rotations, found {len(rotations)}"
                )
            for rotation_index, rotation in enumerate(rotations):
                if not isinstance(rotation, dict):
                    raise SchemaError(
                        f"{mcq_source}:/human-annotated QAs/{qa_index}/options/{rotation_index}: "
                        "expected an object"
                    )
                missing_letters = [letter for letter in ("A", "B", "C", "D") if letter not in rotation]
                if missing_letters:
                    raise SchemaError(
                        f"{mcq_source}:/human-annotated QAs/{qa_index}/options/"
                        f"{rotation_index}: missing choices {missing_letters}"
                    )
                choices = [
                    {"choice_id": letter, "text": str(rotation[letter])}
                    for letter in ("A", "B", "C", "D")
                    if letter in rotation
                ]
                answer_id = str(rotation.get("answer", ""))
                if answer_id not in {"A", "B", "C", "D"}:
                    raise SchemaError(
                        f"{mcq_source}:/human-annotated QAs/{qa_index}/options/"
                        f"{rotation_index}/answer: expected A, B, C, or D"
                    )
                question = {
                    **common_question_fields(row, mcq_source, qa_index, semantic_id, "mcq"),
                    "question_id": stable_id(
                        BENCHMARK_KEY,
                        task_slug,
                        "question",
                        native_id,
                        "mcq",
                        f"rotation_{rotation_index + 1}",
                    ),
                    "variant": {
                        "response_format": "multiple_choice",
                        "rotation_index": rotation_index,
                    },
                    "task": {
                        "category": "multimodal_multi_session_memory",
                        "subcategory": task_subcategory("mcq"),
                        "response_type": "choice",
                    },
                    "choices": choices,
                    "answer": {"text": answer_id, "choice_ids": [answer_id] if answer_id else []},
                }
                question["subset"] = "mcq"
                question["provenance"] = provenance(
                    mcq_source,
                    snapshot,
                    f"/human-annotated QAs/{qa_index}/options/{rotation_index}",
                )
                question["metadata"]["native_semantic_mcq_answer"] = row.get("answer")
                question["metadata"]["native_option_rotation"] = rotation
                questions.append(question)

            open_match = open_by_id.pop(native_id, None)
            if open_match and open_source:
                open_index, open_row = open_match
                question = {
                    **common_question_fields(
                        open_row, open_source, open_index, semantic_id, "open"
                    ),
                    "question_id": stable_id(
                        BENCHMARK_KEY, task_slug, "question", native_id, "open"
                    ),
                    "variant": {"response_format": "open"},
                    "task": {
                        "category": "multimodal_multi_session_memory",
                        "subcategory": task_subcategory("open"),
                        "response_type": "text",
                    },
                    "choices": [],
                    "answer": {"text": str(open_row.get("answer", ""))},
                }
                question["subset"] = "open"
                questions.append(question)

        # Preserve an open-only row rather than silently dropping an imperfect snapshot.
        if open_source:
            for native_id, (open_index, row) in open_by_id.items():
                semantic_id = stable_id(BENCHMARK_KEY, task_slug, "semantic", native_id)
                semantic_ids.add(semantic_id)
                questions.append(
                    {
                        **common_question_fields(
                            row, open_source, open_index, semantic_id, "open"
                        ),
                        "question_id": stable_id(
                            BENCHMARK_KEY, task_slug, "question", native_id, "open"
                        ),
                        "variant": {"response_format": "open"},
                        "task": {
                            "category": "multimodal_multi_session_memory",
                            "subcategory": task_subcategory("open"),
                            "response_type": "text",
                        },
                        "choices": [],
                        "answer": {"text": str(row.get("answer", ""))},
                        "subset": "open",
                    }
                )

    resolved_revision = _git_revision(snapshot)
    manifest = _manifest(revision, resolved_revision, len(semantic_ids), len(questions))
    with BundleWriter(output_root, manifest, overwrite=overwrite) as writer:
        context_memories: dict[str, list[dict[str, Any]]] = {}
        context_questions: dict[str, list[dict[str, Any]]] = {}
        for row in memories:
            context_memories.setdefault(str(row["context_id"]), []).append(row)
        for row in questions:
            context_questions.setdefault(str(row["context_id"]), []).append(row)
        for context in contexts:
            context_id = str(context["context_id"])
            writer.add_context(context)
            for memory in context_memories.get(context_id, []):
                writer.add_memory(memory)
            for question in context_questions.get(context_id, []):
                writer.add_question(question)
        for asset in sorted(assets.values(), key=lambda row: str(row["asset_id"])):
            writer.add_asset(asset)

    report = validate_bundle(output_root)
    return {
        "benchmark": BENCHMARK_NAME,
        "bundle": str(output_root),
        "raw_revision": resolved_revision or revision,
        "semantic_questions": len(semantic_ids),
        "physical_question_rows": len(questions),
        **report,
    }


__all__ = ["convert"]
