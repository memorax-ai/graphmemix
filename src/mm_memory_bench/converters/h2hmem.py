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


BENCHMARK_KEY = "h2hmem"
BENCHMARK_NAME = "H2HMem"
OFFICIAL_DATA_REVISION = "555a613df9dc462b42e4edd53ff97e799572da99"
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SESSION_RE = re.compile(r"^session(\d+)$")
_MULTIPARTY_SESSION_ALIAS_RE = re.compile(r"^S(\d+)-(\d+)$")

_SUBTASKS = {
    "Unimodal Precise Recall": "UPR",
    "Cross-modal Related Retrieval": "CRR",
    "Knowledge Resolution": "KR",
    "Temporal Reasoning": "TR",
    "Multimodal Causal Inference": "MCR",
    "Multimodal Causal Reasoning": "MCR",
    "Reference & Evolution Tracking": "RET",
    "Test-Time Learning": "TTL",
    "Conflict Detection": "CD",
    "Answer Refusal": "AR",
}


def _locate_snapshot(raw_root: Path) -> Path:
    for candidate in (raw_root, raw_root / BENCHMARK_KEY, raw_root / "H2HMEM"):
        if (candidate / "dyadic").is_dir() and (candidate / "multi-party").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"H2HMem snapshot not found under {raw_root}")


def _git_revision(snapshot: Path) -> str:
    git_dir = snapshot / ".git"
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if _GIT_SHA_RE.fullmatch(head):
        return head
    if head.startswith("ref:"):
        ref = head.split(":", 1)[1].strip()
        ref_path = git_dir / ref
        if ref_path.is_file():
            value = ref_path.read_text(encoding="utf-8").strip()
            if _GIT_SHA_RE.fullmatch(value):
                return value
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                parts = line.split(" ", 1)
                if len(parts) == 2 and parts[1] == ref and _GIT_SHA_RE.fullmatch(parts[0]):
                    return parts[0]
    return OFFICIAL_DATA_REVISION


def _session_number(path: Path) -> int:
    match = _SESSION_RE.fullmatch(path.name)
    if not match:
        raise SchemaError(f"invalid H2HMem session directory: {path}")
    return int(match.group(1))


def _canonical_answer_session(
    interaction_type: str, dialogue_name: str, native_session: str
) -> str:
    """Normalize the official multi-party S<dialogue>-<session> aliases."""
    match = _MULTIPARTY_SESSION_ALIAS_RE.fullmatch(native_session)
    dialogue_match = re.fullmatch(r"dialogue(\d+)", dialogue_name)
    if (
        interaction_type == "multi-party"
        and match
        and dialogue_match
        and match.group(1) == dialogue_match.group(1)
    ):
        return f"session{int(match.group(2))}"
    return native_session


def _object(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise SchemaError(f"{path}: expected an object")
    return value


def _rows(value: Any, path: Path, pointer: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise SchemaError(f"{path}:{pointer}: expected an array of objects")
    return [dict(row) for row in value]


def _required_text(value: Any, path: Path, pointer: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{path}:{pointer}: expected a non-empty string")
    return value


def _relative(path: Path, output_root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), output_root.resolve())).as_posix()


def _manifest(revision: str) -> dict[str, Any]:
    return {
        "benchmark": BENCHMARK_NAME,
        "dataset_id": "varib/H2HMEM",
        "description": (
            "Multimodal agent memory over dyadic and multi-party human-human "
            "multi-session interactions."
        ),
        "subsets": ["default"],
        "modalities": ["text", "image"],
        "source": {
            "dataset": "https://huggingface.co/datasets/varib/H2HMEM",
            "repository": "https://github.com/varib1/H2HMEM",
            "paper": "https://arxiv.org/abs/2606.09461",
        },
        "raw_snapshot": {
            "revision_type": "git",
            "git_revision": revision,
            "converter_reference_revision": OFFICIAL_DATA_REVISION,
        },
        "license": {"id": "MIT", "declared_at": "official dataset card"},
        "official_release_counts": {
            "contexts": 25,
            "sessions_reported": 309,
            "session_json_files": 308,
            "memories": 7078,
            "assets": 1300,
            "questions": 2236,
        },
        "protocol": {
            "context_unit": "one complete dialogue",
            "memory_visibility": "all sessions in the dialogue, matching official memory baselines",
            "session0": "cross-session question container; it is not an ingestible session",
            "evaluation": "official release provides lexical metrics and a graded LLM judge; canonical main track uses the locked binary LLM judge",
        },
    }


def convert(
    raw_root: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
    revision: str | None = None,
    strict_release_counts: bool = True,
) -> dict[str, Any]:
    snapshot = _locate_snapshot(Path(raw_root))
    resolved_revision = revision or _git_revision(snapshot)
    manifest = _manifest(resolved_revision)
    registered_assets: dict[Path, str] = {}
    counts = {"session_json_files": 0, "semantic_questions": 0}

    with BundleWriter(output_root, manifest, overwrite=overwrite) as writer:
        def register_asset(path: Path, source: Path, field: str) -> str:
            resolved = path.resolve()
            if resolved in registered_assets:
                return registered_assets[resolved]
            if not resolved.is_file():
                raise FileNotFoundError(f"{source}: referenced H2HMem image is missing: {path}")
            media_type, mime_type = guess_media_type(resolved)
            if resolved.suffix.lower() == ".webp":
                media_type, mime_type = "image", "image/webp"
            if media_type != "image":
                raise SchemaError(f"{source}: expected an image, found {resolved}")
            native = resolved.relative_to(snapshot).as_posix()
            asset_id = stable_id(BENCHMARK_KEY, "asset", native)
            record: dict[str, Any] = {
                "asset_id": asset_id,
                "media_type": "image",
                "path": _relative(resolved, output_root),
                "provenance": provenance(resolved, snapshot),
                "metadata": {"native_path": native, "first_reference_field": field},
            }
            if mime_type:
                record["mime_type"] = mime_type
            writer.add_asset(record)
            registered_assets[resolved] = asset_id
            return asset_id

        for interaction_type in ("dyadic", "multi-party"):
            track_root = snapshot / interaction_type
            dialogue_dirs = sorted(
                (path for path in track_root.glob("dialogue*") if path.is_dir()),
                key=lambda path: int(path.name.removeprefix("dialogue")),
            )
            for dialogue_dir in dialogue_dirs:
                dialogue_name = dialogue_dir.name
                context_id = stable_id(BENCHMARK_KEY, interaction_type, dialogue_name)
                scenes = dialogue_dir / "scenes"
                session_dirs = sorted(
                    (path for path in scenes.glob("session*") if path.is_dir()),
                    key=_session_number,
                )
                ingestible = [path for path in session_dirs if (path / "session.json").is_file()]
                sources = []
                session_memory_ids: dict[str, list[str]] = {}
                session_end_sequence: dict[str, int] = {}
                sequence = 0
                for session_dir in ingestible:
                    source = session_dir / "session.json"
                    session = _object(source)
                    native_session = session_dir.name
                    dialogue = _rows(session.get("dialogue"), source, "/dialogue")
                    counts["session_json_files"] += 1
                    sources.append(
                        {
                            "source_id": native_session,
                            "kind": "conversation_session",
                            "timestamp": str(session.get("timeline_date", "")),
                            "metadata": {
                                key: value
                                for key, value in session.items()
                                if key != "dialogue"
                            },
                        }
                    )
                    ids: list[str] = []
                    for turn_index, turn in enumerate(dialogue):
                        content = turn.get("content")
                        if not isinstance(content, Mapping):
                            raise SchemaError(f"{source}:/dialogue/{turn_index}/content: expected object")
                        speaker = _required_text(turn.get("role"), source, f"/dialogue/{turn_index}/role")
                        parts = [content_text(content.get("text", ""))]
                        image_name = content.get("image", "")
                        if image_name:
                            if not isinstance(image_name, str):
                                raise SchemaError(f"{source}:/dialogue/{turn_index}/content/image: expected string")
                            asset_id = register_asset(
                                session_dir / "image" / image_name,
                                source,
                                "dialogue.content.image",
                            )
                            parts.append(content_asset("image", asset_id, source_field="content.image"))
                        memory_id = stable_id(
                            BENCHMARK_KEY,
                            interaction_type,
                            dialogue_name,
                            native_session,
                            "turn",
                            f"{turn_index:04d}",
                        )
                        memory_record: dict[str, Any] = {
                            "memory_id": memory_id,
                            "context_id": context_id,
                            "session_id": native_session,
                            "sequence": sequence,
                            "kind": "dialogue_message",
                            "role": "participant",
                            "speaker": speaker,
                            "timestamp": str(session.get("timeline_date", "")),
                            "content": parts,
                            "provenance": provenance(
                                source, snapshot, f"/dialogue/{turn_index}"
                            ),
                            "metadata": {
                                "interaction_type": interaction_type,
                                "native_role": speaker,
                                "native_turn_index": turn_index,
                                "session_title": session.get("session_title", ""),
                                "theme": session.get("theme", ""),
                            },
                        }
                        if image_name:
                            # H2HMem has official questions whose answers require
                            # both the session and the original image file name.
                            memory_record["source_id"] = f"{native_session}/{image_name}"
                        writer.add_memory(memory_record)
                        ids.append(memory_id)
                        sequence += 1
                    session_memory_ids[native_session] = ids
                    session_end_sequence[native_session] = sequence - 1

                writer.add_context(
                    {
                        "context_id": context_id,
                        "benchmark": BENCHMARK_NAME,
                        "split": "test",
                        "sources": sources,
                        "metadata": {
                            "interaction_type": interaction_type,
                            "native_dialogue": dialogue_name,
                            "session_count": len(ingestible),
                        },
                    }
                )

                for session_dir in session_dirs:
                    question_source = session_dir / "questions.json"
                    if not question_source.is_file():
                        continue
                    payload = _object(question_source)
                    questions = _rows(payload.get("questions", []), question_source, "/questions")
                    native_question_session = session_dir.name
                    for question_index, row in enumerate(questions):
                        pointer = f"/questions/{question_index}"
                        question_value = row.get("question")
                        if not isinstance(question_value, Mapping):
                            raise SchemaError(f"{question_source}:{pointer}/question: expected object")
                        prompt = [
                            content_text(
                                _required_text(
                                    question_value.get("text"),
                                    question_source,
                                    f"{pointer}/question/text",
                                )
                            )
                        ]
                        question_image = question_value.get("image", "")
                        if question_image:
                            if not isinstance(question_image, str):
                                raise SchemaError(f"{question_source}:{pointer}/question/image: expected string")
                            if native_question_session == "session0":
                                try:
                                    image_session, image_name = question_image.split("/", 1)
                                except ValueError as exc:
                                    raise SchemaError(
                                        f"{question_source}:{pointer}/question/image: session0 image must be session/file"
                                    ) from exc
                                image_path = scenes / image_session / "image" / image_name
                            else:
                                image_path = session_dir / "image" / question_image
                            prompt.append(
                                content_asset(
                                    "image",
                                    register_asset(image_path, question_source, "question.image"),
                                    source_field="question.image",
                                )
                            )

                        question_type = row.get("question_type", {})
                        if not isinstance(question_type, Mapping):
                            raise SchemaError(f"{question_source}:{pointer}/question_type: expected object")
                        sub_type = _required_text(
                            question_type.get("sub_type"), question_source, f"{pointer}/question_type/sub_type"
                        )
                        subtask = _SUBTASKS.get(sub_type)
                        if subtask is None:
                            raise SchemaError(f"{question_source}:{pointer}: unknown subtask {sub_type!r}")
                        main_type = _required_text(
                            question_type.get("main_type"), question_source, f"{pointer}/question_type/main_type"
                        )
                        answer_sessions = row.get("answer_session", [])
                        if not isinstance(answer_sessions, list) or any(
                            not isinstance(value, str) for value in answer_sessions
                        ):
                            raise SchemaError(f"{question_source}:{pointer}/answer_session: expected strings")
                        evidence = [
                            {
                                "native_id": answer_session,
                                "normalized_session_id": _canonical_answer_session(
                                    interaction_type, dialogue_name, answer_session
                                ),
                                "relation": "answer_session",
                                "memory_ids": session_memory_ids.get(
                                    _canonical_answer_session(
                                        interaction_type, dialogue_name, answer_session
                                    ),
                                    [],
                                ),
                            }
                            for answer_session in answer_sessions
                        ]
                        native_id = str(row.get("original_question_id", "")).strip()
                        question_id = stable_id(
                            BENCHMARK_KEY,
                            interaction_type,
                            dialogue_name,
                            native_question_session,
                            native_id or f"question_{question_index:04d}",
                        )
                        query_at: dict[str, Any] = {
                            "session_ids": [
                                _canonical_answer_session(
                                    interaction_type, dialogue_name, answer_session
                                )
                                for answer_session in answer_sessions
                            ]
                        }
                        if native_question_session in session_end_sequence:
                            query_at["native_question_session"] = native_question_session
                        writer.add_question(
                            {
                                "question_id": question_id,
                                "context_id": context_id,
                                "subset": "default",
                                "split": "test",
                                "prompt": prompt,
                                "task": {
                                    "category": main_type.lower().replace(" ", "_"),
                                    "subcategory": subtask,
                                    "response_type": "text",
                                },
                                "choices": [],
                                "answer": {
                                    "text": _required_text(
                                        row.get("original_answer"), question_source, f"{pointer}/original_answer"
                                    ),
                                    "unanswerable": subtask == "AR",
                                },
                                "memory_scope": {"mode": "all"},
                                "query_at": query_at,
                                "evidence": evidence,
                                "provenance": provenance(question_source, snapshot, pointer),
                                "metadata": {
                                    "interaction_type": interaction_type,
                                    "native_dialogue": dialogue_name,
                                    "native_question_session": native_question_session,
                                    "native_question_id": native_id,
                                    "native_question_type": dict(question_type),
                                    "difficulty": row.get("difficulty"),
                                    "validated": row.get("validated"),
                                    "validation_notes": row.get("validation_notes", row.get("notes", "")),
                                    "generated_at": row.get("generated_at"),
                                },
                            }
                        )
                        counts["semantic_questions"] += 1

        # Preserve the complete official image release, including assets not referenced
        # by the current QA/turn files. This makes the canonical snapshot lossless and
        # also detects incomplete LFS downloads against the published count.
        for image_path in sorted(
            path
            for track in (snapshot / "dyadic", snapshot / "multi-party")
            for path in track.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ):
            register_asset(image_path, image_path, "unreferenced_release_asset")

    report = validate_bundle(output_root, check_assets=True)
    expected = {"contexts": 25, "memories": 7078, "assets": 1300, "questions": 2236}
    if strict_release_counts and report["counts"] != expected:
        raise SchemaError(f"H2HMem release-count mismatch: {report['counts']} != {expected}")
    if strict_release_counts and counts["session_json_files"] != 308:
        raise SchemaError(
            f"H2HMem session file mismatch: {counts['session_json_files']} != 308"
        )
    return {
        "benchmark": BENCHMARK_NAME,
        "bundle": str(output_root),
        "raw_revision": resolved_revision,
        "semantic_questions": counts["semantic_questions"],
        **report,
    }


__all__ = ["convert"]
