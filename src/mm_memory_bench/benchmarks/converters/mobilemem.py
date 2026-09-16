from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..bundle import (
    BundleWriter, SchemaError, content_text, provenance, read_json, stable_id,
    validate_bundle,
)

BENCHMARK_KEY = "mobilemem"
BENCHMARK_NAME = "MobileMem"


def _text(value: Any, location: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise SchemaError(f"{location}: expected {'a string' if allow_empty else 'non-empty text'}")
    return value


def _objects(value: Any, location: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise SchemaError(f"{location}: expected a list of objects")
    return value


def _timestamp(value: Any, location: str) -> datetime:
    text = _text(value, location)
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError as exc:
        raise SchemaError(f"{location}: invalid ISO timestamp {text!r}") from exc
    # Naive timestamps share the native trajectory clock; UTC is only a sort key.
    return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc)


def _locate_snapshot(raw_root: Path) -> Path:
    for candidate in (raw_root, raw_root / BENCHMARK_KEY, raw_root / "MobileMem"):
        if (candidate / "text/mobilemem_data.json").is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"MobileMem text/mobilemem_data.json not found under {raw_root}")


def convert(raw_root: Path, output_root: Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Normalize the retained MobileMem text QA track, never the synthesis graph."""
    snapshot = _locate_snapshot(Path(raw_root))
    source = snapshot / "text/mobilemem_data.json"
    trajectories = _objects(read_json(source), str(source))
    if not trajectories:
        raise SchemaError("MobileMem snapshot has no trajectories")
    contexts: list[dict[str, Any]] = []
    memories: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    context_ids: set[str] = set()
    counts: Counter[str] = Counter()
    native_types: Counter[str] = Counter()
    native_forms: Counter[str] = Counter()

    for ti, trajectory in enumerate(trajectories):
        pointer = f"/{ti}"
        person = trajectory.get("person")
        if not isinstance(person, Mapping):
            raise SchemaError(f"{pointer}/person: expected an object with id")
        native_person = _text(person.get("id"), f"{pointer}/person/id")
        context_id = stable_id(BENCHMARK_KEY, "context", native_person)
        if context_id in context_ids:
            raise SchemaError(f"duplicate trajectory id: {native_person}")
        context_ids.add(context_id)
        contexts.append({
            "context_id": context_id, "benchmark": BENCHMARK_NAME,
            "provenance": provenance(source, snapshot, pointer),
            "metadata": {"native_person_id": native_person},
        })
        session_ids: set[str] = set()
        by_native_id: dict[str, str] = {}
        ordered: list[tuple[datetime, int, dict[str, Any]]] = []
        for si, session in enumerate(_objects(trajectory.get("sessions"), f"{pointer}/sessions")):
            sp = f"{pointer}/sessions/{si}"
            session_id = _text(session.get("id"), f"{sp}/id")
            if session_id in session_ids:
                raise SchemaError(f"{sp}: duplicate session id {session_id}")
            session_ids.add(session_id)
            for mi, message in enumerate(_objects(session.get("messages"), f"{sp}/messages")):
                mp = f"{sp}/messages/{mi}"
                native_id = _text(message.get("id"), f"{mp}/id")
                if native_id in by_native_id:
                    raise SchemaError(f"{mp}: duplicate message id {native_id}")
                memory_id = stable_id(BENCHMARK_KEY, native_person, "memory", native_id)
                by_native_id[native_id] = memory_id
                role = _text(message.get("role"), f"{mp}/role")
                if role not in {"system", "user", "assistant"}:
                    raise SchemaError(f"{mp}: unsupported role {role!r}")
                speaker = _text(message.get("name"), f"{mp}/name")
                body = _text(message.get("content"), f"{mp}/content", allow_empty=True)
                stamp = _timestamp(message.get("timestamp"), f"{mp}/timestamp")
                row = {
                    "memory_id": memory_id, "context_id": context_id,
                    "session_id": session_id, "source_id": native_id,
                    "kind": "dialogue_message", "role": role, "speaker": speaker,
                    "timestamp": message["timestamp"],
                    # Keep source attribution even in methods that only consume content.
                    "content": [content_text(f"Source: {speaker} | Role: {role} | Time: {message['timestamp']}"), content_text(body)],
                    "provenance": provenance(source, snapshot, mp),
                    "metadata": {"native_event_id": session.get("event_id")},
                }
                ordered.append((stamp, len(ordered), row))
        for sequence, (_, _, row) in enumerate(sorted(ordered, key=lambda item: (item[0], item[1]))):
            row["sequence"] = sequence
            memories.append(row)
        counts["sessions"] += len(session_ids)
        book = trajectory.get("question_type_toolbook")
        if not isinstance(book, Mapping):
            raise SchemaError(f"{pointer}/question_type_toolbook: missing canonical retained QA source")
        question_ids: set[str] = set()
        for gi, group in enumerate(_objects(book.get("question_types"), f"{pointer}/question_type_toolbook/question_types")):
            gp = f"{pointer}/question_type_toolbook/question_types/{gi}"
            group_name = _text(group.get("name"), f"{gp}/name")
            for qi, q in enumerate(_objects(group.get("qa_pairs"), f"{gp}/qa_pairs")):
                qp = f"{gp}/qa_pairs/{qi}"
                native_id = _text(q.get("id"), f"{qp}/id")
                if native_id in question_ids:
                    raise SchemaError(f"{qp}: duplicate question id {native_id}")
                question_ids.add(native_id)
                question_type = _text(q.get("question_type"), f"{qp}/question_type")
                form = _text(q.get("question_form"), f"{qp}/question_form")
                if form not in {"open_ended", "single_choice", "multiple_choice"}:
                    raise SchemaError(f"{qp}: unsupported question_form {form!r}")
                prompt = _text(q.get("question"), f"{qp}/question")
                answers = q.get("golden_answers")
                if not isinstance(answers, list) or not answers:
                    raise SchemaError(f"{qp}/golden_answers: expected non-empty answer list")
                for ai, answer in enumerate(answers):
                    _text(answer, f"{qp}/golden_answers/{ai}")
                evidence = []
                seen_evidence: set[str] = set()
                for ei, e in enumerate(_objects(q.get("source_evidences", []), f"{qp}/source_evidences")):
                    eid = _text(e.get("id"), f"{qp}/source_evidences/{ei}/id")
                    counts["source_evidence_references"] += 1
                    if eid not in by_native_id:
                        raise SchemaError(f"{qp}: evidence {eid!r} does not reference this trajectory")
                    if eid not in seen_evidence:
                        evidence.append({"memory_id": by_native_id[eid], "native_id": eid, "relation": "supports"})
                        seen_evidence.add(eid)
                if not evidence:
                    counts["questions_without_evidence"] += 1
                native_types[question_type] += 1
                native_forms[form] += 1
                questions.append({
                    "question_id": stable_id(BENCHMARK_KEY, native_person, "question", native_id),
                    "context_id": context_id, "subset": "text", "split": "test",
                    "prompt": [content_text(prompt)],
                    "task": {"category": "mobile_personal_memory", "subcategory": question_type, "response_type": "text"},
                    # Native options are embedded in the question, not a structured array.
                    "choices": [], "answer": {"text": answers[0], "accepted_answers": list(answers)},
                    "memory_scope": {"mode": "all"}, "evidence": evidence,
                    "provenance": provenance(source, snapshot, qp),
                    "metadata": {
                        "native_question_id": native_id, "native_question_form": form,
                        "native_group": group_name, "difficulty": q.get("difficulty"),
                        "num_hops": q.get("num_hops"), "topic": q.get("topic"),
                        "evaluation_private": {key: value for key, value in q.items() if key not in {
                            "id", "question", "question_type", "question_form", "golden_answers",
                            "source_evidences", "difficulty", "num_hops", "topic",
                        }},
                    },
                })
    manifest = {
        "benchmark": BENCHMARK_NAME, "dataset_id": "zjunlp/MobileMem", "subsets": ["text"],
        "modalities": ["text"], "source": {
            "dataset": "https://huggingface.co/datasets/zjunlp/MobileMem",
            "repository": "https://github.com/zjunlp/MobileMem",
            "paper": "https://arxiv.org/abs/2608.13606",
        },
        "raw_snapshot": {"source_file": "text/mobilemem_data.json", "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "normalization": {
            "context_granularity": "one native person trajectory",
            "memory_granularity": "one native message, including empty messages and app/system events",
            "ordering": "stable chronological message order; native naive timestamps use one trajectory clock",
            "private_fields": "person profile, graphs, old QA, revision logs and message side notes are not method inputs",
            "question_forms": "native options stay verbatim in prompt; response_type=text; no inferred option parsing",
        },
        "evaluation": {
            "protocol": "mobilemem-text-common-judge-1.0",
            "memory_visibility": "all sessions ingested before retained questions; effective_timestamp is not a cutoff",
            "reference_answers": "answer.accepted_answers contains alternative valid references; answer.text is the first",
            "scope": "common harness protocol, not a claim of reproducing the upstream MemBase judge",
        },
        "release_counts": {**counts, "question_types": dict(sorted(native_types.items())), "question_forms": dict(sorted(native_forms.items()))},
    }
    with BundleWriter(Path(output_root), manifest, overwrite=overwrite) as writer:
        for row in contexts:
            writer.add_context(row)
        for row in memories:
            writer.add_memory(row)
        for row in questions:
            writer.add_question(row)
    return {"benchmark": BENCHMARK_NAME, "bundle": str(output_root), **validate_bundle(Path(output_root))}
