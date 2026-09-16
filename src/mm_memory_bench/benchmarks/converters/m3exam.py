from __future__ import annotations

import re
from pathlib import Path

from ..bundle import (
    BundleWriter,
    SchemaError,
    content_text,
    provenance,
    read_json,
    stable_id,
    validate_bundle,
)
from ._shared import Assets, as_text, locate, question, source_manifest


def convert(raw_root: Path, output_root: Path, *, overwrite=False):
    root = locate(raw_root, "m3exam", "example_set")
    personas = sorted((root / "example_set").glob("*/sessions.json"))
    if not personas:
        raise SchemaError("M3Exam has no public example personas")
    manifest = {
        "benchmark": "M3Exam",
        "source_snapshot": source_manifest(root),
        "evaluation": {
            "scope": "public examples, not full benchmark",
            "answer_semantics": "native ordered list retained; accepted_answers follow upstream any-match EM",
            "evidence_granularity": "supporting round expanded to user and assistant messages",
            "pdf": "original PDF assets retained; page rendering is a method/reader responsibility",
        },
    }
    with BundleWriter(output_root, manifest, overwrite=overwrite) as w:
        assets = Assets(w, root, output_root, "m3exam")
        for path in personas:
            persona = path.parent.name
            cid = stable_id("m3exam", "context", persona)
            by_round = {}
            seq = 0
            w.add_context(
                {
                    "context_id": cid,
                    "benchmark": "M3Exam",
                    "provenance": provenance(path, root),
                }
            )
            for si, session in enumerate(read_json(path)):
                for ti, turn in enumerate(session["dialogues"]):
                    ids = []
                    for role in ["user", "assistant"]:
                        if role not in turn:
                            continue
                        mid = stable_id(cid, "memory", turn["round"], role)
                        parts = [content_text(turn[role])]
                        if role == "user":
                            for field, directory, kind in [
                                ("img_file", "images", "image"),
                                ("pdf_file", "pdfs", "document"),
                            ]:
                                files = turn.get(field, [])
                                files = [files] if isinstance(files, str) else files
                                for f in files:
                                    part = assets.add(path.parent / directory / f, kind)
                                    # Public asset identity, never derived from question/gold.
                                    part["source_id"] = Path(f).name
                                    parts.append(part)
                        w.add_memory(
                            {
                                "memory_id": mid,
                                "context_id": cid,
                                "session_id": session["session_id"],
                                "round_id": turn["round"],
                                "sequence": seq,
                                "kind": "dialogue_message",
                                "role": role,
                                "timestamp": session.get("date", ""),
                                "content": parts,
                                "provenance": provenance(
                                    path, root, f"/{si}/dialogues/{ti}/{role}"
                                ),
                            }
                        )
                        seq += 1
                        ids.append(mid)
                    by_round[turn["round"]] = ids
            qpath = path.parent / "question.json"
            for i, native in enumerate(read_json(qpath)):
                answers = native["answer"]
                answers = answers if isinstance(answers, list) else [answers]
                q = question(
                    stable_id("m3exam", persona, "question", i),
                    cid,
                    native["question"],
                    as_text(answers[0]),
                    "memory_qa",
                    native["type"],
                    "examples",
                    provenance=provenance(qpath, root, f"/{i}"),
                    metadata={
                        "native_label": native.get("label"),
                        "native_supporting_facts": native.get("supporting_facts"),
                    },
                )
                q["answer"].update(
                    accepted_answers=[as_text(x) for x in answers],
                    native_ordered_answers=answers,
                )
                refs = re.findall(
                    r"D\d+:\d+", as_text(native.get("supporting_facts", []))
                )
                for ref in dict.fromkeys(refs):
                    if ref not in by_round:
                        raise SchemaError(f"{q['question_id']}: unknown round {ref}")
                    q["evidence"].extend(
                        {"memory_id": mid, "native_round_id": ref}
                        for mid in by_round[ref]
                    )
                w.add_question(q)
    return validate_bundle(output_root, check_assets=True)
