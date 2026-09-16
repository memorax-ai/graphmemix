from __future__ import annotations

from collections import defaultdict
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
from ._shared import Assets, choices, locate, question, source_manifest


def convert(raw_root: Path, output_root: Path, *, overwrite=False):
    root = locate(raw_root, "persona_mme", "Persona-MME/Persona-MME.json")
    source = root / "Persona-MME/Persona-MME.json"
    groups = defaultdict(list)
    for i, q in enumerate(read_json(source)):
        groups[q["data_path"]].append((i, q))
    manifest = {
        "benchmark": "Persona-MME",
        "source_snapshot": source_manifest(root),
        "evaluation": {
            "protocol": "persona-mme-history-only-1.0",
            "profile_prior": False,
            "evidence": "not provided by source",
            "tracks": ["main", "alignment"],
            "memory_visibility": "all supplied sessions; question.time is not a cutoff",
        },
    }
    with BundleWriter(output_root, manifest, overwrite=overwrite) as w:
        assets = Assets(w, root, output_root, "persona_mme")
        for data_path, qs in sorted(groups.items()):
            rel = data_path.split("Persona-MME/", 1)[1]
            path = root / "Persona-MME" / rel
            history = read_json(path)
            cid = stable_id("persona_mme", "context", Path(rel).parent.as_posix())
            w.add_context(
                {
                    "context_id": cid,
                    "benchmark": "Persona-MME",
                    "provenance": provenance(path, root),
                    "metadata": {
                        "native_profile": history.get("profile"),
                        "native_persona": history.get("persona"),
                        "profile_exposed": False,
                    },
                }
            )
            images = iter(history.get("imgs", []))
            image_count = 0
            seq = 0
            for si, session in enumerate(history["sessions"]):
                for ti, turn in enumerate(session):
                    for role in ["user", "assistant"]:
                        if role not in turn:
                            continue
                        parts = []
                        chunks = turn[role].split("<img>")
                        for ci, chunk in enumerate(chunks):
                            if chunk:
                                parts.append(content_text(chunk))
                            if ci < len(chunks) - 1:
                                native = next(images, None)
                                if native is None:
                                    raise SchemaError(
                                        f"{path}: more image markers than image paths"
                                    )
                                parts.append(
                                    assets.add(
                                        root / "Persona-MME" / native.removeprefix("./")
                                    )
                                )
                                image_count += 1
                        w.add_memory(
                            {
                                "memory_id": stable_id(cid, "memory", si, ti, role),
                                "context_id": cid,
                                "session_id": stable_id(cid, "session", si),
                                "sequence": seq,
                                "kind": "dialogue_message",
                                "role": role,
                                "timestamp": turn.get("time", ""),
                                "content": parts or [content_text("")],
                                "provenance": provenance(
                                    path, root, f"/sessions/{si}/{ti}/{role}"
                                ),
                            }
                        )
                        seq += 1
            if image_count != len(history.get("imgs", [])):
                raise SchemaError(f"{path}: image count mismatch")
            for i, native in qs:
                qid = stable_id("persona_mme", "question", i)
                q = question(
                    qid,
                    cid,
                    native["query"],
                    native["answer"],
                    native["type"],
                    native["question_type"].lstrip("*"),
                    "main",
                    provenance=provenance(source, root, f"/{i}"),
                    metadata={
                        "native_question_type": native["question_type"],
                        "query_time": native.get("time"),
                        "evidence_status": "unannotated",
                    },
                )
                choices(q, native["choices"], native["answer"])
                w.add_question(q)
                for label in ["chosen", "rejected"]:
                    candidate = native.get("alignment", {}).get(label)
                    if candidate is None:
                        continue
                    prompt = (
                        native["query"]
                        + "\nEvaluate the following response based on the user's personality revealed from the conversation history.\nResponse to Evaluate:\n"
                        + candidate
                        + "\nDoes the provided response align with and adapt to the user's personality?"
                    )
                    a = question(
                        stable_id(qid, "alignment", label),
                        cid,
                        prompt,
                        "",
                        "Alignment",
                        "Personality Alignment",
                        "alignment",
                        metadata={
                            "parent_question_id": qid,
                            "native_alignment_variant": label,
                            "evidence_status": "unannotated",
                        },
                    )
                    choices(
                        a,
                        {"(a)": "Yes, it aligns well.", "(b)": "No, it misaligns."},
                        "(a)" if label == "chosen" else "(b)",
                    )
                    w.add_question(a)
    return validate_bundle(output_root, check_assets=True)
