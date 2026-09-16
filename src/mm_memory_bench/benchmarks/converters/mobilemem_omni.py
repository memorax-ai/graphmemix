"""Convert released Omni dialogue messages and both published QA selections."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..bundle import (
    BundleWriter,
    SchemaError,
    content_text,
    iter_jsonl,
    provenance,
    stable_id,
    validate_bundle,
)
from ._shared import Assets, as_text, locate, question, source_manifest


def resolve_image_path(root: Path, name: str) -> Path:
    """The release uses spaces in dialogue directory names, underscores in ZIP."""
    direct = root / "omni" / name
    if direct.is_file():
        return direct
    parts = Path(name).parts
    normalized = Path(*(part.replace(" ", "_") for part in parts[:-1]), parts[-1])
    return root / "omni" / normalized


def convert(raw_root: Path, output_root: Path, *, overwrite=False):
    root = locate(raw_root, "mobilemem_omni", "omni/data.jsonl")
    source = root / "omni/data.jsonl"
    filtered = {}
    for group in iter_jsonl(root / "omni/filtered_questions.jsonl"):
        for q in group["questions"]:
            filtered[(str(group["uuid"]), q["question_id"])] = q
    groups = {
        str(x["uuid"]): x["questions"]
        for x in iter_jsonl(root / "omni/questions.jsonl")
    }
    manifest = {
        "benchmark": "MobileMem-Omni",
        "source_snapshot": source_manifest(root),
        "evaluation": {
            "scope": "all published questions; filtered and unfiltered_only are disjoint subsets",
            "memory_unit": "one native dialogue message; only image_inline assets ingested",
            "evidence_granularity": "native session expanded to all its messages; not message-level gold",
            "query_images": "image_refs are evidence references, not query attachments (upstream Raw2Locomo)",
            "profile_prior": False,
            "caption": "no generated or annotation captions injected",
            "asset_paths": "ZIP extracted under image/ using GBK metadata encoding; dialogue directory spaces normalized to underscores, basenames unchanged",
        },
    }
    with BundleWriter(output_root, manifest, overwrite=overwrite) as w:
        assets = Assets(w, root, output_root, "mobilemem_omni")
        consumed = set()
        for ri, record in enumerate(iter_jsonl(source)):
            uid = str(record["uuid"])
            cid = stable_id("mobilemem_omni", "context", uid)
            w.add_context(
                {
                    "context_id": cid,
                    "benchmark": "MobileMem-Omni",
                    "provenance": provenance(source, root, f"/{ri}"),
                    "metadata": {"language": record.get("language")},
                }
            )
            sessions = defaultdict(list)
            seq = 0
            for si, session in enumerate(record["sessions"]):
                sid = str(session["session_id"])
                if sid in sessions:
                    raise SchemaError(f"{cid}: duplicate session {sid}")
                sessions[sid] = []
                for ti, turn in enumerate(session["dialogue"]):
                    mid = stable_id(cid, "memory", sid, ti)
                    parts = [content_text(as_text(turn.get("content", "")))]
                    images = turn.get("image_inline") or []
                    if isinstance(images, str):
                        images = [images]
                    for name in images:
                        parts.append(assets.add(resolve_image_path(root, name)))
                    w.add_memory(
                        {
                            "memory_id": mid,
                            "context_id": cid,
                            "session_id": sid,
                            "sequence": seq,
                            "kind": "dialogue_message",
                            "role": turn["role"],
                            "timestamp": session.get("event_start_time", ""),
                            "content": parts,
                            "metadata": {"native_image_paths": images},
                            "provenance": provenance(
                                source, root, f"/{ri}/sessions/{si}/dialogue/{ti}"
                            ),
                        }
                    )
                    sessions[sid].append(mid)
                    seq += 1
            for qi, native in enumerate(groups.pop(uid, [])):
                if native.get("question_format", "open_ended") != "open_ended":
                    raise SchemaError(
                        "Omni converter expects the released open-ended QA track"
                    )
                key = (uid, native["question_id"])
                selected = key in filtered
                if selected and filtered[key] != native:
                    raise SchemaError(
                        f"filtered question differs from full version: {key}"
                    )
                consumed.add(key)
                q = question(
                    stable_id(cid, "question", native["question_id"]),
                    cid,
                    native["question"],
                    as_text(native["answer"]),
                    "memory_qa",
                    native["question_type"],
                    "filtered" if selected else "unfiltered_only",
                    provenance=provenance(
                        root / "omni/questions.jsonl", root, f"/{uid}/questions/{qi}"
                    ),
                    metadata={
                        "native_question_id": native["question_id"],
                        "native_evidence": native.get("evidence", []),
                        "native_image_refs": native.get("image_refs", []),
                        "difficulty": native.get("difficulty"),
                        "native_source_session_ids": native.get(
                            "source_session_ids", []
                        ),
                        "native_source_event_ids": native.get("source_event_ids", []),
                        "evidence_granularity": "session",
                    },
                )
                refs = list(
                    dict.fromkeys(
                        str(e["session_id"])
                        for e in native.get("evidence", [])
                        if isinstance(e, dict) and e.get("session_id") is not None
                    )
                )
                missing = [sid for sid in refs if sid not in sessions]
                if missing:
                    # Keep all questions, but do not invent a gold memory for absent native references.
                    q["metadata"]["unresolved_session_ids"] = missing
                for sid in refs:
                    q["evidence"].extend(
                        {
                            "memory_id": mid,
                            "native_session_id": sid,
                            "granularity": "session",
                        }
                        for mid in sessions.get(sid, [])
                    )
                q["metadata"]["evidence_status"] = (
                    "incomplete_native_references"
                    if missing
                    else ("session_expansion" if refs else "unannotated_or_abstention")
                )
                w.add_question(q)
        if groups or set(filtered) - consumed:
            raise SchemaError("question contexts missing from Omni data")
    return validate_bundle(output_root, check_assets=True)
