"""SMMBench source streams, native evidence assignments, MCQ and call plans."""

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
from ._shared import Assets, as_text, choices, locate, question, source_manifest


def _parts(value, assets, root):
    """Materialize images embedded in native JSON evidence, not just outer images."""
    if isinstance(value, list):
        return [part for item in value for part in _parts(item, assets, root)]
    if isinstance(value, dict):
        if value.get("image_path"):
            rest = {k: v for k, v in value.items() if k != "image_path"}
            return ([content_text(as_text(rest))] if rest else []) + [
                assets.add(root / "Images" / Path(value["image_path"]).name)
            ]
        if value.get("type") in ("image", "text") and "content" in value:
            return _parts(value["content"], assets, root)
        # Tables and other structured records retain keys and values as JSON text.
        if any(
            isinstance(v, (dict, list)) and "image_path" in as_text(v)
            for v in value.values()
        ):
            return [
                part
                for key, v in value.items()
                for part in ([content_text(key)] + _parts(v, assets, root))
            ]
    return [content_text(as_text(value))]


def convert(raw_root: Path, output_root: Path, *, overwrite=False):
    root = locate(raw_root, "smmbench", "Samples")
    files = sorted((root / "Samples").glob("cluster_*/QA_sample.json"))
    if not files:
        raise SchemaError("SMMBench has no cluster QA files")
    manifest = {
        "benchmark": "SMMBench",
        "source_snapshot": source_manifest(root),
        "evaluation": {
            "scope": "all published clusters; MCQ plus function-call plan questions",
            "history": "overall, all source streams sorted by timestamp",
            "evidence": "zero-based native insert_conversation_turn; misleading assignments separate",
            "tools": "descriptions only; no execution; native FC scoring not added",
            "sessions": "source_id denotes a conversation stream, no inferred session splits",
            "captions": "figure/table reference labels exposed; descriptive native captions retained privately, not injected into raw-image track",
        },
    }
    with BundleWriter(output_root, manifest, overwrite=overwrite) as w:
        assets = Assets(w, root, output_root, "smmbench")
        tools = None
        for source in files:
            cluster = source.parent.name
            cid = stable_id("smmbench", "context", cluster)
            w.add_context(
                {
                    "context_id": cid,
                    "benchmark": "SMMBench",
                    "provenance": provenance(source.parent, root),
                }
            )
            lookup = {}
            ordered = []
            for path in sorted(source.parent.glob("*.json")):
                if not path.stem.startswith(("group_chat", "user_assistant")):
                    continue
                history = read_json(path)["conversation"]
                for mi, turn in enumerate(history):
                    stream = turn.get("conversation_name", path.stem)
                    key = (stream, mi)
                    if key in lookup:
                        raise SchemaError(f"{cluster}: duplicate stream/turn {key}")
                    mid = stable_id(cid, "memory", stream, mi)
                    lookup[key] = mid
                    parts = [
                        content_text(
                            f"Source: {stream} | Speaker: {turn.get('sender_name', '')} | Time: {turn.get('timestamp', '')}"
                        )
                    ]
                    references = list(
                        dict.fromkeys(
                            re.findall(
                                r"(?:Fig|Table)\.\s*[0-9a-fA-F]{8}",
                                turn.get("caption", ""),
                            )
                        )
                    )
                    if references:
                        parts.append(content_text("\n".join(references)))
                    parts += _parts(turn.get("content", ""), assets, root)
                    if turn.get("image_path"):
                        parts.append(
                            assets.add(root / "Images" / Path(turn["image_path"]).name)
                        )
                    row = {
                        "memory_id": mid,
                        "context_id": cid,
                        "source_id": stream,
                        "kind": "dialogue_message",
                        "speaker": turn.get("sender_name", ""),
                        "timestamp": turn.get("timestamp", ""),
                        "content": parts,
                        "metadata": {"native_caption": turn.get("caption")},
                        "provenance": provenance(path, root, f"/conversation/{mi}"),
                    }
                    ordered.append(row)
            if not ordered:
                raise SchemaError(f"{cluster}: source conversations missing")
            for sequence, row in enumerate(
                sorted(ordered, key=lambda x: (x["timestamp"], x["memory_id"]))
            ):
                row["sequence"] = sequence
                w.add_memory(row)
            for qi, native in enumerate(read_json(source)):
                fc = native["category"] == "Function_Call"
                q = question(
                    stable_id("smmbench", cluster, native["id"]),
                    cid,
                    native["question"],
                    as_text(native["answer"]),
                    native["category"],
                    native["domain"],
                    "function_call" if fc else "mcq",
                    provenance=provenance(source, root, f"/{qi}"),
                    metadata={
                        "native_id": native["id"],
                        "native_evidence": native.get("evidence"),
                        "native_answer": native["answer"],
                    },
                )
                if fc:
                    if tools is None:
                        tools = read_json(root / "candidate_tools.json")
                    q.update(
                        tools=tools,
                        tool_mode="plan",
                        instruction="Return a JSON list of steps: each has step (integer) and calls (list of objects with name and arguments). Plan only; do not execute tools.",
                    )
                    q["task"]["response_type"] = "structured_json"
                else:
                    mcq = native["multi_choice_QA"]
                    choices(
                        q,
                        {
                            str(i): as_text(x)
                            for i, x in enumerate(mcq["multi_choice_QA_options"])
                        },
                        str(mcq["multi_choice_QA_answer"]),
                    )
                gold, misleading = {}, {}
                for field, refs in native.get("evidence_assignment", {}).items():
                    target = misleading if field.startswith("mis_") else gold
                    for ref in refs:
                        key = (
                            ref["conversation_name"],
                            ref["insert_conversation_turn"],
                        )
                        if key not in lookup:
                            raise SchemaError(
                                f"{q['question_id']}: unknown evidence assignment {key}"
                            )
                        mid = lookup[key]
                        target[mid] = {
                            "memory_id": mid,
                            "native_assignment_type": field,
                            **ref,
                        }
                q["evidence"] = list(gold.values())
                q["misleading_evidence"] = list(misleading.values())
                w.add_question(q)
    return validate_bundle(output_root, check_assets=True)
