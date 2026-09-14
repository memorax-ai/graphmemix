"""PersonaMem-v2 benchmark MCQ and multimodal generative tracks, preserving native history inputs."""

from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import random
from pathlib import Path

from ..bundle import (
    BundleWriter,
    SchemaError,
    provenance,
    read_json,
    stable_id,
    validate_bundle,
)
from ._shared import Assets, as_text, choices, locate, parsed, question, source_manifest


def convert(raw_root: Path, output_root: Path, *, overwrite=False):
    root = locate(raw_root, "personamem_v2", "benchmark/text/benchmark.csv")
    manifest = {
        "benchmark": "PersonaMem-v2",
        "source_snapshot": source_manifest(root),
        "evaluation": {
            "scope": "benchmark split; text/multimodal MCQ x 32k/128k; multimodal generative x 32k/128k",
            "generative": "same native user_query and history as MCQ; no options; preference is evaluation-only metadata; native preference scoring via judge --scoring-protocol personamem_v2_open",
            "options": "deterministic SHA256-seeded permutation",
            "profile_prior": "native system history messages retained, as in upstream loader; CSV profile/preference fields not injected",
            "evidence": "unique exact contiguous snippet match only; otherwise unannotated",
            "sessions": "source has no explicit session IDs",
        },
    }
    csv.field_size_limit(32 * 1024 * 1024)
    with BundleWriter(output_root, manifest, overwrite=overwrite) as w:
        assets = Assets(w, root, output_root, "personamem_v2")
        for mode in ("text", "multimodal"):
            source = root / "benchmark" / mode / "benchmark.csv"
            with source.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for size in ("32k", "128k"):
                histories = {}
                for i, row in enumerate(rows):
                    link = row[f"chat_history_{size}_link"]
                    if link not in histories:
                        path = root / link
                        native = read_json(path)
                        messages = (
                            native
                            if isinstance(native, list)
                            else native["chat_history"]
                        )
                        cid = stable_id(
                            "personamem_v2", mode, size, "context", row["persona_id"]
                        )
                        ids = []
                        w.add_context(
                            {
                                "context_id": cid,
                                "benchmark": "PersonaMem-v2",
                                "provenance": provenance(path, root),
                                "metadata": {
                                    "mode": mode,
                                    "length": size,
                                    "native_persona_id": row["persona_id"],
                                },
                            }
                        )
                        for mi, message in enumerate(messages):
                            mid = stable_id(cid, "memory", mi)
                            role = message["role"]
                            w.add_memory(
                                {
                                    "memory_id": mid,
                                    "context_id": cid,
                                    "sequence": mi,
                                    "kind": "profile"
                                    if role == "system"
                                    else "dialogue_message",
                                    "role": role,
                                    "content": assets.content(message["content"]),
                                    "provenance": provenance(
                                        path, root, f"/chat_history/{mi}"
                                    ),
                                }
                            )
                            ids.append(mid)
                        histories[link] = cid, messages, ids
                    cid, messages, ids = histories[link]
                    semantic = stable_id("personamem_v2", mode, "question", i)
                    query = parsed(row["user_query"])
                    query = query["content"] if isinstance(query, dict) else query
                    q = question(
                        stable_id(semantic, size),
                        cid,
                        as_text(query)
                        + " Please recall my related preferences from our conversation history to give personalized responses.",
                        "",
                        "personalization",
                        row.get("pref_type", ""),
                        f"{mode}_{size}",
                        provenance=provenance(source, root, f"/rows/{i}"),
                        metadata={
                            k: row.get(k)
                            for k in (
                                "persona_id",
                                "topic_query",
                                "topic_preference",
                                "conversation_scenario",
                                "pref_type",
                                "who",
                                "updated",
                                "sensitive_info",
                            )
                        },
                    )
                    q["semantic_question_id"] = semantic
                    distractors = parsed(row["incorrect_answers"])
                    if not isinstance(distractors, list):
                        raise SchemaError(
                            f"{source}:{i}: incorrect_answers must be a list"
                        )
                    options = [(True, row["correct_answer"])] + [
                        (False, as_text(x)) for x in distractors
                    ]
                    random.Random(
                        int(hashlib.sha256(semantic.encode()).hexdigest(), 16)
                    ).shuffle(options)
                    labels = {chr(65 + j): text for j, (_, text) in enumerate(options)}
                    correct = next(
                        chr(65 + j) for j, (gold, _) in enumerate(options) if gold
                    )
                    choices(q, labels, correct)
                    snippet = parsed(row.get("related_conversation_snippet", ""))
                    matches = []
                    if isinstance(snippet, list) and snippet:
                        expected = [(x.get("role"), x.get("content")) for x in snippet]
                        for start in range(len(messages) - len(expected) + 1):
                            if [
                                (x.get("role"), x.get("content"))
                                for x in messages[start : start + len(expected)]
                            ] == expected:
                                matches.append(start)
                    if len(matches) == 1:
                        q["evidence"] = [
                            {"memory_id": mid, "derivation": "unique_exact_snippet"}
                            for mid in ids[matches[0] : matches[0] + len(snippet)]
                        ]
                    q["metadata"].update(
                        evidence_status="exact_snippet"
                        if q["evidence"]
                        else "no_unique_exact_match",
                        native_related_conversation_snippet=snippet,
                    )
                    w.add_question(q)
                    if mode == "multimodal":
                        preference = row.get("preference")
                        if not isinstance(preference, str) or not preference.strip():
                            raise SchemaError(f"{source}:{i}: generative track requires native preference")
                        # Reuse the exact original query, evidence and history; only the
                        # MCQ-specific response instruction and choices are removed.
                        opened = deepcopy(q)
                        opened["question_id"] = q["question_id"] + ":generative"
                        opened["subset"] = f"{mode}_{size}_generative"
                        opened["instruction"] = ""
                        opened["task"]["response_type"] = "text"
                        opened.pop("choices", None)
                        opened["answer"] = {"text": row["correct_answer"]}
                        opened["metadata"].update(
                            native_eval_mode="generative",
                            native_user_query=as_text(query),
                            paired_mcq_question_id=q["question_id"],
                            preference=preference,
                            prev_pref=row.get("prev_pref"),
                            # Match upstream evaluate_narrow_judge dispatch exactly.
                            native_judge_kind="negative"
                            if preference.lower().startswith("do not") else "positive",
                            scoring_target="preference_alignment",
                        )
                        w.add_question(opened)
    return validate_bundle(output_root, check_assets=True)
