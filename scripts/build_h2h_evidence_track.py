#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ALIAS_RE = re.compile(r"^S(\d+)-(\d+)$")


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=Path("data/unified/h2hmem"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/derived/h2hmem/evidence_supported_v1"),
    )
    args = parser.parse_args()

    memories_by_session: dict[tuple[str, str], list[str]] = defaultdict(list)
    for memory in rows(args.bundle / "memories.jsonl"):
        session_id = str(memory.get("session_id", ""))
        if session_id:
            memories_by_session[(str(memory["context_id"]), session_id)].append(
                str(memory["memory_id"])
            )

    selected: list[str] = []
    excluded = Counter()
    recovered_aliases = 0
    total = 0
    for question in rows(args.bundle / "questions.jsonl"):
        total += 1
        context_id = str(question["context_id"])
        evidence = question.get("evidence", [])
        resolved_groups: list[list[str]] = []
        for item in evidence if isinstance(evidence, list) else []:
            memory_ids = item.get("memory_ids", []) if isinstance(item, dict) else []
            resolved = [str(value) for value in memory_ids if isinstance(value, str)]
            native_id = str(item.get("native_id", "")) if isinstance(item, dict) else ""
            if not resolved:
                match = ALIAS_RE.fullmatch(native_id)
                dialogue = str(question.get("metadata", {}).get("native_dialogue", ""))
                dialogue_number = re.search(r"(\d+)$", dialogue)
                if (
                    match
                    and dialogue_number
                    and match.group(1) == dialogue_number.group(1)
                ):
                    resolved = memories_by_session.get(
                        (context_id, f"session{int(match.group(2))}"), []
                    )
                    if resolved:
                        recovered_aliases += 1
            resolved_groups.append(resolved)

        if resolved_groups and all(resolved_groups):
            selected.append(str(question["question_id"]))
            continue

        native_ids = tuple(
            str(item.get("native_id", ""))
            for item in evidence
            if isinstance(item, dict)
        )
        if native_ids == ("session0",):
            excluded["session0_question_container"] += 1
        elif native_ids == ("session6",):
            excluded["missing_dyadic_dialogue3_session6"] += 1
        else:
            excluded["other_unresolved_evidence"] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = args.output_dir / "question_ids.txt"
    ids_path.write_text("".join(f"{value}\n" for value in selected), encoding="utf-8")
    summary = {
        "protocol": "mmmb-h2hmem-evidence-supported-1.0",
        "source_bundle": str(args.bundle),
        "total_questions": total,
        "selected_questions": len(selected),
        "excluded_questions": total - len(selected),
        "recovered_native_session_aliases": recovered_aliases,
        "excluded_by_reason": dict(sorted(excluded.items())),
        "question_ids": str(ids_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
