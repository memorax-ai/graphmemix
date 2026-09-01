from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .harness import MemoryMethod, run_context
from .reader import BundleReader, ContextBatch


def _evidence_ids(question: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    evidence = question.get("evidence")
    if not isinstance(evidence, list):
        return result
    for item in evidence:
        if not isinstance(item, Mapping):
            continue
        memory_id = item.get("memory_id")
        if isinstance(memory_id, str) and memory_id not in result:
            result.append(memory_id)
    return result


def run_diagnostic_smoke(
    method: MemoryMethod,
    bundle_root: Path,
    output_path: Path,
    *,
    subset: str | None = None,
    max_memories: int = 1,
    question_id: str | None = None,
    memory_view: str = "raw",
) -> dict[str, Any]:
    """Run one real query with a tiny evaluator-selected memory slice.

    This is an interface/media diagnostic, never a benchmark score. Gold and
    evidence annotations are used only by this outer selector and remain hidden
    by the normal harness model-visible allowlist.
    """
    if max_memories <= 0:
        raise ValueError("max_memories must be positive")
    reader = BundleReader(bundle_root)
    selected: tuple[ContextBatch, list[str]] | None = None
    for batch in reader.iter_context_batches(subset=subset):
        by_id = {str(memory["memory_id"]): memory for memory in batch.memories}
        for question in batch.questions:
            if question_id is not None and question["question_id"] != question_id:
                continue
            ids = [memory_id for memory_id in _evidence_ids(question) if memory_id in by_id]
            if ids or question_id is not None:
                memories = [by_id[memory_id] for memory_id in ids[:max_memories]]
                if not memories:
                    memories = batch.memories[:max_memories]
                selected = (
                    ContextBatch(batch.context, memories, [question]),
                    [str(memory["memory_id"]) for memory in memories],
                )
                break
        if selected:
            break
    if selected is None:
        for batch in reader.iter_context_batches(subset=subset):
            if batch.memories and batch.questions:
                questions = [q for q in batch.questions if question_id is None or q["question_id"] == question_id]
                if not questions:
                    continue
                memories = batch.memories[:max_memories]
                selected = (
                    ContextBatch(batch.context, memories, [questions[0]]),
                    [str(memory["memory_id"]) for memory in memories],
                )
                break
    if selected is None:
        raise ValueError(f"no smoke candidate found in {bundle_root}")

    batch, memory_ids = selected
    predictions = run_context(method, reader, batch, memory_view=memory_view)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "mmmb-diagnostic-smoke-1.0",
        "not_a_benchmark_score": True,
        "bundle": str(bundle_root),
        "benchmark": batch.context.get("benchmark"),
        "context_id": batch.context["context_id"],
        "question_id": batch.questions[0]["question_id"],
        "selector": "exact question_id" if question_id else "first question with canonical evidence",
        "selected_memory_ids": memory_ids,
        "prediction": predictions[0],
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
