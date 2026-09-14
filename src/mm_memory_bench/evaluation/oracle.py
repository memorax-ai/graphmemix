from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..benchmarks.reader import BundleReader
from ..methods.backends import AnswerModel, ContextWindowExceeded, OpenAICompatibleQwenVL
from ..methods.base import GenerationConfig
from ..methods.media import openai_content_from_parts, question_text
from ..runner.benchmark import _prediction_record, _resolved_memory, _resolved_question


H2H_ALIAS_RE = re.compile(r"^S(\d+)-(\d+)$")


def evidence_memory_ids(
    question: Mapping[str, Any],
    sessions: Mapping[tuple[str, str], Sequence[str]] | None = None,
) -> list[str]:
    """Read evaluator-side evidence IDs without returning any other private field."""
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
        memory_ids = item.get("memory_ids")
        if isinstance(memory_ids, list):
            for candidate in memory_ids:
                if isinstance(candidate, str) and candidate not in result:
                    result.append(candidate)
        if not memory_ids and sessions:
            native_id = str(item.get("native_id", ""))
            match = H2H_ALIAS_RE.fullmatch(native_id)
            dialogue = str(question.get("metadata", {}).get("native_dialogue", ""))
            dialogue_number = re.search(r"(\d+)$", dialogue)
            if match and dialogue_number and match.group(1) == dialogue_number.group(1):
                key = (
                    str(question["context_id"]),
                    f"session{int(match.group(2))}",
                )
                for candidate in sessions.get(key, []):
                    if candidate not in result:
                        result.append(candidate)
    return result


class OracleEvidenceGenerator:
    """Generate from evaluator-selected memories without exposing gold answers."""

    def __init__(
        self,
        generation: GenerationConfig | None = None,
        *,
        answer_model: AnswerModel | None = None,
        video_frames: int = 8,
    ) -> None:
        self.generation = generation or GenerationConfig()
        self.answer_model = answer_model or OpenAICompatibleQwenVL(self.generation)
        self.video_frames = video_frames

    def answer(
        self,
        question: Mapping[str, Any],
        memories: Sequence[Mapping[str, Any]],
    ) -> str:
        tools = question.get("tools")
        tools_text = ""
        if tools and question.get("tool_mode") == "plan":
            tools_text = "\nCandidate tools:\n" + json.dumps(tools, ensure_ascii=False)
            tools = None
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": (
                f"{question.get('instruction', '')}\n"
                f"Question: {question_text(question)}{tools_text}\n"
                "The following memories were selected by an oracle. Answer using only "
                "these memories. Do not assume that every memory is independently sufficient."
            ),
        }]
        # Preserve non-text query media for benchmarks whose question itself is multimodal.
        content.extend(
            openai_content_from_parts(
                [part for part in question.get("prompt", []) if part.get("type") != "text"],
                video_frames=self.video_frames,
            )
        )
        for rank, memory in enumerate(memories, 1):
            identity = {
                key: memory[key]
                for key in ("memory_id", "source_id", "session_id", "speaker", "timestamp")
                if memory.get(key) is not None
            }
            metadata = memory.get("metadata", {})
            content.append({
                "type": "text",
                "text": (
                    f"Oracle memory {rank}: {json.dumps(identity, ensure_ascii=False)}\n"
                    f"Public annotations: {json.dumps(metadata, ensure_ascii=False)}"
                ),
            })
            content.extend(
                openai_content_from_parts(
                    memory.get("content", []), video_frames=self.video_frames
                )
            )
        return self.answer_model.complete([{"role": "user", "content": content}], tools=tools)


def run_oracle_bundle(
    bundle_root: Path,
    output_path: Path,
    *,
    generation: GenerationConfig | None = None,
    answer_model: AnswerModel | None = None,
    memory_view: str = "raw_derived",
    concurrency: int = 8,
    caption_sidecar: Path | None = None,
    pdf_policy: str = "off",
    pdf_page_images: int = 0,
    question_ids_path: Path | None = None,
) -> dict[str, Any]:
    """Run the gold-evidence upper bound while preserving harness visibility rules."""
    if concurrency <= 0:
        raise ValueError("oracle concurrency must be positive")
    reader = BundleReader(bundle_root, caption_sidecar=caption_sidecar,
                          pdf_policy=pdf_policy, pdf_page_images=pdf_page_images)
    generator = OracleEvidenceGenerator(generation, answer_model=answer_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    count = errors = 0
    question_ids: set[str] | None = None
    if question_ids_path is not None:
        values = [
            line.strip()
            for line in question_ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(set(values)) != len(values):
            raise ValueError("oracle question-id allowlist contains duplicates")
        question_ids = set(values)

    def answer_one(values):
        question, by_id, sessions = values
        ids = evidence_memory_ids(question, sessions)
        if not ids:
            return _prediction_record(
                question,
                {
                    "prediction": "",
                    "retrieved_memory_ids": [],
                    "oracle_evidence_count": 0,
                    "protocol": "gold-evidence-oracle",
                    "status": "error",
                    "error_type": "oracle_evidence_unavailable",
                    "error": "benchmark question has no published canonical evidence",
                },
                0.0,
            )
        missing = [memory_id for memory_id in ids if memory_id not in by_id]
        if missing:
            raise ValueError(f"{question['question_id']}: missing oracle memories {missing}")
        visible_question = _resolved_question(reader, question)
        memories = [
            _resolved_memory(reader, by_id[memory_id], memory_view=memory_view)
            for memory_id in ids
        ]
        item_started = time.perf_counter()
        try:
            prediction: Mapping[str, Any] = {
                "prediction": generator.answer(visible_question, memories),
                "retrieved_memory_ids": ids,
                "oracle_evidence_count": len(ids),
                "protocol": "gold-evidence-oracle",
            }
        except ContextWindowExceeded as exc:
            prediction = {
                "prediction": "",
                "retrieved_memory_ids": ids,
                "oracle_evidence_count": len(ids),
                "protocol": "gold-evidence-oracle",
                "status": "error",
                "error_type": "context_window_exceeded",
                "error": str(exc),
                "input_tokens": exc.actual,
                "max_model_len": exc.maximum,
            }
        return _prediction_record(question, prediction, time.perf_counter() - item_started)

    with output_path.open("w", encoding="utf-8") as handle:
        seen_question_ids: set[str] = set()
        for batch in reader.iter_context_batches(question_ids=question_ids):
            by_id = {str(memory["memory_id"]): memory for memory in batch.memories}
            sessions: dict[tuple[str, str], list[str]] = {}
            for memory in batch.memories:
                session_id = str(memory.get("session_id", ""))
                if session_id:
                    sessions.setdefault(
                        (str(memory["context_id"]), session_id), []
                    ).append(str(memory["memory_id"]))
            seen_question_ids.update(
                str(question["question_id"]) for question in batch.questions
            )
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for prediction in pool.map(
                    answer_one,
                    ((question, by_id, sessions) for question in batch.questions),
                ):
                    handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                    handle.flush()
                    count += 1
                    if prediction.get("metadata", {}).get("status") == "error":
                        errors += 1
        if question_ids is not None and seen_question_ids != question_ids:
            missing = sorted(question_ids - seen_question_ids)
            raise ValueError(
                "oracle question-id allowlist contains unknown or unvisited IDs: "
                + ", ".join(missing[:5])
            )
    elapsed = time.perf_counter() - started
    return {
        "protocol": "mmmb-gold-evidence-oracle-1.0",
        "bundle": str(bundle_root),
        "output": str(output_path),
        "predictions": count,
        "errors": errors,
        "concurrency": concurrency,
        "memory_view": memory_view,
        "pdf_policy": pdf_policy,
        "pdf_page_images": pdf_page_images,
        "question_ids": str(question_ids_path) if question_ids_path else None,
        "elapsed_seconds": elapsed,
    }
