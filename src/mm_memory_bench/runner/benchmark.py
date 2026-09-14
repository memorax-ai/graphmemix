from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from ..benchmarks.bundle import TOOL_SENSITIVE_KEYS, normalized_tool_key
from ..benchmarks.reader import BundleReader, ContextBatch


@runtime_checkable
class MemoryMethod(Protocol):
    """The only adapter a memory implementation needs to provide.

    ``answer`` must not persist the evaluation query or its prediction into the
    memory store. Official benchmarks treat questions as independent probes of
    the same memory state.
    """

    def begin_context(self, context: Mapping[str, Any]) -> None: ...

    def ingest(self, memory: Mapping[str, Any]) -> None: ...

    def answer(self, question: Mapping[str, Any]) -> str | Mapping[str, Any]: ...

    def end_context(self) -> None: ...


def _public_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return {}
    visible = metadata.get("agent_visible")
    return dict(visible) if isinstance(visible, Mapping) else {}


def _safe_content(reader: BundleReader, content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve media without exposing asset provenance or annotation payloads."""
    safe = []
    for part in reader.resolve_content(content):
        value = {
            key: item
            for key, item in part.items()
            if key in {"type", "text", "asset_id", "path"}
        }
        safe.append(value)
    return safe


def _safe_tool_value(value: Any) -> Any:
    """Copy JSON-like tool data while recursively removing evaluator-only keys."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_safe_tool_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _safe_tool_value(item)
            for key, item in value.items()
            if isinstance(key, str)
            and normalized_tool_key(key) not in TOOL_SENSITIVE_KEYS
        }
    raise ValueError("question tools must contain only JSON-compatible values")


def _safe_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        raise ValueError("question.tools must be a list")
    safe: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, Mapping):
            raise ValueError(f"question.tools[{index}] must be an object")
        safe.append(_safe_tool_value(tool))
    return safe


def _derived_memory_content(
    memory: Mapping[str, Any],
    *,
    sidecar_captions: list[str] | None = None,
) -> tuple[list[dict[str, Any]] | None, bool]:
    metadata = memory.get("metadata")
    derived = metadata.get("derived") if isinstance(metadata, Mapping) else None
    if not isinstance(derived, Mapping):
        derived = {}
    labels = (
        ("Short summary", "short_summary"),
        ("Summary", "summary"),
        ("Caption", "caption"),
        ("Short caption", "short_caption"),
        ("OCR", "ocr_text"),
        ("Location", "location_name"),
        ("City", "city"),
    )
    lines = [f"{label}: {derived[key]}" for label, key in labels if derived.get(key)]
    image_captions = derived.get("image_captions")
    if isinstance(image_captions, list):
        lines.extend(
            f"Image caption {index}: {caption}"
            for index, caption in enumerate(image_captions, 1)
            if caption
        )
    if sidecar_captions:
        lines.extend(
            f"Generated media caption {index}: {caption}"
            for index, caption in enumerate(sidecar_captions, 1)
            if caption
        )
    tags = derived.get("tags")
    if isinstance(tags, list) and tags:
        lines.append("Tags: " + ", ".join(str(tag) for tag in tags))
    describes_media = any(
        derived.get(key) for key in ("caption", "short_caption", "ocr_text")
    ) or bool(image_captions) or bool(sidecar_captions)
    return (
        ([{"type": "text", "text": "\n".join(lines)}] if lines else None),
        describes_media,
    )


def _safe_derived_metadata(
    memory: Mapping[str, Any], *, sidecar_captions: list[str] | None = None
) -> dict[str, Any]:
    """Expose only model-facing descriptive annotations for an explicit hybrid track."""
    metadata = memory.get("metadata")
    derived = metadata.get("derived") if isinstance(metadata, Mapping) else None
    if not isinstance(derived, Mapping):
        derived = {}
    allowed = {
        "short_summary",
        "summary",
        "caption",
        "short_caption",
        "ocr_text",
        "location_name",
        "city",
        "tags",
        "image_captions",
    }
    result = {key: derived[key] for key in allowed if key in derived}
    if sidecar_captions:
        existing = result.get("image_captions", [])
        existing_values = list(existing) if isinstance(existing, list) else []
        result["image_captions"] = list(
            dict.fromkeys([*existing_values, *sidecar_captions])
        )
    return result


def _resolved_memory(
    reader: BundleReader,
    memory: Mapping[str, Any],
    *,
    memory_view: str = "raw",
) -> dict[str, Any]:
    allowed = {
        "memory_id",
        "context_id",
        "session_id",
        "source_id",
        "sequence",
        "timestamp",
        "kind",
        "role",
        "speaker",
        "round_id",
    }
    value = {key: item for key, item in memory.items() if key in allowed}
    if memory_view not in {"raw", "derived", "raw_derived"}:
        raise ValueError(f"unsupported memory view: {memory_view}")
    raw_content = _safe_content(reader, list(memory["content"]))
    sidecar_captions = reader.captions_for_content(list(memory["content"]))
    derived_content, describes_media = (
        _derived_memory_content(memory, sidecar_captions=sidecar_captions)
        if memory_view == "derived"
        else (None, False)
    )
    if derived_content:
        preserved = [
            part
            for part in raw_content
            if part.get("type") not in {"image", "video", "audio"} or not describes_media
        ]
        value["content"] = preserved + derived_content
    else:
        value["content"] = raw_content
    public_metadata = _public_metadata(memory)
    if memory_view == "raw_derived":
        safe_derived = _safe_derived_metadata(
            memory, sidecar_captions=sidecar_captions
        )
        if safe_derived:
            public_metadata["derived"] = safe_derived
    if public_metadata:
        value["metadata"] = public_metadata
    return value


def _resolved_question(reader: BundleReader, question: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "question_id",
        "semantic_question_id",
        "context_id",
        "subset",
        "split",
        "instruction",
        "tool_mode",
    }
    value = {key: item for key, item in question.items() if key in allowed}
    value["prompt"] = _safe_content(reader, list(question["prompt"]))
    task = question.get("task")
    if isinstance(task, Mapping) and isinstance(task.get("response_type"), str):
        value["task"] = {"response_type": task["response_type"]}
    query_at = question.get("query_at")
    if isinstance(query_at, Mapping):
        safe_query_at = {
            key: item for key, item in query_at.items() if key in {"timestamp", "date"}
        }
        if safe_query_at:
            value["query_at"] = safe_query_at
    choices = question.get("choices")
    if isinstance(choices, list):
        safe_choices = []
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            safe_choice = {
                key: item for key, item in choice.items() if key in {"choice_id", "text"}
            }
            if isinstance(choice.get("content"), list):
                safe_choice["content"] = _safe_content(reader, list(choice["content"]))
            safe_choices.append(safe_choice)
        value["choices"] = safe_choices
    if "tools" in question:
        value["tools"] = _safe_tools(question["tools"])
        # Backward compatibility for bundles converted before tool_mode existed.
        if (
            "tool_mode" not in value
            and value.get("subset") == "function_call"
            and value.get("task", {}).get("response_type") == "structured_json"
        ):
            value["tool_mode"] = "plan"
    public_metadata = _public_metadata(question)
    if public_metadata:
        value["metadata"] = public_metadata
    return value


def _visible_context(context: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"context_id", "benchmark", "title", "persona", "profile"}
    value = {key: item for key, item in context.items() if key in allowed}
    public_metadata = _public_metadata(context)
    if public_metadata:
        value["metadata"] = public_metadata
    return value


def _prediction_record(question: Mapping[str, Any], result: Any, elapsed: float) -> dict[str, Any]:
    if isinstance(result, Mapping):
        prediction = str(result.get("prediction", result.get("answer", "")))
        retrieved = result.get("retrieved_memory_ids", [])
        metadata = {key: value for key, value in result.items() if key not in {"prediction", "answer", "retrieved_memory_ids"}}
    else:
        prediction = str(result)
        retrieved = []
        metadata = {}
    return {
        "question_id": question["question_id"],
        "semantic_question_id": question.get("semantic_question_id", question["question_id"]),
        "context_id": question["context_id"],
        "subset": question.get("subset", "default"),
        "prediction": prediction,
        "retrieved_memory_ids": list(retrieved),
        "latency_seconds": elapsed,
        "metadata": metadata,
    }


def _answer(method: MemoryMethod, reader: BundleReader, question: Mapping[str, Any]) -> dict[str, Any]:
    resolved = _resolved_question(reader, question)
    started = time.perf_counter()
    result = method.answer(resolved)
    return _prediction_record(question, result, time.perf_counter() - started)


def _answer_group(
    method: MemoryMethod,
    reader: BundleReader,
    questions: list[Mapping[str, Any]],
    concurrency: int,
    *,
    continue_on_error: bool = False,
) -> list[dict[str, Any]]:
    def answer_one(question: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            return _answer(method, reader, question)
        except Exception as exc:
            if not continue_on_error:
                raise
            return _prediction_record(
                question,
                {
                    "prediction": "",
                    "retrieved_memory_ids": [],
                    "method_error": f"{type(exc).__name__}: {exc}"[:2000],
                },
                time.perf_counter() - started,
            )

    batch_answer = getattr(method, "answer_many", None)
    if concurrency <= 1 or not callable(batch_answer) or len(questions) <= 1:
        return [answer_one(question) for question in questions]
    resolved = [_resolved_question(reader, question) for question in questions]
    started = time.perf_counter()
    try:
        results = batch_answer(resolved, concurrency=concurrency)
    except Exception:
        if not continue_on_error:
            raise
        return [answer_one(question) for question in questions]
    elapsed = time.perf_counter() - started
    if len(results) != len(questions):
        raise RuntimeError("answer_many returned a different number of results")
    per_item = elapsed / len(questions)
    return [
        _prediction_record(question, result, per_item)
        for question, result in zip(questions, results)
    ]


def run_context(
    method: MemoryMethod,
    reader: BundleReader,
    batch: ContextBatch,
    *,
    memory_view: str = "raw",
    timings: dict[str, float] | None = None,
    query_concurrency: int = 1,
    completed_question_ids: set[str] | frozenset[str] = frozenset(),
    continue_on_query_error: bool = False,
    prediction_sink=None,
) -> list[dict[str, Any]]:
    """Run one context while honoring each question's visible-memory scope."""
    memories = sorted(batch.memories, key=lambda row: row["sequence"])
    all_questions: list[dict[str, Any]] = []
    prefix_questions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    explicit_questions: list[dict[str, Any]] = []

    for question in batch.questions:
        if str(question["question_id"]) in completed_question_ids:
            continue
        scope = question.get("memory_scope", {"mode": "all"})
        mode = scope.get("mode", "all")
        if mode == "all":
            all_questions.append(question)
        elif mode == "prefix":
            cutoff = scope.get("max_sequence")
            if not isinstance(cutoff, int):
                raise ValueError(f"{question['question_id']}: prefix scope needs integer max_sequence")
            prefix_questions[cutoff].append(question)
        elif mode == "ids":
            explicit_questions.append(question)
        else:
            raise ValueError(f"{question['question_id']}: unsupported memory_scope mode {mode!r}")

    predictions: list[dict[str, Any]] = []
    visible_context = _visible_context(batch.context)
    timing = timings if timings is not None else defaultdict(float)

    def digest_call(call, *args, **kwargs):
        started = time.perf_counter()
        try:
            return call(*args, **kwargs)
        finally:
            timing["digest_seconds"] += time.perf_counter() - started

    def synchronize() -> None:
        barrier = getattr(method, "synchronize_memory", None)
        if callable(barrier):
            digest_call(barrier)

    def close_context(primary_error: BaseException | None = None) -> None:
        if primary_error is None:
            digest_call(method.end_context)
            return
        # A method-specific abort path must release resources without retrying
        # the mutation that already failed. Legacy adapters still receive
        # end_context, but a cleanup failure must never replace the root cause.
        cleanup = getattr(method, "abort_context", None)
        if not callable(cleanup):
            cleanup = method.end_context
        try:
            digest_call(cleanup)
        except BaseException as cleanup_error:
            primary_error.add_note(
                "context cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )

    digest_call(method.begin_context, visible_context)
    cursor = 0
    try:
        for cutoff in sorted(prefix_questions):
            while cursor < len(memories) and memories[cursor]["sequence"] <= cutoff:
                digest_call(
                    method.ingest,
                    _resolved_memory(reader, memories[cursor], memory_view=memory_view),
                )
                timing["ingest_calls"] += 1
                cursor += 1
            synchronize()
            for prediction in _answer_group(
                method,
                reader,
                prefix_questions[cutoff],
                query_concurrency,
                continue_on_error=continue_on_query_error,
            ):
                timing["answer_seconds"] += float(prediction["latency_seconds"])
                predictions.append(prediction)
                if prediction_sink:
                    prediction_sink(prediction)
        while cursor < len(memories):
            digest_call(
                method.ingest,
                _resolved_memory(reader, memories[cursor], memory_view=memory_view),
            )
            timing["ingest_calls"] += 1
            cursor += 1
        synchronize()
        for start in range(0, len(all_questions), max(1, query_concurrency)):
            for prediction in _answer_group(
                method,
                reader,
                all_questions[start : start + max(1, query_concurrency)],
                query_concurrency,
                continue_on_error=continue_on_query_error,
            ):
                timing["answer_seconds"] += float(prediction["latency_seconds"])
                predictions.append(prediction)
                if prediction_sink:
                    prediction_sink(prediction)
    except BaseException as exc:
        close_context(exc)
        raise
    else:
        close_context()

    by_id = {memory["memory_id"]: memory for memory in memories}
    for question in explicit_questions:
        selected_ids = question["memory_scope"].get("memory_ids", [])
        digest_call(method.begin_context, visible_context)
        try:
            selected = [by_id[memory_id] for memory_id in selected_ids]
            for memory in sorted(selected, key=lambda row: row["sequence"]):
                digest_call(
                    method.ingest,
                    _resolved_memory(reader, memory, memory_view=memory_view),
                )
                timing["ingest_calls"] += 1
            synchronize()
            prediction = _answer_group(
                method,
                reader,
                [question],
                1,
                continue_on_error=continue_on_query_error,
            )[0]
            timing["answer_seconds"] += float(prediction["latency_seconds"])
            predictions.append(prediction)
            if prediction_sink:
                prediction_sink(prediction)
        except BaseException as exc:
            close_context(exc)
            raise
        else:
            close_context()
    return predictions


def run_bundle(
    method: MemoryMethod,
    bundle_root: Path,
    output_path: Path,
    *,
    subset: str | None = None,
    split: str | None = None,
    task_subcategory: str | None = None,
    memory_view: str = "raw",
    query_concurrency: int = 1,
    caption_sidecar: Path | None = None,
    pdf_policy: str = "off",
    pdf_page_images: int = 0,
    resume_predictions: bool = False,
    continue_on_query_error: bool = False,
    question_ids_path: Path | None = None,
) -> dict[str, Any]:
    reader = BundleReader(bundle_root, caption_sidecar=caption_sidecar,
                          pdf_policy=pdf_policy, pdf_page_images=pdf_page_images)
    from ..preprocessing.pdf import check_pdf_checkpoint
    check_pdf_checkpoint(getattr(method, "checkpoint_dir", None),
                         policy=pdf_policy, page_images=pdf_page_images)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_question_ids: set[str] = set()
    initial_count = 0
    if resume_predictions and output_path.is_file():
        for line_number, line in enumerate(
            output_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"cannot resume malformed prediction line {line_number}: {output_path}"
                ) from exc
            question_id = str(row.get("question_id", ""))
            if not question_id or question_id in completed_question_ids:
                raise RuntimeError(
                    f"cannot resume missing/duplicate question_id at line {line_number}: {output_path}"
                )
            completed_question_ids.add(question_id)
            initial_count += 1
    count = initial_count
    question_ids = None
    if question_ids_path is not None:
        question_ids = {
            line.strip()
            for line in question_ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if not question_ids:
            raise ValueError(f"question ID allowlist is empty: {question_ids_path}")
    started = time.perf_counter()
    timings: dict[str, float] = defaultdict(float)
    mode = "a" if resume_predictions else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        def write_prediction(prediction):
            nonlocal count
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            handle.flush()
            count += 1

        for batch in reader.iter_context_batches(
            subset=subset,
            split=split,
            task_subcategory=task_subcategory,
            question_ids=question_ids,
        ):
            run_context(
                method,
                reader,
                batch,
                memory_view=memory_view,
                timings=timings,
                query_concurrency=query_concurrency,
                completed_question_ids=completed_question_ids,
                continue_on_query_error=continue_on_query_error,
                prediction_sink=write_prediction,
            )
    elapsed = time.perf_counter() - started
    digest_seconds = float(timings["digest_seconds"])
    answer_seconds = float(timings["answer_seconds"])
    return {
        "bundle": str(bundle_root),
        "output": str(output_path),
        "predictions": count,
        "resumed_predictions": initial_count,
        "new_predictions": count - initial_count,
        "task_subcategory": task_subcategory,
        "pdf_policy": pdf_policy,
        "pdf_page_images": pdf_page_images,
        "memory_ingest_calls": int(timings["ingest_calls"]),
        "digest_seconds": digest_seconds,
        "answer_seconds": answer_seconds,
        "overhead_seconds": max(0.0, elapsed - digest_seconds - answer_seconds),
        "elapsed_seconds": elapsed,
    }


def digest_bundle(
    method: MemoryMethod,
    bundle_root: Path,
    *,
    subset: str | None = None,
    split: str | None = None,
    task_subcategory: str | None = None,
    memory_view: str = "raw",
    caption_sidecar: Path | None = None,
    pdf_policy: str = "off",
    pdf_page_images: int = 0,
    memory_ids_path: Path | None = None,
) -> dict[str, Any]:
    """Build every selected context without exposing evaluation questions.

    Method/model construction intentionally happens before this function, so
    the measured interval matches the digest boundary used by ``run_bundle``:
    begin_context, ingest, synchronization, and end_context. Callers that need
    a cold measurement must provide a new, empty method checkpoint directory.
    """

    reader = BundleReader(bundle_root, caption_sidecar=caption_sidecar,
                          pdf_policy=pdf_policy, pdf_page_images=pdf_page_images)
    from ..preprocessing.pdf import check_pdf_checkpoint
    check_pdf_checkpoint(getattr(method, "checkpoint_dir", None),
                         policy=pdf_policy, page_images=pdf_page_images)
    requested_memory_ids: set[str] | None = None
    if memory_ids_path is not None:
        requested_memory_ids = {
            line.strip()
            for line in memory_ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if not requested_memory_ids:
            raise ValueError(f"memory ID allowlist is empty: {memory_ids_path}")
    observed_memory_ids: set[str] = set()
    started = time.perf_counter()
    digest_seconds = 0.0
    memory_ingest_calls = 0
    context_reports: list[dict[str, Any]] = []

    for batch in reader.iter_context_batches(
        subset=subset,
        split=split,
        task_subcategory=task_subcategory,
    ):
        selected_memories = [
            memory
            for memory in batch.memories
            if requested_memory_ids is None
            or str(memory["memory_id"]) in requested_memory_ids
        ]
        if not selected_memories:
            continue
        context_started = time.perf_counter()
        method.begin_context(_visible_context(batch.context))
        try:
            for memory in sorted(selected_memories, key=lambda row: row["sequence"]):
                method.ingest(
                    _resolved_memory(reader, memory, memory_view=memory_view)
                )
                observed_memory_ids.add(str(memory["memory_id"]))
                memory_ingest_calls += 1
            barrier = getattr(method, "synchronize_memory", None)
            if callable(barrier):
                barrier()
        except BaseException as error:
            cleanup = getattr(method, "abort_context", None)
            if not callable(cleanup):
                cleanup = method.end_context
            try:
                cleanup()
            except BaseException as cleanup_error:
                error.add_note(
                    "digest cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise
        else:
            method.end_context()
        context_elapsed = time.perf_counter() - context_started
        digest_seconds += context_elapsed
        context_reports.append(
            {
                "context_id": str(batch.context["context_id"]),
                "memories": len(selected_memories),
                "digest_seconds": context_elapsed,
            }
        )

    if requested_memory_ids is not None:
        missing = requested_memory_ids - observed_memory_ids
        if missing:
            preview = ", ".join(sorted(missing)[:5])
            raise ValueError(
                f"memory ID allowlist has {len(missing)} IDs absent from selected bundle: {preview}"
            )
    elapsed = time.perf_counter() - started
    return {
        "protocol": "mmmb-cold-digest-1.0",
        "bundle": str(bundle_root),
        "subset": subset,
        "split": split,
        "task_subcategory": task_subcategory,
        "pdf_policy": pdf_policy,
        "pdf_page_images": pdf_page_images,
        "memory_view": memory_view,
        "memory_ids": str(memory_ids_path) if memory_ids_path else None,
        "contexts": len(context_reports),
        "memory_ingest_calls": memory_ingest_calls,
        "digest_seconds": digest_seconds,
        "overhead_seconds": max(0.0, elapsed - digest_seconds),
        "elapsed_seconds": elapsed,
        "context_reports": context_reports,
    }
