#!/usr/bin/env python3
"""Generate canonical Memix answers from an immutable retrieval JSONL.

This deliberately bypasses retrieval: every answer is conditioned on the
ordered ``retrieved_memory_ids`` already recorded by a retrieval experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from mm_memory_bench.runner.benchmark import _prediction_record, _resolved_question
from mm_memory_bench.methods.base import GenerationConfig
from mm_memory_bench.methods.backends import (
    ContextWindowExceeded,
    OpenAICompatibleQwenVL,
    openai_tools,
)
from mm_memory_bench.methods.concrete_memix import ConcreteMemixMethod, _reader_data_url
from mm_memory_bench.methods.media import question_text, uniformly_sample_video
from mm_memory_bench.benchmarks.reader import BundleReader
from mm_memory_bench.benchmarks.bundle import resolve_asset_path

from run_memix_retrieval_only import (
    UnusedEmbedder,
    configure_legacy_empty_choices,
    core_hash,
)

READER_RUN_PROTOCOL = "graphmemix-fixed-evidence-reader-v3"


def retrieval_contract_sha256(row: Mapping[str, Any]) -> str:
    """Bind a reader row to IDs, ordering, actions, and selector metadata."""
    payload = json.dumps(
        dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ReaderOnlyIndex:
    """Index placeholder: fixed-evidence answer generation never searches."""

    def reset(self) -> None:
        pass


class SinglePassQwenVL(OpenAICompatibleQwenVL):
    """Avoid a duplicate vision preflight for prompts accepted by vLLM.

    The chat endpoint enforces the same model context boundary.  If it rejects
    a near-limit prompt, fall back to the canonical preflight path, which can
    shrink the output budget or raise ``ContextWindowExceeded`` for the
    existing image-resize recovery policy.
    """

    def complete(self, messages, *, tools=None, response_format=None) -> str:
        self._local.last_preflight_tokens = None
        self._local.last_preflight_source = "single_pass_skipped"
        self._local.last_tokenize_error = None
        normalized_tools = openai_tools(tools)
        payload = self.completion_payload(messages, self.config.max_output_tokens)
        if normalized_tools:
            payload["tools"] = normalized_tools
        if response_format:
            payload["response_format"] = dict(response_format)
        try:
            result = self._transport("/chat/completions", payload)
        except RuntimeError as exc:
            detail = str(exc).lower()
            if not any(
                marker in detail
                for marker in (
                    "maximum context length",
                    "context length",
                    "context window",
                    "too many tokens",
                    "too long",
                )
            ):
                raise
            return super().complete(
                messages, tools=tools, response_format=response_format
            )
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, Mapping) else None
        self._local.last_input_tokens = prompt_tokens if isinstance(prompt_tokens, int) else None
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, Mapping) else None
        self._local.last_completion_tokens = (
            completion_tokens if isinstance(completion_tokens, int) else None
        )
        details = usage.get("prompt_tokens_details") if isinstance(usage, Mapping) else None
        cached_tokens = details.get("cached_tokens") if isinstance(details, Mapping) else None
        self._local.last_cached_input_tokens = (
            cached_tokens if isinstance(cached_tokens, int) else None
        )
        try:
            message = result["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("vLLM response has no assistant message") from exc
        content = message.get("content")
        if content:
            return str(content)
        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            return json.dumps(tool_calls, ensure_ascii=False)
        raise RuntimeError("vLLM response has empty assistant content and no tool calls")


def hydrate_reader(
    *,
    checkpoint_state: Path,
    memix_repo: Path,
    generation: GenerationConfig,
    answer_model: OpenAICompatibleQwenVL,
    source_top_k: int,
    public_source_ids: Mapping[str, str],
    public_asset_paths: Mapping[str, str],
    reader_ocr_budget_chars: int = 0,
    video_frames_override: int = 0,
) -> ConcreteMemixMethod:
    state = json.loads(checkpoint_state.read_text(encoding="utf-8"))
    config = dict(state.get("config", {}))
    expected_hash = str(config.get("core_sha256", ""))
    actual_hash = core_hash(memix_repo)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Memix core hash mismatch: checkpoint={expected_hash} current={actual_hash}"
        )
    method = ConcreteMemixMethod(
        generation,
        answer_model=answer_model,
        embedder=UnusedEmbedder(int(config.get("embedding_dimension", 1536))),
        index=ReaderOnlyIndex(),
        memix_repo=memix_repo,
        checkpoint_dir=None,
        source_top_k=source_top_k,
        video_frames=(
            video_frames_override
            if video_frames_override > 0
            else int(config.get("video_frames", 8))
        ),
        media_atomic_limit=int(config.get("media_atomic_limit", 0)),
        reader_ocr_budget_chars=reader_ocr_budget_chars,
    )
    method.begin_context({"context_id": str(state["context_id"])})
    method._records = [dict(record) for record in state["records"]]
    for record in method._records:
        source_id = public_source_ids.get(str(record["item_id"]))
        if source_id is not None:
            record["source_id"] = source_id
            metadata = record.get("metadata")
            if isinstance(metadata, dict):
                metadata["source_id"] = source_id
        # Checkpoints are intentionally reusable across machines, but older
        # checkpoints may contain absolute media paths from the machine where
        # they were created.  The canonical bundle asset table is the source
        # of truth, so rebase every checkpoint media part by asset_id.
        for part in record.get("content", []):
            asset_id = str(part.get("asset_id", ""))
            public_path = public_asset_paths.get(asset_id)
            if public_path is not None:
                part["path"] = public_path
    method._record_by_id = {
        str(record["item_id"]): record for record in method._records
    }
    # Fixed-evidence answer generation reads only ``_record_by_id``. Building
    # the full Memix retrieval index here can take minutes for large contexts
    # and is both unused and semantically irrelevant to the reader prompt.
    return method


def read_jsonl(path: Path, key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            value = str(row[key])
            if value in rows:
                raise ValueError(f"duplicate {key}={value!r} in {path}:{line_number}")
            rows[value] = row
    return rows


def checkpoint_states(root: Path) -> dict[str, Path]:
    candidates = list(root.glob("state.json"))
    candidates.extend(root.glob("contexts/*/state.json"))
    result: dict[str, Path] = {}
    for path in sorted(candidates):
        state = json.loads(path.read_text(encoding="utf-8"))
        context_id = str(state["context_id"])
        if context_id in result:
            raise ValueError(f"duplicate checkpoint for context {context_id!r}")
        result[context_id] = path
    if not result:
        raise ValueError(f"no checkpoint states below {root}")
    return result


def append_jsonl(handle, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    handle.flush()


def representation_actions(retrieval_row: Mapping[str, Any]) -> dict[str, str]:
    """Read the explicit per-memory representation selected by a packer."""
    metadata = retrieval_row.get("metadata", {})
    rows = metadata.get("selected_actions", []) if isinstance(metadata, Mapping) else []
    actions: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("memory_id"):
            continue
        memory_id = str(row["memory_id"])
        action = str(row.get("action", ""))
        if action not in {"text", "low", "high"}:
            raise ValueError(f"invalid representation action {action!r} for {memory_id}")
        if memory_id in actions:
            raise ValueError(f"duplicate representation action for {memory_id}")
        actions[memory_id] = action
    return actions


def derived_graph_evidence(retrieval_row: Mapping[str, Any]) -> str:
    """Return bounded, explicitly attributed graph evidence for the reader.

    Retrieval methods may preserve a short proof assembled from their graph.
    Keeping it in metadata avoids manufacturing a memory record while still
    allowing the reader to inspect the aggregation that caused raw memories to
    be selected.  The hard limit keeps this channel cheap and prevents an
    unbounded agent trace from entering the answer prompt.
    """
    metadata = retrieval_row.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return ""
    value = metadata.get("derived_graph_evidence", "")
    if not isinstance(value, str):
        return ""
    return value.strip()[:4000]


def generate_action_aware_answer(
    method: ConcreteMemixMethod,
    question: Mapping[str, Any],
    evidence_ids: list[str],
    actions: Mapping[str, str],
    *,
    low_image_max_edge: int,
    high_image_max_edge: int = 0,
) -> str:
    """Generate from fixed IDs while honoring text/low/high per-item actions."""
    content: list[dict[str, Any]] = []
    for part in question.get("prompt", []):
        if part.get("type") == "image":
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": _reader_data_url(str(part["path"]), high_image_max_edge)
                },
            })
        elif part.get("type") == "video":
            content.extend(
                {"type": "image_url", "image_url": {"url": frame}}
                for frame in uniformly_sample_video(
                    str(part["path"]), method.video_frames, max_edge=high_image_max_edge
                )
            )
    evidence_content: list[dict[str, Any]] = []
    for rank, memory_id in enumerate(evidence_ids, 1):
        action = actions.get(memory_id)
        if action is None:
            raise ValueError(f"missing representation action for {memory_id}")
        record = method._record_by_id[memory_id]
        evidence_content.append({
            "type": "text",
            "text": (
                f"Evidence {rank}; memory_id={memory_id}; "
                f"source_id={record.get('source_id')}; "
                f"modality={record.get('modality', 'unknown')}:\n"
                f"{method._reader_text(record, query=question_text(question))}"
            ),
        })
        if action == "text":
            continue
        image_edge = low_image_max_edge if action == "low" else high_image_max_edge
        for part in record.get("content", []):
            if part.get("type") == "image" and part.get("path"):
                evidence_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": _reader_data_url(str(part["path"]), image_edge)
                    },
                })
            elif part.get("type") == "video" and part.get("path"):
                try:
                    frames = uniformly_sample_video(
                        str(part["path"]), method.video_frames, max_edge=image_edge
                    )
                except RuntimeError:
                    continue
                evidence_content.append({
                    "type": "text",
                    "text": (
                        f"The following {len(frames)} images are sampled frames "
                        f"from the video in Evidence {rank}."
                    ),
                })
                evidence_content.extend(
                    {"type": "image_url", "image_url": {"url": frame}}
                    for frame in frames
                )
    tools = question.get("tools") if question.get("tool_mode") != "plan" else None
    plan_tools = ""
    if question.get("tools") and question.get("tool_mode") == "plan":
        plan_tools = "\nCandidate tools:\n" + json.dumps(
            question["tools"], ensure_ascii=False
        )
    prompt = (
        f"{question.get('instruction', '')}\nQuestion: {question_text(question)}"
        f"{plan_tools}\n\nUse only the following Memix evidence packet."
    )
    content.extend([{"type": "text", "text": prompt}, *evidence_content])
    return method.answer_model.complete([{"role": "user", "content": content}], tools=tools)


def generate_text_only_answer(
    method: ConcreteMemixMethod,
    question: Mapping[str, Any],
    evidence_ids: list[str],
) -> str:
    """Answer from the canonical textual evidence views without media payloads."""
    query = question_text(question)
    evidence = []
    for rank, memory_id in enumerate(evidence_ids, 1):
        record = method._record_by_id[memory_id]
        evidence.append(
            f"Evidence {rank}; memory_id={memory_id}; "
            f"source_id={record.get('source_id')}; "
            f"modality={record.get('modality', 'unknown')}:\n"
            f"{method._reader_text(record, query=query)}"
        )
    tools = question.get("tools") if question.get("tool_mode") != "plan" else None
    plan_tools = ""
    if question.get("tools") and question.get("tool_mode") == "plan":
        plan_tools = "\nCandidate tools:\n" + json.dumps(
            question["tools"], ensure_ascii=False
        )
    prompt = (
        f"{question.get('instruction', '')}\nQuestion: {query}{plan_tools}\n\n"
        "Use only the following Memix evidence packet.\n\n"
        + "\n\n".join(evidence)
    )
    return method.answer_model.complete([{"role": "user", "content": prompt}], tools=tools)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--memix-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8091/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--timeout-seconds", type=float, default=1200,
        help="HTTP timeout per model request; this does not alter the reader prompt contract.",
    )
    parser.add_argument(
        "--single-pass-normal",
        action="store_true",
        help="Skip the separate /tokenize pass unless chat reports a context-limit error.",
    )
    parser.add_argument(
        "--token-count-mode",
        choices=("auto", "local", "remote"),
        default="auto",
        help=(
            "Context preflight source. auto uses local conservative estimates for "
            "hosted gateways and remote /tokenize for local vLLM servers."
        ),
    )
    parser.add_argument("--max-output-tokens", type=int, default=1000)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument(
        "--reasoning-effort", default="minimal",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
        help="Reasoning effort sent to hosted GPT-5-compatible models.",
    )
    parser.add_argument(
        "--overflow-image-max-edge",
        type=int,
        default=2500,
        help="After a 32K overflow, retry with images capped to this longest edge.",
    )
    parser.add_argument("--source-top-k", type=int, default=200)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--question-ids",
        type=Path,
        help="Optional newline-delimited allowlist of question IDs.",
    )
    parser.add_argument(
        "--reader-instruction-suffix",
        default="",
        help="Optional instruction appended to every selected question before answer generation.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="With --resume, discard recorded error rows and regenerate them.",
    )
    parser.add_argument("--legacy-empty-choices", action="store_true")
    parser.add_argument("--allow-empty-retrieval", action="store_true")
    parser.add_argument(
        "--answer-empty-retrieval",
        action="store_true",
        help="Generate from the question alone when a variable-cardinality pack selects zero evidence.",
    )
    parser.add_argument(
        "--allow-variable-retrieval",
        action="store_true",
        help="Accept any non-empty evidence count up to source-top-k instead of exactly ten.",
    )
    parser.add_argument(
        "--respect-selected-actions",
        action="store_true",
        help="Honor metadata.selected_actions text/low/high choices in the reader packet.",
    )
    parser.add_argument("--low-image-max-edge", type=int, default=448)
    parser.add_argument(
        "--video-frames-override",
        type=int,
        default=0,
        help="Override checkpoint video frames per evidence; zero preserves the checkpoint setting.",
    )
    parser.add_argument(
        "--reader-ocr-budget-chars",
        type=int,
        default=0,
        help=(
            "Per-evidence OCR character budget after query-aware line de-duplication; "
            "zero preserves the checkpoint text unchanged."
        ),
    )
    parser.add_argument(
        "--reader-media-mode",
        choices=("full", "text"),
        default="full",
        help="Use full media payloads or only caption/OCR/metadata evidence text.",
    )
    parser.add_argument(
        "--record-errors",
        action="store_true",
        help="Write canonical empty predictions after retries instead of leaving them pending.",
    )
    parser.add_argument("--progress-interval", type=int, default=25)
    args = parser.parse_args()
    configure_legacy_empty_choices(args.legacy_empty_choices)

    if (
        args.concurrency <= 0
        or args.timeout_seconds <= 0
        or args.retries <= 0
        or args.progress_interval <= 0
        or args.overflow_image_max_edge < 0
        or args.low_image_max_edge <= 0
        or args.reader_ocr_budget_chars < 0
        or args.video_frames_override < 0
    ):
        parser.error("concurrency, retries, and progress-interval must be positive")

    started = time.perf_counter()
    reader = BundleReader(args.bundle)
    questions = read_jsonl(args.bundle / "questions.jsonl", "question_id")
    retrieval = read_jsonl(args.retrieval, "question_id")
    unknown = set(retrieval) - set(questions)
    if unknown:
        raise ValueError(f"retrieval contains {len(unknown)} unknown question IDs")
    selected_question_ids: set[str] | None = None
    if args.question_ids is not None:
        values = [
            line.strip()
            for line in args.question_ids.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(values) != len(set(values)):
            raise ValueError("question ID allowlist contains duplicates")
        selected_question_ids = set(values)
        missing_question_ids = selected_question_ids - set(retrieval)
        if missing_question_ids:
            raise KeyError(
                "question ID allowlist contains IDs absent from retrieval: "
                + ", ".join(sorted(missing_question_ids)[:5])
            )

    def valid_length(length: int) -> bool:
        if length == 0:
            return args.allow_empty_retrieval or args.answer_empty_retrieval
        if args.allow_variable_retrieval:
            return length <= args.source_top_k
        return length == 10

    invalid = {
        question_id: len(row.get("retrieved_memory_ids", []))
        for question_id, row in retrieval.items()
        if not valid_length(len(row.get("retrieved_memory_ids", [])))
    }
    if invalid:
        preview = list(invalid.items())[:5]
        expected = "1..source-top-k" if args.allow_variable_retrieval else "exactly 10"
        raise ValueError(f"retrieval rows must contain {expected} IDs: {preview}")

    states = checkpoint_states(args.checkpoint_root)
    reader_contract = {
        "reader_run_protocol": READER_RUN_PROTOCOL,
        "model": args.model,
        "base_url": args.base_url.rstrip("/"),
        "max_output_tokens": args.max_output_tokens,
        "max_model_len": args.max_model_len,
        "reasoning_effort": args.reasoning_effort,
        "overflow_image_max_edge": args.overflow_image_max_edge,
        "source_top_k": args.source_top_k,
        "single_pass_normal": args.single_pass_normal,
        "token_count_mode": args.token_count_mode,
        "reader_instruction_suffix": args.reader_instruction_suffix,
        "legacy_empty_choices": args.legacy_empty_choices,
        "allow_empty_retrieval": args.allow_empty_retrieval,
        "answer_empty_retrieval": args.answer_empty_retrieval,
        "allow_variable_retrieval": args.allow_variable_retrieval,
        "respect_selected_actions": args.respect_selected_actions,
        "low_image_max_edge": args.low_image_max_edge,
        "reader_ocr_budget_chars": args.reader_ocr_budget_chars,
        "reader_media_mode": args.reader_media_mode,
        "memix_core_sha256": core_hash(args.memix_repo),
    }
    if args.video_frames_override > 0:
        reader_contract["video_frames_override"] = args.video_frames_override
    state_hashes = {
        context_id: hashlib.sha256(path.read_bytes()).hexdigest()
        for context_id, path in states.items()
    }

    completed = read_jsonl(args.output, "question_id") if args.resume and args.output.is_file() else {}
    incompatible_completed: list[str] = []
    for question_id, row in completed.items():
        retrieval_row = retrieval.get(question_id)
        question = questions.get(question_id)
        if retrieval_row is None or question is None:
            incompatible_completed.append(question_id)
            continue
        expected_ids = [str(value) for value in retrieval_row.get("retrieved_memory_ids", [])]
        actual_ids = [str(value) for value in row.get("retrieved_memory_ids", [])]
        metadata = row.get("metadata", {})
        stored_contract = (
            metadata.get("reader_run_contract", {})
            if isinstance(metadata, Mapping) else {}
        )
        expected_contract = {
            **reader_contract,
            "checkpoint_state_sha256": state_hashes.get(str(question["context_id"])),
            "retrieval_row_sha256": retrieval_contract_sha256(retrieval_row),
        }
        if actual_ids != expected_ids or stored_contract != expected_contract:
            incompatible_completed.append(question_id)
    if incompatible_completed:
        raise RuntimeError(
            f"{args.output} contains {len(incompatible_completed)} rows whose evidence "
            "IDs or Reader run contract differ from this invocation; use a new output "
            "path instead of resuming"
        )
    existing_error_ids = [
        question_id
        for question_id, row in completed.items()
        if row.get("metadata", {}).get("status") == "error"
        or not str(row.get("prediction", "")).strip()
    ]
    if existing_error_ids and not args.retry_errors:
        raise RuntimeError(
            f"{args.output} contains {len(existing_error_ids)} error or empty rows; "
            "resume with --retry-errors instead of treating them as complete"
        )
    if args.retry_errors:
        if not args.resume:
            parser.error("--retry-errors requires --resume")
        completed = {
            question_id: row
            for question_id, row in completed.items()
            if row.get("metadata", {}).get("status") != "error"
        }
    selected_rows = [
        (questions[question_id], row)
        for question_id, row in retrieval.items()
        if question_id not in completed
        and (selected_question_ids is None or question_id in selected_question_ids)
    ]
    if args.limit:
        selected_rows = selected_rows[: args.limit]

    needed_contexts = {str(question["context_id"]) for question, _ in selected_rows}
    missing_states = needed_contexts - set(states)
    if missing_states:
        raise ValueError(f"missing checkpoints for contexts: {sorted(missing_states)[:5]}")

    public_source_ids = {
        str(row["memory_id"]): str(row["source_id"])
        for row in read_jsonl(args.bundle / "memories.jsonl", "memory_id").values()
        if row.get("source_id") is not None
    }
    public_asset_paths = {
        str(asset_id): str(resolve_asset_path(args.bundle, asset))
        for asset_id, asset in reader.assets.items()
    }
    generation = GenerationConfig(
        model=args.model,
        base_url=args.base_url,
        temperature=0,
        top_k=10,
        max_model_len=args.max_model_len,
        max_output_tokens=args.max_output_tokens,
        overflow_policy="error",
        reasoning_effort=args.reasoning_effort,
    )
    answer_model = (
        SinglePassQwenVL(
            generation,
            timeout_seconds=args.timeout_seconds,
            token_count_mode=args.token_count_mode,
        )
        if args.single_pass_normal
        else OpenAICompatibleQwenVL(
            generation,
            timeout_seconds=args.timeout_seconds,
            token_count_mode=args.token_count_mode,
        )
    )

    rows_by_context: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for question, retrieval_row in selected_rows:
        rows_by_context.setdefault(str(question["context_id"]), []).append(
            (question, retrieval_row)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_mode = "a" if args.resume and args.output.exists() and not args.retry_errors else "w"
    done = 0
    failures: list[tuple[str, str]] = []
    with args.output.open(output_mode, encoding="utf-8") as handle:
        if args.retry_errors:
            for row in completed.values():
                append_jsonl(handle, row)
        for context_index, (context_id, rows) in enumerate(rows_by_context.items(), 1):
            method = hydrate_reader(
                checkpoint_state=states[context_id],
                memix_repo=args.memix_repo,
                generation=generation,
                answer_model=answer_model,
                source_top_k=args.source_top_k,
                public_source_ids=public_source_ids,
                public_asset_paths=public_asset_paths,
                reader_ocr_budget_chars=args.reader_ocr_budget_chars,
                video_frames_override=args.video_frames_override,
            )

            def answer_one(values):
                question, retrieval_row = values
                question_id = str(question["question_id"])
                question_contract = {
                    **reader_contract,
                    "checkpoint_state_sha256": state_hashes[str(question["context_id"])],
                    "retrieval_row_sha256": retrieval_contract_sha256(retrieval_row),
                }
                resolved = _resolved_question(reader, question)
                graph_evidence = derived_graph_evidence(retrieval_row)
                if graph_evidence:
                    resolved = dict(resolved)
                    original_instruction = str(resolved.get("instruction", "")).strip()
                    resolved["instruction"] = "\n\n".join(
                        value for value in (
                            original_instruction,
                            "Graph-derived evidence (verify it against the raw evidence below):\n"
                            + graph_evidence,
                        ) if value
                    )
                if args.reader_instruction_suffix:
                    resolved = dict(resolved)
                    original_instruction = str(resolved.get("instruction", "")).strip()
                    resolved["instruction"] = "\n".join(
                        value
                        for value in (
                            original_instruction,
                            args.reader_instruction_suffix.strip(),
                        )
                        if value
                    )
                evidence_ids = [str(value) for value in retrieval_row["retrieved_memory_ids"]]
                actions = (
                    representation_actions(retrieval_row)
                    if args.respect_selected_actions else {}
                )
                if not evidence_ids and not args.answer_empty_retrieval:
                    method_error = str(
                        retrieval_row.get("metadata", {}).get(
                            "method_error", "fixed retrieval row is empty"
                        )
                    )
                    return _prediction_record(
                        question,
                        {
                            "prediction": "",
                            "retrieved_memory_ids": [],
                            "status": "error",
                            "error_type": "OriginalMethodError",
                            "error": method_error[:2000],
                            "retrieval_metadata": retrieval_row.get("metadata", {}),
                            "reader_run_contract": question_contract,
                        },
                        0.0,
                    )
                missing = [value for value in evidence_ids if value not in method._record_by_id]
                if missing:
                    raise KeyError(f"evidence IDs absent from checkpoint: {missing[:3]}")
                last_error: Exception | None = None
                resize_after_transport_error = False
                for attempt in range(1, args.retries + 1):
                    request_started = time.perf_counter()
                    try:
                        overflow: ContextWindowExceeded | None = None
                        if resize_after_transport_error:
                            prediction = (
                                generate_text_only_answer(method, resolved, evidence_ids)
                                if args.reader_media_mode == "text"
                                else
                                generate_action_aware_answer(
                                    method,
                                    resolved,
                                    evidence_ids,
                                    actions,
                                    low_image_max_edge=args.low_image_max_edge,
                                    high_image_max_edge=args.overflow_image_max_edge,
                                )
                                if args.respect_selected_actions
                                else method._generate_answer(
                                    resolved,
                                    evidence_ids,
                                    image_max_edge=args.overflow_image_max_edge,
                                )
                            )
                        else:
                            try:
                                prediction = (
                                    generate_text_only_answer(method, resolved, evidence_ids)
                                    if args.reader_media_mode == "text"
                                    else
                                    generate_action_aware_answer(
                                        method,
                                        resolved,
                                        evidence_ids,
                                        actions,
                                        low_image_max_edge=args.low_image_max_edge,
                                    )
                                    if args.respect_selected_actions
                                    else method._generate_answer(resolved, evidence_ids)
                                )
                            except ContextWindowExceeded as exc:
                                if args.overflow_image_max_edge <= 0:
                                    raise
                                overflow = exc
                                prediction = (
                                    generate_text_only_answer(method, resolved, evidence_ids)
                                    if args.reader_media_mode == "text"
                                    else
                                    generate_action_aware_answer(
                                        method,
                                        resolved,
                                        evidence_ids,
                                        actions,
                                        low_image_max_edge=args.low_image_max_edge,
                                        high_image_max_edge=args.overflow_image_max_edge,
                                    )
                                    if args.respect_selected_actions
                                    else method._generate_answer(
                                        resolved,
                                        evidence_ids,
                                        image_max_edge=args.overflow_image_max_edge,
                                    )
                                )
                        result = {
                            "prediction": prediction,
                            "retrieved_memory_ids": evidence_ids,
                            "retrieval_metadata": retrieval_row.get("metadata", {}),
                            "retrieval_latency_seconds": retrieval_row.get("latency_seconds"),
                            "answer_attempts": attempt,
                            "reader_input_tokens_actual": answer_model.last_input_tokens,
                            "reader_completion_tokens_actual": answer_model.last_completion_tokens,
                            "reader_cached_input_tokens_actual": answer_model.last_cached_input_tokens,
                            "reader_preflight_tokens": answer_model.last_preflight_tokens,
                            "reader_preflight_source": answer_model.last_preflight_source,
                            "reader_tokenize_error": answer_model.last_tokenize_error,
                            "reader_run_contract": question_contract,
                        }
                        if overflow is not None:
                            result.update({
                                "overflow_recovered": True,
                                "original_input_tokens": overflow.actual,
                                "overflow_image_max_edge": args.overflow_image_max_edge,
                            })
                        if resize_after_transport_error:
                            result.update({
                                "transport_resize_recovered": True,
                                "overflow_image_max_edge": args.overflow_image_max_edge,
                            })
                        return _prediction_record(
                            question, result, time.perf_counter() - request_started
                        )
                    except Exception as exc:  # retry transient endpoint failures
                        last_error = exc
                        if isinstance(exc, ContextWindowExceeded):
                            break
                        detail = str(exc).lower()
                        if args.overflow_image_max_edge > 0 and any(
                            marker in detail
                            for marker in (
                                "eof occurred in violation of protocol",
                                "remote end closed connection",
                                "broken pipe",
                                "request entity too large",
                                "payload too large",
                                "exceeds model's maximum context length",
                                "exceeds the model's maximum context length",
                            )
                        ):
                            resize_after_transport_error = True
                        if attempt < args.retries:
                            time.sleep(min(2 ** (attempt - 1), 4))
                raise RuntimeError(f"{question_id}: {type(last_error).__name__}: {last_error}")

            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                future_to_id = {
                    pool.submit(answer_one, row): str(row[0]["question_id"])
                    for row in rows
                }
                for future in as_completed(future_to_id):
                    question_id = future_to_id[future]
                    try:
                        prediction_row = future.result()
                        append_jsonl(handle, prediction_row)
                        metadata = prediction_row.get("metadata", {})
                        if (
                            metadata.get("status") == "error"
                            or not str(prediction_row.get("prediction", "")).strip()
                        ):
                            failures.append((
                                question_id,
                                str(metadata.get("error", "empty prediction"))[:2000],
                            ))
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        if args.record_errors:
                            retrieval_row = retrieval[question_id]
                            append_jsonl(handle, _prediction_record(
                                questions[question_id],
                                {
                                    "prediction": "",
                                    "retrieved_memory_ids": retrieval_row[
                                        "retrieved_memory_ids"
                                    ],
                                    "status": "error",
                                    "error_type": type(exc).__name__,
                                    "error": error[:2000],
                                    "retrieval_metadata": retrieval_row.get(
                                        "metadata", {}
                                    ),
                                    "reader_run_contract": {
                                        **reader_contract,
                                        "checkpoint_state_sha256": state_hashes[
                                            str(questions[question_id]["context_id"])
                                        ],
                                    },
                                },
                                0.0,
                            ))
                        failures.append((question_id, error))
                    done += 1
                    if done % args.progress_interval == 0 or done == len(selected_rows):
                        print(json.dumps({
                            "event": "answers",
                            "done": done,
                            "total": len(selected_rows),
                            "failures": len(failures),
                            "context": context_id,
                            "context_index": context_index,
                            "contexts": len(rows_by_context),
                            "elapsed_seconds": time.perf_counter() - started,
                        }), flush=True)
            method.end_context()

    summary = {
        "event": "complete" if not failures else "incomplete",
        "output": str(args.output),
        "new_answers": done - len(failures),
        "previous_answers": len(completed),
        "failures": failures[:20],
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
