#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

from mm_memory_bench.harness import _resolved_question
from mm_memory_bench.methods import (
    ConcreteMemixMethod,
    FaissFlatIPIndex,
    GMEQwen2VLEmbedder,
    GenerationConfig,
)
from mm_memory_bench.methods import concrete_memix as concrete_memix_module
from mm_memory_bench.methods.media import question_text as canonical_question_text
from mm_memory_bench.reader import BundleReader


CORE_FILES = (
    "scripts/QA_Agent/MMRAG/memix_memory.py",
    "scripts/QA_Agent/MMRAG/iterative_reasoning_retrieval_eval.py",
    "scripts/QA_Agent/MMRAG/task_aware_cascade_retrieval_eval.py",
)
SOURCE_PRIOR_PROTOCOL = "memix-source-prior-v2"


class UnusedAnswerModel:
    def complete(self, messages, *, tools=None, response_format=None):
        raise RuntimeError("retrieval-only run must not call the answer model")


class UnusedEmbedder:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def encode_units(self, units):
        raise RuntimeError("rerank worker must use cached source priors")


_WORKER_METHOD: ConcreteMemixMethod | None = None


def configure_legacy_empty_choices(enabled: bool) -> None:
    if not enabled:
        return

    def legacy_question_text(question: Mapping[str, Any]) -> str:
        text = canonical_question_text(question)
        return f"{text}\nChoices:\n" if question.get("choices") == [] else text

    concrete_memix_module.question_text = legacy_question_text


def core_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for relative in CORE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_manifest_hash(root: Path) -> str:
    """Hash model identity/config without rereading multi-GB weight shards."""
    digest = hashlib.sha256()
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        stat = root.stat()
        digest.update(root.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        return digest.hexdigest()
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = str(path.relative_to(root))
        stat = path.stat()
        digest.update(relative.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        if path.name in {
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
        }:
            digest.update(path.read_bytes())
    return digest.hexdigest()


def read_jsonl(path: Path | None, key: str) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
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


def append_jsonl(handle, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    handle.flush()


def load_questions(bundle: Path) -> tuple[BundleReader, list[dict[str, Any]], list[dict[str, Any]]]:
    reader = BundleReader(bundle)
    batches = list(reader.iter_context_batches())
    if len(batches) != 1:
        raise ValueError(f"expected one ATM context, got {len(batches)}")
    raw_questions = batches[0].questions
    resolved = [_resolved_question(reader, question) for question in raw_questions]
    return reader, raw_questions, resolved


def hydrate_method(
    *,
    checkpoint_state: Path,
    embedding_model: Path,
    memix_repo: Path,
    source_top_k: int,
) -> tuple[ConcreteMemixMethod, dict[str, Any]]:
    state = json.loads(checkpoint_state.read_text(encoding="utf-8"))
    config = dict(state.get("config", {}))
    expected_hash = str(config.get("core_sha256", ""))
    actual_hash = core_hash(memix_repo)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Memix core hash mismatch: checkpoint={expected_hash} current={actual_hash}"
        )
    dimension = int(config.get("embedding_dimension", 1536))
    video_frames = int(config.get("video_frames", 8))
    media_atomic_limit = int(config.get("media_atomic_limit", 0))

    embedder = GMEQwen2VLEmbedder(embedding_model)
    if embedder.dimension != dimension:
        raise RuntimeError(
            f"embedding dimension mismatch: checkpoint={dimension} model={embedder.dimension}"
        )
    method = ConcreteMemixMethod(
        GenerationConfig(top_k=10),
        answer_model=UnusedAnswerModel(),
        embedder=embedder,
        index=FaissFlatIPIndex(dimension),
        memix_repo=memix_repo,
        checkpoint_dir=None,
        source_top_k=source_top_k,
        video_frames=video_frames,
        media_atomic_limit=media_atomic_limit,
    )
    method.begin_context({"context_id": str(state["context_id"])})
    method._records = [dict(record) for record in state["records"]]
    method._record_by_id = {
        str(record["item_id"]): record for record in method._records
    }
    method._embedding_units = [dict(unit) for unit in state["embedding_units"]]
    method._processed_memory_ids = {
        str(value) for value in state.get("processed_memory_ids", [])
    }
    method.index.load(checkpoint_state.parent / str(state["index_file"]))
    if method.index.size != len(method._embedding_units):
        raise RuntimeError("checkpoint index/embedding-unit row mismatch")
    method._rebuild_memory_index()
    return method, state


def hydrate_reranker(
    *, checkpoint_state: Path, memix_repo: Path, source_top_k: int,
    public_source_ids: Mapping[str, str] | None = None,
) -> ConcreteMemixMethod:
    state = json.loads(checkpoint_state.read_text(encoding="utf-8"))
    config = dict(state.get("config", {}))
    expected_hash = str(config.get("core_sha256", ""))
    actual_hash = core_hash(memix_repo)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Memix core hash mismatch: checkpoint={expected_hash} current={actual_hash}"
        )
    dimension = int(config.get("embedding_dimension", 1536))
    method = ConcreteMemixMethod(
        GenerationConfig(top_k=10),
        answer_model=UnusedAnswerModel(),
        embedder=UnusedEmbedder(dimension),
        index=FaissFlatIPIndex(dimension),
        memix_repo=memix_repo,
        checkpoint_dir=None,
        source_top_k=source_top_k,
        video_frames=int(config.get("video_frames", 8)),
        media_atomic_limit=int(config.get("media_atomic_limit", 0)),
    )
    method.begin_context({"context_id": str(state["context_id"])})
    method._records = [dict(record) for record in state["records"]]
    if public_source_ids:
        for record in method._records:
            source_id = public_source_ids.get(str(record["item_id"]))
            if source_id is not None:
                record["source_id"] = source_id
                metadata = record.get("metadata")
                if isinstance(metadata, dict):
                    metadata["source_id"] = source_id
    method._record_by_id = {
        str(record["item_id"]): record for record in method._records
    }
    method._rebuild_memory_index()
    return method


def init_rerank_worker(
    checkpoint_state: Path,
    memix_repo: Path,
    source_top_k: int,
    legacy_empty_choices: bool,
    public_source_ids: Mapping[str, str] | None = None,
) -> None:
    global _WORKER_METHOD
    configure_legacy_empty_choices(legacy_empty_choices)
    _WORKER_METHOD = hydrate_reranker(
        checkpoint_state=checkpoint_state,
        memix_repo=memix_repo,
        source_top_k=source_top_k,
        public_source_ids=public_source_ids,
    )


def process_select(values):
    if _WORKER_METHOD is None:
        raise RuntimeError("rerank worker was not initialized")
    raw, resolved, source_detail = values
    query_started = time.perf_counter()
    selected_ids, filter_counts = _WORKER_METHOD._select_evidence(
        resolved, source_detail=source_detail
    )
    return raw, selected_ids, filter_counts, time.perf_counter() - query_started


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fresh GME query encoding and Memix reranking without answer generation."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint-state", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--memix-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-priors", type=Path, required=True)
    parser.add_argument("--source-top-k", type=int, default=200)
    parser.add_argument(
        "--source-priors-only",
        action="store_true",
        help="Stop after fresh query encoding and top-k source-prior generation.",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--legacy-empty-choices",
        action="store_true",
        help="Reproduce the 2026-07-15 adapter that appended an empty Choices header.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=0,
        help="Use isolated spawn workers for CPU-bound Memix reranking.",
    )
    parser.add_argument("--progress-interval", type=int, default=25)
    args = parser.parse_args()

    if (
        args.source_top_k < 10
        or args.concurrency <= 0
        or args.processes < 0
        or args.progress_interval <= 0
    ):
        parser.error("invalid source-top-k, concurrency, or progress-interval")

    started = time.perf_counter()
    configure_legacy_empty_choices(args.legacy_empty_choices)
    _, raw_questions, questions = load_questions(args.bundle)
    completed = read_jsonl(args.output, "question_id")
    cached_priors = read_jsonl(args.source_priors, "question_id")
    checkpoint_payload = json.loads(args.checkpoint_state.read_text(encoding="utf-8"))
    checkpoint_index = (
        args.checkpoint_state.parent / str(checkpoint_payload["index_file"])
    )
    source_prior_contract = {
        "source_prior_protocol": SOURCE_PRIOR_PROTOCOL,
        "source_top_k": args.source_top_k,
        "legacy_empty_choices": args.legacy_empty_choices,
        "checkpoint_state_sha256": file_hash(args.checkpoint_state),
        "checkpoint_index_sha256": file_hash(checkpoint_index),
        "memix_core_sha256": core_hash(args.memix_repo),
        "embedding_model_manifest_sha256": model_manifest_hash(args.embedding_model),
        "questions_sha256": file_hash(args.bundle / "questions.jsonl"),
    }
    question_ids = {str(question["question_id"]) for question in raw_questions}
    if not set(completed).issubset(question_ids):
        raise ValueError("output contains question IDs outside this bundle")
    if not set(cached_priors).issubset(question_ids):
        raise ValueError("source-prior cache contains question IDs outside this bundle")
    incompatible_priors = [
        question_id
        for question_id, row in cached_priors.items()
        if any(row.get(key) != value for key, value in source_prior_contract.items())
    ]
    if incompatible_priors:
        raise ValueError(
            "source-prior cache was produced under a different source-prior contract; "
            "use a new cache path"
        )
    incompatible_completed = [
        question_id
        for question_id, row in completed.items()
        if (
            not isinstance(row.get("metadata"), Mapping)
            or row["metadata"].get("source_prior_contract") != source_prior_contract
        )
    ]
    if incompatible_completed:
        raise ValueError(
            "retrieval output was produced under a different source-prior contract; "
            "use a new output path"
        )

    method, state = hydrate_method(
        checkpoint_state=args.checkpoint_state,
        embedding_model=args.embedding_model,
        memix_repo=args.memix_repo,
        source_top_k=args.source_top_k,
    )
    pending = [
        (raw, resolved)
        for raw, resolved in zip(raw_questions, questions)
        if str(raw["question_id"]) not in completed
    ]
    print(json.dumps({
        "event": "ready",
        "questions": len(raw_questions),
        "pending": len(pending),
        "cached_priors": len(cached_priors),
        "records": len(method._records),
        "embedding_units": len(method._embedding_units),
        "index_rows": method.index.size,
        "checkpoint_format": state.get("config", {}).get("format"),
        "core_sha256": state.get("config", {}).get("core_sha256"),
        "legacy_empty_choices": args.legacy_empty_choices,
    }), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.source_priors.parent.mkdir(parents=True, exist_ok=True)
    source_details: list[dict[str, Any]] = []
    prior_mode = "a" if args.source_priors.exists() else "w"
    with args.source_priors.open(prior_mode, encoding="utf-8") as prior_handle:
        for index, (raw, resolved) in enumerate(pending, 1):
            question_id = str(raw["question_id"])
            cached = cached_priors.get(question_id)
            if cached is None:
                query_started = time.perf_counter()
                detail = method._source_detail(resolved)
                cached = {
                    "question_id": question_id,
                    "retrieval_ids": detail["retrieval_ids"],
                    "retrieval_scores": detail["retrieval_scores"],
                    "latency_seconds": time.perf_counter() - query_started,
                    "legacy_empty_choices": args.legacy_empty_choices,
                    **source_prior_contract,
                }
                append_jsonl(prior_handle, cached)
                cached_priors[question_id] = cached
            source_details.append({
                "retrieval_ids": list(cached["retrieval_ids"]),
                "retrieval_scores": list(cached["retrieval_scores"]),
            })
            if index % args.progress_interval == 0 or index == len(pending):
                print(json.dumps({
                    "event": "source_priors",
                    "done": index,
                    "total": len(pending),
                    "elapsed_seconds": time.perf_counter() - started,
                }), flush=True)

    if args.source_priors_only:
        print(json.dumps({
            "event": "complete_source_priors",
            "source_priors": str(args.source_priors),
            "questions": len(raw_questions),
            "elapsed_seconds": time.perf_counter() - started,
        }), flush=True)
        return 0

    def select(values):
        raw, resolved, source_detail = values
        query_started = time.perf_counter()
        selected_ids, filter_counts = method._select_evidence(
            resolved, source_detail=source_detail
        )
        return raw, selected_ids, filter_counts, time.perf_counter() - query_started

    output_mode = "a" if args.output.exists() else "w"
    selection_inputs = [
        (raw, resolved, detail)
        for (raw, resolved), detail in zip(pending, source_details)
    ]
    with args.output.open(output_mode, encoding="utf-8") as output_handle:
        if args.processes:
            context = mp.get_context("spawn")
            pool_context = context.Pool(
                processes=args.processes,
                initializer=init_rerank_worker,
                initargs=(
                    args.checkpoint_state,
                    args.memix_repo,
                    args.source_top_k,
                    args.legacy_empty_choices,
                    None,
                ),
            )
            selections = pool_context.imap(process_select, selection_inputs, chunksize=1)
        else:
            pool_context = ThreadPoolExecutor(max_workers=args.concurrency)
            selections = pool_context.map(select, selection_inputs)
        try:
            for index, (raw, selected_ids, filter_counts, latency) in enumerate(
                selections, 1
            ):
                append_jsonl(output_handle, {
                    "question_id": raw["question_id"],
                    "semantic_question_id": raw.get(
                        "semantic_question_id", raw["question_id"]
                    ),
                    "context_id": raw["context_id"],
                    "subset": raw.get("subset", "default"),
                    "prediction": "",
                    "retrieved_memory_ids": selected_ids,
                    "latency_seconds": latency,
                    "metadata": {
                        "method": "memix-canonical-source-prior-retrieval-only",
                        "source_top_k": args.source_top_k,
                        "filter_counts": filter_counts,
                        "legacy_empty_choices": args.legacy_empty_choices,
                        "source_prior_contract": source_prior_contract,
                    },
                })
                if index % args.progress_interval == 0 or index == len(pending):
                    print(json.dumps({
                        "event": "memix_rerank",
                        "done": index,
                        "total": len(pending),
                        "elapsed_seconds": time.perf_counter() - started,
                    }), flush=True)
        finally:
            if args.processes:
                pool_context.close()
                pool_context.join()
            else:
                pool_context.shutdown(wait=True)

    print(json.dumps({
        "event": "complete",
        "output": str(args.output),
        "source_priors": str(args.source_priors),
        "questions": len(raw_questions),
        "elapsed_seconds": time.perf_counter() - started,
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
