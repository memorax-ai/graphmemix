#!/usr/bin/env python3
"""Score one GraphMemix candidate set per question with Qwen3-VL verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from mm_memory_bench.runner.benchmark import _resolved_question
from mm_memory_bench.methods.backends import OpenAICompatibleQwenVL
from mm_memory_bench.methods.base import GenerationConfig
from mm_memory_bench.methods.concrete_memix import CAPTION_PROMPT, _reader_data_url
from mm_memory_bench.methods.media import question_text
from mm_memory_bench.benchmarks.reader import BundleReader

from graphmemix_core import (
    MEMORY_SNIPPET_PROTOCOL, candidate_pool, file_sha256, load_adjacency, memory_snippet,
    memory_location, question_native_captions, read_jsonl, source_score_map,
)


def fold(question_id: str, folds: int = 5) -> int:
    digest = hashlib.sha256(question_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        left, right = text.find("{"), text.rfind("}")
        if left < 0 or right <= left:
            raise
        value = json.loads(text[left:right + 1])
    if not isinstance(value, dict):
        raise ValueError("verifier returned non-object JSON")
    return value


def validate_selected_rows(
    value: Any, *, valid_aliases: set[str],
) -> list[tuple[str, float]]:
    """Validate the sparse node-verifier response before accepting a cache row."""
    if not isinstance(value, list):
        raise ValueError("verifier selected field must be a list")
    result: list[tuple[str, float]] = []
    seen: set[str] = set()
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError("verifier selected row must be an object")
        alias = str(row.get("id", ""))
        if alias not in valid_aliases:
            raise ValueError(f"verifier returned unknown candidate id: {alias!r}")
        if alias in seen:
            raise ValueError(f"verifier returned duplicate candidate id: {alias!r}")
        seen.add(alias)
        try:
            score = float(row.get("score"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"verifier returned invalid score for {alias}") from exc
        if not 0.0 <= score <= 5.0:
            raise ValueError(f"verifier score out of range for {alias}")
        result.append((alias, score))
    return result


def modality_tags(memory: Mapping[str, Any]) -> list[str]:
    values = {str(memory.get("kind", "memory"))}
    values.update(str(part.get("type")) for part in memory.get("content", []) if part.get("type"))
    return sorted(values)


def caption_question_images(
    model: OpenAICompatibleQwenVL, question: Mapping[str, Any], native: list[str],
) -> list[str]:
    image_parts = [part for part in question.get("prompt", []) if part.get("type") == "image"]
    if len(native) == len(image_parts):
        return native
    result: list[str] = []
    native_iter = iter(native)
    for part in image_parts:
        annotations = part.get("annotations", {})
        has_native = bool(
            isinstance(annotations, Mapping) and (
                annotations.get("native_image_caption") or annotations.get("caption")
                or (isinstance(annotations.get("derived"), Mapping)
                    and (annotations["derived"].get("text") or annotations["derived"].get("caption")))
            )
        )
        if has_native:
            result.append(next(native_iter))
            continue
        path = part.get("path")
        if not path:
            raise ValueError("resolved query image lacks path")
        caption = model.complete([{
            "role": "user",
            "content": [
                {"type": "text", "text": CAPTION_PROMPT},
                {"type": "image_url", "image_url": {"url": _reader_data_url(str(path), 2500)}},
            ],
        }]).strip()
        if not caption:
            raise RuntimeError("empty query-image caption")
        result.append(caption)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-priors", type=Path, required=True)
    parser.add_argument("--relation-store", type=Path, required=True)
    parser.add_argument("--graph-mode", choices=("none", "explicit", "full"), required=True)
    parser.add_argument("--source-top-l", type=int, default=24)
    parser.add_argument("--candidate-limit", type=int, default=48)
    parser.add_argument(
        "--candidate-override",
        type=Path,
        help="Optional JSONL mapping question_id to a frozen candidate_ids list.",
    )
    parser.add_argument("--semantic-k", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--question-reference",
        type=Path,
        help="Optional JSONL whose question IDs define the exact evaluation subset.",
    )
    parser.add_argument(
        "--question-subcategory",
        help="Optionally restrict scoring to question.task.subcategory.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:18096/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--include-dev-fold", type=int, default=0)
    parser.add_argument("--heldout-limit", type=int, default=100)
    parser.add_argument(
        "--all-questions", action="store_true",
        help="Score every eligible question; existing output rows are reused on resume.",
    )
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument(
        "--reasoning-effort", default="minimal",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
        help="Reasoning effort sent to hosted GPT-5-compatible models.",
    )
    parser.add_argument("--snippet-chars", type=int, default=420)
    parser.add_argument("--ocr-chars", type=int, default=120)
    parser.add_argument(
        "--question-type-mode",
        choices=("empty", "subcategory"),
        default="empty",
        help=(
            "Question-type field exposed to the verifier. 'empty' preserves the "
            "original protocol; 'subcategory' uses question.task.subcategory."
        ),
    )
    args = parser.parse_args()
    if not 0 <= args.include_dev_fold < 5:
        parser.error("--include-dev-fold must be in [0, 4]")
    if args.source_top_l <= 0 or args.candidate_limit < args.source_top_l:
        parser.error("invalid source/candidate limits")
    if args.snippet_chars <= 0 or args.ocr_chars < 0:
        parser.error("--snippet-chars must be positive and --ocr-chars non-negative")

    reader = BundleReader(args.bundle)
    raw_questions = {
        str(question["question_id"]): question
        for batch in reader.iter_context_batches() for question in batch.questions
    }
    resolved = {qid: _resolved_question(reader, row) for qid, row in raw_questions.items()}
    memories = read_jsonl(args.bundle / "memories.jsonl", "memory_id")
    priors = read_jsonl(args.source_priors, "question_id")
    candidate_override = (
        read_jsonl(args.candidate_override, "question_id")
        if args.candidate_override is not None else {}
    )
    adjacency = load_adjacency(args.relation_store, args.graph_mode, args.semantic_k)
    run_contract = {
        "memory_snippet_protocol": MEMORY_SNIPPET_PROTOCOL,
        "verifier_model": args.model,
        "graph_mode": args.graph_mode,
        "source_top_l": args.source_top_l,
        "candidate_limit": args.candidate_limit,
        "semantic_k": args.semantic_k,
        "snippet_chars": args.snippet_chars,
        "ocr_chars": args.ocr_chars,
        "question_type_mode": args.question_type_mode,
        "reasoning_effort": args.reasoning_effort,
        "source_priors_sha256": file_sha256(args.source_priors),
        "relation_store_sha256": file_sha256(args.relation_store),
        "candidate_override_sha256": (
            file_sha256(args.candidate_override)
            if args.candidate_override is not None else None
        ),
    }
    eligible = sorted(set(raw_questions) & set(priors))
    dev = [qid for qid in eligible if fold(qid) == args.include_dev_fold]
    heldout = sorted(
        (qid for qid in eligible if fold(qid) != args.include_dev_fold),
        key=lambda qid: hashlib.sha256(("graphmemix-heldout-v1\0" + qid).encode()).digest(),
    )[:args.heldout_limit]
    if args.question_reference is not None:
        reference = read_jsonl(args.question_reference, "question_id")
        wanted = sorted(set(eligible) & set(reference))
    else:
        wanted = eligible if args.all_questions else dev + heldout
    if args.question_subcategory is not None:
        wanted = [
            qid for qid in wanted
            if str(raw_questions[qid].get("task", {}).get("subcategory", ""))
            == args.question_subcategory
        ]
    if args.max_questions is not None:
        if args.max_questions <= 0:
            parser.error("--max-questions must be positive")
        wanted = wanted[:args.max_questions]

    completed = read_jsonl(args.output, "question_id") if args.output.is_file() else {}
    incompatible = [
        qid for qid, row in completed.items()
        if any(row.get(key) != value for key, value in run_contract.items())
    ]
    if incompatible:
        raise RuntimeError(
            f"{args.output} contains {len(incompatible)} rows from an incompatible "
            "memory snippet protocol; use a new output path instead of resuming it"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    local = threading.local()
    config = GenerationConfig(
        model=args.model, base_url=args.base_url, temperature=0,
        max_output_tokens=8192, max_model_len=args.max_model_len,
        reasoning_effort=args.reasoning_effort,
    )

    def get_model() -> OpenAICompatibleQwenVL:
        if not hasattr(local, "model"):
            local.model = OpenAICompatibleQwenVL(config, timeout_seconds=1200)
        return local.model

    def score(qid: str) -> dict[str, Any]:
        started = time.perf_counter()
        model = get_model()
        if candidate_override:
            if qid not in candidate_override:
                raise KeyError(f"candidate override missing question_id={qid}")
            candidates = list(dict.fromkeys(map(
                str, candidate_override[qid].get("candidate_ids", [])
            )))
            if not 1 <= len(candidates) <= args.candidate_limit:
                raise ValueError(
                    f"candidate override for {qid} has {len(candidates)} unique IDs; "
                    f"expected between 1 and {args.candidate_limit}"
                )
            unknown = [mid for mid in candidates if mid not in memories]
            if unknown:
                raise KeyError(f"candidate override contains unknown memories: {unknown[:3]}")
        else:
            candidates = candidate_pool(
                priors[qid], adjacency, source_top_l=args.source_top_l,
                candidate_limit=args.candidate_limit,
            )
        raw_scores = source_score_map(priors[qid])
        question = resolved[qid]
        query_text = question_text(question)
        captions = caption_question_images(
            model, question, question_native_captions(raw_questions[qid])
        )
        aliases = {memory_id: f"C{index:02d}" for index, memory_id in enumerate(candidates)}
        reverse_aliases = {alias: memory_id for memory_id, alias in aliases.items()}
        payload = {
            "question": query_text,
            "question_image_captions": captions,
            "question_type": (
                str(raw_questions[qid].get("task", {}).get("subcategory", ""))
                if args.question_type_mode == "subcategory" else ""
            ),
            "candidates": [{
                "id": aliases[memory_id],
                "modalities": modality_tags(memories[memory_id]),
                "date": str(memories[memory_id].get("timestamp", ""))[:10],
                "location": memory_location(memories[memory_id])[:120],
                "retrieval_score": round(float(raw_scores.get(memory_id, min(raw_scores.values()) - 0.01)), 4),
                "snippet": memory_snippet(
                    memories[memory_id], args.snippet_chars,
                    query=query_text, ocr_chars=args.ocr_chars,
                ),
            } for memory_id in candidates],
            "instructions": [
                "Score final or necessary supporting evidence for the question.",
                "Use only candidate IDs provided.",
                "Be recall-oriented for list/count/multi-hop tasks.",
                "Omit candidates with score 0.",
                "Do not apply benchmark-specific rules.",
                "Treat question-image captions as noisy visual descriptions, not ground truth.",
                "Return only candidate id and numeric score. Never return reasons or explanations.",
                "Return strict JSON only.",
            ],
            "output_schema": {"selected": [{"id": "candidate id", "score": "0-5 evidence usefulness"}]},
        }
        messages = [
            {"role": "system", "content": "You verify personal-memory evidence candidates. Return strict JSON only."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        last_error: Exception | None = None
        parsed: dict[str, Any] | None = None
        validated: list[tuple[str, float]] | None = None
        retry_messages = messages
        for attempt in range(3):
            try:
                raw = model.complete(retry_messages, response_format={"type": "json_object"})
                parsed = parse_json_object(raw)
                validated = validate_selected_rows(
                    parsed.get("selected"), valid_aliases=set(reverse_aliases)
                )
                break
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                last_error = exc
                retry_messages = messages + [{
                    "role": "user",
                    "content": (
                        f"Correction attempt {attempt + 2}: the previous response failed "
                        f"strict validation ({exc}). Produce a fresh JSON object. Use only "
                        f"these exact candidate IDs: {sorted(reverse_aliases)}. Do not emit "
                        "duplicates, unpadded aliases, unknown IDs, prose, or markdown."
                    ),
                }]
        if parsed is None or validated is None:
            raise RuntimeError(f"verifier returned invalid JSON three times: {last_error}")
        scores: dict[str, float] = {}
        for alias, value in validated:
            if value > 0:
                memory_id = reverse_aliases[alias]
                scores[memory_id] = max(scores.get(memory_id, 0.0), value)
        return {
            "question_id": qid, "context_id": raw_questions[qid]["context_id"],
            "split": (
                "dev" if fold(qid) == args.include_dev_fold
                else ("heldout" if args.all_questions else "heldout100")
            ),
            "candidate_ids": candidates, "atomic_scores": [raw_scores.get(mid) for mid in candidates],
            "provided_anchor_ids": (
                list(map(str, candidate_override[qid].get("anchor_ids", [])))
                if candidate_override else []
            ),
            "verifier_scores": scores, "query_captions": captions,
            "input_tokens": model.last_input_tokens,
            "completion_tokens": model.last_completion_tokens,
            "latency_seconds": time.perf_counter() - started,
            "graph_mode": args.graph_mode, "source_top_l": args.source_top_l,
            "candidate_limit": args.candidate_limit, "semantic_k": args.semantic_k,
            "snippet_chars": args.snippet_chars,
            "ocr_chars": args.ocr_chars,
            "question_type_mode": args.question_type_mode,
            "reasoning_effort": args.reasoning_effort,
            "question_type": payload["question_type"],
            "memory_snippet_protocol": MEMORY_SNIPPET_PROTOCOL,
            "verifier_model": args.model,
            "source_priors_sha256": run_contract["source_priors_sha256"],
            "relation_store_sha256": run_contract["relation_store_sha256"],
            "candidate_override_sha256": run_contract["candidate_override_sha256"],
        }

    todo = [qid for qid in wanted if qid not in completed]
    mode = "a" if args.output.exists() else "w"
    started = time.perf_counter()
    with args.output.open(mode, encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(score, qid): qid for qid in todo}
            done = 0
            errors = 0
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as exc:
                    errors += 1
                    print(json.dumps({
                        "event": "question_error", "question_id": futures[future],
                        "error": repr(exc),
                    }), flush=True)
                    continue
                with lock:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                done += 1
                if done % 25 == 0 or done == len(todo):
                    print(json.dumps({
                        "event": "progress", "done": done, "todo": len(todo),
                        "total": len(wanted), "elapsed_seconds": time.perf_counter() - started,
                    }), flush=True)
    print(json.dumps({
        "event": "complete", "questions": len(wanted), "new": len(todo),
        "dev": len(dev), "heldout100": len(heldout), "errors": errors,
        "output": str(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
