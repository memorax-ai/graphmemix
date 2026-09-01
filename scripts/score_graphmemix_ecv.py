#!/usr/bin/env python3
"""Score direct node support and anchor-conditioned explicit-edge support."""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from mm_memory_bench.methods.media import question_text

from graphmemix_core import (
    MEMORY_SNIPPET_PROTOCOL,
    file_sha256,
    memory_location,
    memory_snippet,
    read_jsonl,
)
from mm_memory_bench.methods.backends import estimate_multimodal_input_tokens


POSITIVE_ROLES = {"new_fact", "clarification", "corroboration"}
ALL_ROLES = POSITIVE_ROLES | {"redundant", "conflict", "irrelevant"}
ECV_PARSER_PROTOCOL = "graphmemix-ecv-structured-normalize-2"


def direct_support_value(row: Mapping[str, Any], *, edge_only: bool) -> float:
    """Enforce the edge-only contract in code, not merely in the prompt."""
    if edge_only:
        return 0.0
    try:
        return max(0.0, min(5.0, float(row.get("direct_support", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def validate_candidate_rows(
    value: Any,
    *,
    expected_aliases: set[str],
    eligible_anchors: Mapping[str, set[str]],
) -> list[Mapping[str, Any]]:
    """Reject incomplete, duplicate, or schema-invalid structured ECV rows."""
    if not isinstance(value, list):
        raise ValueError("ECV candidates must be a list")
    rows: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    ignored_unknown_rows = 0
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError("ECV candidate row must be an object")
        alias = str(row.get("id", ""))
        if alias not in expected_aliases:
            ignored_unknown_rows += 1
            continue
        if alias in seen:
            raise ValueError(f"ECV returned duplicate candidate id: {alias!r}")
        seen.add(alias)
        role = str(row.get("role", "")).lower()
        invalid_role = role not in ALL_ROLES
        if invalid_role:
            role = "irrelevant"
        try:
            direct = float(row.get("direct_support", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ECV returned invalid direct support for {alias}") from exc
        if not 0.0 <= direct <= 5.0:
            raise ValueError(f"ECV direct support out of range for {alias}")
        anchor = row.get("best_anchor_id")
        ineligible_anchor = (
            anchor is not None
            and str(anchor) not in eligible_anchors.get(alias, set())
        )
        try:
            incremental = float(row.get("incremental_support", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ECV returned invalid incremental support for {alias}") from exc
        if not 0.0 <= incremental <= 5.0:
            raise ValueError(f"ECV incremental support out of range for {alias}")
        normalized = dict(row)
        normalizations: list[str] = []
        if invalid_role:
            normalized["role"] = "irrelevant"
            normalizations.append("invalid_role_replaced_with_irrelevant")
        if ineligible_anchor:
            normalized["best_anchor_id"] = None
            normalized["incremental_support"] = 0.0
            normalized["role"] = "irrelevant"
            normalizations.append("ineligible_anchor_cleared_edge_fields")
        elif anchor is None and (incremental != 0.0 or role != "irrelevant"):
            normalized["incremental_support"] = 0.0
            normalized["role"] = "irrelevant"
            normalizations.append("null_anchor_cleared_edge_fields")
        elif role in POSITIVE_ROLES and incremental <= 0.0:
            normalized["role"] = "irrelevant"
            normalizations.append("zero_incremental_cleared_positive_role")
        elif role not in POSITIVE_ROLES and incremental > 0.0:
            normalized["incremental_support"] = 0.0
            normalizations.append("nonpositive_role_cleared_incremental")
        if normalizations:
            normalized["_normalizations"] = normalizations
        rows.append(normalized)
    if seen != expected_aliases:
        missing = sorted(expected_aliases - seen)
        raise ValueError(f"ECV omitted {len(missing)} candidate rows: {missing[:3]}")
    if ignored_unknown_rows and rows:
        first = dict(rows[0])
        first["_normalizations"] = [
            *map(str, first.get("_normalizations", [])),
            *(["ignored_unknown_candidate_row"] * ignored_unknown_rows),
        ]
        rows[0] = first
    return rows


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


def modality_tags(memory: Mapping[str, Any]) -> list[str]:
    values = {str(memory.get("kind", "memory"))}
    values.update(str(part.get("type")) for part in memory.get("content", []) if part.get("type"))
    return sorted(values)


def visible_question_text(question: Mapping[str, Any]) -> str:
    return question_text(question)


class ChatClient:
    def __init__(
        self, base_url: str, model: str, max_model_len: int,
        max_output_tokens: int, reasoning_effort: str,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.max_model_len = max_model_len
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.last_input_tokens: int | None = None
        self.last_completion_tokens: int | None = None
        self.last_finish_reason: str | None = None

    def complete(self, messages: list[dict[str, str]]) -> str:
        estimated = estimate_multimodal_input_tokens(messages)
        if estimated >= self.max_model_len:
            raise RuntimeError(
                f"ECV input context estimate {estimated} exceeds hard limit "
                f"{self.max_model_len}"
            )
        output_budget = min(self.max_output_tokens, self.max_model_len - estimated)
        request_payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if "gpt-5" in self.model.lower():
            request_payload["max_completion_tokens"] = output_budget
            request_payload["reasoning_effort"] = self.reasoning_effort
        else:
            request_payload["temperature"] = 0
            request_payload["max_tokens"] = output_budget
            if "deepseek-v4" in self.model.lower() and self.reasoning_effort == "none":
                request_payload["thinking"] = {"type": "disabled"}
        payload = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=payload, headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }, method="POST",
        )
        with urllib.request.urlopen(request, timeout=1200) as response:
            value = json.loads(response.read().decode("utf-8"))
        usage = value.get("usage", {})
        self.last_input_tokens = usage.get("prompt_tokens") or estimated
        self.last_completion_tokens = usage.get("completion_tokens")
        self.last_finish_reason = value["choices"][0].get("finish_reason")
        return str(value["choices"][0]["message"]["content"])


def explicit_relations_from_store(path: Path) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            relations = [str(value) for value in row.get("explicit_relations", []) if value]
            if not relations:
                continue
            left, right = str(row["left"]), str(row["right"])
            result.setdefault(left, {})[right] = relations
            result.setdefault(right, {})[left] = relations
    return result


def explicit_relations_from_memories(
    memories: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    edges: dict[tuple[str, str], list[str]] = {}

    def add(left: str, right: str, relation: str) -> None:
        if left == right:
            return
        values = edges.setdefault(tuple(sorted((left, right))), [])
        if relation not in values:
            values.append(relation)

    by_round: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_session: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in memories.values():
        context = str(row["context_id"])
        mid = str(row["memory_id"])
        session = str(row.get("session_id") or "")
        round_id = str(row.get("metadata", {}).get("native_round_id") or "")
        if round_id:
            by_round[(context, round_id)].append(mid)
        if session:
            by_session[(context, session)].append(row)
    for values in by_round.values():
        for index, left in enumerate(values):
            for right in values[index + 1:]:
                add(left, right, "same_round")
    for values in by_session.values():
        ordered = sorted(values, key=lambda row: (int(row.get("sequence", 0)), str(row["memory_id"])))
        for left, right in zip(ordered, ordered[1:]):
            add(str(left["memory_id"]), str(right["memory_id"]), "consecutive_turn")
    result: dict[str, dict[str, list[str]]] = {}
    for (left, right), values in edges.items():
        result.setdefault(left, {})[right] = values
        result.setdefault(right, {})[left] = values
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--relation-store", type=Path)
    parser.add_argument(
        "--anchor-mode", choices=("atomic", "verifier", "reward", "provided"), default="atomic",
        help="Rank anchors by frozen Atomic order, node-verifier score, or calibrated node reward.",
    )
    parser.add_argument("--source-priors", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8096/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--anchor-count", type=int, default=10)
    parser.add_argument("--snippet-chars", type=int, default=420)
    parser.add_argument("--ocr-chars", type=int, default=120)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument(
        "--reasoning-effort", default="minimal",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
        help="Reasoning effort sent to hosted GPT-5-compatible models.",
    )
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=0,
        help="Maximum candidates per verifier request; 0 keeps all candidates in one request.",
    )
    parser.add_argument("--question-id", action="append", help="Run only the specified question ID; repeatable.")
    parser.add_argument("--edge-only", action="store_true")
    parser.add_argument(
        "--question-type-mode",
        choices=("empty", "subcategory"),
        default="empty",
        help="Expose no type or question.task.subcategory to ECV.",
    )
    parser.add_argument("--split", choices=("all", "dev", "heldout", "heldout100"), default="all")
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    if (
        args.anchor_count <= 0
        or args.snippet_chars <= 0
        or args.ocr_chars < 0
        or args.concurrency <= 0
        or args.candidate_batch_size < 0
    ):
        parser.error("anchor-count, snippet-chars and concurrency must be positive; ocr-chars and candidate-batch-size must be non-negative")
    if args.anchor_mode == "reward" and (
        args.source_priors is None or args.calibration is None
    ):
        parser.error("--anchor-mode reward requires --source-priors and --calibration")

    raw_questions = read_jsonl(args.bundle / "questions.jsonl", "question_id")
    memories = read_jsonl(args.bundle / "memories.jsonl", "memory_id")
    cached = read_jsonl(args.candidate_cache, "question_id")
    incompatible_candidates = [
        qid for qid, row in cached.items()
        if row.get("memory_snippet_protocol") != MEMORY_SNIPPET_PROTOCOL
    ]
    if incompatible_candidates:
        raise RuntimeError(
            f"{args.candidate_cache} contains {len(incompatible_candidates)} rows "
            "from an incompatible memory snippet protocol; regenerate node scores"
        )
    priors = (
        read_jsonl(args.source_priors, "question_id")
        if args.source_priors is not None else {}
    )
    coefficients: dict[str, float] = {}
    if args.calibration is not None:
        calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
        coefficients = {
            key: float(calibration["coefficients"][key])
            for key in ("a", "b", "c")
        }
    relations = (
        explicit_relations_from_store(args.relation_store)
        if args.relation_store is not None
        else explicit_relations_from_memories(memories)
    )
    run_contract = {
        "memory_snippet_protocol": MEMORY_SNIPPET_PROTOCOL,
        "ecv_parser_protocol": ECV_PARSER_PROTOCOL,
        "ecv_model": args.model,
        "anchor_count": args.anchor_count,
        "snippet_chars": args.snippet_chars,
        "ocr_chars": args.ocr_chars,
        "max_model_len": args.max_model_len,
        "max_output_tokens": args.max_output_tokens,
        "reasoning_effort": args.reasoning_effort,
        "candidate_batch_size": args.candidate_batch_size,
        "edge_only": args.edge_only,
        "anchor_mode": args.anchor_mode,
        "question_type_mode": args.question_type_mode,
        "candidate_cache_sha256": file_sha256(args.candidate_cache),
        "relation_source_sha256": file_sha256(
            args.relation_store
            if args.relation_store is not None
            else args.bundle / "memories.jsonl"
        ),
        "source_priors_sha256": (
            file_sha256(args.source_priors) if args.source_priors is not None else None
        ),
        "calibration_sha256": (
            file_sha256(args.calibration) if args.calibration is not None else None
        ),
    }
    wanted = [
        qid for qid in sorted(cached)
        if qid in raw_questions and (args.split == "all" or cached[qid].get("split") == args.split)
    ]
    if args.question_id:
        allowed = set(map(str, args.question_id))
        wanted = [qid for qid in wanted if qid in allowed]
    if args.max_questions is not None:
        if args.max_questions <= 0:
            parser.error("max-questions must be positive")
        wanted = wanted[:args.max_questions]

    completed = read_jsonl(args.output, "question_id") if args.output.is_file() else {}
    incompatible_completed = [
        qid for qid, row in completed.items()
        if any(row.get(key) != value for key, value in run_contract.items())
    ]
    if incompatible_completed:
        raise RuntimeError(
            f"{args.output} contains {len(incompatible_completed)} rows from an "
            "incompatible ECV run contract; use a new output path"
        )
    todo = [qid for qid in wanted if qid not in completed]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    local = threading.local()
    lock = threading.Lock()
    def get_model() -> ChatClient:
        if not hasattr(local, "model"):
            local.model = ChatClient(
                args.base_url, args.model, args.max_model_len,
                args.max_output_tokens, args.reasoning_effort,
            )
        return local.model

    def score(qid: str) -> dict[str, Any]:
        started = time.perf_counter()
        model = get_model()
        source = cached[qid]
        query_text = visible_question_text(raw_questions[qid])
        candidate_ids = [str(value) for value in source["candidate_ids"]]
        candidate_rank = {mid: index for index, mid in enumerate(candidate_ids)}
        verifier_scores = {
            str(mid): float(value)
            for mid, value in source.get("verifier_scores", {}).items()
        }
        if args.anchor_mode == "provided":
            supplied = list(dict.fromkeys(map(
                str, source.get("provided_anchor_ids", [])
            )))
            unknown = [mid for mid in supplied if mid not in candidate_rank]
            if unknown:
                raise ValueError(
                    f"provided anchors for {qid} are outside the candidate pool: {unknown[:3]}"
                )
            if not supplied:
                raise ValueError(f"provided anchor mode has no anchors for {qid}")
            anchor_order = supplied
        elif args.anchor_mode == "atomic":
            anchor_order = candidate_ids
        elif args.anchor_mode == "verifier":
            anchor_order = sorted(
                candidate_ids,
                key=lambda mid: (-verifier_scores.get(mid, 0.0), candidate_rank[mid]),
            )
        else:
            prior = priors[qid]
            source_scores = dict(zip(
                map(str, prior.get("retrieval_ids", [])),
                map(float, prior.get("retrieval_scores", [])),
            ))
            floor = min(source_scores.values()) - 0.01
            a, b, c = (coefficients[key] for key in ("a", "b", "c"))

            def reward(mid: str) -> float:
                logit = (
                    a * source_scores.get(mid, floor)
                    + b * verifier_scores.get(mid, 0.0)
                    + c
                )
                return 1.0 / (1.0 + math.exp(-logit))

            anchor_order = sorted(
                candidate_ids,
                key=lambda mid: (-reward(mid), candidate_rank[mid]),
            )
        anchors = anchor_order[:min(args.anchor_count, len(anchor_order))]
        anchor_set = set(anchors)
        aliases = {mid: f"C{index:02d}" for index, mid in enumerate(candidate_ids)}
        reverse = {alias: mid for mid, alias in aliases.items()}
        anchor_aliases = {aliases[mid] for mid in anchors}
        candidates: list[dict[str, Any]] = []
        for mid in candidate_ids:
            options = []
            for anchor in anchors:
                if anchor == mid or anchor not in relations.get(mid, {}):
                    continue
                options.append({
                    "anchor_id": aliases[anchor],
                    "relation_types": relations[mid][anchor],
                })
            candidates.append({
                "id": aliases[mid],
                "is_atomic_anchor": mid in anchor_set,
                "modalities": modality_tags(memories[mid]),
                "date": str(memories[mid].get("timestamp", ""))[:10],
                "location": memory_location(memories[mid])[:120],
                "snippet": memory_snippet(
                    memories[mid], args.snippet_chars,
                    query=query_text, ocr_chars=args.ocr_chars,
                ),
                "eligible_anchor_relations": options,
            })
        edge_candidates = [row for row in candidates if row["eligible_anchor_relations"]]
        expected_edge_ids = {reverse[str(row["id"])] for row in edge_candidates}
        if args.edge_only and not edge_candidates:
            return {
                "question_id": qid, "context_id": source["context_id"], "split": source.get("split"),
                "candidate_ids": candidate_ids, "atomic_scores": source["atomic_scores"],
                "direct_scores": {}, "edge_scores": {}, "anchor_ids": anchors,
                "query_captions": source.get("query_captions", []), "returned_candidates": 0,
                "eligible_edge_candidates": 0, "input_tokens": 0,
                "latency_seconds": time.perf_counter() - started,
                "anchor_count": args.anchor_count, "snippet_chars": args.snippet_chars,
                "ocr_chars": args.ocr_chars,
                "edge_only": True, "anchor_mode": args.anchor_mode,
                "memory_snippet_protocol": MEMORY_SNIPPET_PROTOCOL,
                **{
                    key: value
                    for key, value in run_contract.items()
                    if key != "memory_snippet_protocol"
                },
            }
        rows_to_score = edge_candidates if args.edge_only else candidates
        batch_size = args.candidate_batch_size or len(rows_to_score)
        batches = [
            rows_to_score[index:index + batch_size]
            for index in range(0, len(rows_to_score), batch_size)
        ]
        parsed_rows: list[Any] = []
        total_input_tokens = 0
        total_completion_tokens = 0
        finish_reasons: list[str] = []
        for batch in batches:
            expected_batch_aliases = {str(row["id"]) for row in batch}
            eligible_batch_anchors = {
                str(row["id"]): {
                    str(option["anchor_id"])
                    for option in row.get("eligible_anchor_relations", [])
                }
                for row in batch
            }
            payload = {
                "question": query_text,
                "question_image_captions": source.get("query_captions", []),
                "question_type": (
                    str(raw_questions[qid].get("task", {}).get("subcategory", ""))
                    if args.question_type_mode == "subcategory" else ""
                ),
                "task": (
                    "Assess conditional evidence-chain value for schema-linked candidates."
                    if args.edge_only else "Assess direct evidence and conditional evidence-chain value."
                ),
                "definitions": {
                    "direct_support": "0-5 support for answering the question from this candidate itself.",
                    "incremental_support": "0-5 NEW answer-relevant information added by this candidate when the chosen anchor is already known; relevance or adjacency alone is not enough.",
                    "best_anchor_id": "One ID from eligible_anchor_relations, or null.",
                    "role": "new_fact, clarification, corroboration, redundant, conflict, or irrelevant.",
                },
                "anchor_evidence": [
                    {
                        "id": aliases[mid],
                        "date": str(memories[mid].get("timestamp", ""))[:10],
                        "location": memory_location(memories[mid])[:120],
                        "snippet": memory_snippet(
                            memories[mid], args.snippet_chars,
                            query=query_text, ocr_chars=args.ocr_chars,
                        ),
                    }
                    for mid in anchors
                ] if args.edge_only else [],
                "anchor_selection": args.anchor_mode,
                "candidates": batch,
                "instructions": [
                    "Return one row for every listed candidate and use only provided IDs.",
                    (
                        "Set direct_support to 0; this edge-only experiment reuses the locked original node verifier."
                        if args.edge_only else "Score direct_support independently of graph relations."
                    ),
                    "Choose an anchor only from that candidate's eligible_anchor_relations.",
                    "Use new_fact/clarification/corroboration only when the candidate adds useful information beyond the anchor.",
                    "Use redundant when it merely repeats the anchor and conflict for incompatible or stale evidence.",
                    "Question-image captions are noisy descriptions, not ground truth.",
                    "Return strict JSON only without explanations.",
                ],
                "output_schema": {
                    "candidates": [{
                        "id": "candidate id", "direct_support": "0-5",
                        "best_anchor_id": "eligible anchor id or null",
                        "incremental_support": "0-5", "role": "one allowed role",
                    }]
                },
            }
            messages = [
                {"role": "system", "content": "You are a conservative personal-memory evidence-chain verifier. Return strict JSON only."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
            parsed: dict[str, Any] | None = None
            validated_rows: list[Mapping[str, Any]] | None = None
            last_error: Exception | None = None
            retry_messages = messages
            for attempt in range(3):
                try:
                    parsed = parse_json_object(model.complete(retry_messages))
                    validated_rows = validate_candidate_rows(
                        parsed.get("candidates"),
                        expected_aliases=expected_batch_aliases,
                        eligible_anchors=eligible_batch_anchors,
                    )
                    break
                except (json.JSONDecodeError, ValueError, TypeError) as exc:
                    last_error = exc
                    retry_messages = messages + [{
                        "role": "user",
                        "content": (
                            f"Correction attempt {attempt + 2}: the previous response failed "
                            f"strict validation ({exc}). Produce a fresh JSON object with "
                            "exactly one row for every candidate ID in this list and no other "
                            f"IDs: {sorted(expected_batch_aliases)}. Do not omit or duplicate "
                            "rows, and do not include prose or markdown."
                        ),
                    }]
            if parsed is None or validated_rows is None:
                raise RuntimeError(f"ECV returned invalid JSON three times: {last_error}")
            parsed_rows.extend(validated_rows)
            total_input_tokens += int(model.last_input_tokens or 0)
            total_completion_tokens += int(model.last_completion_tokens or 0)
            finish_reasons.append(str(model.last_finish_reason))

        direct: dict[str, float] = {}
        edge_scores: dict[str, dict[str, Any]] = {}
        returned: set[str] = set()
        normalization_counts: Counter[str] = Counter()
        for row in parsed_rows:
            if not isinstance(row, Mapping):
                continue
            alias = str(row.get("id", ""))
            if alias not in reverse:
                continue
            mid = reverse[alias]
            normalization_counts.update(map(str, row.get("_normalizations", [])))
            if args.edge_only and mid not in expected_edge_ids:
                continue
            returned.add(mid)
            direct[mid] = direct_support_value(row, edge_only=args.edge_only)
            anchor_alias = row.get("best_anchor_id")
            role = str(row.get("role", "irrelevant")).lower()
            if role not in ALL_ROLES or not anchor_alias or str(anchor_alias) not in anchor_aliases:
                continue
            anchor = reverse[str(anchor_alias)]
            if anchor not in relations.get(mid, {}):
                continue
            try:
                incremental = max(0.0, min(5.0, float(row.get("incremental_support", 0.0))))
            except (TypeError, ValueError):
                continue
            key = "\t".join(sorted((mid, anchor)))
            current = edge_scores.get(key)
            if current is None or incremental > float(current["incremental_support"]):
                edge_scores[key] = {
                    "left": min(mid, anchor), "right": max(mid, anchor),
                    "incremental_support": incremental, "role": role,
                    "relation_types": relations[mid][anchor],
                }
        if args.edge_only and returned != expected_edge_ids:
            missing = len(expected_edge_ids - returned)
            raise RuntimeError(
                f"edge-only ECV row coverage mismatch: expected={len(expected_edge_ids)} "
                f"returned={len(returned)} missing={missing}"
            )
        for mid in candidate_ids:
            direct.setdefault(mid, 0.0)
        return {
            "question_id": qid, "context_id": source["context_id"], "split": source.get("split"),
            "candidate_ids": candidate_ids, "atomic_scores": source["atomic_scores"],
            "direct_scores": direct, "edge_scores": edge_scores,
            "anchor_ids": anchors, "query_captions": source.get("query_captions", []),
            "returned_candidates": len(returned), "input_tokens": total_input_tokens,
            "completion_tokens": total_completion_tokens,
            "finish_reason": ",".join(finish_reasons),
            "latency_seconds": time.perf_counter() - started,
            "anchor_count": args.anchor_count, "snippet_chars": args.snippet_chars,
            "ocr_chars": args.ocr_chars,
            "eligible_edge_candidates": len(edge_candidates), "edge_only": args.edge_only,
            "anchor_mode": args.anchor_mode,
            "normalization_counts": dict(sorted(normalization_counts.items())),
            "memory_snippet_protocol": MEMORY_SNIPPET_PROTOCOL,
            **{
                key: value
                for key, value in run_contract.items()
                if key != "memory_snippet_protocol"
            },
        }

    mode = "a" if args.output.exists() else "w"
    errors = 0
    started = time.perf_counter()
    with args.output.open(mode, encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(score, qid): qid for qid in todo}
            done = 0
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as exc:
                    errors += 1
                    print(json.dumps({"event": "question_error", "question_id": futures[future], "error": repr(exc)}), flush=True)
                    continue
                with lock:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                done += 1
                if done % 25 == 0 or done == len(todo):
                    print(json.dumps({"event": "progress", "done": done, "todo": len(todo), "elapsed_seconds": time.perf_counter() - started}), flush=True)
    print(json.dumps({"event": "complete", "questions": len(wanted), "new": len(todo), "errors": errors, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
