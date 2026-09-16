from __future__ import annotations

import json
import hashlib
import os
import time
import warnings
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Condition, Lock, Thread, get_ident
from typing import Any, Mapping, Sequence

import numpy as np

from ..preprocessing.captions import (
    CAPTION_PROMPT,
    caption_cache_key,
    load_cached_caption,
    save_cached_caption,
)
from .amem import AMemMethod
from .backends import (
    AnswerModel,
    OpenAICompatibleQwenVL,
    SentenceTransformerEmbedder,
    TextEmbedder,
    data_url,
)
from .base import GenerationConfig
from .concrete_memguide import public_captions
from .answer_input import build_answer_task
from .media import openai_content_from_parts, question_text, text_from_parts, uniformly_sample_video
from .vector_index import FaissFlatIPIndex, VectorIndex


@dataclass
class AMemoryNote:
    note_id: str
    source_memory_id: str
    content: str
    context: str
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    timestamp: str = ""

    def retrieval_text(self) -> str:
        return (
            f"content: {self.content}\ncontext: {self.context}\n"
            f"keywords: {', '.join(self.keywords)}\ntags: {', '.join(self.tags)}"
        )


class MemoryResponseFormatError(RuntimeError):
    """The model responded, but its structured payload was not valid JSON."""


class ConcreteAMemMethod(AMemMethod):
    """A-Mem note construction/evolution with FAISS exact retrieval."""

    def __init__(
        self,
        generation: GenerationConfig | None = None,
        *,
        answer_model: AnswerModel | None = None,
        memory_model: AnswerModel | None = None,
        caption_model: AnswerModel | None = None,
        caption_cache_dir: str | Path | None = None,
        embedder: TextEmbedder | None = None,
        index: VectorIndex | None = None,
        follow_links: bool = True,
        evolution_neighbors: int = 5,
        video_frames: int = 8,
        memory_workers: int = 1,
        max_pending_memories: int | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 100,
        memory_retry_count: int = 3,
        memory_retry_backoff: float = 5.0,
        max_consecutive_fallbacks: int = 10,
    ) -> None:
        super().__init__(generation)
        self.answer_model = answer_model or OpenAICompatibleQwenVL(self.generation)
        self.memory_model = memory_model or self.answer_model
        self.caption_model = caption_model
        self.caption_cache_dir = Path(caption_cache_dir) if caption_cache_dir else None
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.index = index or FaissFlatIPIndex(self.embedder.dimension)
        self.follow_links = follow_links
        self.evolution_neighbors = evolution_neighbors
        self.video_frames = video_frames
        if memory_workers <= 0:
            raise ValueError("memory_workers must be positive")
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if memory_retry_count <= 0:
            raise ValueError("memory_retry_count must be positive")
        if max_consecutive_fallbacks <= 0:
            raise ValueError("max_consecutive_fallbacks must be positive")
        self.memory_workers = memory_workers
        self.max_pending_memories = max_pending_memories or max(2, memory_workers * 2)
        self.checkpoint_dir = Path(checkpoint_dir).resolve() if checkpoint_dir else None
        self.checkpoint_interval = checkpoint_interval
        self.memory_retry_count = memory_retry_count
        self.memory_retry_backoff = memory_retry_backoff
        self.max_consecutive_fallbacks = max_consecutive_fallbacks
        self.notes: list[AMemoryNote] = []
        self._processed_memory_ids: set[str] = set()
        self._pending: deque[
            tuple[Mapping[str, Any], Future[Sequence[Mapping[str, Any]]]]
        ] = deque()
        self._executor: ThreadPoolExecutor | None = None
        self._commit_condition = Condition()
        self._commit_thread: Thread | None = None
        self._commit_active = False
        self._commit_stop = False
        self._commit_error: BaseException | None = None
        self._commits_since_checkpoint = 0
        self._checkpoint_context_id = ""
        self._memory_fallbacks = 0
        self._consecutive_memory_fallbacks = 0
        self._fallback_lock = Lock()

    def _begin_context(self, context: Mapping[str, Any]) -> None:
        self.notes = []
        self.index.reset()
        self._processed_memory_ids = set()
        self._pending.clear()
        self._commits_since_checkpoint = 0
        self._memory_fallbacks = 0
        self._consecutive_memory_fallbacks = 0
        self._checkpoint_context_id = str(context.get("context_id", "context"))
        self._commit_active = False
        self._commit_stop = False
        self._commit_error = None
        if self.memory_workers > 1:
            self._executor = ThreadPoolExecutor(
                max_workers=self.memory_workers, thread_name_prefix="amem-digest"
            )
        self._load_checkpoint(self._checkpoint_context_id)
        if self._executor is not None:
            self._commit_thread = Thread(
                target=self._commit_loop,
                name="amem-ordered-evolution",
                daemon=True,
            )
            self._commit_thread.start()

    def _end_context(self) -> None:
        try:
            self._flush_pending()
            self._save_checkpoint(force=True)
        finally:
            if self._commit_thread is not None:
                with self._commit_condition:
                    self._commit_stop = True
                    self._commit_condition.notify_all()
                self._commit_thread.join()
                self._commit_thread = None
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None
            self.notes = []
            self.index.reset()
            self._processed_memory_ids = set()

    def _ingest(self, memory: Mapping[str, Any]) -> None:
        memory_id = str(memory["memory_id"])
        if memory_id in self._processed_memory_ids:
            return
        if self._executor is None:
            future: Future[Sequence[Mapping[str, Any]]] = Future()
            try:
                future.set_result(self.memory_to_notes(memory))
            except BaseException as exc:
                future.set_exception(exc)
        else:
            future = self._executor.submit(self.memory_to_notes, memory)
        if self._commit_thread is None:
            self._pending.append((dict(memory), future))
            if len(self._pending) >= self.max_pending_memories:
                self._commit_one()
            return
        with self._commit_condition:
            while (
                len(self._pending) >= self.max_pending_memories
                and self._commit_error is None
            ):
                self._commit_condition.wait()
            self._raise_commit_error()
            self._pending.append((dict(memory), future))
            self._commit_condition.notify_all()

    def _answer(self, question: Mapping[str, Any]):
        self._flush_pending()
        return super()._answer(question)

    def _synchronize_memory(self) -> None:
        self._flush_pending()

    def _commit_one(self) -> None:
        source_memory, future = self._pending.popleft()
        notes = future.result()
        self.add_and_evolve(notes, source_memory=source_memory)
        self._processed_memory_ids.add(str(source_memory["memory_id"]))
        self._commits_since_checkpoint += 1
        self._save_checkpoint()

    def _flush_pending(self) -> None:
        if self._commit_thread is not None:
            with self._commit_condition:
                while (
                    (self._pending or self._commit_active)
                    and self._commit_error is None
                ):
                    self._commit_condition.wait()
                self._raise_commit_error()
            return
        while self._pending:
            self._commit_one()

    def _raise_commit_error(self) -> None:
        if self._commit_error is not None:
            raise RuntimeError("A-Mem ordered evolution worker failed") from self._commit_error

    def _commit_loop(self) -> None:
        while True:
            with self._commit_condition:
                while not self._pending and not self._commit_stop:
                    self._commit_condition.wait()
                if self._commit_stop and not self._pending:
                    return
                source_memory, future = self._pending.popleft()
                self._commit_active = True
                self._commit_condition.notify_all()
            try:
                notes = future.result()
                self.add_and_evolve(notes, source_memory=source_memory)
                self._processed_memory_ids.add(str(source_memory["memory_id"]))
                self._commits_since_checkpoint += 1
                self._save_checkpoint()
            except BaseException as exc:
                with self._commit_condition:
                    self._commit_error = exc
                    self._commit_active = False
                    self._commit_condition.notify_all()
                return
            with self._commit_condition:
                self._commit_active = False
                self._commit_condition.notify_all()

    def _checkpoint_path(self) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        digest = hashlib.sha256(self._checkpoint_context_id.encode("utf-8")).hexdigest()[:20]
        return self.checkpoint_dir / "contexts" / f"{digest}.json"

    def _legacy_checkpoint_path(self) -> Path | None:
        return self.checkpoint_dir / "state.json" if self.checkpoint_dir else None

    def _checkpoint_config(self) -> dict[str, Any]:
        backend_config = getattr(self.memory_model, "config", None)
        config = {
            "format": "mmmb-amem-checkpoint-1",
            "embedding_dimension": self.embedder.dimension,
            "evolution_neighbors": self.evolution_neighbors,
            "follow_links": self.follow_links,
            "memory_model": getattr(backend_config, "model", None),
            "memory_base_url": getattr(backend_config, "base_url", None),
            "memory_max_output_tokens": getattr(backend_config, "max_output_tokens", None),
            "note_prompt_version": 2,
        }
        if self.caption_model is not None:
            caption_config = getattr(self.caption_model, "config", None)
            config.update({
                "caption_model": getattr(caption_config, "model", None),
                "caption_base_url": getattr(caption_config, "base_url", None),
                "caption_prompt_version": 1,
            })
        return config

    def _caption_key(self, part: Mapping[str, Any]) -> str:
        return caption_cache_key(part, video_frames=self.video_frames)

    def _caption(self, part: Mapping[str, Any]) -> str:
        if self.caption_model is None:
            raise RuntimeError("A-Mem caption fallback requires a caption model")
        key = self._caption_key(part)
        cache_path = self.caption_cache_dir / f"{key}.json" if self.caption_cache_dir else None
        if cache_path:
            cached = load_cached_caption(cache_path)
            if cached:
                return cached
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": CAPTION_PROMPT,
        }]
        kind, path = str(part["type"]), str(part["path"])
        if kind == "image":
            content.append({"type": "image_url", "image_url": {"url": data_url(path)}})
        else:
            content.extend(
                {"type": "image_url", "image_url": {"url": frame}}
                for frame in uniformly_sample_video(path, self.video_frames)
            )
        caption = self.caption_model.complete([{"role": "user", "content": content}]).strip()
        if not caption:
            raise RuntimeError(f"Qwen3-VL returned an empty A-Mem caption for {path}")
        if cache_path:
            save_cached_caption(cache_path, caption=caption, kind=kind)
        return caption

    def _load_checkpoint(self, context_id: str) -> None:
        path = self._checkpoint_path()
        if path is None:
            return
        if not path.is_file():
            # ATM-Bench checkpoints created before context isolation remain readable.
            legacy = self._legacy_checkpoint_path()
            if legacy is None or not legacy.is_file():
                return
            legacy_state = json.loads(legacy.read_text(encoding="utf-8"))
            if legacy_state.get("context_id") != context_id:
                return
            path = legacy
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("config") != self._checkpoint_config():
            raise RuntimeError(f"incompatible A-Mem checkpoint: {path}")
        if state.get("context_id") != context_id:
            raise RuntimeError(f"A-Mem checkpoint belongs to another context: {path}")
        self.notes = [AMemoryNote(**row) for row in state.get("notes", [])]
        self._processed_memory_ids = {
            str(value) for value in state.get("processed_memory_ids", [])
        }
        self._memory_fallbacks = int(state.get("memory_fallbacks", 0))
        if self.notes:
            vectors = self.embedder.encode_texts([note.retrieval_text() for note in self.notes])
            self.index.add(vectors)

    def _save_checkpoint(self, *, force: bool = False) -> None:
        path = self._checkpoint_path()
        if path is None or (not force and self._commits_since_checkpoint < self.checkpoint_interval):
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "config": self._checkpoint_config(),
            "context_id": self._checkpoint_context_id,
            "processed_memory_ids": sorted(self._processed_memory_ids),
            "memory_fallbacks": self._memory_fallbacks,
            "notes": [vars(note) for note in self.notes],
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
        self._commits_since_checkpoint = 0

    @staticmethod
    def _json_object(value: str) -> Mapping[str, Any]:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end < start:
            raise MemoryResponseFormatError(f"model did not return JSON: {value[:200]}")
        try:
            parsed = json.loads(value[start : end + 1])
        except json.JSONDecodeError as exc:
            raise MemoryResponseFormatError(
                f"model returned malformed JSON: {value[:200]}"
            ) from exc
        if not isinstance(parsed, Mapping):
            raise MemoryResponseFormatError("model JSON response is not an object")
        return parsed

    @staticmethod
    def _json_schema(properties: Mapping[str, Any], required: Sequence[str]) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": dict(properties),
                    "required": list(required),
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

    def _memory_json(self, messages, *, response_format) -> Mapping[str, Any]:
        last_error: BaseException | None = None
        for attempt in range(self.memory_retry_count):
            try:
                parsed = self._json_object(
                    self.memory_model.complete(
                        messages, response_format=response_format
                    )
                )
                self._record_structured_success()
                return parsed
            except (RuntimeError, OSError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < self.memory_retry_count:
                    time.sleep(self.memory_retry_backoff * (attempt + 1))
        assert last_error is not None
        raise last_error

    def _record_format_fallback(self, operation: str, item_id: str, exc: BaseException) -> None:
        with self._fallback_lock:
            self._memory_fallbacks += 1
            self._consecutive_memory_fallbacks += 1
            consecutive = self._consecutive_memory_fallbacks
        warnings.warn(f"A-Mem {operation} fallback for {item_id}: {exc}", RuntimeWarning)
        if consecutive >= self.max_consecutive_fallbacks:
            raise RuntimeError(
                f"A-Mem circuit breaker: {consecutive} consecutive malformed model responses"
            ) from exc

    def _record_structured_success(self) -> None:
        with self._fallback_lock:
            self._consecutive_memory_fallbacks = 0

    def memory_to_notes(self, memory: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        parts = list(memory.get("content", []))
        raw_text = text_from_parts(parts)
        has_visual_media = any(
            part.get("type") in {"image", "video"} for part in parts
        )
        # Ordinary dialogue accompanying an image is not an image description.
        # Only an explicit benchmark caption suppresses method-owned visual
        # description generation.  In the formal ``derived`` view an available
        # benchmark caption has already replaced the raw media, so this also
        # naturally avoids sending those assets to the digest VLM.
        needs_description = has_visual_media and not public_captions(memory)
        cached_descriptions: list[str] = []
        if needs_description and self.caption_model is not None:
            cached_descriptions = [
                self._caption(part)
                for part in parts
                if part.get("type") in {"image", "video"}
            ]
        inline_visual_description = needs_description and not cached_descriptions
        public_metadata = memory.get("metadata", {})
        identity = {
            key: memory[key]
            for key in ("memory_id", "source_id", "session_id", "speaker", "timestamp")
            if memory.get(key) is not None
        }
        if cached_descriptions:
            visible = [{
                "type": "text",
                "text": "\n".join(
                    value for value in (raw_text, *cached_descriptions) if value
                ),
            }]
        else:
            visible = openai_content_from_parts(parts, video_frames=self.video_frames)
        visible.append(
            {
                "type": "text",
                "text": (
                    "Create one A-Mem note as JSON with keys keywords (string array), "
                    "context (exactly one concise sentence, at most 40 words), and tags "
                    "(string array). Return 3 to 8 short keywords and 3 to 8 short tags; "
                    "never copy the full input into context. "
                    + ("Also include content as a faithful visual description. " if inline_visual_description else "")
                    + "Preserve every supplied identifier and "
                    "public metadata value verbatim. Do not invent details.\n"
                    f"Identifiers: {json.dumps(identity, ensure_ascii=False)}\n"
                    f"Public metadata: {json.dumps(public_metadata, ensure_ascii=False)}"
                ),
            }
        )
        properties: dict[str, Any] = {
            "keywords": {
                "type": "array",
                "items": {"type": "string", "maxLength": 64},
                "minItems": 3,
                "maxItems": 8,
            },
            "context": {"type": "string", "maxLength": 256},
            "tags": {
                "type": "array",
                "items": {"type": "string", "maxLength": 64},
                "minItems": 3,
                "maxItems": 8,
            },
        }
        required = ["keywords", "context", "tags"]
        if inline_visual_description:
            properties["content"] = {"type": "string"}
            required.append("content")
        try:
            analysis = self._memory_json(
                [{"role": "user", "content": visible}],
                response_format=self._json_schema(properties, required),
            )
        except MemoryResponseFormatError as exc:
            # Match the upstream ATM A-Mem behavior: retain the source content
            # and use an unstructured note instead of aborting the full index.
            self._record_format_fallback("note", str(memory["memory_id"]), exc)
            analysis = {"keywords": [], "context": "General", "tags": []}
        identity_text = json.dumps(identity, ensure_ascii=False)
        metadata_text = json.dumps(public_metadata, ensure_ascii=False)
        generated_description = "\n".join(cached_descriptions).strip()
        if inline_visual_description:
            generated_description = str(analysis.get("content", "")).strip()
        generated = "\n".join(
            value
            for value in (
                raw_text,
                (
                    f"Visual description: {generated_description}"
                    if needs_description and generated_description
                    else ""
                ),
            )
            if value
        )
        content = f"Identifiers: {identity_text}\nPublic metadata: {metadata_text}\n{generated}"
        if not content:
            raise RuntimeError(f"A-Mem note for {memory['memory_id']} has empty content")
        return [
            {
                "note_id": str(memory["memory_id"]),
                "source_memory_id": str(memory["memory_id"]),
                "content": content,
                "context": str(analysis.get("context", "General")),
                "keywords": [str(v) for v in analysis.get("keywords", [])],
                "tags": [str(v) for v in analysis.get("tags", [])],
                "timestamp": str(memory.get("timestamp", "")),
            }
        ]

    def add_and_evolve(self, notes, *, source_memory) -> None:
        for raw in notes:
            note = AMemoryNote(**dict(raw))
            if self.notes:
                query = self.embedder.encode_texts([note.content])
                _, ids = self.index.search(query, self.evolution_neighbors)
                neighbors = [self.notes[int(i)] for i in ids[0]]
                prompt = {
                    "role": "user",
                    "content": (
                        "Decide A-Mem evolution. Return JSON: should_evolve boolean, "
                        "suggested_connections array of note_id strings, tags_to_update "
                        "array, and neighbor_updates array of objects with note_id, context, tags.\n"
                        "Every context must be one sentence of at most 40 words; every tags array "
                        "must contain at most 8 short tags. Do not copy full memory text into a field.\n"
                        f"New note: {note.retrieval_text()}\n"
                        "Neighbors:\n" + "\n".join(n.retrieval_text() for n in neighbors)
                    ),
                }
                try:
                    decision = self._memory_json(
                        [prompt],
                        response_format=self._json_schema(
                        {
                                "should_evolve": {"type": "boolean"},
                                "suggested_connections": {
                                    "type": "array",
                                    "items": {"type": "string", "maxLength": 256},
                                    "maxItems": 5,
                                },
                                "tags_to_update": {
                                    "type": "array",
                                    "items": {"type": "string", "maxLength": 64},
                                    "maxItems": 8,
                                },
                                "neighbor_updates": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "note_id": {"type": "string", "maxLength": 256},
                                            "context": {"type": "string", "maxLength": 256},
                                            "tags": {
                                                "type": "array",
                                                "items": {"type": "string", "maxLength": 64},
                                                "maxItems": 8,
                                            },
                                        },
                                        "required": ["note_id", "context", "tags"],
                                        "additionalProperties": False,
                                    },
                                    "maxItems": 5,
                                },
                        },
                        [
                            "should_evolve",
                            "suggested_connections",
                            "tags_to_update",
                            "neighbor_updates",
                        ],
                        ),
                    )
                except MemoryResponseFormatError as exc:
                    # Upstream treats an unparsable evolution decision as no-op.
                    self._record_format_fallback("evolution", note.note_id, exc)
                    decision = {"should_evolve": False}
                if decision.get("should_evolve"):
                    valid = {n.note_id for n in neighbors}
                    note.links = [
                        str(v) for v in decision.get("suggested_connections", []) if str(v) in valid
                    ]
                    if decision.get("tags_to_update"):
                        note.tags = [str(v) for v in decision["tags_to_update"]]
                    by_id = {n.note_id: n for n in neighbors}
                    for update in decision.get("neighbor_updates", []):
                        if not isinstance(update, Mapping) or str(update.get("note_id")) not in by_id:
                            continue
                        target = by_id[str(update["note_id"])]
                        target.context = str(update.get("context", target.context))
                        if update.get("tags"):
                            target.tags = [str(v) for v in update["tags"]]
                        target_row = self.notes.index(target)
                        target_vector = self.embedder.encode_texts([target.retrieval_text()])
                        self.index.update(target_row, target_vector)
            self.notes.append(note)
            vector = self.embedder.encode_texts([note.retrieval_text()])
            self.index.add(vector)

    def search_notes(self, question, *, top_k):
        if not self.notes:
            return []
        query = self.embedder.encode_texts([question_text(question)])
        scores, ids = self.index.search(query, top_k)
        selected = [self.notes[int(i)] for i in ids[0]]
        if self.follow_links:
            by_id = {note.note_id: note for note in self.notes}
            seen = {note.note_id for note in selected}
            for note in list(selected):
                for linked in note.links:
                    if linked in by_id and linked not in seen:
                        selected.append(by_id[linked])
                        seen.add(linked)
        return [
            {
                "memory_id": note.source_memory_id,
                "content": note.content,
                "context": note.context,
                "timestamp": note.timestamp,
            }
            for note in selected
        ]

    def generate_answer(self, question, notes):
        evidence = "\n\n".join(json.dumps(note, ensure_ascii=False) for note in notes)
        task = build_answer_task(question)
        messages = task.messages(
            f"{task.text}\nEvidence:\n{evidence}",
            default_system="Answer only from retrieved memory evidence.",
        )
        return self.answer_model.complete(messages, tools=task.api_tools)
