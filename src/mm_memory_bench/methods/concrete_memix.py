from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .backends import AnswerModel, MultiModalEmbedder, OpenAICompatibleQwenVL, data_url
from ..preprocessing.captions import CAPTION_PROMPT, caption_cache_key, load_cached_caption, save_cached_caption
from .base import BaseMemoryMethod, GenerationConfig, MethodCapabilities, MethodResult
from .media import question_text, uniformly_sample_video
from ..preprocessing.text import clean_labeled_ocr_block
from .vector_index import FaissFlatIPIndex, VectorIndex


@lru_cache(maxsize=512)
def _reader_data_url(path: str, max_edge: int = 0) -> str:
    """Encode a reader image, optionally downscaling its longest edge."""
    if max_edge <= 0:
        return data_url(path)
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        if max(source.size) <= max_edge:
            return data_url(path)
        image = ImageOps.exif_transpose(source)
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _load_memix_core(repo: Path):
    root = repo.resolve()
    core_path = root / "scripts" / "QA_Agent" / "MMRAG" / "memix_memory.py"
    if not core_path.is_file():
        raise RuntimeError(f"Memix core is missing: {core_path}")
    module_name = "mmmb_external_memix_memory"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    os.environ.setdefault("ATMBENCH_SUPPRESS_API_KEY_WARNINGS", "1")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    spec = importlib.util.spec_from_file_location(module_name, core_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load Memix core: {core_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _safe_context_dir(root: Path, context_id: str) -> Path:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", context_id).strip("._") or "context"
    suffix = hashlib.sha256(context_id.encode()).hexdigest()[:12]
    return root / "contexts" / f"{clean[:80]}-{suffix}"


def _visible_derived(memory: Mapping[str, Any]) -> dict[str, Any]:
    metadata = memory.get("metadata")
    derived = metadata.get("derived") if isinstance(metadata, Mapping) else None
    return dict(derived) if isinstance(derived, Mapping) else {}


def _without_ocr(value: str) -> str:
    """Remove adapter-added OCR blocks while preserving following metadata fields."""
    return re.sub(
        r"(?ms)^OCR:.*?(?=^(?:Location|City|Tags|Image caption \d+):|\Z)",
        "",
        value,
    ).strip()


def _atomic_text_chunks(value: str, limit: int) -> list[str]:
    """Split record text into stable line/sentence units for multi-vector retrieval."""
    if limit <= 0:
        return []
    chunks: list[str] = []
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        chunks.extend(
            part.strip()
            for part in re.split(r"(?<=[.!?。！？])\s+", line)
            if part.strip()
        )
    return list(dict.fromkeys(chunks))[:limit]


class _VerifierInputClient:
    """Enrich only Memix verifier calls with query/candidate visual context."""

    def __init__(self, delegate: Any, context: Any) -> None:
        self._delegate = delegate
        self._context = context

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    @staticmethod
    def _is_verifier_call(messages: Sequence[Mapping[str, Any]]) -> bool:
        return any(
            "verify personal-memory evidence candidates" in str(message.get("content", ""))
            for message in messages
            if message.get("role") == "system"
        )

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
    ) -> tuple[dict[str, Any], str]:
        context = self._context()
        if not context or not self._is_verifier_call(messages):
            return self._delegate.chat_json(messages, max_tokens=max_tokens)
        user_index = next(
            (index for index in range(len(messages) - 1, -1, -1)
             if messages[index].get("role") == "user"),
            None,
        )
        if user_index is None or not isinstance(messages[user_index].get("content"), str):
            return self._delegate.chat_json(messages, max_tokens=max_tokens)
        try:
            payload = json.loads(str(messages[user_index]["content"]))
        except json.JSONDecodeError:
            return self._delegate.chat_json(messages, max_tokens=max_tokens)
        captions = list(context.get("captions", []))
        payload["question_image_captions"] = captions
        payload.setdefault("instructions", []).append(
            "Use query-image captions to compare subjects, actions, objects, and scene meaning. "
            "Treat captions as noisy visual descriptions, not ground truth."
        )
        enriched = [dict(message) for message in messages]
        if context.get("mode") == "query_caption":
            enriched[user_index] = {
                **enriched[user_index],
                "content": json.dumps(payload, ensure_ascii=False),
            }
        else:
            content: list[dict[str, Any]] = [{
                "type": "text",
                "text": json.dumps(payload, ensure_ascii=False),
            }]
            for index, path in enumerate(context.get("query_image_paths", []), 1):
                content.extend([
                    {"type": "text", "text": f"Query image {index}:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": _reader_data_url(
                            str(path), int(context.get("image_max_edge", 448))
                        )},
                    },
                ])
            records = context.get("records", {})
            for candidate in payload.get("candidates", []):
                record = records.get(str(candidate.get("id", "")))
                if not isinstance(record, Mapping) or not record.get("image_path"):
                    continue
                content.extend([
                    {
                        "type": "text",
                        "text": f"Candidate image for id={candidate.get('id')}:",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _reader_data_url(
                            str(record["image_path"]),
                            int(context.get("image_max_edge", 448)),
                        )},
                    },
                ])
            enriched[user_index] = {**enriched[user_index], "content": content}
        return self._delegate.chat_json(enriched, max_tokens=max_tokens)


class ConcreteMemixMethod(BaseMemoryMethod):
    """Memix dynamic evidence boundary over canonical Main Track records.

    The original dataset scripts consume a precomputed broad source-retrieval
    file. This adapter generates the same query-local source prior from an
    injected multimodal embedder and exact FAISS, then delegates task planning,
    predicate filtering, metadata boosts, and boundary completion to the
    original Memix core without exposing evaluator-only annotations.
    """

    capabilities = MethodCapabilities(
        memory_modalities=frozenset({"text", "image", "video", "table", "document"}),
        query_modalities=frozenset({"text", "image", "video"}),
        supports_tools=True,
        native_multimodal_retrieval=True,
        method_owned_media_processing=True,
    )

    def __init__(
        self,
        generation: GenerationConfig | None = None,
        *,
        answer_model: AnswerModel | None = None,
        embedder: MultiModalEmbedder,
        index: VectorIndex | None = None,
        memix_repo: str | Path = "vendor/memix-core",
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 100,
        ingest_batch_size: int = 16,
        source_top_k: int = 200,
        video_frames: int = 8,
        use_llm_planner: bool = False,
        use_llm_verify: bool = False,
        drop_reader_ocr: bool = False,
        reader_ocr_budget_chars: int = 0,
        media_atomic_limit: int = 0,
        verifier_input_mode: str = "query_caption",
        query_caption_cache_dir: str | Path | None = None,
        verifier_image_max_edge: int = 448,
    ) -> None:
        super().__init__(generation)
        if checkpoint_interval <= 0 or ingest_batch_size <= 0:
            raise ValueError("Memix checkpoint interval and ingest batch size must be positive")
        if source_top_k < self.generation.top_k or video_frames <= 0:
            raise ValueError("Memix source_top_k must cover final top-k and video_frames must be positive")
        if media_atomic_limit < 0:
            raise ValueError("Memix media_atomic_limit must be non-negative")
        if reader_ocr_budget_chars < 0:
            raise ValueError("Memix reader_ocr_budget_chars must be non-negative")
        if verifier_input_mode not in {"current", "query_caption", "full_vl"}:
            raise ValueError("Memix verifier_input_mode must be current, query_caption, or full_vl")
        if verifier_image_max_edge <= 0:
            raise ValueError("Memix verifier_image_max_edge must be positive")
        self.answer_model = answer_model or OpenAICompatibleQwenVL(self.generation)
        self.embedder = embedder
        self.index = index or FaissFlatIPIndex(embedder.dimension)
        self.memix_repo = Path(memix_repo)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_interval = checkpoint_interval
        self.ingest_batch_size = ingest_batch_size
        self.source_top_k = source_top_k
        self.video_frames = video_frames
        self.use_llm_planner = use_llm_planner
        self.use_llm_verify = use_llm_verify
        self.drop_reader_ocr = drop_reader_ocr
        self.reader_ocr_budget_chars = reader_ocr_budget_chars
        self.media_atomic_limit = media_atomic_limit
        self.verifier_input_mode = verifier_input_mode
        self.query_caption_cache_dir = (
            Path(query_caption_cache_dir) if query_caption_cache_dir else None
        )
        self.verifier_image_max_edge = verifier_image_max_edge
        self._verifier_local = threading.local()
        self._query_caption_memory_cache: dict[str, str] = {}
        self.core = _load_memix_core(self.memix_repo)
        local_provider = "vllm" if (use_llm_planner or use_llm_verify) else "none"
        endpoint = self.generation.base_url.rstrip("/") + "/chat/completions"
        local_client = self.core.base.ChatClient(
            provider=local_provider,
            model=self.generation.model,
            endpoint=endpoint if local_provider != "none" else None,
            temperature=0,
            max_tokens=1400,
            timeout=1200,
        )
        self._local_client = _VerifierInputClient(
            local_client,
            lambda: getattr(self._verifier_local, "context", None),
        )
        self._config = self._make_config()
        self._context_id = ""
        self._records: list[dict[str, Any]] = []
        self._record_by_id: dict[str, dict[str, Any]] = {}
        self._embedding_units: list[dict[str, Any]] = []
        self._pending_records: list[dict[str, Any]] = []
        self._pending_embedding_units: list[dict[str, Any]] = []
        self._processed_memory_ids: set[str] = set()
        self._commits_since_checkpoint = 0
        self._memory_index = None
        self._media_failures: list[dict[str, str]] = []

    def _make_config(self):
        # These are the benchmark-neutral defaults of the current Memix core.
        # LLM planning/verification remain optional query-time components; the
        # standard local track uses deterministic fallback planning.
        return self.core.MemixConfig(
            disable_llm_planner=not self.use_llm_planner,
            disable_llm_verify=not self.use_llm_verify,
        )

    def _begin_context(self, context: Mapping[str, Any]) -> None:
        self._context_id = str(context.get("context_id", "context"))
        self.index.reset()
        self._records = []
        self._record_by_id = {}
        self._embedding_units = []
        self._pending_records = []
        self._pending_embedding_units = []
        self._processed_memory_ids = set()
        self._commits_since_checkpoint = 0
        self._memory_index = None
        self._media_failures = []
        self._load_checkpoint()
        self._rebuild_memory_index()

    def _end_context(self) -> None:
        self._flush_pending()
        self._save_checkpoint(force=True)
        self.index.reset()
        self._memory_index = None

    def _abort_context(self) -> None:
        # Completed batches are checkpoint-safe. Do not commit a partially
        # encoded batch while preserving the original failure.
        self._pending_records = []
        self._pending_embedding_units = []
        self.index.reset()
        self._memory_index = None

    def _synchronize_memory(self) -> None:
        self._flush_pending()
        self._save_checkpoint(force=True)

    def _ingest(self, memory: Mapping[str, Any]) -> None:
        memory_id = str(memory["memory_id"])
        if memory_id in self._processed_memory_ids:
            # Canonical public identifiers may be repaired without changing the
            # media/text embedding. Hydrate old checkpoints so answer evidence
            # immediately sees the same source_id as a fresh run.
            source_id = memory.get("source_id")
            record = self._record_by_id.get(memory_id)
            if (
                record is not None
                and source_id is not None
                and record.get("source_id") != source_id
            ):
                record["source_id"] = source_id
                metadata = record.get("metadata")
                if isinstance(metadata, dict):
                    metadata["source_id"] = source_id
                for unit in self._embedding_units:
                    if str(unit.get("memory_id")) == memory_id:
                        unit["source_id"] = source_id
                self._memory_index = None
            return
        record, embedding_units = self._adapt_memory(memory)
        self._pending_records.append(record)
        self._pending_embedding_units.extend(embedding_units)
        if len(self._pending_records) >= self.ingest_batch_size:
            self._flush_pending()

    def _flush_pending(self) -> None:
        if not self._pending_records:
            return
        records = self._pending_records
        units = self._pending_embedding_units
        vectors = self.embedder.encode_units(units)
        if len(vectors) != len(units):
            raise RuntimeError("Memix source embedder returned a different number of rows")
        self.index.add(vectors)
        self._records.extend(records)
        self._record_by_id.update({str(record["item_id"]): record for record in records})
        self._embedding_units.extend(units)
        self._processed_memory_ids.update(str(record["item_id"]) for record in records)
        self._commits_since_checkpoint += len(records)
        self._pending_records = []
        self._pending_embedding_units = []
        self._memory_index = None
        self._save_checkpoint()

    def _canonical_modality(self, memory: Mapping[str, Any]) -> str:
        kinds = {str(part.get("type", "")) for part in memory.get("content", [])}
        kind = str(memory.get("kind", "")).lower()
        if "video" in kinds:
            return "video"
        if "image" in kinds:
            return "image"
        if "email" in kind:
            return "email"
        if memory.get("session_id") or memory.get("role") or "dialogue" in kind:
            return "conversation"
        return "document"

    def _adapt_memory(
        self, memory: Mapping[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        memory_id = str(memory["memory_id"])
        modality = self._canonical_modality(memory)
        derived = _visible_derived(memory)
        text_parts: list[str] = []
        image_paths: list[str] = []
        video_paths: list[str] = []
        for part in memory.get("content", []):
            kind = part.get("type")
            if kind in {"text", "table", "document"} and part.get("text"):
                text_parts.append(str(part["text"]))
            elif kind == "image" and part.get("path"):
                image_paths.append(str(part["path"]))
            elif kind == "video" and part.get("path"):
                video_paths.append(str(part["path"]))
            elif kind == "audio":
                raise NotImplementedError("Memix Main Track does not transcribe audio")
        labels = (
            ("Short Summary", "short_summary"),
            ("Summary", "summary"),
            ("Caption", "caption"),
            ("Short Caption", "short_caption"),
            ("OCR", "ocr_text"),
            ("Location", "location_name"),
            ("City", "city"),
        )
        for label, key in labels:
            value = derived.get(key)
            if value:
                text_parts.append(f"{label}: {value}")
        image_captions = derived.get("image_captions")
        if isinstance(image_captions, list):
            text_parts.extend(
                f"Image caption {index}: {value}"
                for index, value in enumerate(image_captions, 1)
                if value
            )
        tags = derived.get("tags")
        if isinstance(tags, list) and tags:
            text_parts.append("Tags: " + ", ".join(str(value) for value in tags))
        header = " ".join(
            f"{key}={memory[key]}"
            for key in ("source_id", "timestamp", "session_id", "speaker", "round_id")
            if memory.get(key) is not None
        )
        text = "\n".join(value for value in (header, *text_parts) if value).strip()
        if not text:
            text = f"source_id={memory.get('source_id', memory_id)} modality={modality}"
        metadata = {
            "source": modality,
            "source_id": memory.get("source_id"),
            "timestamp": memory.get("timestamp"),
            "session_id": memory.get("session_id"),
            "round": memory.get("round_id"),
            "location": derived.get("location_name") or derived.get("city") or "",
            "scenario": self._context_id,
        }
        record = {
            "item_id": memory_id,
            "modality": modality,
            "text": text,
            "image_path": image_paths[0] if image_paths else None,
            "video_path": video_paths[0] if video_paths else None,
            "metadata": metadata,
            "content": [dict(part) for part in memory.get("content", [])],
            "source_id": memory.get("source_id"),
        }
        units: list[dict[str, Any]] = []
        if image_paths:
            units.extend(
                {"memory_id": memory_id, "text": text, "image": path}
                for path in image_paths
            )
        for path in video_paths:
            try:
                frames = uniformly_sample_video(path, self.video_frames)
            except RuntimeError as exc:
                self._media_failures.append(
                    {"memory_id": memory_id, "path": path, "error": str(exc)}
                )
                continue
            units.extend(
                {
                    "memory_id": memory_id,
                    "text": f"{text}\nvideo_frame={index}",
                    "image": self._frame_path(memory_id, index, frame),
                }
                for index, frame in enumerate(frames)
            )
        if not units:
            units.append({"memory_id": memory_id, "text": text})
        if modality in {"image", "video"}:
            units.extend(
                {
                    "memory_id": memory_id,
                    "text": chunk,
                    "representation": "media_atomic_text",
                }
                for chunk in _atomic_text_chunks(text, self.media_atomic_limit)
            )
        return record, units

    def _frame_path(self, memory_id: str, frame_index: int, frame: str) -> str:
        state_dir = self._state_dir()
        if state_dir is None or not frame.startswith("data:"):
            return frame
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", memory_id).strip("._")
        path = state_dir / "frames" / f"{safe[:120]}_{frame_index:02d}.jpg"
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(base64.b64decode(frame.split(",", 1)[1]))
            os.replace(temporary, path)
        return str(path)

    def _retrieval_items(self) -> list[Any]:
        item_class = self.core.base.RetrievalItem
        return [
            item_class(
                item_id=str(record["item_id"]),
                modality=str(record["modality"]),
                text=self._reader_text(record),
                image_path=Path(record["image_path"]) if record.get("image_path") else None,
                video_path=Path(record["video_path"]) if record.get("video_path") else None,
                metadata=dict(record.get("metadata", {})),
            )
            for record in self._records
        ]

    def _reader_text(self, record: Mapping[str, Any], query: str = "") -> str:
        value = str(record["text"])
        if self.drop_reader_ocr:
            return _without_ocr(value)
        if self.reader_ocr_budget_chars:
            return clean_labeled_ocr_block(
                value, query=query, max_chars=self.reader_ocr_budget_chars
            )
        return value

    def _rebuild_memory_index(self) -> None:
        self._memory_index = (
            self.core.MemoryIndex.from_items(self._retrieval_items(), scope_mode="global")
            if self._records else None
        )

    def _question_units(self, question: Mapping[str, Any]) -> list[dict[str, Any]]:
        text = question_text(question)
        units: list[dict[str, Any]] = []
        for part in question.get("prompt", []):
            if part.get("type") == "image":
                units.append({"text": text, "image": str(part["path"])})
            elif part.get("type") == "video":
                try:
                    frames = uniformly_sample_video(str(part["path"]), self.video_frames)
                except RuntimeError:
                    continue
                units.extend({"text": text, "image": frame} for frame in frames)
        return units or [{"text": text}]

    def _source_detail(self, question: Mapping[str, Any]) -> dict[str, Any]:
        if not self._embedding_units:
            return {"retrieval_ids": [], "retrieval_scores": []}
        queries = self.embedder.encode_units(self._question_units(question))
        # A video contributes up to ``video_frames`` rows. Over-fetch enough
        # rows that deduplication can still produce the requested source-level
        # candidate pool instead of returning 200 frames from far fewer videos.
        per_query_k = min(
            self.source_top_k * max(2, self.video_frames, self.media_atomic_limit),
            len(self._embedding_units),
        )
        scores, rows = self.index.search(queries, per_query_k)
        best: dict[str, float] = {}
        for query_scores, query_rows in zip(scores, rows):
            for score, row in zip(query_scores, query_rows):
                memory_id = str(self._embedding_units[int(row)]["memory_id"])
                best[memory_id] = max(best.get(memory_id, -float("inf")), float(score))
        ranked = sorted(best, key=lambda item_id: (-best[item_id], item_id))[: self.source_top_k]
        return {
            "retrieval_ids": ranked,
            "retrieval_scores": [best[item_id] for item_id in ranked],
        }

    def _answer(self, question: Mapping[str, Any]) -> MethodResult:
        self._flush_pending()
        selected_ids, filter_counts = self._select_evidence(question)
        prediction = self._generate_answer(question, selected_ids)
        return MethodResult(
            prediction=prediction,
            diagnostics=self._diagnostics(selected_ids, filter_counts),
        )

    def _query_image_caption(self, part: Mapping[str, Any]) -> str:
        annotations = part.get("annotations")
        annotations = annotations if isinstance(annotations, Mapping) else {}
        for value in (
            part.get("caption"),
            annotations.get("native_image_caption"),
            annotations.get("caption"),
        ):
            if value:
                return str(value).strip()
        path = str(part.get("path", ""))
        if not path:
            return ""
        key = caption_cache_key(part, video_frames=self.video_frames)
        if key in self._query_caption_memory_cache:
            return self._query_caption_memory_cache[key]
        cache_path = (
            self.query_caption_cache_dir / f"{key}.json"
            if self.query_caption_cache_dir else None
        )
        if cache_path:
            cached = load_cached_caption(cache_path)
            if cached:
                self._query_caption_memory_cache[key] = cached
                return cached
        caption = self.answer_model.complete([{
            "role": "user",
            "content": [
                {"type": "text", "text": CAPTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": _reader_data_url(path)},
                },
            ],
        }]).strip()
        if not caption:
            raise RuntimeError(f"Qwen3-VL returned an empty Memix query caption for {path}")
        self._query_caption_memory_cache[key] = caption
        if cache_path:
            save_cached_caption(cache_path, caption=caption, kind="image")
        return caption

    def _verifier_context(self, question: Mapping[str, Any]) -> dict[str, Any] | None:
        if self.verifier_input_mode == "current":
            return None
        image_parts = [
            part for part in question.get("prompt", [])
            if part.get("type") == "image" and part.get("path")
        ]
        if not image_parts:
            return None
        return {
            "mode": self.verifier_input_mode,
            "captions": [self._query_image_caption(part) for part in image_parts],
            "query_image_paths": [str(part["path"]) for part in image_parts],
            "records": self._record_by_id,
            "image_max_edge": self.verifier_image_max_edge,
        }

    def _select_evidence(
        self,
        question: Mapping[str, Any],
        *,
        source_detail: Mapping[str, Any] | None = None,
    ) -> tuple[list[str], dict[str, Any]]:
        if self._memory_index is None:
            self._rebuild_memory_index()
        if self._memory_index is None:
            selected_ids: list[str] = []
            filter_counts: dict[str, Any] = {}
        else:
            query = question_text(question)
            qa = {
                "id": str(question.get("question_id", "query")),
                "question": query,
                # Do not expose benchmark task labels or gold annotations.
                "qtype": "",
                "point": "",
                "scenario": self._context_id,
                "question_image": any(
                    part.get("type") == "image" for part in question.get("prompt", [])
                ),
            }
            self._verifier_local.context = self._verifier_context(question)
            try:
                result = self.core.retrieve_one(
                    qa=qa,
                    source_detail=(
                        dict(source_detail)
                        if source_detail is not None
                        else self._source_detail(question)
                    ),
                    memory_index=self._memory_index,
                    scope_id="global",
                    client=self._local_client,
                    cache_dir=None,
                    config=self._config,
                    debug_top_n=0,
                    gt_ids=[],
                )
            finally:
                self._verifier_local.context = None
            selected_ids = [
                str(state.item.item_id)
                for state in result["ordered"][: self.generation.top_k]
            ]
            filter_counts = dict(result.get("filter_counts", {}))
            if self.use_llm_verify:
                filter_counts["llm_verify_selected"] = list(
                    result.get("verify_selected", [])
                )
        return selected_ids, filter_counts

    def _diagnostics(
        self, selected_ids: Sequence[str], filter_counts: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "retrieved_memory_ids": list(selected_ids),
            "method": "memix-canonical-source-prior",
            "source_top_k": self.source_top_k,
            "filter_counts": dict(filter_counts),
            "media_failures": len(self._media_failures),
            "drop_reader_ocr": self.drop_reader_ocr,
            "reader_ocr_budget_chars": self.reader_ocr_budget_chars,
            "llm_evidence_verify": self.use_llm_verify,
            "verifier_input_mode": self.verifier_input_mode,
        }

    def answer_many(
        self, questions: Sequence[Mapping[str, Any]], *, concurrency: int = 8
    ) -> list[dict[str, Any]]:
        """Encode queries serially, then parallelize independent LLM stages.

        GME query encoding shares one GPU model and remains ordered.  Once its
        source priors have been materialized, Memix retrieval only reads the
        immutable memory index; its optional HTTP verifier is therefore safe to
        run concurrently, just like the final readers.
        """
        self._require_active("answer_many")
        self._flush_pending()
        workers = max(1, concurrency)
        source_details = [self._source_detail(question) for question in questions]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            selections = list(
                pool.map(
                    lambda values: self._select_evidence(
                        values[0], source_detail=values[1]
                    ),
                    zip(questions, source_details),
                )
            )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            predictions = list(
                pool.map(
                    lambda values: self._generate_answer(*values),
                    (
                        (question, selected_ids)
                        for question, (selected_ids, _) in zip(questions, selections)
                    ),
                )
            )
        return [
            {
                "prediction": prediction,
                **self._diagnostics(selected_ids, filter_counts),
            }
            for prediction, (selected_ids, filter_counts) in zip(predictions, selections)
        ]

    def _generate_answer(
        self,
        question: Mapping[str, Any],
        selected_ids: Sequence[str],
        *,
        image_max_edge: int = 0,
    ) -> str:
        content: list[dict[str, Any]] = []
        for part in question.get("prompt", []):
            if part.get("type") == "image":
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _reader_data_url(str(part["path"]), image_max_edge)
                        },
                    }
                )
            elif part.get("type") == "video":
                content.extend(
                    {"type": "image_url", "image_url": {"url": frame}}
                    for frame in uniformly_sample_video(
                        str(part["path"]), self.video_frames, max_edge=image_max_edge
                    )
                )
        evidence_content: list[dict[str, Any]] = []
        query = question_text(question)
        for rank, memory_id in enumerate(selected_ids, 1):
            record = self._record_by_id[memory_id]
            evidence_content.append(
                {
                    "type": "text",
                    "text": (
                        f"Evidence {rank}; memory_id={memory_id}; "
                        f"source_id={record.get('source_id')}; "
                        f"modality={record.get('modality', 'unknown')}:\n"
                        f"{self._reader_text(record, query=query)}"
                    ),
                }
            )
            for part in record.get("content", []):
                if part.get("type") == "image" and part.get("path"):
                    evidence_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _reader_data_url(
                                    str(part["path"]), image_max_edge
                                )
                            },
                        }
                    )
                elif part.get("type") == "video" and part.get("path"):
                    try:
                        frames = uniformly_sample_video(
                            str(part["path"]), self.video_frames,
                            max_edge=image_max_edge,
                        )
                    except RuntimeError:
                        continue
                    evidence_content.append(
                        {
                            "type": "text",
                            "text": (
                                f"The following {len(frames)} images are sampled frames "
                                f"from the video in Evidence {rank}."
                            ),
                        }
                    )
                    evidence_content.extend(
                        {"type": "image_url", "image_url": {"url": frame}}
                        for frame in frames
                    )
        tools = question.get("tools") if question.get("tool_mode") != "plan" else None
        plan_tools = ""
        if question.get("tools") and question.get("tool_mode") == "plan":
            plan_tools = "\nCandidate tools:\n" + json.dumps(question["tools"], ensure_ascii=False)
        prompt = (
            f"{question.get('instruction', '')}\nQuestion: {question_text(question)}{plan_tools}\n\n"
            "Use only the following Memix evidence packet."
        )
        content.extend([{"type": "text", "text": prompt}, *evidence_content])
        return self.answer_model.complete([{"role": "user", "content": content}], tools=tools)

    def _state_dir(self) -> Path | None:
        return (
            _safe_context_dir(self.checkpoint_dir, self._context_id)
            if self.checkpoint_dir else None
        )

    def _core_hash(self) -> str:
        digest = hashlib.sha256()
        for relative in (
            "scripts/QA_Agent/MMRAG/memix_memory.py",
            "scripts/QA_Agent/MMRAG/iterative_reasoning_retrieval_eval.py",
            "scripts/QA_Agent/MMRAG/task_aware_cascade_retrieval_eval.py",
        ):
            path = self.memix_repo / relative
            digest.update(relative.encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _checkpoint_config(self) -> dict[str, Any]:
        config = {
            "format": "mmmb-memix-checkpoint-2",
            "record_text_schema": "raw+summary+caption+ocr+location-v2",
            "core_sha256": self._core_hash(),
            "embedding_dimension": self.embedder.dimension,
            "video_frames": self.video_frames,
        }
        # Preserve compatibility with existing baseline checkpoints when the
        # optional strategy is disabled.
        if self.media_atomic_limit:
            config["media_atomic_limit"] = self.media_atomic_limit
        return config

    def _save_checkpoint(self, *, force: bool = False) -> None:
        state_dir = self._state_dir()
        if state_dir is None or (
            not force and self._commits_since_checkpoint < self.checkpoint_interval
        ):
            return
        state_dir.mkdir(parents=True, exist_ok=True)
        version = len(self._processed_memory_ids)
        index_name = f"index-{version:08d}.faiss"
        temporary_index = state_dir / f"{index_name}.tmp"
        self.index.save(temporary_index)
        os.replace(temporary_index, state_dir / index_name)
        state = {
            "config": self._checkpoint_config(),
            "context_id": self._context_id,
            "processed_memory_ids": sorted(self._processed_memory_ids),
            "records": self._records,
            "embedding_units": self._embedding_units,
            "media_failures": self._media_failures,
            "index_file": index_name,
        }
        state_path = state_dir / "state.json"
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, state_path)
        for old in state_dir.glob("index-*.faiss"):
            if old.name != index_name:
                old.unlink()
        self._commits_since_checkpoint = 0

    def _load_checkpoint(self) -> None:
        state_dir = self._state_dir()
        state_path = state_dir / "state.json" if state_dir else None
        if state_path is None or not state_path.is_file():
            return
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config") != self._checkpoint_config():
            raise RuntimeError(f"incompatible Memix checkpoint: {state_path}")
        if state.get("context_id") != self._context_id:
            raise RuntimeError(f"Memix checkpoint belongs to another context: {state_path}")
        self._records = [dict(record) for record in state.get("records", [])]
        self._record_by_id = {
            str(record["item_id"]): record for record in self._records
        }
        self._embedding_units = [
            dict(unit) for unit in state.get("embedding_units", [])
        ]
        self._processed_memory_ids = {
            str(value) for value in state.get("processed_memory_ids", [])
        }
        self._media_failures = [
            {str(key): str(value) for key, value in dict(item).items()}
            for item in state.get("media_failures", [])
        ]
        self.index.load(state_dir / str(state["index_file"]))
        if self.index.size != len(self._embedding_units):
            raise RuntimeError("Memix checkpoint index/embedding-unit row mismatch")
        if len(self._records) != len(self._processed_memory_ids):
            raise RuntimeError("Memix checkpoint record/processed-id mismatch")
