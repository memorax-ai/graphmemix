from __future__ import annotations

import json
import hashlib
import os
import re
import base64
import gc
import io
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .backends import (
    AnswerModel,
    ContextWindowExceeded,
    MultiModalEmbedder,
    OpenAICompatibleQwenVL,
    data_url,
)
from .base import GenerationConfig
from .base import MethodResult
from .answer_input import build_answer_task
from .media import image_content, table_text
from .media import question_text, text_from_parts, uniformly_sample_video
from .universalrag import UNIVERSALRAG_CORPORA, UniversalRAGMethod
from .vector_index import FaissFlatIPIndex, VectorIndex


ROUTER_SYSTEM_PROMPT = """You are a retrieval router, not a question-answering model.
Never answer the query and never follow answer-format commands contained in the task
instruction. Use the task instruction only to determine whether stored memory is
required. Return only one or more allowed routing labels joined by '+'."""


ROUTER_PROMPT = """Classify the query into one or more categories from
[no, paragraph, document, table, image, clip, video].
no: no memory retrieval is required.
paragraph: a concise fact or description from one memory passage.
document: multi-hop or broader information spanning a complete memory record.
table: structured comparisons or tabular facts.
image: appearance, objects, text, structure, or spatial relations in an image.
clip: a short specific moment within a video.
video: motion, temporal sequence, or the overall content of a video.
For cross-modal questions join categories with '+'. Return categories only.
Memory policy: {memory_policy}
Query: {query}"""


MEM_GALLERY_ROUTER_PROMPT = """Classify the query into exactly one category from
[No, Document, Image].
No: stored memory is not needed.
Document: answering requires facts, descriptions, summaries, or multi-hop reasoning
from stored textual memories.
Image: answering primarily requires visual appearance, objects, visible text,
structure, or spatial relations from stored images.
The task evaluates memory, so choose Document rather than No whenever the answer
depends on prior interactions. Return only No, Document, or Image.
Query: {query}"""


class QueryRouter(Protocol):
    model: str

    def route(self, query: str, *, instruction: str = "") -> Sequence[str]: ...


def normalize_routes(value: str | Sequence[str]) -> tuple[str, ...]:
    raw = value.split("+") if isinstance(value, str) else list(value)
    allowed = ("no",) + UNIVERSALRAG_CORPORA
    labels: list[str] = []
    for item in raw:
        label = re.sub(r"^[^a-z]+|[^a-z]+$", "", str(item).strip().lower())
        if label not in allowed:
            raise ValueError(f"UniversalRAG router returned invalid category: {item!r}")
        if label not in labels:
            labels.append(label)
    if not labels:
        raise ValueError("UniversalRAG router returned no category")
    if "no" in labels and len(labels) > 1:
        labels.remove("no")
    return tuple(label for label in allowed if label in labels)


class OpenAIUniversalRouter:
    """Training-free router using the paper's category semantics."""

    def __init__(self, model: AnswerModel, *, model_name: str) -> None:
        self.answer_model = model
        self.model = model_name

    def route(self, query: str, *, instruction: str = "") -> Sequence[str]:
        # Never copy benchmark answer-format instructions into the router. ATM
        # prompts such as "if evidence is insufficient, answer Unknown" can be
        # mistaken for a routing answer. Preserve only the controlled fact that
        # this task explicitly requires stored memory.
        memory_policy = (
            "The task explicitly requires retrieval from stored memory."
            if instruction.strip()
            else "No separate memory requirement was supplied."
        )
        response = self.answer_model.complete(
            [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": ROUTER_PROMPT.format(
                    memory_policy=memory_policy,
                    query=json.dumps(query, ensure_ascii=False),
                )},
            ]
        )
        return normalize_routes(response)


class MemGalleryUniversalRouter:
    """Deterministic version of Mem-Gallery's no/document/image LLM router."""

    def __init__(self, model: AnswerModel, *, model_name: str) -> None:
        self.answer_model = model
        self.model = f"{model_name}:mem-gallery-3way"

    def route(self, query: str, *, instruction: str = "") -> Sequence[str]:
        response = self.answer_model.complete(
            [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": MEM_GALLERY_ROUTER_PROMPT.format(
                        query=json.dumps(query, ensure_ascii=False)
                    ),
                },
            ]
        ).strip().lower()
        # Mem-Gallery maps unsupported labels into its closest three-way class.
        # Use a deterministic document fallback instead of its random error path
        # so repeated benchmark runs remain reproducible.
        if "image" in response or "visual" in response or "video" in response or "clip" in response:
            return ("image",)
        if "document" in response or "text" in response or "paragraph" in response or "table" in response:
            return ("document",)
        if re.fullmatch(r"[^a-z]*(no|none)[^a-z]*", response):
            return ("no",)
        return ("document",)


class OfficialQwen3TextEmbedder:
    """Adapter for UniversalRAG's Qwen3-Embedding text corpora."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B") -> None:
        from sentence_transformers import SentenceTransformer

        model_kwargs: dict[str, Any] = {"dtype": "auto", "device_map": "auto"}
        attention = os.environ.get("UNIVERSALRAG_ATTN_IMPLEMENTATION", "").strip()
        if attention:
            model_kwargs["attn_implementation"] = attention
        self.model = SentenceTransformer(
            model_name,
            model_kwargs=model_kwargs,
            tokenizer_kwargs={"padding_side": "left"},
        )
        self._dimension = int(self.model.get_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_units(self, units: Sequence[Mapping[str, Any]]) -> np.ndarray:
        texts = [str(unit.get("text", "")).strip() or " " for unit in units]
        return np.asarray(
            self.model.encode(texts, batch_size=32, show_progress_bar=False),
            dtype=np.float32,
        )

    def encode_query_units(self, units: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Match the official query preprocessing (`prompt_name='query'`)."""
        texts = [str(unit.get("text", "")).strip() or " " for unit in units]
        return np.asarray(
            self.model.encode(
                texts, batch_size=32, show_progress_bar=False, prompt_name="query"
            ),
            dtype=np.float32,
        )


class OfficialVLM2VecEmbedder:
    """Adapter for UniversalRAG's VLM2Vec-V2 image/clip/video corpora."""

    def __init__(
        self,
        model_name: str = "VLM2Vec/VLM2Vec-V2.0",
        *,
        official_repo: str | Path = "sources/universalrag",
    ) -> None:
        import sys

        source = str((Path(official_repo) / "src").resolve())
        if source not in sys.path:
            sys.path.insert(0, source)
        from universalrag.embedding.vlm2vecv2 import VLM2VecV2EmbeddingModel

        self.model = VLM2VecV2EmbeddingModel(model_name=model_name, device="cuda")
        self._dimension = int(self.model.get_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_units(self, units: Sequence[Mapping[str, Any]]) -> np.ndarray:
        rows: list[np.ndarray | None] = [None] * len(units)
        groups = {
            "video": [i for i, unit in enumerate(units) if unit.get("video")],
            "image": [i for i, unit in enumerate(units) if unit.get("image") and not unit.get("video")],
            "text": [i for i, unit in enumerate(units) if not unit.get("image") and not unit.get("video")],
        }
        encoders = {
            "video": lambda ids: self._encode_videos_resilient(units, ids),
            "image": lambda ids: self._encode_images_resilient(units, ids),
            "text": lambda ids: self.model.encode_text(
                [str(units[i].get("text", "")).strip() or " " for i in ids]
            ),
        }
        for kind, ids in groups.items():
            if not ids:
                continue
            values = np.asarray(encoders[kind](ids), dtype=np.float32)
            for index, value in zip(ids, values):
                rows[index] = value
        if any(row is None for row in rows):
            raise RuntimeError("VLM2Vec failed to encode every UniversalRAG unit")
        return np.asarray(rows, dtype=np.float32)

    def _encode_images_resilient(
        self,
        units: Sequence[Mapping[str, Any]],
        ids: Sequence[int],
    ) -> np.ndarray:
        """Encode one image at a time and close decoded pixels deterministically."""
        rows: list[np.ndarray] = []
        for index in ids:
            image = _universal_image(str(units[index]["image"]))
            try:
                value = np.asarray(
                    self.model.encode_image([image], batch_size=1), dtype=np.float32
                )
                rows.append(value[0])
            finally:
                image.close()
        return np.asarray(rows, dtype=np.float32)

    def _encode_videos_resilient(
        self,
        units: Sequence[Mapping[str, Any]],
        ids: Sequence[int],
    ) -> np.ndarray:
        import torch

        rows: list[np.ndarray] = []
        for index in ids:
            unit = units[index]
            path = str(unit["video"])
            last_error: str | None = None
            try:
                rows.append(
                    np.asarray(self.model.encode_video([path]), dtype=np.float32)[0]
                )
            except (MemoryError, torch.cuda.OutOfMemoryError):
                # Resource exhaustion is not a corrupt-video condition and must
                # remain visible instead of silently changing representations.
                raise
            except Exception as exc:
                # Store text rather than the exception object: its traceback can
                # retain decoded video tensors and poison subsequent batches.
                last_error = f"{type(exc).__name__}: {exc}"
            finally:
                gc.collect()
                torch.cuda.empty_cache()
            if last_error is None:
                continue
            frames = []
            try:
                frames = [_universal_image(frame) for frame in uniformly_sample_video(path, 8)]
                frame_rows = np.asarray(
                    self.model.encode_image(frames, batch_size=1), dtype=np.float32
                )
                value = frame_rows.mean(axis=0)
                norm = float(np.linalg.norm(value))
                rows.append(value / norm if norm else value)
                warnings.warn(
                    f"UniversalRAG video encoder used sampled-frame fallback for {path}: {last_error}",
                    RuntimeWarning,
                )
            except (MemoryError, torch.cuda.OutOfMemoryError):
                raise
            except Exception as frame_error:
                rows.append(
                    np.asarray(
                        self.model.encode_text([str(unit.get("text", "")).strip() or path]),
                        dtype=np.float32,
                    )[0]
                )
                warnings.warn(
                    "UniversalRAG video encoder used text fallback after repeated decode "
                    f"failures for {path}: {last_error}; {frame_error}",
                    RuntimeWarning,
                )
            finally:
                for frame in frames:
                    frame.close()
                gc.collect()
                torch.cuda.empty_cache()
        return np.asarray(rows, dtype=np.float32)


def _universal_image(value: str):
    from PIL import Image

    if value.startswith("data:"):
        return Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[1]))).convert("RGB")
    with Image.open(value) as image:
        return image.convert("RGB")


class ConcreteUniversalRAGMethod(UniversalRAGMethod):
    """Benchmark adapter for modality/granularity-routed UniversalRAG.

    Each corpus has its own embedder and FAISS index. This is intentionally not
    implemented as one shared multimodal index.
    """

    def __init__(
        self,
        generation: GenerationConfig | None = None,
        *,
        answer_model: AnswerModel | None = None,
        router: QueryRouter,
        corpus_embedders: Mapping[str, MultiModalEmbedder],
        indexes: Mapping[str, VectorIndex] | None = None,
        ingest_batch_size: int = 16,
        video_frames: int = 8,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 100,
        adapter_profile: str = "official",
    ) -> None:
        super().__init__(generation)
        if ingest_batch_size <= 0 or checkpoint_interval <= 0:
            raise ValueError("batch size and checkpoint interval must be positive")
        missing = set(UNIVERSALRAG_CORPORA) - set(corpus_embedders)
        if missing:
            raise ValueError(f"missing UniversalRAG corpus embedders: {sorted(missing)}")
        if adapter_profile not in {"official", "mem-gallery"}:
            raise ValueError(f"unsupported UniversalRAG adapter profile: {adapter_profile}")
        self.answer_model = answer_model or OpenAICompatibleQwenVL(self.generation)
        self.router = router
        self.embedders = dict(corpus_embedders)
        self.indexes = dict(indexes or {})
        for corpus, embedder in self.embedders.items():
            if corpus not in self.indexes:
                self.indexes[corpus] = FaissFlatIPIndex(embedder.dimension)
        self.ingest_batch_size = ingest_batch_size
        self.video_frames = video_frames
        self.checkpoint_dir = Path(checkpoint_dir).resolve() if checkpoint_dir else None
        self.checkpoint_interval = checkpoint_interval
        self.adapter_profile = adapter_profile
        self.units: dict[str, list[dict[str, Any]]] = {name: [] for name in UNIVERSALRAG_CORPORA}
        self._pending_memories: list[dict[str, Any]] = []
        self._processed_memory_ids: set[str] = set()
        self._context_id = ""
        self._commits_since_checkpoint = 0

    def _begin_context(self, context: Mapping[str, Any]) -> None:
        self.units = {name: [] for name in UNIVERSALRAG_CORPORA}
        for index in self.indexes.values():
            index.reset()
        self._pending_memories = []
        self._processed_memory_ids = set()
        self._context_id = str(context.get("context_id", "context"))
        self._commits_since_checkpoint = 0
        self._load_checkpoint()

    def _end_context(self) -> None:
        self._flush_pending()
        self._save_checkpoint(force=True)
        self._clear_context_state()

    def _abort_context(self) -> None:
        # Never retry a failed pending multimodal batch while unwinding an
        # exception. Checkpoints contain only fully committed transactions.
        if not self._pending_memories:
            # Query-time failures happen after synchronize_memory. Persist the
            # final committed tail so a router/generator failure does not force
            # those memories to be embedded again on resume.
            self._save_checkpoint(force=True)
        self._pending_memories = []
        self._clear_context_state()

    def _clear_context_state(self) -> None:
        self.units = {name: [] for name in UNIVERSALRAG_CORPORA}
        for index in self.indexes.values():
            index.reset()

    def index_memory(self, memory: Mapping[str, Any]) -> None:
        memory_id = str(memory["memory_id"])
        if memory_id in self._processed_memory_ids:
            return
        self._pending_memories.append(dict(memory))
        if len(self._pending_memories) >= self.ingest_batch_size:
            self._flush_pending()

    def _memory_units(self, memory: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
        memory_id = str(memory["memory_id"])
        parts = list(memory.get("content", []))
        prefix = " ".join(
            f"{key}={memory[key]}"
            for key in ("source_id", "session_id", "speaker", "timestamp")
            if memory.get(key) is not None
        )
        public_metadata = {
            key: value
            for key, value in dict(memory.get("metadata", {})).items()
            if key != "derived"
        }
        if public_metadata:
            prefix = "\n".join(
                value
                for value in (
                    prefix,
                    "public_metadata=" + json.dumps(public_metadata, ensure_ascii=False),
                )
                if value
            )
        base = {"memory_id": memory_id, "source_id": memory.get("source_id")}
        derived = memory.get("metadata", {}).get("derived", {})
        media_auxiliary = self._media_auxiliary_text(derived)
        result = {name: [] for name in UNIVERSALRAG_CORPORA}
        paragraphs = [
            str(part.get("text", "")).strip()
            for part in parts
            if part.get("type") in {"text", "document"} and part.get("text")
        ]
        for text in paragraphs:
            result["paragraph"].append(
                {**base, "text": "\n".join(value for value in (prefix, text) if value)}
            )
        document_text = "\n".join(value for value in (prefix, "\n".join(paragraphs)) if value)
        if self.adapter_profile == "mem-gallery":
            return self._mem_gallery_memory_units(
                memory_id=memory_id,
                source_id=memory.get("source_id"),
                parts=parts,
                document_text=document_text,
                media_auxiliary=media_auxiliary,
            )
        # Paragraph/document are text corpora. Media-only records must not create
        # source-id/timestamp placeholder rows in either text index.
        if paragraphs and document_text:
            result["document"].append({**base, "text": document_text})
        for part in parts:
            kind = part.get("type")
            structured_text = table_text(part)
            if structured_text or kind == "table":
                text = structured_text or str(part.get("path") or "")
                if text:
                    result["table"].append({**base, "text": "\n".join(v for v in (prefix, text) if v)})
            elif kind == "image":
                result["image"].append({
                    **base,
                    "text": document_text,
                    "media_auxiliary": media_auxiliary,
                    "image": str(part["path"]),
                    "media_source_id": part.get("source_id"),
                })
            elif kind == "video":
                video = str(part["path"])
                result["video"].append({
                    **base,
                    "text": document_text,
                    "media_auxiliary": media_auxiliary,
                    "video": video,
                })
                for frame_index, frame in enumerate(self._video_clip_frames(video)):
                    result["clip"].append(
                        {
                            **base,
                            "text": document_text,
                            "media_auxiliary": media_auxiliary,
                            "image": self._clip_frame_path(memory_id, frame_index, frame),
                            "clip_index": frame_index,
                        }
                    )
        return result

    def _mem_gallery_memory_units(
        self,
        *,
        memory_id: str,
        source_id: Any,
        parts: Sequence[Mapping[str, Any]],
        document_text: str,
        media_auxiliary: str,
    ) -> dict[str, list[dict[str, Any]]]:
        """Build Mem-Gallery's shared-GME document/image storage view."""
        result = {name: [] for name in UNIVERSALRAG_CORPORA}
        rendered_text = "\n".join(
            value for value in (document_text, media_auxiliary) if value
        ).strip()
        base = {
            "memory_id": memory_id,
            "source_id": source_id,
            "text": rendered_text,
            "media_auxiliary": media_auxiliary,
        }
        # Mem-Gallery stores every observation in the text/document index,
        # including image observations whose text happens to be empty.
        result["document"].append({**base, "embedding_text": rendered_text or " "})
        for part in parts:
            kind = part.get("type")
            if kind == "image":
                # Its image index uses pure image embeddings, not fused text+image.
                result["image"].append(
                    {**base, "embedding_text": "", "image": str(part["path"])}
                )
            elif kind == "video":
                # Mem-Gallery folds clip/video routes into image. Preserve that
                # routing contract while keeping frame extraction method-owned.
                for frame_index, frame in enumerate(
                    self._video_clip_frames(str(part["path"]))
                ):
                    result["image"].append(
                        {
                            **base,
                            "embedding_text": "",
                            "image": self._clip_frame_path(memory_id, frame_index, frame),
                            "clip_index": frame_index,
                        }
                    )
        return result

    @staticmethod
    def _media_auxiliary_text(derived: Any) -> str:
        if not isinstance(derived, Mapping):
            return ""
        values: list[str] = []
        for label, key in (
            ("Caption", "caption"),
            ("Short caption", "short_caption"),
            ("OCR", "ocr_text"),
        ):
            if derived.get(key):
                values.append(f"{label}: {derived[key]}")
        image_captions = derived.get("image_captions")
        if isinstance(image_captions, list):
            values.extend(
                f"Image caption {index}: {value}"
                for index, value in enumerate(image_captions, 1)
                if value
            )
        return "\n".join(values)

    def _video_clip_frames(self, video: str) -> list[str]:
        last_error: str | None = None
        for attempt in range(3):
            try:
                return uniformly_sample_video(video, self.video_frames)
            except MemoryError:
                raise
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                gc.collect()
                time.sleep(2 * (attempt + 1))
        warnings.warn(
            f"UniversalRAG omitted clip proxies after repeated decode failure for {video}: {last_error}",
            RuntimeWarning,
        )
        return []

    def _clip_frame_path(self, memory_id: str, frame_index: int, frame: str) -> str:
        if self.checkpoint_dir is None:
            return frame
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", memory_id)
        path = self.checkpoint_dir / "frames" / f"{safe}_{frame_index:02d}.jpg"
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(base64.b64decode(frame.split(",", 1)[1]))
            os.replace(temporary, path)
        return str(path)

    def _flush_pending(self) -> None:
        if not self._pending_memories:
            return
        grouped = {name: [] for name in UNIVERSALRAG_CORPORA}
        for memory in self._pending_memories:
            built = self._memory_units(memory)
            for corpus in UNIVERSALRAG_CORPORA:
                grouped[corpus].extend(built[corpus])
        # Encode every corpus before mutating any index. This makes one memory
        # batch transactional: a video/image failure cannot leave text indexes
        # partially advanced and then duplicate them on checkpoint resume.
        encoded: dict[str, np.ndarray] = {}
        for corpus, values in grouped.items():
            if not values:
                continue
            embeddings = self.embedders[corpus].encode_units(values)
            if len(embeddings) != len(values):
                raise RuntimeError(f"{corpus} embedder returned a different number of rows")
            encoded[corpus] = np.asarray(embeddings, dtype=np.float32)
        for corpus, embeddings in encoded.items():
            values = grouped[corpus]
            self.indexes[corpus].add(embeddings)
            self.units[corpus].extend(values)
        committed = {str(memory["memory_id"]) for memory in self._pending_memories}
        self._processed_memory_ids.update(committed)
        self._commits_since_checkpoint += len(committed)
        self._pending_memories = []
        self._save_checkpoint()

    def route(self, question: Mapping[str, Any]) -> Sequence[str]:
        instruction = str(question.get("instruction", "")).strip()
        values = self.router.route(question_text(question), instruction=instruction)
        return self._profile_routes(values)

    def _profile_routes(self, values: str | Sequence[str]) -> tuple[str, ...]:
        routes = normalize_routes(values)
        if self.adapter_profile != "mem-gallery":
            return routes
        mapped = []
        for route in routes:
            value = {
                "paragraph": "document",
                "table": "document",
                "clip": "image",
                "video": "image",
            }.get(route, route)
            if value not in mapped:
                mapped.append(value)
        if "no" in mapped and len(mapped) > 1:
            mapped.remove("no")
        return tuple(mapped or ["document"])

    def _synchronize_memory(self) -> None:
        self._flush_pending()

    def _query_unit(self, question: Mapping[str, Any], corpus: str) -> dict[str, Any]:
        text = question_text(question)
        if corpus in {"image", "clip", "video"}:
            for part in question.get("prompt", []):
                if part.get("type") == "image":
                    value = {"text": text, "image": str(part["path"])}
                    if self.adapter_profile == "mem-gallery":
                        value["embedding_text"] = ""
                    return value
                if part.get("type") == "video":
                    return {"text": text, "video": str(part["path"])}
        return {"text": text}

    def _encode_queries(
        self, corpus: str, units: Sequence[Mapping[str, Any]]
    ) -> np.ndarray:
        embedder = self.embedders[corpus]
        encode = getattr(embedder, "encode_query_units", None)
        return np.asarray(
            encode(units) if callable(encode) else embedder.encode_units(units),
            dtype=np.float32,
        )

    def retrieve_routed(self, question, *, routes, top_k):
        self._flush_pending()
        per_corpus: list[list[dict[str, Any]]] = []
        # The official implementation retrieves top-k independently per routed
        # corpus; scores across distinct embedding spaces are not comparable.
        # Interleave ranks across corpora, then enforce the harness-wide global
        # top-k contract. This preserves corpus coverage without comparing scores.
        routed = list(self._profile_routes(routes))
        for corpus in routed:
            if corpus == "no" or not self.units[corpus]:
                continue
            query = self._encode_queries(corpus, [self._query_unit(question, corpus)])
            scores, ids = self.indexes[corpus].search(query, top_k)
            per_corpus.append([
                {**self.units[corpus][int(row_id)], "corpus": corpus, "score": float(score)}
                for score, row_id in zip(scores[0], ids[0])
            ])
        evidence = self._merge_corpus_candidates(per_corpus, top_k)
        if (
            self.adapter_profile == "mem-gallery"
            and not evidence
            and "image" in routed
            and self.units["document"]
        ):
            query = self._encode_queries("document", [self._query_unit(question, "document")])
            scores, ids = self.indexes["document"].search(query, top_k)
            evidence = [
                {
                    **self.units["document"][int(row_id)],
                    "corpus": "document",
                    "score": float(score),
                }
                for score, row_id in zip(scores[0], ids[0])
            ]
        return evidence

    @staticmethod
    def _merge_corpus_candidates(
        per_corpus: Sequence[Sequence[Mapping[str, Any]]], top_k: int
    ) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        seen_units: set[tuple[str, str]] = set()
        cursors = [0] * len(per_corpus)
        while len(evidence) < top_k:
            added = False
            for corpus_index, candidates in enumerate(per_corpus):
                candidate = None
                while cursors[corpus_index] < len(candidates):
                    value = candidates[cursors[corpus_index]]
                    cursors[corpus_index] += 1
                    memory_id = str(value.get("memory_id", ""))
                    identity = (memory_id, str(value.get("image") or value.get("video") or "text"))
                    if not memory_id or identity not in seen_units:
                        candidate = value
                        break
                if candidate is None:
                    continue
                memory_id = str(candidate.get("memory_id", ""))
                evidence.append(candidate)
                added = True
                if memory_id:
                    seen_units.add((memory_id, str(candidate.get("image") or candidate.get("video") or "text")))
                if len(evidence) == top_k:
                    return evidence
            if not added:
                break
        return evidence

    def answer_many(
        self, questions: Sequence[Mapping[str, Any]], *, concurrency: int = 8
    ) -> list[dict[str, str]]:
        """Batch UniversalRAG routing/embedding/retrieval and concurrent generation."""
        self._require_active("answer_many")
        self._flush_pending()
        if not questions:
            return []
        workers = max(1, concurrency)
        def route_one(question):
            try:
                return self.route(question), None
            except ValueError as exc:
                # Match the official UniversalRAG evaluator's treatment of an
                # out-of-vocabulary router label: it is a query-level routing
                # error, not a substitute corpus choice and not a run-level
                # failure. The placeholder `no` route only keeps later batch
                # bookkeeping well-formed; generation is skipped below.
                return ("no",), {
                    "prediction": "",
                    "status": "error",
                    "error_type": "invalid_router_output",
                    "error": str(exc),
                }

        with ThreadPoolExecutor(max_workers=workers) as pool:
            routed = list(pool.map(route_one, questions))
        routes = [value[0] for value in routed]
        route_errors = [value[1] for value in routed]

        candidates: list[dict[str, list[dict[str, Any]]]] = [dict() for _ in questions]
        for corpus in UNIVERSALRAG_CORPORA:
            indices = [
                index
                for index, values in enumerate(routes)
                if corpus in normalize_routes(values) and self.units[corpus]
            ]
            if not indices:
                continue
            query_units = [self._query_unit(questions[index], corpus) for index in indices]
            query_vectors = self._encode_queries(corpus, query_units)
            scores, rows = self.indexes[corpus].search(query_vectors, self.generation.top_k)
            for batch_row, question_index in enumerate(indices):
                candidates[question_index][corpus] = [
                    {
                        **self.units[corpus][int(row_id)],
                        "corpus": corpus,
                        "score": float(score),
                    }
                    for score, row_id in zip(scores[batch_row], rows[batch_row])
                ]

        evidence = [
            self._merge_corpus_candidates(
                [candidates[index][corpus] for corpus in normalize_routes(routes[index])
                 if corpus != "no" and corpus in candidates[index]],
                self.generation.top_k,
            )
            for index in range(len(questions))
        ]
        def generate_one(values):
            question, selected, route_error = values
            if route_error is not None:
                return route_error
            try:
                return {"prediction": self.generate_answer(question, selected)}
            except ContextWindowExceeded as exc:
                # A single oversized multimodal query must remain a visible
                # benchmark failure without cancelling the other queries in
                # the concurrent batch. Do not truncate evidence or silently
                # reduce top-k: both would change the configured method.
                return {
                    "prediction": "",
                    "status": "error",
                    "error_type": "context_window_exceeded",
                    "error": str(exc),
                    "input_tokens": exc.actual,
                    "max_model_len": exc.maximum,
                }

        with ThreadPoolExecutor(max_workers=workers) as pool:
            generations = list(pool.map(
                generate_one,
                zip(questions, evidence, route_errors),
            ))
        # Batch calls do not pass through BaseMemoryMethod.answer(), so return
        # the already-unwrapped public result shape expected by the harness.
        return [
            {
                **generations[index],
                "retrieved_memory_ids": [item["memory_id"] for item in evidence[index]],
                "routes": [] if route_errors[index] is not None else list(routes[index]),
                "retrieved_corpora": [item.get("corpus") for item in evidence[index]],
                "retrieval_scores": [item.get("score") for item in evidence[index]],
            }
            for index in range(len(generations))
        ]

    def generate_answer(self, question, evidence):
        task = build_answer_task(question)
        grounding = (
            "Use only the following retrieved evidence."
            if evidence
            else "No memory evidence is available in this request."
        )
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": (
                f"{task.text}\n"
                f"{grounding}"
            ),
        }]
        for rank, unit in enumerate(evidence, 1):
            content.append({
                "type": "text",
                "text": (
                    f"Evidence {rank}; corpus={unit.get('corpus')}; "
                    f"memory_id={unit.get('memory_id')}; source_id={unit.get('source_id')}: "
                    f"{unit.get('text', '')}"
                ),
            })
            if unit.get("media_auxiliary"):
                content.append({
                    "type": "text",
                    "text": "Dataset-provided media annotation:\n" + str(unit["media_auxiliary"]),
                })
            if unit.get("image"):
                image = str(unit["image"])
                content.extend(image_content({"path": image, "source_id": unit.get("media_source_id")}))
            elif unit.get("video"):
                for frame in uniformly_sample_video(str(unit["video"]), self.video_frames):
                    content.append({"type": "image_url", "image_url": {"url": frame}})
        return self.answer_model.complete(task.messages(content), tools=task.api_tools)

    def _state_path(self) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", self._context_id).strip("._") or "context"
        suffix = hashlib.sha256(self._context_id.encode()).hexdigest()[:12]
        return self.checkpoint_dir / "contexts" / f"{safe[:80]}-{suffix}" / "state.json"

    def _legacy_state_path(self) -> Path | None:
        return self.checkpoint_dir / "state.json" if self.checkpoint_dir else None

    def _checkpoint_config(self) -> dict[str, Any]:
        return {
            "format": "mmmb-universalrag-checkpoint-3",
            "router_model": self.router.model,
            "adapter_profile": self.adapter_profile,
            "dimensions": {name: self.embedders[name].dimension for name in UNIVERSALRAG_CORPORA},
            "video_frames": self.video_frames,
        }

    def _save_checkpoint(self, *, force: bool = False) -> None:
        path = self._state_path()
        if path is None or (not force and self._commits_since_checkpoint < self.checkpoint_interval):
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        version = len(self._processed_memory_ids)
        index_files: dict[str, str] = {}
        for corpus in UNIVERSALRAG_CORPORA:
            name = f"{corpus}-{version:08d}.faiss"
            temporary = path.parent / f".{name}.tmp"
            self.indexes[corpus].save(temporary)
            os.replace(temporary, path.parent / name)
            index_files[corpus] = name
        state = {
            "config": self._checkpoint_config(),
            "context_id": self._context_id,
            "processed_memory_ids": sorted(self._processed_memory_ids),
            "units": self.units,
            "index_files": index_files,
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
        for old_index in path.parent.glob("*.faiss"):
            if old_index.name not in set(index_files.values()):
                old_index.unlink()
        self._commits_since_checkpoint = 0

    def _load_checkpoint(self) -> None:
        path = self._state_path()
        if path is None:
            return
        if not path.is_file():
            legacy = self._legacy_state_path()
            if legacy is None or not legacy.is_file():
                return
            legacy_state = json.loads(legacy.read_text(encoding="utf-8"))
            if legacy_state.get("context_id") != self._context_id:
                return
            path = legacy
        state = json.loads(path.read_text(encoding="utf-8"))
        stored_config = dict(state.get("config", {}))
        stored_config.setdefault("adapter_profile", "official")
        current_config = self._checkpoint_config()
        # The router is used only at query time. Changing the common reader/router
        # model must not invalidate frozen memory units or their embedding indexes.
        # Index compatibility is fully determined by the adapter profile, corpus
        # dimensions, frame sampling policy, and checkpoint format below.
        stored_config.pop("router_model", None)
        current_config.pop("router_model", None)
        if stored_config != current_config:
            raise RuntimeError(f"incompatible UniversalRAG checkpoint: {path}")
        if state.get("context_id") != self._context_id:
            raise RuntimeError(f"UniversalRAG checkpoint belongs to another context: {path}")
        self._processed_memory_ids = {str(value) for value in state.get("processed_memory_ids", [])}
        self.units = {
            corpus: [dict(unit) for unit in state.get("units", {}).get(corpus, [])]
            for corpus in UNIVERSALRAG_CORPORA
        }
        for corpus in UNIVERSALRAG_CORPORA:
            self.indexes[corpus].load(path.parent / str(state["index_files"][corpus]))
            if self.indexes[corpus].size != len(self.units[corpus]):
                raise RuntimeError(f"UniversalRAG {corpus} checkpoint index/unit mismatch")
