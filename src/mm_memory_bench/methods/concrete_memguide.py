from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .backends import (
    AnswerModel,
    OpenAICompatibleQwenVL,
    SentenceTransformerEmbedder,
    TextEmbedder,
    data_url,
)
from .base import BaseMemoryMethod, GenerationConfig, MethodCapabilities, MethodResult
from .answer_input import build_answer_task
from .media import question_text, uniformly_sample_video
from .vector_index import FaissFlatIPIndex, VectorIndex


class NVEmbedV2TextEmbedder:
    """Official MemGuide NV-Embed-v2 retrieval configuration."""

    query_instruction = "Given a question, retrieve passages that answer the question"

    def __init__(
        self,
        model_name: str = "nvidia/NV-Embed-v2",
        *,
        max_length: int = 32768,
        batch_size: int = 8,
    ) -> None:
        try:
            import torch
            import torch.nn.functional as functional
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for NV-Embed-v2") from exc
        self.torch = torch
        self.functional = functional
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        ).eval()
        self._dimension = int(
            getattr(self.model.config, "hidden_size", 0)
            or getattr(self.model.config, "sentence_embedding_dimension", 0)
        )
        if not self._dimension:
            probe = self.encode_texts(["dimension probe"])
            self._dimension = int(probe.shape[1])

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode(self, texts: Sequence[str], instruction: str) -> np.ndarray:
        rows = []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                values = self.model.encode(
                    list(texts[start : start + self.batch_size]),
                    instruction=instruction,
                    max_length=self.max_length,
                )
                values = self.functional.normalize(values, p=2, dim=1)
                rows.append(values.detach().float().cpu().numpy())
        return np.asarray(np.concatenate(rows, axis=0), dtype=np.float32)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "")

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        prefix = f"Instruct: {self.query_instruction}\nQuery: "
        return self._encode(texts, prefix)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                result.extend(_strings(item.get("text", item.get("caption", ""))))
            else:
                result.extend(_strings(item))
        return result
    if isinstance(value, Mapping):
        for key in ("final_text", "full_text", "text", "caption", "description"):
            result = _strings(value.get(key))
            if result:
                return result
    return []


def public_captions(memory: Mapping[str, Any]) -> list[str]:
    """Read only captions carried by the canonical public benchmark record."""
    metadata = memory.get("metadata")
    if not isinstance(metadata, Mapping):
        return []
    derived = metadata.get("derived", {})
    if not isinstance(derived, Mapping):
        return []
    result: list[str] = []
    for key in (
        "caption",
        "short_caption",
        "image_caption",
        "image_captions",
        "video_caption",
        "video_captions",
        "blip_caption",
        "blip_captions",
    ):
        result.extend(_strings(derived.get(key)))
    # Stable de-duplication preserves the benchmark-provided ordering.
    return list(dict.fromkeys(result))


class ConcreteMemGuideMethod(BaseMemoryMethod):
    """MemGuide adapted to caption-mediated multimodal benchmarks.

    The original method is text-only and assumes QA-formatted memories. Media is
    therefore converted inside this method: public dataset captions take
    precedence, otherwise Qwen3-VL captions the original image or sampled video.
    Query-time processing follows MemGuide's two stages: intent-aligned vector
    retrieval and missing-information guided LLM filtering.
    """

    capabilities = MethodCapabilities(
        memory_modalities=frozenset({"text", "image", "video", "table", "document"}),
        query_modalities=frozenset({"text", "image", "video"}),
        supports_tools=True,
        native_multimodal_retrieval=False,
        method_owned_media_processing=True,
    )

    def __init__(
        self,
        generation: GenerationConfig | None = None,
        *,
        answer_model: AnswerModel | None = None,
        guide_model: AnswerModel | None = None,
        caption_model: AnswerModel | None = None,
        question_model: AnswerModel | None = None,
        embedder: TextEmbedder | None = None,
        index: VectorIndex | None = None,
        caption_cache_dir: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 100,
        ingest_batch_size: int = 32,
        video_frames: int = 8,
        retrieval_pool_size: int = 10,
        question_mode: str = "fixed",
        question_workers: int = 8,
    ) -> None:
        super().__init__(generation)
        self.answer_model = answer_model or OpenAICompatibleQwenVL(self.generation)
        self.guide_model = guide_model or self.answer_model
        self.caption_model = caption_model or self.answer_model
        self.question_model = question_model or self.answer_model
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.index = index or FaissFlatIPIndex(self.embedder.dimension)
        self.caption_cache_dir = Path(caption_cache_dir) if caption_cache_dir else None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if checkpoint_interval <= 0 or ingest_batch_size <= 0:
            raise ValueError("MemGuide checkpoint interval and batch size must be positive")
        self.checkpoint_interval = checkpoint_interval
        self.ingest_batch_size = ingest_batch_size
        self.video_frames = video_frames
        self.retrieval_pool_size = max(self.generation.top_k, retrieval_pool_size)
        if question_mode not in {"fixed", "llm"}:
            raise ValueError("MemGuide question mode must be fixed or llm")
        if question_workers <= 0:
            raise ValueError("MemGuide question workers must be positive")
        self.question_mode = question_mode
        self.question_workers = question_workers
        self.units: list[dict[str, Any]] = []
        self._context_id = ""
        self._processed_memory_ids: set[str] = set()
        self._commits_since_checkpoint = 0
        self._pending_units: list[dict[str, Any]] = []

    def _begin_context(self, context: Mapping[str, Any]) -> None:
        self._context_id = str(context.get("context_id", "context"))
        self.units = []
        self.index.reset()
        self._processed_memory_ids = set()
        self._commits_since_checkpoint = 0
        self._pending_units = []
        self._load_checkpoint()

    def _end_context(self) -> None:
        self._flush_pending()
        self._save_checkpoint(force=True)
        self.units = []
        self.index.reset()

    def _ingest(self, memory: Mapping[str, Any]) -> None:
        memory_id = str(memory["memory_id"])
        if memory_id in self._processed_memory_ids:
            return
        self._pending_units.append(self._memory_unit(memory))
        if len(self._pending_units) >= self.ingest_batch_size:
            self._flush_pending()

    def _flush_pending(self) -> None:
        if not self._pending_units:
            return
        values = self._pending_units
        if self.question_mode == "llm":
            with ThreadPoolExecutor(max_workers=self.question_workers) as pool:
                generated = list(pool.map(self._generate_memory_question, values))
            for unit, (question, latency, fallback) in zip(values, generated):
                unit["text"] = f"Q: {question}\nA: {unit.pop('answer_text')}"
                unit["generated_question"] = question
                unit["question_source"] = "fixed_fallback" if fallback else "llm"
                unit["question_latency_seconds"] = latency
        vectors = self.embedder.encode_texts([unit["text"] for unit in values])
        if len(vectors) != len(values):
            raise RuntimeError("MemGuide embedder returned a different number of rows")
        self.index.add(vectors)
        self.units.extend(values)
        self._processed_memory_ids.update(unit["memory_id"] for unit in values)
        self._commits_since_checkpoint += len(values)
        self._pending_units = []
        self._save_checkpoint()

    def _synchronize_memory(self) -> None:
        self._flush_pending()
        self._save_checkpoint(force=True)

    def _embedding_identity(self) -> str:
        return str(
            getattr(self.embedder, "model_name", "")
            or getattr(getattr(self.embedder, "model", None), "model_name_or_path", "")
            or self.embedder.__class__.__qualname__
        )

    def _checkpoint_config(self) -> dict[str, Any]:
        config = {
            "format": "mmmb-memguide-checkpoint-1",
            "embedding": self._embedding_identity(),
            "embedding_dimension": self.embedder.dimension,
            "video_frames": self.video_frames,
            "ingest_batch_size": self.ingest_batch_size,
        }
        # Preserve compatibility with historical fixed-template checkpoints.
        if self.question_mode != "fixed":
            config.update({
                "question_mode": self.question_mode,
                "question_prompt": "content-specific-v1",
            })
        return config

    def _state_path(self) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", self._context_id).strip("._") or "context"
        suffix = hashlib.sha256(self._context_id.encode()).hexdigest()[:12]
        return self.checkpoint_dir / "contexts" / f"{safe[:80]}-{suffix}" / "state.json"

    def _save_checkpoint(self, *, force: bool = False) -> None:
        state_path = self._state_path()
        if state_path is None or (
            not force and self._commits_since_checkpoint < self.checkpoint_interval
        ):
            return
        state_path.parent.mkdir(parents=True, exist_ok=True)
        version = len(self.units)
        index_name = f"index-{version:08d}.faiss"
        index_path = state_path.parent / index_name
        index_tmp = index_path.with_suffix(".tmp")
        self.index.save(index_tmp)
        os.replace(index_tmp, index_path)
        state = {
            "config": self._checkpoint_config(),
            "context_id": self._context_id,
            "processed_memory_ids": sorted(self._processed_memory_ids),
            "units": self.units,
            "index_file": index_name,
        }
        state_tmp = state_path.with_suffix(".tmp")
        state_tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        os.replace(state_tmp, state_path)
        # The state file now atomically points at the new index; older versions
        # are no longer needed and NV-Embed-v2 indexes are large (~16 KiB/item).
        for old_index in state_path.parent.glob("index-*.faiss"):
            if old_index.name != index_name:
                old_index.unlink()
        self._commits_since_checkpoint = 0

    def _load_checkpoint(self) -> None:
        state_path = self._state_path()
        if state_path is None or not state_path.is_file():
            return
        state = json.loads(state_path.read_text(encoding="utf-8"))
        stored_config = dict(state.get("config", {}))
        # retrieval_pool_size was present in the first checkpoint format even
        # though it affects only query-time selection, not stored units or
        # embeddings. Ignore it so changing top-k never forces a re-digest.
        stored_config.pop("retrieval_pool_size", None)
        if stored_config != self._checkpoint_config():
            raise RuntimeError(f"incompatible MemGuide checkpoint: {state_path}")
        if state.get("context_id") != self._context_id:
            raise RuntimeError(f"MemGuide checkpoint belongs to another context: {state_path}")
        self.units = [dict(unit) for unit in state.get("units", [])]
        self._processed_memory_ids = {
            str(value) for value in state.get("processed_memory_ids", [])
        }
        self.index.load(state_path.parent / str(state["index_file"]))
        if self.index.size != len(self.units):
            raise RuntimeError("MemGuide checkpoint index/unit row mismatch")
        if len(self._processed_memory_ids) != len(self.units):
            raise RuntimeError("MemGuide checkpoint processed/unit count mismatch")

    def _memory_unit(self, memory: Mapping[str, Any]) -> dict[str, Any]:
        parts: list[str] = []
        media: list[Mapping[str, Any]] = []
        for part in memory.get("content", []):
            kind = part.get("type")
            if kind in {"text", "table", "document"} and part.get("text"):
                parts.append(str(part["text"]))
            elif kind in {"image", "video"}:
                media.append(part)
            elif kind == "audio":
                raise NotImplementedError("MemGuide caption track does not transcribe audio")

        captions = public_captions(memory)
        if captions:
            parts.extend(f"[Dataset caption] {caption}" for caption in captions)
        else:
            for part in media:
                parts.append(f"[Generated caption] {self._caption(part)}")

        header = " ".join(
            f"{key}={memory[key]}"
            for key in ("source_id", "session_id", "speaker", "timestamp")
            if memory.get(key) is not None
        )
        text = "\n".join(value for value in (header, *parts) if value).strip()
        if not text:
            raise RuntimeError(f"memory {memory['memory_id']} produced no MemGuide text")
        # A benchmark interaction is the smallest lossless QA-summary surrogate:
        # its provenance is the implicit question, and its content is the answer.
        qa_text = f"Q: What happened in memory {memory['memory_id']}?\nA: {text}"
        unit = {
            "memory_id": str(memory["memory_id"]),
            "source_id": memory.get("source_id"),
            "caption_source": "dataset" if captions else ("generated" if media else "none"),
        }
        if self.question_mode == "llm":
            unit["answer_text"] = text
        else:
            unit["text"] = qa_text
            unit["question_source"] = "fixed"
        return unit

    def _generate_memory_question(
        self, unit: Mapping[str, Any]
    ) -> tuple[str, float, bool]:
        """Create one content-specific QA question and expose its wall latency."""
        answer_text = str(unit["answer_text"])
        prompt = (
            "Convert the memory below into exactly one concise, information-seeking question "
            "that can be answered solely from the memory. Ask about its most specific useful "
            "fact, such as a person, object, action, date, place, amount, identifier, reason, "
            "or event. Do not mention the words memory or evidence, do not expose source IDs, "
            "and do not provide the answer. Return only the question as one sentence.\n\n"
            f"Memory:\n{answer_text}"
        )
        started = time.perf_counter()
        raw = self.question_model.complete(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        ).strip()
        latency = time.perf_counter() - started
        question = re.sub(r"^(?:question\s*:\s*|q\s*:\s*)", "", raw, flags=re.I).strip()
        question = question.splitlines()[0].strip() if question else ""
        fallback = not question
        if fallback:
            question = f"What happened in memory {unit['memory_id']}?"
        return question, latency, fallback

    def _caption_key(self, part: Mapping[str, Any]) -> str:
        path = str(part["path"])
        digest = hashlib.sha256()
        digest.update(str(part.get("type", "media")).encode())
        digest.update(b"\0")
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(f"\0frames={self.video_frames}\0prompt=v1".encode())
        return digest.hexdigest()

    def _caption(self, part: Mapping[str, Any]) -> str:
        key = self._caption_key(part)
        cache_path = self.caption_cache_dir / f"{key}.json" if self.caption_cache_dir else None
        if cache_path and cache_path.is_file():
            value = json.loads(cache_path.read_text(encoding="utf-8"))
            if value.get("caption"):
                return str(value["caption"])

        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": (
                "Describe this benchmark memory faithfully and densely. Include visible text/OCR, "
                "people, objects, attributes, spatial relations, actions, and temporal changes. "
                "Do not infer facts that are not visible. Return only the description."
            ),
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
            raise RuntimeError(f"Qwen3-VL returned an empty caption for {path}")
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps({"caption": caption, "kind": kind}, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(temporary, cache_path)
        return caption

    def _answer(self, question: Mapping[str, Any]) -> MethodResult:
        self._flush_pending()
        query = question_text(question)
        if not self.units:
            evidence: list[dict[str, Any]] = []
        else:
            encode_queries = getattr(self.embedder, "encode_queries", self.embedder.encode_texts)
            vector = encode_queries([query])
            scores, rows = self.index.search(vector, self.retrieval_pool_size)
            candidates = [self.units[int(row)] for row in rows[0]]
            evidence = self._missing_information_filter(query, candidates)

        prediction = self._generate_answer(question, evidence)
        return MethodResult(
            prediction=prediction,
            diagnostics={
                "retrieved_memory_ids": [unit["memory_id"] for unit in evidence[: self.generation.top_k]],
                "method": "memguide-caption",
            },
        )

    def _generate_answer(
        self,
        question: Mapping[str, Any],
        evidence: Sequence[Mapping[str, Any]],
    ) -> str:
        rendered = "\n\n".join(
            f"Evidence {rank} (memory_id={unit['memory_id']}):\n{unit['text']}"
            for rank, unit in enumerate(evidence[: self.generation.top_k], 1)
        ) or "No relevant memory was retrieved."
        task = build_answer_task(question)
        prompt = (
            f"{task.text}\n\n"
            "Answer using only the selected memory evidence.\n\n" + rendered
        )
        return self.answer_model.complete(
            task.messages([{"type": "text", "text": prompt}]),
            tools=task.api_tools,
        )

    def answer_many(
        self, questions: Sequence[Mapping[str, Any]], *, concurrency: int = 8
    ) -> list[dict[str, Any]]:
        """Batch query embeddings, then parallelize MemGuide filtering/generation."""
        self._require_active("answer_many")
        self._flush_pending()
        if not questions:
            return []
        queries = [question_text(question) for question in questions]
        candidate_groups: list[list[dict[str, Any]]] = [[] for _ in questions]
        if self.units:
            encode_queries = getattr(self.embedder, "encode_queries", self.embedder.encode_texts)
            vectors = encode_queries(queries)
            _, rows = self.index.search(vectors, self.retrieval_pool_size)
            candidate_groups = [
                [self.units[int(row)] for row in query_rows]
                for query_rows in rows
            ]
        workers = max(1, concurrency)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            evidence_groups = list(pool.map(
                lambda pair: self._missing_information_filter(*pair),
                zip(queries, candidate_groups),
            ))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            predictions = list(pool.map(
                lambda values: self._generate_answer(*values),
                zip(questions, evidence_groups),
            ))
        return [
            {
                "prediction": prediction,
                "retrieved_memory_ids": [
                    unit["memory_id"] for unit in evidence[: self.generation.top_k]
                ],
                "method": "memguide-caption",
            }
            for prediction, evidence in zip(predictions, evidence_groups)
        ]

    def _missing_information_filter(
        self, query: str, candidates: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        listing = "\n\n".join(
            f"[{index}] {unit['text']}" for index, unit in enumerate(candidates)
        )
        prompt = (
            "You are the missing-information guided filter from MemGuide. Infer the user's "
            "goal and the facts/slots needed to answer it, then select the candidate memories "
            f"with the greatest completion gain. Select at most {self.generation.top_k}. "
            "Return JSON only as {\"selected\":[integer indices]}.\n\n"
            f"Question: {query}\n\nCandidates:\n{listing}"
        )
        raw = self.guide_model.complete(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            response_format={"type": "json_object"},
        )
        try:
            selected = json.loads(raw).get("selected", [])
            ids = [int(value) for value in selected]
        except (ValueError, TypeError, json.JSONDecodeError):
            ids = list(range(min(self.generation.top_k, len(candidates))))
        unique = list(dict.fromkeys(i for i in ids if 0 <= i < len(candidates)))
        if not unique:
            unique = list(range(min(self.generation.top_k, len(candidates))))
        return [dict(candidates[i]) for i in unique[: self.generation.top_k]]
