from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..preprocessing.captions import CAPTION_PROMPT, caption_cache_key, load_cached_caption, save_cached_caption
from .backends import AnswerModel, OpenAICompatibleQwenVL, data_url
from .base import BaseMemoryMethod, GenerationConfig, MethodCapabilities, MethodResult
from .concrete_memguide import public_captions
from .media import question_text, uniformly_sample_video


MEMORY_ID_RE = re.compile(r"\[memory_id=([^\]\n]+)\]")
OCR_LINE_RE = re.compile(r"^(\s*(?:\[Dataset caption\]\s*)?OCR\s*:\s*)(.*)$", re.I)
INLINE_OCR_RE = re.compile(
    r"(\bOCR\s*:\s*)(.*?)(?=\s+(?:Location|City|Country|Final output|Tags)\s*:|$)",
    re.I | re.S,
)
NUMERIC_OCR_TOKEN_RE = re.compile(r"[+-]?\d+(?:[.,:/-]\d+)*%?")
TRANSPORT_SOURCE_ID_RE = re.compile(
    r"\bsource\s*_?\s*id\s*=\s*.*?(?=\s+timestamp\s*=)", re.I | re.S
)
ARCHIVE_FIELD_RE = re.compile(
    r"^\s*(?:Caption|Short caption|OCR|Location|City|Tags|Image caption\s+\d+)\s*:",
    re.I,
)
SINGLETON_REPAIR_ATTEMPTS = 3

ARCHIVE_EXTRACTION_PROMPT = """
You are the factual extraction stage of LightMem. The input is an ordered stream
of benchmark archive records, not a conversation. Process every user record and extract all
answerable facts, including names, numbers, dates, visible/OCR details, lists,
and relationships. Never combine facts from different source records. Use the
integer message prefix as source_id; provenance is attached structurally by the
runtime and must not be copied into the fact text. Source IDs may begin at a
nonzero value: copy the exact displayed integer and never renumber a call from zero.
Return at most 24 distinct facts for each source record. Never repeat or
paraphrase the same fact to fill the response. After the final useful fact,
close the JSON object immediately.

Return exactly one JSON object of this form:
{"data": [{"source_id": 0, "fact": "complete standalone fact"}]}
""".strip()

SINGLETON_EXTRACTION_PROMPT = """
You are repairing one benchmark archive record that a previous LightMem batch
could not index. Return exactly one concise, standalone fact summarizing the
most answerable information in this user record. The fact must be at most 80
words. Copy the exact integer message prefix as source_id; do not renumber it,
and do not include provenance syntax inside the fact. Close the JSON object
immediately after the single fact.

Return exactly this shape and no markdown:
{"data": [{"source_id": 0, "fact": "one concise standalone fact"}]}
""".strip()


class LightMemBackend(Protocol):
    def add_memory(self, messages, **kwargs): ...

    def retrieve(self, query: str, limit: int = 10) -> list[str]: ...

    def get_token_statistics(self) -> Mapping[str, Any]: ...

    def source_memory_ids_for_batch(self, batch_id: str) -> set[str]: ...

    def rollback_batch(self, batch_id: str) -> None: ...


class _OfficialLightMemBackend:
    def __init__(self, memory: Any) -> None:
        self.memory = memory

    def add_memory(self, messages, **kwargs):
        return self.memory.add_memory(messages, **kwargs)

    def retrieve(self, query: str, limit: int = 10) -> list[str]:
        return self.memory.retrieve(query, limit=limit)

    def get_token_statistics(self) -> Mapping[str, Any]:
        return self.memory.get_token_statistics()

    def _batch_points(self, batch_id: str) -> list[Any]:
        points: list[Any] = []
        offset = None
        while True:
            rows, offset = self.memory.embedding_retriever.scroll(
                scroll_filter={"ingest_batch_id": batch_id},
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(rows)
            if offset is None:
                return points

    def source_memory_ids_for_batch(self, batch_id: str) -> set[str]:
        return {
            str(point.payload.get("source_memory_id"))
            for point in self._batch_points(batch_id)
            if point.payload and point.payload.get("source_memory_id")
        }

    def rollback_batch(self, batch_id: str) -> None:
        for point in self._batch_points(batch_id):
            self.memory.embedding_retriever.delete(point.id)


def _official_backend(
    *,
    official_repo: Path,
    collection_name: str,
    qdrant_path: Path,
    generation: GenerationConfig,
    llmlingua_model: str,
    embedding_model: str,
    device: str,
) -> LightMemBackend:
    source_root = str((official_repo / "src").resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    try:
        from lightmem.memory.lightmem import LightMemory
    except ImportError as exc:
        raise RuntimeError(
            "official LightMem dependencies are unavailable; use the isolated LightMem runtime"
        ) from exc

    qdrant_path.mkdir(parents=True, exist_ok=True)
    config = {
        "pre_compress": True,
        "pre_compressor": {
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": {
                    "model_name": llmlingua_model,
                    "device_map": device,
                    "use_llmlingua2": True,
                }
            },
        },
        "topic_segment": True,
        "precomp_topic_shared": True,
        "topic_segmenter": {"model_name": "llmlingua-2"},
        "messages_use": "user_only",
        "metadata_generate": True,
        "text_summary": True,
        "memory_manager": {
            "model_name": "openai",
            "configs": {
                "model": generation.model,
                "api_key": os.environ.get("OPENAI_API_KEY", "EMPTY"),
                "max_tokens": generation.max_output_tokens,
                "temperature": generation.temperature,
                "openai_base_url": generation.base_url,
            },
        },
        "extract_threshold": 0.1,
        "index_strategy": "embedding",
        "text_embedder": {
            "model_name": "huggingface",
            "configs": {
                "model": embedding_model,
                "embedding_dims": 384,
                "model_kwargs": {"device": device},
            },
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": collection_name,
                "embedding_model_dims": 384,
                "path": str(qdrant_path),
                "on_disk": True,
            },
        },
        # The paper's efficient path performs consolidation asynchronously.
        # We keep extraction/indexing but do not run the optional quadratic
        # queue construction and LLM merge pass during benchmark ingestion.
        "update": "offline",
    }
    return _OfficialLightMemBackend(LightMemory.from_config(config))


class ConcreteLightMemMethod(BaseMemoryMethod):
    """Official LightMem pipeline behind the common benchmark lifecycle.

    Each canonical memory is represented by one user turn followed by a neutral
    assistant placeholder. The pair is required by upstream LightMem's
    ``source_id * 2`` provenance mapping; only user turns are extracted.
    """

    capabilities = MethodCapabilities(
        memory_modalities=frozenset({"text", "image", "video", "table", "document"}),
        query_modalities=frozenset({"text", "image", "video"}),
        native_multimodal_retrieval=False,
        method_owned_media_processing=True,
    )

    def __init__(
        self,
        generation: GenerationConfig | None = None,
        *,
        memory_generation: GenerationConfig | None = None,
        answer_model: AnswerModel | None = None,
        caption_model: AnswerModel | None = None,
        caption_cache_dir: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        official_repo: str | Path = "sources/lightmem",
        llmlingua_model: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        ingest_batch_size: int = 32,
        video_frames: int = 8,
        backend_factory: Callable[..., LightMemBackend] | None = None,
    ) -> None:
        super().__init__(generation)
        if ingest_batch_size <= 0:
            raise ValueError("LightMem ingest batch size must be positive")
        self.answer_model = answer_model or OpenAICompatibleQwenVL(self.generation)
        self.memory_generation = memory_generation or self.generation
        self.caption_model = caption_model or self.answer_model
        self.caption_cache_dir = Path(caption_cache_dir) if caption_cache_dir else None
        self.checkpoint_dir = Path(checkpoint_dir or "runs/lightmem/checkpoint")
        self.official_repo = Path(official_repo)
        self.llmlingua_model = llmlingua_model
        self.embedding_model = embedding_model
        self.device = device
        self.ingest_batch_size = ingest_batch_size
        self.video_frames = video_frames
        self.backend_factory = backend_factory or _official_backend
        self.backend: LightMemBackend | None = None
        self._context_id = ""
        self._processed_memory_ids: set[str] = set()
        self._pending: list[tuple[str, list[dict[str, Any]]]] = []
        self._sequence = 0
        self.timing: dict[str, float] = {}

    def _context_root(self) -> Path:
        digest = hashlib.sha256(self._context_id.encode()).hexdigest()[:20]
        return self.checkpoint_dir / "contexts" / digest

    def _state_path(self) -> Path:
        return self._context_root() / "state.json"

    def _begin_context(self, context: Mapping[str, Any]) -> None:
        self._context_id = str(context.get("context_id", "context"))
        root = self._context_root()
        root.mkdir(parents=True, exist_ok=True)
        self._processed_memory_ids = set()
        self._pending = []
        self._sequence = 0
        self.timing = {"caption_seconds": 0.0, "lightmem_add_seconds": 0.0}
        state_path = self._state_path()
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("context_id") != self._context_id:
                raise RuntimeError("LightMem checkpoint belongs to another context")
            self._processed_memory_ids = {
                str(value) for value in state.get("processed_memory_ids", [])
            }
            self._sequence = int(state.get("sequence", len(self._processed_memory_ids)))
            stored_timing = state.get("timing", {})
            if isinstance(stored_timing, Mapping):
                for key in self.timing:
                    self.timing[key] = float(stored_timing.get(key, 0.0))
        collection = "mmmb_" + hashlib.sha256(self._context_id.encode()).hexdigest()[:16]
        self.backend = self.backend_factory(
            official_repo=self.official_repo,
            collection_name=collection,
            qdrant_path=root / "qdrant",
            generation=self.memory_generation,
            llmlingua_model=self.llmlingua_model,
            embedding_model=self.embedding_model,
            device=self.device,
        )

    def _save_state(self) -> None:
        state_path = self._state_path()
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "format": "mmmb-lightmem-checkpoint-1",
                    "context_id": self._context_id,
                    "processed_memory_ids": sorted(self._processed_memory_ids),
                    "sequence": self._sequence,
                    "timing": self.timing,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, state_path)

    @staticmethod
    def _safe_timestamp(value: Any, sequence: int) -> str:
        raw = str(value or "").strip()
        if raw:
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(
                    tzinfo=None
                ).isoformat(timespec="seconds")
            except ValueError:
                pass
        base = datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=sequence)
        return base.replace(tzinfo=None).isoformat(timespec="seconds")

    def _caption(self, part: Mapping[str, Any]) -> str:
        key = caption_cache_key(part, video_frames=self.video_frames)
        cache_path = self.caption_cache_dir / f"{key}.json" if self.caption_cache_dir else None
        if cache_path:
            cached = load_cached_caption(cache_path)
            if cached:
                return cached
        content: list[dict[str, Any]] = [{"type": "text", "text": CAPTION_PROMPT}]
        kind, path = str(part["type"]), str(part["path"])
        if kind == "image":
            content.append({"type": "image_url", "image_url": {"url": data_url(path)}})
        else:
            content.extend(
                {"type": "image_url", "image_url": {"url": frame}}
                for frame in uniformly_sample_video(path, self.video_frames)
            )
        started = time.perf_counter()
        caption = self.caption_model.complete([{"role": "user", "content": content}]).strip()
        self.timing["caption_seconds"] += time.perf_counter() - started
        if not caption:
            raise RuntimeError(f"Qwen3-VL returned an empty LightMem caption for {path}")
        if cache_path:
            save_cached_caption(cache_path, caption=caption, kind=kind)
        return caption

    @staticmethod
    def _compact_numeric_ocr_runs(tokens: Sequence[str]) -> list[str]:
        """Collapse enumerated OCR IDs without touching ordinary numbers."""
        compacted: list[str] = []
        index = 0
        while index < len(tokens):
            normalized = tokens[index].strip("()[]{}<>,;.!?\"'")
            if not NUMERIC_OCR_TOKEN_RE.fullmatch(normalized):
                compacted.append(tokens[index])
                index += 1
                continue
            cursor = index + 1
            while cursor < len(tokens):
                candidate = tokens[cursor].strip("()[]{}<>,;.!?\"'")
                if not NUMERIC_OCR_TOKEN_RE.fullmatch(candidate):
                    break
                cursor += 1
            run = tokens[index:cursor]
            if len(run) >= 16:
                sample = " ".join(run[:6])
                compacted.append(
                    f"[numeric OCR list compacted: {len(run)} values; sample {sample}]"
                )
            else:
                compacted.extend(run)
            index = cursor
        return compacted

    @classmethod
    def _compact_ocr_payload(cls, payload: str) -> str:
        tokens = cls._compact_numeric_ocr_runs(payload.split())
        if len(tokens) <= 128 and len(" ".join(tokens)) <= 1024:
            return " ".join(tokens)
        counts: dict[str, int] = {}
        kept: list[str] = []
        kept_chars = 0
        for token in tokens:
            key = token.casefold()
            count = counts.get(key, 0)
            if count >= 3:
                continue
            counts[key] = count + 1
            addition = len(token) + (1 if kept else 0)
            if kept_chars + addition > 1024:
                break
            kept.append(token)
            kept_chars += addition
        return " ".join(kept).rstrip() + " [repetitive OCR compacted]"

    @classmethod
    def _compact_archive_text(cls, text: str) -> str:
        """Bound pathological OCR repetition without rewriting ordinary records."""
        raw_text = str(text).replace("\x00", " ")
        # ATM derived records are sometimes serialized as one long line, so
        # line-anchored OCR handling alone misses the pathological field.
        raw_text = INLINE_OCR_RE.sub(
            lambda match: match.group(1) + cls._compact_ocr_payload(match.group(2)),
            raw_text,
        )
        source_lines = raw_text.splitlines()
        lines: list[str] = []
        index = 0
        while index < len(source_lines):
            raw_line = source_lines[index]
            line = " ".join(raw_line.split())
            match = OCR_LINE_RE.match(line)
            if not match:
                lines.append(line)
                index += 1
                continue
            prefix, payload = match.groups()
            # OCR annotations can contain embedded newlines. Consume the whole
            # field up to the next canonical annotation label before measuring
            # repetition; otherwise each repeated line looks harmless alone.
            block = [payload]
            cursor = index + 1
            while cursor < len(source_lines):
                candidate = " ".join(source_lines[cursor].split())
                if ARCHIVE_FIELD_RE.match(candidate):
                    break
                block.append(candidate)
                cursor += 1
            payload = " ".join(value for value in block if value)
            tokens = payload.split()
            if len(tokens) <= 128 and len(payload) <= 1024:
                lines.append(prefix + payload)
                index = cursor
                continue
            lines.append(prefix + cls._compact_ocr_payload(payload))
            index = cursor
        return "\n".join(line for line in lines if line)

    def _memory_text(self, memory: Mapping[str, Any]) -> str:
        values: list[str] = []
        media: list[Mapping[str, Any]] = []
        for part in memory.get("content", []):
            kind = part.get("type")
            if kind in {"text", "table", "document"} and part.get("text"):
                # Canonical ATM derived text can contain a transport-level
                # ``source_id=...`` immediately before the timestamp. LightMem
                # uses small local integer source IDs in its extraction JSON;
                # exposing both IDs makes the VLM copy the transport ID and
                # breaks structured provenance assignment.
                visible = TRANSPORT_SOURCE_ID_RE.sub("", str(part["text"]))
                values.append(self._compact_archive_text(visible))
            elif kind in {"image", "video"}:
                media.append(part)
            elif kind == "audio":
                raise NotImplementedError("LightMem caption track does not transcribe audio")
        captions = public_captions(memory)
        if captions:
            values.extend(f"[Dataset caption] {caption}" for caption in captions)
        else:
            values.extend(f"[Generated caption] {self._caption(part)}" for part in media)
        identity = " ".join(
            f"{key}={memory[key]}"
            for key in ("session_id", "speaker", "timestamp")
            if memory.get(key) is not None
        )
        memory_id = str(memory["memory_id"])
        text = "\n".join(v for v in (identity, *values) if v)
        if not text.strip():
            raise RuntimeError(f"memory {memory_id} produced no LightMem text")
        return text

    def _ingest(self, memory: Mapping[str, Any]) -> None:
        memory_id = str(memory["memory_id"])
        if memory_id in self._processed_memory_ids:
            return
        timestamp = self._safe_timestamp(memory.get("timestamp"), self._sequence)
        speaker = str(memory.get("speaker") or memory.get("kind") or "Archive")
        messages = [
            {
                "role": "user",
                "content": self._memory_text(memory),
                "time_stamp": timestamp,
                "speaker_name": speaker,
                "speaker_id": speaker,
                "canonical_memory_id": memory_id,
            },
            {
                "role": "assistant",
                "content": "Memory recorded.",
                "time_stamp": timestamp,
                "speaker_name": "Recorder",
                "speaker_id": "recorder",
            },
        ]
        self._pending.append((memory_id, messages))
        self._sequence += 1
        if len(self._pending) >= self.ingest_batch_size:
            self._flush_pending()

    def _flush_pending(self) -> None:
        if not self._pending:
            return
        if self.backend is None:
            raise RuntimeError("LightMem backend is not initialized")
        pending = self._pending
        messages = [message for _, pair in pending for message in pair]
        batch_id = hashlib.sha256(
            (self._context_id + "\0" + "\0".join(memory_id for memory_id, _ in pending)).encode()
        ).hexdigest()[:24]
        for message in messages:
            message["ingest_batch_id"] = batch_id
        # A previous process may have died after Qdrant insertion but before
        # the atomic state checkpoint. Deterministic batch IDs make retry safe.
        self.backend.rollback_batch(batch_id)
        started = time.perf_counter()
        try:
            result = self.backend.add_memory(
                messages,
                METADATA_GENERATE_PROMPT=ARCHIVE_EXTRACTION_PROMPT,
                force_segment=True,
                force_extract=True,
            )
            if not isinstance(result, Mapping) or int(result.get("api_call_nums", 0)) <= 0:
                raise RuntimeError("LightMem extraction produced no LLM calls; batch not committed")
            raw_outputs = result.get("add_output_prompt", [])
            if not isinstance(raw_outputs, list) or any(
                not isinstance(value, str) or not value.strip() for value in raw_outputs
            ):
                raise RuntimeError("LightMem extraction returned an empty LLM response; batch not committed")
            expected_ids = {memory_id for memory_id, _ in pending}
            indexed_ids = self.backend.source_memory_ids_for_batch(batch_id)
            missing_ids = sorted(expected_ids - indexed_ids)
            # Batched factual extraction can occasionally merge adjacent
            # records and omit a source label even though the input was valid.
            # Repair only those omissions one record at a time.  Successful
            # batch facts remain untouched, and the whole transaction is still
            # rolled back if any singleton extraction also fails.
            pending_by_id = {memory_id: pair for memory_id, pair in pending}
            repair_errors: list[str] = []
            for missing_id in missing_ids:
                for attempt in range(1, SINGLETON_REPAIR_ATTEMPTS + 1):
                    try:
                        repair_result = self.backend.add_memory(
                            pending_by_id[missing_id],
                            METADATA_GENERATE_PROMPT=SINGLETON_EXTRACTION_PROMPT,
                            force_segment=True,
                            force_extract=True,
                        )
                        if not isinstance(repair_result, Mapping) or int(
                            repair_result.get("api_call_nums", 0)
                        ) <= 0:
                            raise RuntimeError("extraction produced no LLM call")
                        repair_outputs = repair_result.get("add_output_prompt", [])
                        if not isinstance(repair_outputs, list) or any(
                            not isinstance(value, str) or not value.strip()
                            for value in repair_outputs
                        ):
                            raise RuntimeError("extraction returned an empty response")
                        raw_outputs.extend(repair_outputs)
                    except Exception as exc:
                        repair_errors.append(
                            f"{missing_id} attempt {attempt}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    if missing_id in self.backend.source_memory_ids_for_batch(batch_id):
                        break
                    repair_errors.append(
                        f"{missing_id} attempt {attempt}: no indexed provenance fact"
                    )
            indexed_ids = self.backend.source_memory_ids_for_batch(batch_id)
            missing_ids = sorted(expected_ids - indexed_ids)
            if missing_ids:
                diagnostics_dir = self._context_root() / "failed_batches"
                diagnostics_dir.mkdir(parents=True, exist_ok=True)
                diagnostics_path = diagnostics_dir / f"{batch_id}.json"
                diagnostics_path.write_text(
                    json.dumps(
                        {
                            "batch_id": batch_id,
                            "expected_memory_ids": sorted(expected_ids),
                            "indexed_memory_ids": sorted(indexed_ids),
                            "missing_memory_ids": missing_ids,
                            "raw_outputs": raw_outputs,
                            "repair_errors": repair_errors,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                raise RuntimeError(
                    "LightMem extraction created no indexed fact for "
                    f"{len(missing_ids)} memories (first: {missing_ids[0]}); batch rolled back"
                )
        except BaseException:
            self.backend.rollback_batch(batch_id)
            raise
        self.timing["lightmem_add_seconds"] += time.perf_counter() - started
        self._processed_memory_ids.update(memory_id for memory_id, _ in pending)
        self._pending = []
        self._save_state()

    def _synchronize_memory(self) -> None:
        self._flush_pending()

    def _answer(self, question: Mapping[str, Any]) -> MethodResult:
        self._flush_pending()
        if self.backend is None:
            raise RuntimeError("LightMem backend is not initialized")
        query = question_text(question)
        evidence = self.backend.retrieve(query, limit=self.generation.top_k)
        rendered = "\n\n".join(
            f"Evidence {index}:\n{value}" for index, value in enumerate(evidence, 1)
        ) or "No relevant memory was retrieved."
        prompt = (
            f"{question.get('instruction', '')}\nQuestion: {query}\n\n"
            "Answer using only the retrieved LightMem evidence.\n\n" + rendered
        )
        prediction = self.answer_model.complete(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        )
        ids = list(
            dict.fromkeys(
                match.group(1)
                for value in evidence
                for match in MEMORY_ID_RE.finditer(value)
            )
        )
        return MethodResult(
            prediction=prediction,
            diagnostics={
                "retrieved_memory_ids": ids,
                "method": "lightmem-official-caption",
            },
        )

    def _end_context(self) -> None:
        self._flush_pending()
        self._save_state()
        self.backend = None
        self._pending = []

    def _abort_context(self) -> None:
        # Do not retry a failed LightMem batch during exception cleanup.
        self.backend = None
        self._pending = []
