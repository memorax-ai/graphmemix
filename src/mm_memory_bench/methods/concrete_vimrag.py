from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from .backends import AnswerModel, MultiModalEmbedder, OpenAICompatibleQwenVL
from .base import BaseMemoryMethod, GenerationConfig, MethodCapabilities, MethodResult
from .media import question_text, text_from_parts
from .vector_index import FaissFlatIPIndex, VectorIndex


class VimRAGAgent(Protocol):
    def run(self, sample: dict[str, Any]): ...


def normalize_vimrag_action(
    name: str | None, arguments: Mapping[str, Any] | None
) -> tuple[str | None, dict[str, Any] | None]:
    """Validate required actions and repair non-semantic graph bookkeeping."""
    if name is None or not isinstance(arguments, Mapping):
        return None, None
    values = dict(arguments)
    if name == "add_search_node":
        query = str(values.get("query", "")).strip()
        if not query:
            return None, None
        values["query"] = query
        if not str(values.get("id", "")).strip():
            digest = hashlib.sha256(query.encode()).hexdigest()[:10]
            values["id"] = f"search_{digest}"
        if not isinstance(values.get("parent_ids"), list):
            values["parent_ids"] = ["root"]
    elif name == "summarize_and_memorize":
        if "summarize" not in values:
            return None, None
        if not isinstance(values.get("memorize"), list):
            values["memorize"] = []
    elif name == "add_answer_node":
        if "answer" not in values:
            return None, None
        if not isinstance(values.get("parent_ids"), list):
            values["parent_ids"] = []
    else:
        return None, None
    return name, values


def _load_official_demo(official_repo: Path):
    demo = str((official_repo / "demo").resolve())
    if demo not in sys.path:
        sys.path.insert(0, demo)
    return (
        importlib.import_module("vimrag_agent"),
        importlib.import_module("vimrag_prompt"),
        importlib.import_module("vimrag_utils"),
    )


def official_vimrag_agent_factory(
    *,
    official_repo: Path,
    answer_model: AnswerModel,
    generation: GenerationConfig,
    search: Callable[[str, int], dict[str, Any]],
    query_media: list[dict[str, Any]],
    search_top_k: int,
    memory_top_k: int,
    max_steps: int,
    video_frames: int,
) -> VimRAGAgent:
    agent_module, prompt_module, utils_module = _load_official_demo(official_repo)

    class LocalVimRAG(agent_module.VimRAG):
        def __init__(self) -> None:
            super().__init__(
                base_url=generation.base_url,
                search_url="local://faiss",
                model_name=generation.model,
                search_top_k=search_top_k,
                max_mem_steps=max_steps,
                enable_thinking=False,
            )
            self.memory_top_k = memory_top_k
            # The official demo defaults to 32 frames per retrieved video.
            # Keep video budgeting method-owned, but use the explicit Main
            # Track cap so top-k videos cannot silently multiply into hundreds
            # of visual inputs.
            self.max_frames_video = video_frames
            self.parse_repairs = 0

        def search(self, queries, top_k=None):
            values = [queries] if isinstance(queries, str) else list(queries)
            return [search(str(query), int(top_k or self.search_top_k)) for query in values]

        def _model_generate(self, messages):
            content = answer_model.complete(messages)
            yield {"type": "done", "reasoning": "", "content": content}

        @staticmethod
        def _tool_object(response: str):
            candidates = [response]
            candidates.extend(
                match.group(1)
                for match in re.finditer(r"```(?:json|json5)?\s*(.*?)```", response, re.S | re.I)
            )
            object_match = re.search(r"\{.*\}", response, re.S)
            if object_match:
                candidates.append(object_match.group(0))
            for candidate in candidates:
                try:
                    value = agent_module.json5.loads(candidate.strip())
                except Exception:
                    continue
                if not isinstance(value, Mapping):
                    continue
                function = value.get("function")
                if isinstance(function, Mapping):
                    value = function
                name = value.get("name")
                arguments = value.get("arguments")
                if name in {"add_search_node", "summarize_and_memorize", "add_answer_node"} and isinstance(arguments, Mapping):
                    return str(name), dict(arguments)
            return None, None

        def _parse_response(self, response):
            name, arguments = super()._parse_response(response)
            if name is not None:
                name, arguments = normalize_vimrag_action(name, arguments)
                if name is not None:
                    return name, arguments
            name, arguments = self._tool_object(response)
            if name is not None:
                name, arguments = normalize_vimrag_action(name, arguments)
                if name is not None:
                    return name, arguments
            repair_prompt = (
                "Convert the candidate agent action below into exactly one valid VimRAG "
                "tool call. Preserve its intent and factual answer; do not add facts. "
                "Allowed names: add_search_node, summarize_and_memorize, add_answer_node. "
                "Return only <tool_call>{\"name\":...,\"arguments\":{...}}</tool_call>.\n\n"
                "Candidate action:\n" + response
            )
            repaired = answer_model.complete(
                [{"role": "user", "content": [{"type": "text", "text": repair_prompt}]}]
            )
            self.parse_repairs += 1
            name, arguments = super()._parse_response(repaired)
            if name is not None:
                return normalize_vimrag_action(name, arguments)
            name, arguments = self._tool_object(repaired)
            return normalize_vimrag_action(name, arguments)

        def _build_initial_messages(self, question, action_graph):
            messages = super()._build_initial_messages(question, action_graph)
            messages[-1]["content"].extend(dict(item) for item in query_media)
            return messages

        def _update_messages_with_memory(self, messages, action_graph, multimodal_memory):
            if not multimodal_memory:
                return
            energies = utils_module.calculate_intuitive_memory_energy(
                action_graph, multimodal_memory
            )
            content = utils_module.generate_multimodal_messages(
                action_graph,
                energies,
                top_k=self.memory_top_k,
                S_total=self.memory_buffer_pixels,
                video_frame_max_pixels=self.max_pixels_video,
                image_max_pixels=self.max_pixels_image,
            )
            messages[-1]["content"].append(
                {
                    "type": "text",
                    "text": "### Multimodal Memory\nHistorical information for reference only.\n",
                }
            )
            messages[-1]["content"].extend(content)

    return LocalVimRAG()


class Qwen3VLVimRAGEmbedder:
    """Official VimRAG Qwen3-VL embedding wrapper."""

    def __init__(self, model_name: str, *, official_repo: str | Path) -> None:
        root = Path(official_repo).resolve()
        search_root = str((root / "search_engine").resolve())
        if search_root not in sys.path:
            sys.path.insert(0, search_root)
        try:
            import torch
            from models.Qwen3_VL_Embedding.qwen3_vl_embedding import Qwen3VLEmbedder
        except ImportError as exc:
            raise RuntimeError("official VimRAG Qwen3-VL embedding dependencies are missing") from exc
        self.model = Qwen3VLEmbedder(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )
        self._dimension = 2048 if "2B" in model_name else 4096

    @property
    def dimension(self) -> int:
        return self._dimension

    @staticmethod
    def _sample_video_frames(path: str, *, max_frames: int = 8) -> list[Any]:
        """Decode a bounded frame sequence without qwen-vl-utils' batch-wide fallback."""
        import cv2
        from PIL import Image

        decoder = cv2.VideoCapture(path)
        try:
            if not decoder.isOpened():
                raise RuntimeError("OpenCV could not open the video")
            total = int(decoder.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                raise RuntimeError("video reports no frames")
            samples = min(max_frames, total)
            # Qwen-VL video preprocessing requires at least two frames. Duplicate
            # a single-frame clip rather than failing the complete ingest batch.
            # Container metadata commonly reports one non-decodable sentinel
            # frame at ``total - 1``. Sample through the last reliable candidate
            # and retry preceding positions before declaring the video broken.
            last_candidate = max(0, total - 2)
            positions = np.linspace(0, last_candidate, num=max(2, samples), dtype=int)
            frames: list[Any] = []
            for position in positions:
                ok = False
                frame = None
                for candidate in range(int(position), max(-1, int(position) - 4), -1):
                    decoder.set(cv2.CAP_PROP_POS_FRAMES, candidate)
                    ok, frame = decoder.read()
                    if ok:
                        break
                if not ok or frame is None:
                    raise RuntimeError(f"failed reading frame near {position}/{total}")
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            return frames
        finally:
            decoder.release()

    def encode_units(self, units: Sequence[Mapping[str, Any]]) -> np.ndarray:
        values: list[dict[str, Any]] = []
        for unit in units:
            value: dict[str, Any] = {}
            if unit.get("text"):
                value["text"] = str(unit["text"])
            if unit.get("image"):
                value["image"] = str(unit["image"])
            if unit.get("video"):
                video = str(unit["video"])
                try:
                    value["video"] = self._sample_video_frames(video, max_frames=8)
                except Exception as exc:
                    # A media record also has a caption/timestamp text unit when
                    # the benchmark provides one. Keep this row aligned but make
                    # the failed raw-video proxy deliberately low-information so
                    # one corrupt asset cannot abort the entire context.
                    warnings.warn(
                        f"VimRAG video embedding fell back to text for {video}: "
                        f"{type(exc).__name__}: {exc}",
                        RuntimeWarning,
                    )
                    value["text"] = "unavailable video proxy source_id=" + str(
                        unit.get("source_id", Path(video).stem)
                    )
            values.append(value)
        return np.asarray(
            self.model.process(values, normalize=True).detach().float().cpu().numpy(),
            dtype=np.float32,
        )


class ConcreteVimRAGMethod(BaseMemoryMethod):
    """Official VimRAG inference graph over a canonical context-local FAISS corpus."""

    capabilities = MethodCapabilities(
        memory_modalities=frozenset({"text", "image", "video", "document", "table"}),
        query_modalities=frozenset({"text", "image", "video"}),
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
        official_repo: str | Path = "sources/vimrag",
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 100,
        ingest_batch_size: int = 16,
        max_steps: int = 20,
        video_frames: int = 8,
        agent_factory: Callable[..., VimRAGAgent] = official_vimrag_agent_factory,
        embedding_name: str = "unknown",
    ) -> None:
        super().__init__(generation)
        if (
            checkpoint_interval <= 0
            or ingest_batch_size <= 0
            or max_steps <= 0
            or video_frames <= 0
        ):
            raise ValueError(
                "VimRAG checkpoint interval, batch size, max steps, and video frames "
                "must be positive"
            )
        self.answer_model = answer_model or OpenAICompatibleQwenVL(self.generation)
        self.embedder = embedder
        self.index = index or FaissFlatIPIndex(embedder.dimension)
        self.official_repo = Path(official_repo).resolve()
        self.checkpoint_dir = Path(checkpoint_dir).resolve() if checkpoint_dir else None
        self.checkpoint_interval = checkpoint_interval
        self.ingest_batch_size = ingest_batch_size
        self.max_steps = max_steps
        self.video_frames = video_frames
        self.agent_factory = agent_factory
        self.embedding_name = embedding_name
        self._context_id = ""
        self.units: list[dict[str, Any]] = []
        self._pending_units: list[dict[str, Any]] = []
        self._pending_memory_ids: list[str] = []
        self._processed_memory_ids: set[str] = set()
        self._commits_since_checkpoint = 0

    def _begin_context(self, context: Mapping[str, Any]) -> None:
        self._context_id = str(context.get("context_id", "context"))
        self.units = []
        self._pending_units = []
        self._pending_memory_ids = []
        self._processed_memory_ids = set()
        self._commits_since_checkpoint = 0
        self.index.reset()
        self._load_checkpoint()

    def _end_context(self) -> None:
        self._flush_pending()
        self._save_checkpoint(force=True)
        self.units = []
        self.index.reset()

    def _abort_context(self) -> None:
        self._pending_units = []
        self._pending_memory_ids = []
        self.units = []
        self.index.reset()

    def _ingest(self, memory: Mapping[str, Any]) -> None:
        memory_id = str(memory["memory_id"])
        if memory_id in self._processed_memory_ids:
            return
        self._pending_units.extend(self._memory_units(memory))
        self._pending_memory_ids.append(memory_id)
        if len(self._pending_units) >= self.ingest_batch_size:
            self._flush_pending()

    def _synchronize_memory(self) -> None:
        self._flush_pending()
        # Queries can fail after ingestion. Persist the final sub-interval batch
        # before answering so recovery never re-embeds the tail of a context.
        self._save_checkpoint(force=True)

    def _flush_pending(self) -> None:
        if not self._pending_units:
            return
        values = self._pending_units
        embeddings = self.embedder.encode_units(values)
        if len(embeddings) != len(values):
            raise RuntimeError("VimRAG embedder returned a different number of rows")
        self.index.add(embeddings)
        self.units.extend(values)
        self._processed_memory_ids.update(self._pending_memory_ids)
        self._commits_since_checkpoint += len(self._pending_memory_ids)
        self._pending_units = []
        self._pending_memory_ids = []
        self._save_checkpoint()

    def _memory_units(self, memory: Mapping[str, Any]) -> list[dict[str, Any]]:
        parts = list(memory.get("content", []))
        prefix = " ".join(
            f"{key}={memory[key]}"
            for key in ("source_id", "session_id", "speaker", "timestamp", "sequence")
            if memory.get(key) is not None
        )
        body = text_from_parts(parts)
        metadata = memory.get("metadata", {})
        derived = metadata.get("derived", {}) if isinstance(metadata, Mapping) else {}
        media_text: list[str] = []
        if isinstance(derived, Mapping):
            for label, key in (
                ("Caption", "caption"),
                ("Short caption", "short_caption"),
                ("OCR", "ocr_text"),
            ):
                value = derived.get(key)
                if value:
                    media_text.append(f"{label}: {value}")
            image_captions = derived.get("image_captions")
            if isinstance(image_captions, list):
                media_text.extend(
                    f"Image caption {index}: {value}"
                    for index, value in enumerate(image_captions, 1)
                    if value
                )
        body = "\n".join(value for value in (body, *media_text) if value)
        text = "\n".join(value for value in (prefix, body) if value)
        common = {
            "memory_id": str(memory["memory_id"]),
            "source_id": memory.get("source_id"),
        }
        units: list[dict[str, Any]] = []
        if text:
            units.append({"type": "text", "content": text, "text": text, **common})
        for part in parts:
            kind = part.get("type")
            if kind == "image":
                units.append(
                    {
                        "type": "image",
                        "file_path": str(part["path"]),
                        "image": str(part["path"]),
                        **common,
                    }
                )
            elif kind == "video":
                units.append(
                    {
                        "type": "video",
                        "file_path": str(part["path"]),
                        "video": str(part["path"]),
                        **common,
                    }
                )
            elif kind in {"document", "table"} and not part.get("text"):
                rendered = f"{kind}: {part.get('path', '')}"
                units.append({"type": "text", "content": rendered, "text": rendered, **common})
            elif kind == "audio":
                raise NotImplementedError("VimRAG has no audio retrieval path")
        if not units:
            raise RuntimeError(f"memory {memory['memory_id']} produced no VimRAG corpus unit")
        return units

    def _search(self, query: str, top_k: int) -> dict[str, Any]:
        self._flush_pending()
        if not self.units:
            return {"score": [], "indice": [], "data": []}
        query_embedding = self.embedder.encode_units([{"text": query}])
        scores, rows = self.index.search(query_embedding, top_k)
        ids = [int(value) for value in rows[0]]
        return {
            "score": [float(value) for value in scores[0]],
            "indice": ids,
            "data": [self.units[row] for row in ids],
        }

    @staticmethod
    def _query_media(
        question: Mapping[str, Any], *, video_frames: int
    ) -> list[dict[str, Any]]:
        media: list[dict[str, Any]] = []
        for part in question.get("prompt", []):
            if part.get("type") == "image":
                media.append({"type": "image", "image": str(part["path"])})
            elif part.get("type") == "video":
                media.append(
                    {
                        "type": "video",
                        "video": str(part["path"]),
                        "fps": 1.0,
                        "max_frames": video_frames,
                    }
                )
        return media

    def _answer(self, question: Mapping[str, Any]) -> MethodResult:
        self._flush_pending()
        agent = self.agent_factory(
            official_repo=self.official_repo,
            answer_model=self.answer_model,
            generation=self.generation,
            search=self._search,
            query_media=self._query_media(question, video_frames=self.video_frames),
            search_top_k=self.generation.top_k,
            memory_top_k=self.generation.top_k,
            max_steps=self.max_steps,
            video_frames=self.video_frames,
        )
        query = question_text(question)
        instruction = str(question.get("instruction", "")).strip()
        if instruction:
            query = f"{query}\n\nAnswer requirements: {instruction}"
        sample = {"query": query}
        errors: list[str] = []
        answer = None
        completed_sample: Mapping[str, Any] = {}
        repeated_error = ""
        repeated_error_count = 0
        events = agent.run(sample)
        try:
            for event in events:
                if event.get("event") == "answer":
                    answer = str(event.get("content", ""))
                    completed_sample = event.get("sample", {})
                    break
                if event.get("event") != "error":
                    continue
                detail = str(event.get("content", ""))
                errors.append(detail)

                # The official demo treats every exception as recoverable and
                # advances to another agent step. A context-window violation is
                # deterministic for the unchanged message history, so retrying
                # it up to max_steps only repeats the same expensive request.
                if "input context has " in detail and "hard limit is " in detail:
                    raise RuntimeError(f"VimRAG context window exceeded: {detail}")

                if detail == repeated_error:
                    repeated_error_count += 1
                else:
                    repeated_error = detail
                    repeated_error_count = 1
                if repeated_error_count >= 3:
                    raise RuntimeError(
                        "VimRAG aborted after three identical agent errors: " + detail
                    )
        finally:
            close = getattr(events, "close", None)
            if callable(close):
                close()
        if answer is None:
            detail = errors[-1] if errors else "maximum reasoning steps reached"
            raise RuntimeError(f"VimRAG failed to produce an answer: {detail}")
        retrieved: list[str] = []
        for result in completed_sample.get("search_results", []):
            for unit in result.get("data", []):
                memory_id = str(unit.get("memory_id", ""))
                if memory_id and memory_id not in retrieved:
                    retrieved.append(memory_id)
        return MethodResult(
            answer,
            {
                "retrieved_memory_ids": retrieved,
                "vimrag_steps": int(completed_sample.get("generate_times", 0)) + 1,
                "vimrag_searches": len(completed_sample.get("search_results", [])),
                "vimrag_parse_repairs": int(getattr(agent, "parse_repairs", 0)),
                "vimrag_adapter": "official-demo-inference/qwen3vl8b-local",
            },
        )

    def _checkpoint_config(self) -> dict[str, Any]:
        return {
            "format": "mmmb-vimrag-checkpoint-1",
            "embedding_name": self.embedding_name,
            "embedding_dimension": self.embedder.dimension,
        }

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
        index_name = f"index-{len(self._processed_memory_ids):08d}.faiss"
        index_path = state_path.parent / index_name
        temporary_index = index_path.with_suffix(".tmp")
        self.index.save(temporary_index)
        os.replace(temporary_index, index_path)
        state = {
            "config": self._checkpoint_config(),
            "context_id": self._context_id,
            "processed_memory_ids": sorted(self._processed_memory_ids),
            "units": self.units,
            "index_file": index_name,
        }
        temporary_state = state_path.with_suffix(".tmp")
        temporary_state.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary_state, state_path)
        for old in state_path.parent.glob("index-*.faiss"):
            if old.name != index_name:
                old.unlink()
        self._commits_since_checkpoint = 0

    def _load_checkpoint(self) -> None:
        state_path = self._state_path()
        if state_path is None or not state_path.is_file():
            return
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config") != self._checkpoint_config():
            raise RuntimeError(f"incompatible VimRAG checkpoint: {state_path}")
        if state.get("context_id") != self._context_id:
            raise RuntimeError(f"VimRAG checkpoint belongs to another context: {state_path}")
        self.units = [dict(unit) for unit in state.get("units", [])]
        self._processed_memory_ids = {
            str(value) for value in state.get("processed_memory_ids", [])
        }
        self.index.load(state_path.parent / str(state["index_file"]))
        if self.index.size != len(self.units):
            raise RuntimeError("VimRAG checkpoint index/unit row mismatch")
