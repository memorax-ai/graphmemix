from __future__ import annotations

from abc import abstractmethod
from typing import Any, Mapping, Sequence

from .base import BaseMemoryMethod, MethodCapabilities, MethodResult


UNIVERSALRAG_CORPORA = ("paragraph", "document", "table", "image", "clip", "video")


class UniversalRAGMethod(BaseMemoryMethod):
    """UniversalRAG family boundary: route first, then retrieve per corpus.

    Unlike a unified multimodal index, every modality/granularity corpus owns an
    independent embedding space and index. Concrete adapters also own text
    chunking, document grouping, image handling, and video/clip construction.
    """

    capabilities = MethodCapabilities(
        memory_modalities=frozenset({"text", "image", "video", "document", "table"}),
        query_modalities=frozenset({"text", "image", "video", "document", "table"}),
        native_multimodal_retrieval=True,
    )

    def _ingest(self, memory: Mapping[str, Any]) -> None:
        self.index_memory(memory)

    def _answer(self, question: Mapping[str, Any]) -> MethodResult:
        routes = self.route(question)
        evidence = self.retrieve_routed(question, routes=routes, top_k=self.generation.top_k)
        return MethodResult(
            prediction=self.generate_answer(question, evidence),
            diagnostics={
                "retrieved_memory_ids": [item["memory_id"] for item in evidence],
                "routes": list(routes),
                "retrieved_corpora": [item.get("corpus") for item in evidence],
                "retrieval_scores": [item.get("score") for item in evidence],
            },
        )

    @abstractmethod
    def index_memory(self, memory: Mapping[str, Any]) -> None: ...

    @abstractmethod
    def route(self, question: Mapping[str, Any]) -> Sequence[str]: ...

    @abstractmethod
    def retrieve_routed(
        self,
        question: Mapping[str, Any],
        *,
        routes: Sequence[str],
        top_k: int,
    ) -> Sequence[Mapping[str, Any]]: ...

    @abstractmethod
    def generate_answer(
        self,
        question: Mapping[str, Any],
        evidence: Sequence[Mapping[str, Any]],
    ) -> str: ...
