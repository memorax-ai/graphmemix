from __future__ import annotations

from abc import abstractmethod
from typing import Any, Mapping, Sequence

from .base import BaseMemoryMethod, MethodCapabilities, MethodResult


class AMemMethod(BaseMemoryMethod):
    """A-Mem family boundary.

    Media-to-note conversion belongs here because A-Mem reasons over evolving
    structured notes. A concrete implementation may use Qwen3-VL to describe
    raw media before creating/linking/evolving its notes.
    """

    capabilities = MethodCapabilities(
        memory_modalities=frozenset({"text", "image", "video", "audio", "document", "table"}),
        query_modalities=frozenset({"text", "image", "video", "document", "table"}),
        native_multimodal_retrieval=False,
    )

    def _ingest(self, memory: Mapping[str, Any]) -> None:
        notes = self.memory_to_notes(memory)
        self.add_and_evolve(notes, source_memory=memory)

    def _answer(self, question: Mapping[str, Any]) -> MethodResult:
        notes = self.search_notes(question, top_k=self.generation.top_k)
        return MethodResult(prediction=self.generate_answer(question, notes))

    @abstractmethod
    def memory_to_notes(self, memory: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        """Perform A-Mem-owned text/media understanding and note construction."""

    @abstractmethod
    def add_and_evolve(
        self,
        notes: Sequence[Mapping[str, Any]],
        *,
        source_memory: Mapping[str, Any],
    ) -> None: ...

    @abstractmethod
    def search_notes(
        self, question: Mapping[str, Any], *, top_k: int
    ) -> Sequence[Mapping[str, Any]]: ...

    @abstractmethod
    def generate_answer(
        self,
        question: Mapping[str, Any],
        notes: Sequence[Mapping[str, Any]],
    ) -> str: ...
