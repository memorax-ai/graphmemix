from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class GenerationConfig:
    """Shared answer-model policy, independent of a memory algorithm."""

    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    base_url: str = "http://127.0.0.1:8091/v1"
    temperature: float = 0.0
    top_k: int = 10
    max_model_len: int = 32768
    max_output_tokens: int = 1000
    overflow_policy: str = "error"
    reasoning_effort: str = "minimal"

    def __post_init__(self) -> None:
        if self.temperature != 0:
            raise ValueError("the canonical deterministic track requires temperature=0")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.overflow_policy != "error":
            raise ValueError("canonical runs must fail visibly instead of silently truncating")
        if self.reasoning_effort not in {
            "none", "minimal", "low", "medium", "high", "xhigh",
        }:
            raise ValueError("unsupported reasoning_effort")


@dataclass(frozen=True)
class MethodCapabilities:
    """Descriptive capabilities used for preflight checks, not data conversion."""

    memory_modalities: frozenset[str]
    query_modalities: frozenset[str]
    supports_tools: bool = True
    native_multimodal_retrieval: bool = False
    method_owned_media_processing: bool = True


@dataclass(frozen=True)
class MethodResult:
    """Only prediction is required for formal correctness evaluation."""

    prediction: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class BaseMemoryMethod(ABC):
    """State-safe base class shared by memory-method implementations.

    Canonical records remain lossless and benchmark-neutral. Subclasses own all
    algorithmic transformations, including captioning, video frame sampling,
    note construction, embedding, indexing, evidence selection, and prompting.
    """

    capabilities: MethodCapabilities

    def __init__(self, generation: GenerationConfig | None = None) -> None:
        self.generation = generation or GenerationConfig()
        self._active = False

    def begin_context(self, context: Mapping[str, Any]) -> None:
        if self._active:
            raise RuntimeError("begin_context called before the previous context ended")
        self._active = True
        try:
            self._begin_context(context)
        except BaseException:
            self._active = False
            raise

    def ingest(self, memory: Mapping[str, Any]) -> None:
        self._require_active("ingest")
        self._ingest(memory)

    def answer(self, question: Mapping[str, Any]) -> str | Mapping[str, Any]:
        self._require_active("answer")
        result = self._answer(question)
        if isinstance(result, MethodResult):
            # Diagnostics are deliberately opt-in and are never judge input.
            return {"prediction": result.prediction, **dict(result.diagnostics)}
        return result

    def synchronize_memory(self) -> None:
        """Wait until all previously submitted memory mutations are committed."""
        self._require_active("synchronize_memory")
        self._synchronize_memory()

    def end_context(self) -> None:
        self._require_active("end_context")
        try:
            self._end_context()
        finally:
            self._active = False

    def abort_context(self) -> None:
        """Release an active context after an operation failed.

        Implementations whose normal ``end_context`` commits buffered work can
        override ``_abort_context`` so exception cleanup never retries the
        failed mutation and masks the original error.
        """
        self._require_active("abort_context")
        try:
            self._abort_context()
        finally:
            self._active = False

    def _require_active(self, operation: str) -> None:
        if not self._active:
            raise RuntimeError(f"{operation} requires an active context")

    @abstractmethod
    def _begin_context(self, context: Mapping[str, Any]) -> None: ...

    @abstractmethod
    def _ingest(self, memory: Mapping[str, Any]) -> None: ...

    @abstractmethod
    def _answer(
        self, question: Mapping[str, Any]
    ) -> str | Mapping[str, Any] | MethodResult: ...

    @abstractmethod
    def _end_context(self) -> None: ...

    def _abort_context(self) -> None:
        # Preserve the historical cleanup behavior for methods that have not
        # opted into a distinct rollback path. The harness still preserves the
        # primary exception if this cleanup also fails.
        self._end_context()

    def _synchronize_memory(self) -> None:
        """Optional timing barrier for asynchronous concrete methods."""
