"""Benchmark-neutral interfaces for memory-system implementations."""

from .base import (
    BaseMemoryMethod,
    GenerationConfig,
    MethodCapabilities,
    MethodResult,
)
from .amem import AMemMethod
from .concrete_amem import ConcreteAMemMethod
from .gme import GMEQwen2VLEmbedder
from .concrete_memguide import ConcreteMemGuideMethod, NVEmbedV2TextEmbedder
from ..preprocessing.captions import public_captions
from .concrete_lightmem import ConcreteLightMemMethod
from .concrete_vimrag import (
    ConcreteVimRAGMethod,
    Qwen3VLVimRAGEmbedder,
    official_vimrag_agent_factory,
)
from .concrete_memix import ConcreteMemixMethod
from .universalrag import UNIVERSALRAG_CORPORA, UniversalRAGMethod
from .concrete_universalrag import (
    ConcreteUniversalRAGMethod,
    MemGalleryUniversalRouter,
    OfficialQwen3TextEmbedder,
    OfficialVLM2VecEmbedder,
    OpenAIUniversalRouter,
    normalize_routes,
)
from .vector_index import FaissFlatIPIndex

__all__ = [
    "AMemMethod",
    "BaseMemoryMethod",
    "ConcreteAMemMethod",
    "ConcreteMemGuideMethod",
    "ConcreteLightMemMethod",
    "ConcreteVimRAGMethod",
    "ConcreteMemixMethod",
    "NVEmbedV2TextEmbedder",
    "ConcreteUniversalRAGMethod",
    "MemGalleryUniversalRouter",
    "FaissFlatIPIndex",
    "GenerationConfig",
    "MethodCapabilities",
    "MethodResult",
    "OpenAIUniversalRouter",
    "OfficialQwen3TextEmbedder",
    "OfficialVLM2VecEmbedder",
    "UNIVERSALRAG_CORPORA",
    "UniversalRAGMethod",
    "GMEQwen2VLEmbedder",
    "normalize_routes",
    "public_captions",
    "official_vimrag_agent_factory",
    "Qwen3VLVimRAGEmbedder",
]
