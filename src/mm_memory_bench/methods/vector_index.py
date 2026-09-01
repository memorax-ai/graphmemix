from __future__ import annotations

from typing import Protocol
from pathlib import Path

import numpy as np


class VectorIndex(Protocol):
    @property
    def size(self) -> int: ...

    def add(self, vectors: np.ndarray) -> None: ...

    def update(self, row_id: int, vector: np.ndarray) -> None: ...

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]: ...

    def reset(self) -> None: ...

    def save(self, path: str | Path) -> None: ...

    def load(self, path: str | Path) -> None: ...


def normalized_float32(vectors: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(vectors, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError(f"embeddings must be rank 2, got shape={value.shape}")
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-length embedding cannot be cosine-normalized")
    return value / norms


class FaissFlatIPIndex:
    """Exact cosine search using FAISS IndexFlatIP.

    Vectors are normalized before add/search, so inner product equals cosine
    similarity. FAISS owns contiguous storage; no repeated concatenation occurs.
    """

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "FAISS is required; install the 'faiss-cpu' runtime dependency"
            ) from exc
        self.dimension = dimension
        self._faiss = faiss
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        self._next_id = 0

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    def add(self, vectors: np.ndarray) -> None:
        value = normalized_float32(vectors)
        if value.shape[1] != self.dimension:
            raise ValueError(f"expected dimension {self.dimension}, got {value.shape[1]}")
        ids = np.arange(self._next_id, self._next_id + len(value), dtype=np.int64)
        self._index.add_with_ids(value, ids)
        self._next_id += len(value)

    def update(self, row_id: int, vector: np.ndarray) -> None:
        value = normalized_float32(vector)
        if value.shape != (1, self.dimension):
            raise ValueError(f"update expects shape (1, {self.dimension}), got {value.shape}")
        removed = self._index.remove_ids(np.asarray([row_id], dtype=np.int64))
        if removed != 1:
            raise KeyError(f"FAISS row id does not exist: {row_id}")
        self._index.add_with_ids(value, np.asarray([row_id], dtype=np.int64))

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        value = normalized_float32(queries)
        if value.shape[1] != self.dimension:
            raise ValueError(f"expected dimension {self.dimension}, got {value.shape[1]}")
        k = min(top_k, self.size)
        if k == 0:
            shape = (value.shape[0], 0)
            return np.empty(shape, np.float32), np.empty(shape, np.int64)
        scores, indices = self._index.search(value, k)
        return scores, indices

    def reset(self) -> None:
        self._index.reset()
        self._next_id = 0

    def save(self, path: str | Path) -> None:
        self._faiss.write_index(self._index, str(path))

    def load(self, path: str | Path) -> None:
        index = self._faiss.read_index(str(path))
        if index.d != self.dimension:
            raise ValueError(f"expected saved dimension {self.dimension}, got {index.d}")
        self._index = index
        self._next_id = int(index.ntotal)
