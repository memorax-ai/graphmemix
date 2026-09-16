from __future__ import annotations

import json
import weakref
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from mm_memory_bench.methods.concrete_lightmem import (
    ConcreteLightMemMethod,
    _OfficialLightMemBackend,
)


class LocalRetriever:
    """Exercise storage ownership with a real client, without model dependencies."""

    def __init__(self, path):
        qdrant = pytest.importorskip("qdrant_client")
        self.client = qdrant.QdrantClient(path=str(path))
        from qdrant_client.http import models
        self.models = models
        if not self.client.collection_exists("facts"):
            self.client.create_collection(
                "facts", vectors_config=self.models.VectorParams(
                    size=1, distance=self.models.Distance.COSINE,
                ),
            )

    def insert(self, *, vectors, payloads, ids):
        self.client.upsert("facts", points=[
            self.models.PointStruct(id=pid, vector=vector, payload=payload)
            for pid, vector, payload in zip(ids, vectors, payloads)
        ])

    def scroll(self, *, scroll_filter, **kwargs):
        filters = self.models.Filter(must=[
            self.models.FieldCondition(key=key, match=self.models.MatchValue(value=value))
            for key, value in scroll_filter.items()
        ])
        return self.client.scroll("facts", scroll_filter=filters, **kwargs)

    def delete(self, point_id):
        self.client.delete("facts", points_selector=[point_id])


def make_method(tmp_path):
    retrievers = []

    def factory(*, qdrant_path, **kwargs):
        retriever = LocalRetriever(qdrant_path)
        retrievers.append(retriever)
        return _OfficialLightMemBackend(SimpleNamespace(embedding_retriever=retriever))

    method = ConcreteLightMemMethod(checkpoint_dir=tmp_path, backend_factory=factory)
    method.begin_context({"context_id": "c1"})
    return method, retrievers[0]


@pytest.mark.parametrize("exit_method", ["end_context", "abort_context"])
def test_context_exit_releases_real_qdrant_lock(tmp_path, exit_method):
    method, retriever = make_method(tmp_path)
    backend = method.backend  # Keep it alive: releasing the lock must be explicit.
    try:
        getattr(method, exit_method)()
        reopened = LocalRetriever(method._context_root() / "qdrant")
        reopened.client.close()
        assert method.backend is None
        assert not method._active
        backend.close()
    finally:
        retriever.client.close()


def test_close_restores_insert_and_releases_backend_without_gc():
    client = SimpleNamespace(close=Mock())
    retriever = SimpleNamespace(client=client, insert=Mock())
    original_insert = retriever.insert
    backend = _OfficialLightMemBackend(SimpleNamespace(embedding_retriever=retriever))
    backend._sources["source"] = {"source_memory_id": "m1"}
    ref = weakref.ref(backend)
    backend.close()
    backend.close()
    assert retriever.insert is original_insert
    assert backend._sources == {}
    client.close.assert_called_once_with()
    del backend
    assert ref() is None


def test_close_failure_still_removes_wrapper_and_can_be_retried():
    client = SimpleNamespace(close=Mock(side_effect=[RuntimeError("close failed"), None]))
    retriever = SimpleNamespace(client=client, insert=Mock())
    original_insert = retriever.insert
    backend = _OfficialLightMemBackend(SimpleNamespace(embedding_retriever=retriever))
    backend._sources["source"] = {"source_memory_id": "m1"}
    with pytest.raises(RuntimeError, match="close failed"):
        backend.close()
    assert retriever.insert is original_insert
    assert not backend._sources
    backend.close()
    backend.close()
    assert client.close.call_count == 2


def test_close_restores_class_method_without_creating_retriever_cycle():
    class Retriever:
        client = SimpleNamespace(close=Mock())

        def insert(self, **kwargs):
            pass

    retriever = Retriever()
    backend = _OfficialLightMemBackend(SimpleNamespace(embedding_retriever=retriever))
    ref = weakref.ref(retriever)
    backend.close()
    assert "insert" not in vars(retriever)
    del backend, retriever
    assert ref() is None


@pytest.mark.parametrize("failing_stage", ["_flush_pending", "_save_state"])
def test_end_failure_still_releases_real_qdrant_lock(tmp_path, monkeypatch, failing_stage):
    method, retriever = make_method(tmp_path)
    business_error = RuntimeError("business failure")
    monkeypatch.setattr(method, failing_stage, Mock(side_effect=business_error))
    try:
        with pytest.raises(RuntimeError) as caught:
            method.end_context()
        assert caught.value is business_error
        reopened = LocalRetriever(method._context_root() / "qdrant")
        reopened.client.close()
        assert method.backend is None
        assert not method._active
    finally:
        retriever.client.close()


@pytest.mark.parametrize("failing_stage", ["_flush_pending", "_save_state"])
def test_end_cleanup_error_does_not_replace_business_error(tmp_path, monkeypatch, failing_stage):
    backend = SimpleNamespace(close=Mock(side_effect=RuntimeError("close failed")))
    method = ConcreteLightMemMethod(checkpoint_dir=tmp_path, backend_factory=lambda **_: backend)
    method.begin_context({"context_id": "c1"})
    business_error = ValueError("business failure")
    monkeypatch.setattr(method, failing_stage, Mock(side_effect=business_error))
    with pytest.raises(ValueError) as caught:
        method.end_context()
    assert caught.value is business_error
    if hasattr(business_error, "add_note"):
        assert any("close failed" in note for note in business_error.__notes__)
    backend.close.assert_called_once_with()
    assert method.backend is None
    assert not method._pending


@pytest.mark.parametrize("exit_method", ["end_context", "abort_context"])
def test_cleanup_failure_is_visible_and_detaches_context(tmp_path, exit_method):
    backend = SimpleNamespace(close=Mock(side_effect=RuntimeError("close failed")))
    method = ConcreteLightMemMethod(checkpoint_dir=tmp_path, backend_factory=lambda **_: backend)
    method.begin_context({"context_id": "c1"})
    with pytest.raises(RuntimeError, match="close failed"):
        getattr(method, exit_method)()
    assert method.backend is None
    assert not method._active
    assert not method._pending


def test_normal_end_flushes_and_saves_before_close(tmp_path, monkeypatch):
    events = []
    backend = SimpleNamespace(close=lambda: events.append("close"))
    method = ConcreteLightMemMethod(checkpoint_dir=tmp_path, backend_factory=lambda **_: backend)
    method.begin_context({"context_id": "c1"})

    def flush():
        events.append("flush")
        method._processed_memory_ids.add("m1")

    def close():
        assert json.loads(method._state_path().read_text())["processed_memory_ids"] == ["m1"]
        events.append("close")

    backend.close = close
    monkeypatch.setattr(method, "_flush_pending", flush)
    method.end_context()
    assert events == ["flush", "close"]


def test_abort_discards_pending_without_retrying(tmp_path, monkeypatch):
    backend = SimpleNamespace(close=Mock())
    method = ConcreteLightMemMethod(checkpoint_dir=tmp_path, backend_factory=lambda **_: backend)
    method.begin_context({"context_id": "c1"})
    method.ingest({"memory_id": "m1", "content": [{"type": "text", "text": "blue"}]})
    flush = Mock(side_effect=AssertionError("abort must not flush"))
    monkeypatch.setattr(method, "_flush_pending", flush)
    method.abort_context()
    flush.assert_not_called()
    backend.close.assert_called_once_with()
    assert not method._pending
    assert not method._state_path().exists()


def test_provenance_and_batch_rollback_survive_close_and_reopen(tmp_path):
    retriever = LocalRetriever(tmp_path / "qdrant")

    def add(messages, **kwargs):
        # Stand in for upstream source_id resolution; the real client persists
        # the payload produced by our actual provenance insertion wrapper.
        for message in reversed(messages):
            retriever.insert(vectors=[[0.25]], ids=[str(uuid4())], payloads=[{
                "speaker_id": message["speaker_id"], "memory": "extracted fact",
            }])

    backend = _OfficialLightMemBackend(SimpleNamespace(
        embedding_retriever=retriever, add_memory=add,
    ))
    try:
        for batch, ids in [("keep", ["m1", "m2"]), ("failed", ["m3"])]:
            backend.add_memory([
                dict(canonical_memory_id=mid, ingest_batch_id=batch, speaker_id="Alice")
                for mid in ids
            ])
        assert backend.source_memory_ids_for_batch("keep") == {"m1", "m2"}
        backend.rollback_batch("failed")
        assert backend.source_memory_ids_for_batch("failed") == set()
        backend.close()
        reopened = LocalRetriever(tmp_path / "qdrant")
        try:
            rows, _ = reopened.scroll(scroll_filter={"ingest_batch_id": "keep"})
            assert {row.payload["source_memory_id"] for row in rows} == {"m1", "m2"}
            for row in rows:
                assert row.payload["speaker_id"] == "Alice"
                assert row.payload["memory"] == (
                    f"[memory_id={row.payload['source_memory_id']}]\nextracted fact"
                )
            assert reopened.client.count("facts").count == 2
        finally:
            reopened.client.close()
    finally:
        retriever.client.close()
