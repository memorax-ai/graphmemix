"""Keep source speaker identity visible to method adapters."""

from copy import deepcopy

import pytest

from mm_memory_bench.benchmarks.reader import BundleReader
from mm_memory_bench.methods.concrete_lightmem import ConcreteLightMemMethod
from mm_memory_bench.runner.benchmark import _resolved_memory


@pytest.fixture
def reader(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "assets.jsonl").write_text("")
    return BundleReader(tmp_path)


@pytest.mark.parametrize("memory_view", ["raw", "derived", "raw_derived"])
@pytest.mark.parametrize(
    ("identity", "expected_identity"),
    [
        ({"speaker": "Alice", "role": "user"}, {"speaker": "Alice", "role": "user"}),
        ({"speaker": "Alice"}, {"speaker": "Alice"}),
        ({"role": "user"}, {"speaker": "user", "role": "user"}),
        ({"speaker": "", "role": "assistant"}, {"speaker": "assistant", "role": "assistant"}),
        ({"speaker": None, "role": "user"}, {"speaker": "user", "role": "user"}),
        ({}, {}),
        ({"speaker": "", "role": None}, {"speaker": "", "role": None}),
    ],
)
def test_resolved_memory_preserves_source_and_prefers_speaker(
    reader, memory_view, identity, expected_identity
):
    memory = {
        "memory_id": "m1",
        "context_id": "c1",
        "content": [{"type": "text", "text": "I like blue."}],
        **identity,
    }
    original = deepcopy(memory)

    resolved = _resolved_memory(reader, memory, memory_view=memory_view)

    assert memory == original
    assert {key: resolved[key] for key in ("speaker", "role") if key in resolved} == expected_identity
    assert resolved["memory_id"] == "m1"
    assert resolved["content"] == original["content"]


def test_role_only_memories_remain_distinct_in_method_input(reader):
    method = ConcreteLightMemMethod.__new__(ConcreteLightMemMethod)
    memory = {
        "memory_id": "m1",
        "context_id": "c1",
        "content": [{"type": "text", "text": "I like blue."}],
    }
    user = _resolved_memory(reader, {**memory, "role": "user"})
    assistant = _resolved_memory(reader, {**memory, "role": "assistant"})

    user_text = method._memory_text(user)
    assistant_text = method._memory_text(assistant)

    assert user_text != assistant_text
    assert "user" in user_text
    assert "assistant" in assistant_text
