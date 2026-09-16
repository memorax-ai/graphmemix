from types import SimpleNamespace

import pytest

from mm_memory_bench.cli import main
from mm_memory_bench.methods.backends import OpenAICompatibleQwenVL
from mm_memory_bench.methods.base import GenerationConfig


@pytest.mark.parametrize("extra,expected", [([], 32768), (["--reader-max-model-len", "65536"], 65536)])
def test_run_method_passes_reader_limit_without_changing_router(monkeypatch, tmp_path, extra, expected):
    import mm_memory_bench.methods as methods
    import mm_memory_bench.runner.benchmark as runner
    captured = {}

    def factory(generation, **kwargs):
        captured["reader"] = generation
        captured["router"] = kwargs["router"].answer_model.config
        return object()

    monkeypatch.setattr(methods, "OfficialQwen3TextEmbedder", lambda *a, **kw: SimpleNamespace(dimension=1))
    monkeypatch.setattr(methods, "OfficialVLM2VecEmbedder", lambda *a, **kw: SimpleNamespace(dimension=1))
    monkeypatch.setattr(methods, "ConcreteUniversalRAGMethod", factory)
    monkeypatch.setattr(runner, "run_bundle", lambda *a, **kw: {})
    assert main(["run-method", "universalrag", str(tmp_path), "--output", str(tmp_path / "out"), *extra]) == 0
    assert captured["reader"].max_model_len == expected
    assert captured["reader"].max_output_tokens == 1000
    assert captured["router"].max_model_len == 32768


def test_nonpositive_reader_limit_fails_before_model_loading(tmp_path):
    assert main(["run-method", "universalrag", str(tmp_path), "--output", str(tmp_path / "out"),
                 "--reader-max-model-len", "0"]) == 2


@pytest.mark.parametrize("count,budget", [(43754, 1000), (65000, 536)])
def test_large_reader_input_is_preserved_and_output_fits_window(monkeypatch, count, budget):
    calls = []
    messages = [{"role": "user", "content": "original full evidence"}]

    def transport(route, payload):
        calls.append(payload)
        return {"choices": [{"message": {"content": "A"}}]}

    model = OpenAICompatibleQwenVL(GenerationConfig(max_model_len=65536), transport=transport)
    monkeypatch.setattr(model, "count_input_tokens", lambda *args: count)
    assert model.complete(messages) == "A"
    assert calls[0]["messages"] == messages
    assert calls[0]["max_tokens"] == budget
