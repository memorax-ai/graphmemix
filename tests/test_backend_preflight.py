from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from mm_memory_bench.methods.backends import (
    OpenAICompatibleQwenVL,
    estimate_multimodal_input_tokens,
)
from mm_memory_bench.methods.base import GenerationConfig


def image_url(width: int, height: int) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_local_multimodal_estimate_uses_patch_dimensions() -> None:
    messages = [{
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": image_url(720, 480)}}],
    }]
    # ceil(720/32) * ceil(480/32) = 345 patches; 1.25 safety multiplier.
    assert estimate_multimodal_input_tokens(messages) == 16 + 432


def test_hosted_auto_mode_skips_private_tokenize_route() -> None:
    routes: list[str] = []

    def transport(route, payload):
        routes.append(route)
        return {
            "choices": [{"message": {"content": "answer"}}],
            "usage": {"prompt_tokens": 123},
        }

    model = OpenAICompatibleQwenVL(
        GenerationConfig(model="gpt-5.6-sol", base_url="https://gateway.example/v1"),
        transport=transport,
        token_count_mode="auto",
    )
    assert model.complete([{"role": "user", "content": "question"}]) == "answer"
    assert routes == ["/chat/completions"]
    assert model.last_preflight_source == "hosted_gateway_local_estimate"
    assert model.last_preflight_tokens is not None
    assert model.last_input_tokens == 123


def test_empty_assistant_content_is_an_error() -> None:
    def transport(route, payload):
        return {"choices": [{"message": {"content": ""}}]}

    model = OpenAICompatibleQwenVL(
        GenerationConfig(model="gpt-5.6-sol", base_url="https://gateway.example/v1"),
        transport=transport,
        token_count_mode="local",
    )
    with pytest.raises(RuntimeError, match="empty assistant content"):
        model.complete([{"role": "user", "content": "question"}])


def test_deepseek_none_disables_thinking_and_records_completion_tokens() -> None:
    payloads = []

    def transport(route, payload):
        payloads.append(payload)
        return {
            "choices": [{"message": {"content": "answer"}}],
            "usage": {"prompt_tokens": 41, "completion_tokens": 7},
        }

    model = OpenAICompatibleQwenVL(
        GenerationConfig(
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            reasoning_effort="none",
        ),
        transport=transport,
        token_count_mode="local",
    )
    assert model.complete([{"role": "user", "content": "question"}]) == "answer"
    assert payloads[0]["thinking"] == {"type": "disabled"}
    assert payloads[0]["temperature"] == 0
    assert model.last_input_tokens == 41
    assert model.last_completion_tokens == 7
