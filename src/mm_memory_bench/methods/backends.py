from __future__ import annotations

import base64
import io
import json
import math
import mimetypes
import os
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from .base import GenerationConfig


class TextEmbedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...


class MultiModalEmbedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def encode_units(self, units: Sequence[Mapping[str, Any]]) -> np.ndarray: ...


class AnswerModel(Protocol):
    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> str: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for A-Mem") from exc
        self.model = SentenceTransformer(model_name)
        self._dimension = int(self.model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=False),
            dtype=np.float32,
        )


def data_url(path: str) -> str:
    value = Path(path)
    mime = mimetypes.guess_type(value.name)[0] or "application/octet-stream"
    raw = value.read_bytes()
    if mime == "application/octet-stream":
        if raw.startswith(b"\xff\xd8\xff"):
            mime = "image/jpeg"
        elif raw.startswith(b"\x89PNG\r\n\x1a\n"):
            mime = "image/png"
        elif raw.startswith((b"GIF87a", b"GIF89a")):
            mime = "image/gif"
        elif raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
            mime = "image/webp"
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class OpenAICompatibleQwenVL:
    """Minimal vLLM client with a configurable strict token preflight."""

    def __init__(
        self,
        config: GenerationConfig | None = None,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 1200,
        transport: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
        token_count_mode: str = "auto",
    ) -> None:
        if token_count_mode not in {"auto", "local", "remote"}:
            raise ValueError("token_count_mode must be auto, local, or remote")
        self.config = config or GenerationConfig()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.timeout_seconds = timeout_seconds
        self._transport = transport or self._post
        self._local = threading.local()
        self.token_count_mode = token_count_mode
        self._tokenize_disabled_reason: str | None = None
        self._tokenize_warning_emitted = False
        self._tokenize_lock = threading.Lock()

    @property
    def last_input_tokens(self) -> int | None:
        """Input tokens from this thread's most recent completion preflight."""
        value = getattr(self._local, "last_input_tokens", None)
        return int(value) if isinstance(value, int) else None

    @property
    def last_completion_tokens(self) -> int | None:
        """Completion tokens reported by the most recent chat request."""
        value = getattr(self._local, "last_completion_tokens", None)
        return int(value) if isinstance(value, int) else None

    @property
    def last_cached_input_tokens(self) -> int | None:
        """Cached input tokens reported by the most recent chat request."""
        value = getattr(self._local, "last_cached_input_tokens", None)
        return int(value) if isinstance(value, int) else None

    @property
    def last_preflight_tokens(self) -> int | None:
        value = getattr(self._local, "last_preflight_tokens", None)
        return int(value) if isinstance(value, int) else None

    @property
    def last_preflight_source(self) -> str | None:
        value = getattr(self._local, "last_preflight_source", None)
        return str(value) if value else None

    @property
    def last_tokenize_error(self) -> str | None:
        value = getattr(self._local, "last_tokenize_error", None)
        return str(value) if value else None

    def _post(self, route: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        base = self.config.base_url.rstrip("/")
        # vLLM exposes OpenAI chat under /v1 but its tokenizer at the server root.
        if route == "/tokenize" and base.endswith("/v1"):
            base = base[:-3]
        url = base + route
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
                try:
                    result = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"vLLM {route} returned non-JSON content: {raw[:200]!r}"
                    ) from exc
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM {route} failed ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            raise RuntimeError(
                f"vLLM {route} transport failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(result, Mapping):
            raise RuntimeError(f"vLLM {route} returned a non-object response")
        return result

    def count_input_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> int:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
            "add_generation_prompt": True,
        }
        normalized_tools = openai_tools(tools)
        if normalized_tools:
            payload["tools"] = normalized_tools
        self._local.last_tokenize_error = None
        use_local = self.token_count_mode == "local"
        local_reason = "configured_local_estimate"
        if self.token_count_mode == "auto":
            host = (urllib.parse.urlparse(self.config.base_url).hostname or "").lower()
            if host not in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}:
                use_local = True
                local_reason = "hosted_gateway_local_estimate"
            elif self._tokenize_disabled_reason is not None:
                use_local = True
                local_reason = "remote_tokenize_disabled_after_error"
                self._local.last_tokenize_error = self._tokenize_disabled_reason
        if use_local:
            count = estimate_multimodal_input_tokens(messages, normalized_tools)
            self._local.last_preflight_tokens = count
            self._local.last_preflight_source = local_reason
            return count
        try:
            result = self._transport("/tokenize", payload)
        except Exception as exc:
            if self.token_count_mode != "auto":
                raise
            reason = f"{type(exc).__name__}: {exc}"
            with self._tokenize_lock:
                self._tokenize_disabled_reason = reason
                if not self._tokenize_warning_emitted:
                    print(
                        "WARNING: remote /tokenize failed; switching this client "
                        f"to local conservative estimates: {reason}",
                        file=sys.stderr,
                        flush=True,
                    )
                    self._tokenize_warning_emitted = True
            count = estimate_multimodal_input_tokens(messages, normalized_tools)
            self._local.last_tokenize_error = reason
            self._local.last_preflight_tokens = count
            self._local.last_preflight_source = "local_estimate_after_remote_error"
            return count
        count = result.get("count")
        if not isinstance(count, int):
            tokens = result.get("tokens")
            if not isinstance(tokens, list):
                raise RuntimeError("vLLM /tokenize did not return count or tokens")
            count = len(tokens)
        self._local.last_preflight_tokens = count
        self._local.last_preflight_source = "remote_tokenize"
        return count

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> str:
        normalized_tools = openai_tools(tools)
        count = self.count_input_tokens(messages, normalized_tools)
        self._local.last_input_tokens = count
        if count >= self.config.max_model_len:
            raise ContextWindowExceeded(count, self.config.max_model_len)
        # vLLM applies max_model_len to prompt + generated tokens. Preserve the
        # complete input and shrink only the output allowance when a valid
        # near-limit prompt leaves less room than the configured maximum.
        output_budget = min(
            self.config.max_output_tokens,
            self.config.max_model_len - count,
        )
        payload = self.completion_payload(messages, output_budget)
        if normalized_tools:
            payload["tools"] = normalized_tools
        if response_format:
            payload["response_format"] = dict(response_format)
        result = self._transport("/chat/completions", payload)
        usage = result.get("usage")
        if isinstance(usage, Mapping):
            prompt_tokens = usage.get("prompt_tokens")
            if isinstance(prompt_tokens, int):
                self._local.last_input_tokens = prompt_tokens
            completion_tokens = usage.get("completion_tokens")
            self._local.last_completion_tokens = (
                completion_tokens if isinstance(completion_tokens, int) else None
            )
            details = usage.get("prompt_tokens_details")
            cached_tokens = details.get("cached_tokens") if isinstance(details, Mapping) else None
            self._local.last_cached_input_tokens = (
                cached_tokens if isinstance(cached_tokens, int) else None
            )
        else:
            self._local.last_completion_tokens = None
            self._local.last_cached_input_tokens = None
        try:
            message = result["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("vLLM response has no assistant message") from exc
        content = message.get("content")
        if content:
            return str(content)
        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            return json.dumps(tool_calls, ensure_ascii=False)
        raise RuntimeError("vLLM response has empty assistant content and no tool calls")

    def completion_payload(
        self,
        messages: Sequence[Mapping[str, Any]],
        output_budget: int,
    ) -> dict[str, Any]:
        """Build a request accepted by both local vLLM and hosted GPT-5 APIs."""
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
        }
        if "gpt-5" in self.config.model.lower():
            payload["max_completion_tokens"] = output_budget
            payload["reasoning_effort"] = self.config.reasoning_effort
        else:
            payload["temperature"] = 0
            payload["max_tokens"] = output_budget
            if (
                "deepseek-v4" in self.config.model.lower()
                and self.config.reasoning_effort == "none"
            ):
                payload["thinking"] = {"type": "disabled"}
        return payload


def estimate_multimodal_input_tokens(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> int:
    """Conservative fallback for hosted gateways without a tokenize endpoint."""
    text_chars = 0
    visual_tokens = 0
    for message in messages:
        content = message.get("content")
        parts = content if isinstance(content, list) else [content]
        for part in parts:
            if isinstance(part, str):
                text_chars += len(part)
            elif isinstance(part, Mapping):
                if part.get("type") in {"image", "image_url", "video"}:
                    visual_tokens += estimate_visual_input_tokens(part)
                else:
                    text_chars += len(json.dumps(part, ensure_ascii=False))
    if tools:
        text_chars += len(json.dumps(list(tools), ensure_ascii=False))
    return 16 + (text_chars + 2) // 3 + visual_tokens


def estimate_visual_input_tokens(part: Mapping[str, Any]) -> int:
    """Estimate hosted-VLM image tokens from 32px patches.

    The 1.25 safety multiplier is deliberately conservative and the 1536-patch
    cap matches the common patch-budget regime. Unknown or non-data URLs retain
    the older 1024-token fallback rather than being treated as free.
    """
    image = part.get("image_url")
    url = image.get("url") if isinstance(image, Mapping) else image
    if not isinstance(url, str) or not url.startswith("data:") or ";base64," not in url:
        return 1024
    try:
        from PIL import Image

        encoded = url.split(";base64,", 1)[1]
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as value:
            width, height = value.size
        if width <= 0 or height <= 0:
            raise ValueError("non-positive image dimensions")
        patches = math.ceil(width / 32) * math.ceil(height / 32)
        return max(1, math.ceil(1.25 * min(1536, patches)))
    except Exception:
        return 1024


class ContextWindowExceeded(RuntimeError):
    def __init__(self, actual: int, maximum: int) -> None:
        super().__init__(f"input context has {actual} tokens; hard limit is {maximum}")
        self.actual = actual
        self.maximum = maximum


def openai_tools(tools: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize public benchmark tool declarations for an OpenAI endpoint."""
    normalized: list[dict[str, Any]] = []
    for tool in tools or []:
        if tool.get("type") == "function" and isinstance(tool.get("function"), Mapping):
            normalized.append(dict(tool))
            continue
        if isinstance(tool.get("name"), str):
            name = str(tool["name"])
            description = str(tool.get("description", ""))
            parameters = tool.get("input_schema", tool.get("parameters"))
        elif isinstance(tool.get("function_name"), str):
            name = str(tool["function_name"])
            defaults = tool.get("default_arguments", {})
            description = str(tool.get("function_comment", ""))
            if defaults:
                description += "\nDefault arguments: " + json.dumps(defaults, ensure_ascii=False)
            parameters = None
        else:
            raise ValueError(f"unsupported public tool declaration: {tool}")
        if not isinstance(parameters, Mapping):
            parameters = {"type": "object", "additionalProperties": True}
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": dict(parameters),
                },
            }
        )
    return normalized
