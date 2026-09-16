"""Small format helpers shared by the additional benchmark converters."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from ..bundle import SchemaError, content_asset, content_text, read_json, stable_id


def locate(raw_root: Path, name: str, marker: str) -> Path:
    for root in (Path(raw_root) / name, Path(raw_root)):
        if (root / marker).exists():
            return root.resolve()
    raise FileNotFoundError(f"{name}: missing {marker} under {raw_root}")


def parsed(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value


def as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def source_manifest(root: Path) -> dict:
    path = root / "download-manifest.json"
    snapshot = (
        read_json(path) if path.exists() else {"scope": "user-supplied local snapshot"}
    )
    if snapshot.get("errors"):
        raise SchemaError(
            f"{path}: download has unresolved errors; repair it before converting"
        )
    return snapshot


class Assets:
    def __init__(self, writer, root: Path, output: Path, benchmark: str):
        self.writer = writer
        self.root = root
        self.output = output
        self.benchmark = benchmark
        self.seen = {}

    def add(self, path: Path, kind="image") -> dict:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing {kind} asset: {path}")
        relative = os.path.relpath(path, self.output.resolve())
        key = (relative, kind)
        if key not in self.seen:
            aid = stable_id(
                self.benchmark,
                "asset",
                hashlib.sha256(relative.encode()).hexdigest()[:24],
            )
            self.writer.add_asset(
                {
                    "asset_id": aid,
                    "media_type": kind,
                    "path": relative,
                    "mime_type": mimetypes.guess_type(path.name)[0],
                }
            )
            self.seen[key] = aid
        return content_asset(kind, self.seen[key])

    def data_image(self, value: str) -> dict:
        if not value.startswith("data:image/") or ";base64," not in value:
            raise SchemaError("expected an embedded base64 image, not an unfetched URL")
        header, encoded = value.split(",", 1)
        raw = base64.b64decode(encoded, validate=True)
        suffix = mimetypes.guess_extension(header[5:].split(";")[0]) or ".bin"
        path = self.root / "decoded_assets" / (hashlib.sha256(raw).hexdigest() + suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(raw)
        return self.add(path)

    def content(self, value: Any) -> list[dict]:
        if isinstance(value, str):
            return [content_text(value)]
        if not isinstance(value, list):
            raise SchemaError(f"unsupported message content: {type(value)}")
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(content_text(item))
                continue
            if item.get("type") == "text":
                parts.append(content_text(item["text"]))
            elif item.get("type") in {"image_url", "image"}:
                image = item.get("image_url", item.get("image"))
                if isinstance(image, dict):
                    image = image["url"]
                parts.append(self.data_image(image))
            else:
                raise SchemaError(f"unsupported content type {item.get('type')}")
        return parts or [content_text("")]


def question(
    qid: str,
    cid: str,
    prompt: str,
    answer: str,
    category: str,
    subtype: str,
    subset="default",
    **kwargs,
):
    return {
        "question_id": qid,
        "semantic_question_id": qid,
        "context_id": cid,
        "subset": subset,
        "prompt": [content_text(prompt)],
        "task": {"category": category, "subcategory": subtype, "response_type": "text"},
        "answer": {"text": answer},
        "evidence": [],
        "memory_scope": {"mode": "all"},
        **kwargs,
    }


def choices(q: dict, values: dict[str, str], correct: str):
    if correct not in values:
        raise SchemaError(f"answer {correct!r} not in choices")
    q["choices"] = [{"choice_id": k, "text": v} for k, v in values.items()]
    q["answer"] = {
        "text": values[correct],
        "choice_id": correct,
        "native_label": correct,
    }
    q["task"]["response_type"] = "choice"
    q["instruction"] = "Select one option. Return its label."
