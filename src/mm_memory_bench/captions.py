from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from threading import get_ident
from typing import Any, Mapping


CAPTION_PROMPT_VERSION = 1
CAPTION_PROMPT = (
    "Describe this benchmark memory faithfully and densely. Include visible text/OCR, "
    "people, objects, attributes, spatial relations, and actions. Do not infer facts "
    "that are not visible. Return only the description."
)


def caption_cache_key(part: Mapping[str, Any], *, video_frames: int = 8) -> str:
    path = str(part["path"])
    digest = hashlib.sha256()
    digest.update(str(part.get("type", "media")).encode())
    digest.update(b"\0")
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(
        f"\0frames={video_frames}\0prompt=v{CAPTION_PROMPT_VERSION}".encode()
    )
    return digest.hexdigest()


def load_cached_caption(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    caption = value.get("caption")
    return str(caption) if caption else None


def save_cached_caption(path: Path, *, caption: str, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.{get_ident()}.tmp")
    temporary.write_text(
        json.dumps({"caption": caption, "kind": kind}, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)
