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


def caption_with_source(part: Mapping[str, Any], caption: str) -> str:
    """Bind a caption to its public asset identifier outside the caption cache."""
    source_id = part.get("source_id")
    prefix = f"Image source_id: {source_id}\n"
    return prefix + caption if source_id and not caption.startswith(prefix) else caption


def image_source_inventory(memory: Mapping[str, Any]) -> str:
    """Keep public identities even when unstructured captions cannot be paired."""
    ids = list(dict.fromkeys(str(p["source_id"]) for p in memory.get("content", [])
                           if p.get("type") == "image" and p.get("source_id")))
    return "Image source_ids (no caption ordering implied): " + ", ".join(ids) if ids else ""


def caption_metadata_with_sources(memory: Mapping[str, Any], derived: Mapping[str, Any]) -> dict:
    """Bind explicit asset references; never infer multi-image alignment by position."""
    media = [p for p in memory.get("content", []) if p.get("type") in {"image", "video", "audio"}]
    images = [p for p in media if p.get("type") == "image" and p.get("source_id")]

    def render(value):
        if isinstance(value, list):
            return [render(item) for item in value]
        if isinstance(value, Mapping):
            refs = {key: value[key] for key in ("asset_id", "source_id", "path") if value.get(key)}
            matches = [p for p in images if refs and all(p.get(k) == v for k, v in refs.items())]
            text = next((value[k] for k in ("final_text", "full_text", "text", "caption", "description")
                         if isinstance(value.get(k), str) and value[k].strip()), "")
            # Discard arbitrary annotation fields; expose descriptive text only.
            return caption_with_source(matches[0], text) if len(matches) == 1 and text else text
        if isinstance(value, str) and len(media) == 1 and len(images) == 1:
            return caption_with_source(images[0], value) if value.strip() else value
        return value

    result = dict(derived)
    for key in ("caption", "short_caption", "image_caption", "image_captions",
                "blip_caption", "blip_captions"):
        if key in result:
            result[key] = render(result[key])
    return result


def _strings(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                result.extend(_strings(item.get("text", item.get("caption", ""))))
            else:
                result.extend(_strings(item))
        return result
    if isinstance(value, Mapping):
        for key in ("final_text", "full_text", "text", "caption", "description"):
            result = _strings(value.get(key))
            if result:
                return result
    return []


def public_captions(memory: Mapping[str, Any]) -> list[str]:
    """Read only captions carried by the canonical public benchmark record."""
    metadata = memory.get("metadata")
    if not isinstance(metadata, Mapping):
        return []
    derived = metadata.get("derived", {})
    if not isinstance(derived, Mapping):
        return []
    derived = caption_metadata_with_sources(memory, derived)
    result: list[str] = []
    for key in (
        "caption",
        "short_caption",
        "image_caption",
        "image_captions",
        "video_caption",
        "video_captions",
        "blip_caption",
        "blip_captions",
    ):
        result.extend(_strings(derived.get(key)))
    # Stable de-duplication preserves the benchmark-provided ordering.
    return list(dict.fromkeys(result))
