from __future__ import annotations

import base64
from typing import Any, Mapping, Sequence

from .backends import data_url


def text_from_parts(parts: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        str(part.get("text", ""))
        for part in parts
        if part.get("type") in {"text", "table", "document"} and part.get("text")
    ).strip()


def uniformly_sample_video(
    path: str, frame_count: int = 8, *, max_edge: int = 0
) -> list[str]:
    """Return JPEG data URLs, optionally capped on the longest frame edge."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for raw video sampling") from exc
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"failed to open video: {path}")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise RuntimeError(f"video has no readable frames: {path}")
        wanted = min(frame_count, total)
        positions = [round(i * (total - 1) / max(wanted - 1, 1)) for i in range(wanted)]
        frames: list[str] = []
        for position in positions:
            # Container frame counts and random seeks are occasionally inaccurate,
            # especially around damaged GOPs. Prefer the requested frame, then the
            # nearest earlier/later readable frame instead of aborting the dataset.
            max_offset = min(128, total - 1)
            candidates = [position]
            for offset in range(1, max_offset + 1):
                if position - offset >= 0:
                    candidates.append(position - offset)
                if position + offset < total:
                    candidates.append(position + offset)
            frame = None
            actual_position = position
            for candidate in candidates:
                capture.set(cv2.CAP_PROP_POS_FRAMES, candidate)
                ok, value = capture.read()
                if ok and value is not None:
                    frame = value
                    actual_position = candidate
                    break
            if frame is None:
                raise RuntimeError(
                    f"failed to read frame near {position} (searched +/-{max_offset}) from {path}"
                )
            if max_edge > 0 and max(frame.shape[:2]) > max_edge:
                scale = max_edge / max(frame.shape[:2])
                frame = cv2.resize(
                    frame,
                    (
                        max(1, round(frame.shape[1] * scale)),
                        max(1, round(frame.shape[0] * scale)),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            encoded_ok, encoded = cv2.imencode(".jpg", frame)
            if not encoded_ok:
                raise RuntimeError(f"failed to JPEG-encode frame {actual_position} from {path}")
            value = base64.b64encode(encoded.tobytes()).decode("ascii")
            frames.append(f"data:image/jpeg;base64,{value}")
        return frames
    finally:
        capture.release()


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


def label_retrieved_images(content, units):
    """Attach asset IDs to upstream agent image parts without replacing its tools."""
    labels = {str(unit["file_path"]): unit["media_source_id"] for unit in units
              if unit.get("type") == "image" and unit.get("media_source_id")}
    rendered = []
    for part in content:
        if part.get("type") == "image" and str(part.get("image")) in labels:
            rendered.append({"type": "text", "text":
                             f"Image source_id: {labels[str(part['image'])]}"})
        rendered.append(part)
    return rendered


def image_content(part: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Render public asset identity next to its pixels; no gold/metadata lookup."""
    result = []
    source_id = part.get("source_id")
    if source_id:
        result.append({"type": "text", "text": f"Image source_id: {source_id}"})
    path = str(part["path"])
    result.append({"type": "image_url", "image_url": {
        "url": path if path.startswith("data:") else data_url(path)}})
    return result


def table_text(part: Mapping[str, Any]) -> str:
    """Recognize explicitly marked tables and legacy structured JSON tables."""
    import json
    text = str(part.get("text") or "")
    if part.get("type") == "table" or part.get("annotations", {}).get("format") == "table":
        return text
    # Old SMMBench bundles serialized tables as text; avoid rebuilding the bundle
    # merely to recognize their unambiguous native header/rows schema.
    if part.get("type") == "text" and text.lstrip().startswith("{"):
        try:
            value = json.loads(text)
        except ValueError:
            return ""
        if isinstance(value, dict) and isinstance(value.get("table_header"), list) and isinstance(value.get("table_rows"), list):
            return text
    return ""


def openai_content_from_parts(
    parts: Sequence[Mapping[str, Any]], *, video_frames: int = 8
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for part in parts:
        kind = part.get("type")
        if kind == "text":
            content.append({"type": "text", "text": str(part.get("text", ""))})
        elif kind == "image":
            content.extend(image_content(part))
        elif kind == "video":
            for frame in uniformly_sample_video(str(part["path"]), video_frames):
                content.append({"type": "image_url", "image_url": {"url": frame}})
        elif kind in {"document", "table"}:
            if part.get("text"):
                content.append({"type": "text", "text": str(part["text"])})
            else:
                content.append({"type": "text", "text": f"[{kind}: {part.get('path', '')}]"})
        elif kind == "audio":
            raise NotImplementedError("the default Qwen3-VL method track does not transcribe audio")
    return content


def question_text(question: Mapping[str, Any]) -> str:
    text = text_from_parts(question.get("prompt", []))
    choices = question.get("choices")
    if isinstance(choices, list) and choices:
        rendered = "\n".join(
            f"{choice.get('choice_id')}: {choice.get('text', '')}" for choice in choices
        )
        text = f"{text}\nChoices:\n{rendered}"
    return text
