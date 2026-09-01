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


def openai_content_from_parts(
    parts: Sequence[Mapping[str, Any]], *, video_frames: int = 8
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for part in parts:
        kind = part.get("type")
        if kind == "text":
            content.append({"type": "text", "text": str(part.get("text", ""))})
        elif kind == "image":
            content.append(
                {"type": "image_url", "image_url": {"url": data_url(str(part["path"]))}}
            )
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
