from __future__ import annotations

import base64
import io
from typing import Any, Mapping, Sequence

import numpy as np


def _load_image(value: str):
    from PIL import Image

    if value.startswith("data:"):
        payload = value.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    return Image.open(value).convert("RGB")


class GMEQwen2VLEmbedder:
    """Adapter for GME-Qwen2-VL multimodal embeddings."""

    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        *,
        dimension: int = 1536,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for GME") from exc
        self.torch = torch
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_units(self, units: Sequence[Mapping[str, Any]]) -> np.ndarray:
        rows: list[np.ndarray | None] = [None] * len(units)
        with self.torch.no_grad():
            text_indices = []
            image_indices = []
            fused_indices = []
            for index, unit in enumerate(units):
                text = str(unit.get("embedding_text", unit.get("text", ""))).strip()
                image_value = unit.get("image")
                if image_value and text:
                    fused_indices.append(index)
                elif image_value:
                    image_indices.append(index)
                else:
                    text_indices.append(index)
            groups = (
                (
                    fused_indices,
                    lambda ids: self.model.get_fused_embeddings(
                        texts=[
                            str(units[i].get("embedding_text", units[i].get("text", ""))).strip()
                            for i in ids
                        ],
                        images=[_load_image(str(units[i]["image"])) for i in ids],
                    ),
                ),
                (
                    image_indices,
                    lambda ids: self.model.get_image_embeddings(
                        images=[_load_image(str(units[i]["image"])) for i in ids]
                    ),
                ),
                (
                    text_indices,
                    lambda ids: self.model.get_text_embeddings(
                        texts=[
                            str(units[i].get("embedding_text", units[i].get("text", ""))).strip()
                            or " "
                            for i in ids
                        ]
                    ),
                ),
            )
            for indices, encode in groups:
                if not indices:
                    continue
                values = encode(indices).detach().float().cpu().numpy()
                for index, value in zip(indices, values):
                    rows[index] = value
        if any(row is None for row in rows):
            raise RuntimeError("GME failed to encode every retrieval unit")
        return np.asarray(rows, dtype=np.float32)
