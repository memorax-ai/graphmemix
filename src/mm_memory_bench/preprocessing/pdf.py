"""Shared, opt-in PDF views. Never consumes questions or gold annotations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

PDF_POLICIES = ("off", "native_only", "native_then_ocr", "ocr_pages")


class PDFProcessor:
    def __init__(self, policy: str = "off", *, page_images: int = 0,
                 cache_dir: Path | None = None) -> None:
        if policy not in PDF_POLICIES:
            raise ValueError(f"unknown PDF policy: {policy}")
        if page_images < 0 or (page_images and policy == "off"):
            raise ValueError("PDF page images require an enabled policy and a nonnegative limit")
        self.policy = policy
        self.page_images = page_images
        self.cache_dir = cache_dir
        self._cache: dict[str, dict[str, Any]] = {}

    def process(self, part: Mapping[str, Any]) -> list[dict[str, Any]]:
        value = dict(part)
        path = Path(str(value.get("path", "")))
        asset = value.get("asset", {})
        if (self.policy == "off" or value.get("type") != "document"
                or not (path.suffix.lower() == ".pdf" or asset.get("mime_type") == "application/pdf")):
            return [value]
        # Content addressing prevents stale views when a PDF changes at the same path.
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in self._cache:
            self._cache[digest] = self._extract(path, digest)
        result = self._cache[digest]
        text = "\n\n".join(
            f"[PDF {path.name}; page {page['page']}]\n{page['text']}"
            for page in result["pages"] if page["text"]
        )
        value["text"] = "\n\n".join(x for x in (str(value.get("text", "")), text) if x)
        value["pdf_processing"] = {
            "policy": self.policy, "sha256": digest,
            "page_count": len(result["pages"]),
            "ocr_pages": [p["page"] for p in result["pages"] if p["ocr"]],
            "rendered_pages": len(result["images"]),
        }
        if not value["text"] and not result["images"]:
            raise ValueError(f"PDF produced no readable text: {path}; enable OCR or page images")
        return [value, *[
            {"type": "image", "path": image, "pdf_page": i,
             "source_asset_id": value.get("asset_id"), "source_pdf": str(path)}
            for i, image in enumerate(result["images"], 1)
        ]]

    def _extract(self, path: Path, digest: str) -> dict[str, Any]:
        try:
            import pymupdf as fitz
        except ImportError as exc:
            raise RuntimeError('PDF processing requires pip install -e ".[pdf]"') from exc
        pages, images = [], []
        with fitz.open(path) as doc:
            for i, page in enumerate(doc):
                native = page.get_text("text").strip()
                ocr = self.policy == "ocr_pages" or (self.policy == "native_then_ocr" and len(native) < 400)
                if ocr:
                    try:
                        import pytesseract
                        from PIL import Image
                        pix = page.get_pixmap(dpi=150, alpha=False, colorspace=fitz.csRGB)
                        recognized = pytesseract.image_to_string(
                            Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                        ).strip()
                        if recognized and recognized != native:
                            native = "\n".join(x for x in (native, recognized) if x)
                    except Exception as exc:
                        raise RuntimeError(f"PDF OCR failed at {path}, page {i + 1}: {exc}") from exc
                pages.append({"page": i + 1, "text": native, "ocr": ocr})
                if i < self.page_images:
                    if self.cache_dir is None:
                        raise ValueError("PDF page rendering requires a cache directory")
                    target = self.cache_dir / digest / f"page-{i + 1}-144dpi.png"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        page.get_pixmap(dpi=144, alpha=False).save(target)
                    images.append(str(target.resolve()))
        return {"pages": pages, "images": images}


def check_pdf_checkpoint(checkpoint_dir: Path | None, *, policy: str,
                         page_images: int) -> None:
    """Reject reuse of memories built with a different PDF input policy."""
    if checkpoint_dir is None:
        return
    root = Path(checkpoint_dir)
    marker = root / "pdf-input.json"
    expected = {"version": 1, "policy": policy, "page_images": page_images}
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            raise ValueError("PDF input policy differs from checkpoint; use a fresh checkpoint directory")
    elif policy != "off":
        if root.exists() and any(root.iterdir()):
            raise ValueError("Existing checkpoint has no PDF input policy; use a fresh checkpoint directory")
        root.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(expected) + "\n")
