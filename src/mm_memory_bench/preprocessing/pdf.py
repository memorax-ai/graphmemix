"""Shared, opt-in PDF views. Never consumes questions or gold annotations."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

PDF_POLICIES = ("off", "native_only", "native_then_ocr", "ocr_pages")
# Defaults from M³Exam's extract_pdf_pages_for_eval.
_NATIVE_MAX_PAGES = 100
_OCR_FALLBACK_MAX_PAGES = 15
_MIN_NATIVE_CHARS = 400
# PyMuPDF must not run concurrently, including across separate bundle readers.
_PDF_EXTRACT_LOCK = Lock()


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
        self._cache: dict[tuple[str, str, int], Future[dict[str, Any]]] = {}
        self._lock = Lock()
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False

    def close(self) -> None:
        with self._lock:
            self._closed = True
            executor = self._executor
        if executor is not None:
            executor.shutdown(wait=True)
        with self._lock:
            self._cache.clear()

    def __enter__(self) -> PDFProcessor:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def process(self, part: Mapping[str, Any]) -> list[dict[str, Any]]:
        value = dict(part)
        path = Path(str(value.get("path", "")))
        asset = value.get("asset", {})
        if (self.policy == "off" or value.get("type") != "document"
                or not (path.suffix.lower() == ".pdf" or asset.get("mime_type") == "application/pdf")):
            return [value]
        # Content addressing prevents stale views when a PDF changes at the same path.
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        key = (digest, self.policy, self.page_images)
        with self._lock:
            if self._closed:
                raise RuntimeError("PDF processor is closed")
            future = self._cache.get(key)
            if future is None:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="pdf-preprocess"
                    )
                future = self._executor.submit(self._run_extract, path, digest)
                self._cache[key] = future
        try:
            result = future.result()
        except BaseException:
            # Existing waiters observe this failure; a later call may retry it.
            with self._lock:
                if self._cache.get(key) is future:
                    del self._cache[key]
            raise
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

    def _run_extract(self, path: Path, digest: str) -> dict[str, Any]:
        with _PDF_EXTRACT_LOCK:
            return self._extract(path, digest)

    def _extract(self, path: Path, digest: str) -> dict[str, Any]:
        try:
            import pymupdf as fitz
        except ImportError as exc:
            raise RuntimeError('PDF processing requires pip install -e ".[pdf]"') from exc
        pages, images = [], []
        with fitz.open(path) as doc:
            text_page_count = min(len(doc), _NATIVE_MAX_PAGES)
            if self.policy != "ocr_pages":
                pages = [
                    {"page": i + 1, "text": doc[i].get_text("text").strip(), "ocr": False}
                    for i in range(text_page_count)
                ]
            fallback = (self.policy == "native_then_ocr"
                        and sum(len(page["text"]) for page in pages) < _MIN_NATIVE_CHARS)
            if self.policy == "ocr_pages" or fallback:
                if fallback:
                    text_page_count = min(text_page_count, _OCR_FALLBACK_MAX_PAGES)
                # The official fallback replaces native text, including with empty OCR.
                pages = []
                for i in range(text_page_count):
                    try:
                        import pytesseract
                        from PIL import Image
                        pix = doc[i].get_pixmap(dpi=150, alpha=False, colorspace=fitz.csRGB)
                        recognized = pytesseract.image_to_string(
                            Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                        ).strip()
                    except Exception as exc:
                        raise RuntimeError(f"PDF OCR failed at {path}, page {i + 1}: {exc}") from exc
                    pages.append({"page": i + 1, "text": recognized, "ocr": True})
            for i in range(min(len(doc), self.page_images)):
                if self.cache_dir is None:
                    raise ValueError("PDF page rendering requires a cache directory")
                target = self.cache_dir / digest / f"page-{i + 1}-144dpi.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    with tempfile.NamedTemporaryFile(
                        suffix=".png", dir=target.parent, delete=False
                    ) as handle:
                        temporary = Path(handle.name)
                    try:
                        doc[i].get_pixmap(dpi=144, alpha=False).save(temporary)
                        os.replace(temporary, target)
                    finally:
                        temporary.unlink(missing_ok=True)
                images.append(str(target.resolve()))
        return {"pages": pages, "images": images}


def check_pdf_checkpoint(checkpoint_dir: Path | None, *, policy: str,
                         page_images: int) -> None:
    """Reject reuse of memories built with a different PDF input policy."""
    if checkpoint_dir is None:
        return
    root = Path(checkpoint_dir)
    marker = root / "pdf-input.json"
    expected = {"version": 2, "policy": policy, "page_images": page_images}
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            raise ValueError("PDF input policy differs from checkpoint; use a fresh checkpoint directory")
    elif policy != "off":
        if root.exists() and any(root.iterdir()):
            raise ValueError("Existing checkpoint has no PDF input policy; use a fresh checkpoint directory")
        root.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(expected) + "\n")
