"""PDF text policies use document-level fallback and bounded page scans."""
from pathlib import Path
from unittest.mock import Mock

import pytest

from mm_memory_bench.preprocessing.pdf import PDFProcessor


@pytest.fixture
def pdf_input(tmp_path, monkeypatch):
    fitz = pytest.importorskip("pymupdf")
    tesseract = pytest.importorskip("pytesseract")
    from PIL import Image

    class Document:
        def __init__(self, pages):
            self.pages_list = pages
            self.page_count = len(pages)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __len__(self):
            return self.page_count

        def __iter__(self):
            return iter(self.pages_list)

        def __getitem__(self, index):
            return self.pages_list[index]

        def load_page(self, index):
            return self.pages_list[index]

    def create(texts, *, recognized="OCR replacement"):
        path = tmp_path / "source.pdf"
        path.write_bytes(b"controlled PDF input")
        pixmap = Mock(width=1, height=1, samples=b"\xff\xff\xff")
        pixmap.save.side_effect = lambda target: Image.new("RGB", (1, 1)).save(target)
        pages = []
        for text in texts:
            page = Mock()
            page.get_text.return_value = text
            page.get_pixmap.return_value = pixmap
            pages.append(page)
        monkeypatch.setattr(fitz, "open", Mock(return_value=Document(pages)))
        ocr = Mock(return_value=recognized)
        monkeypatch.setattr(tesseract, "image_to_string", ocr)
        return {"type": "document", "path": str(path)}, pages, ocr

    return create


@pytest.mark.parametrize("total_characters", [399, 400])
def test_fallback_threshold_sums_stripped_text_across_pages(pdf_input, total_characters):
    first, second = "a" * 200, "b" * (total_characters - 200)
    part, pages, ocr = pdf_input([f" \n{first} \n", f"\n {second}\n "])
    with PDFProcessor("native_then_ocr") as processor:
        result = processor.process(part)[0]
    if total_characters == 400:
        ocr.assert_not_called()
        assert first in result["text"] and second in result["text"]
        assert result["pdf_processing"]["ocr_pages"] == []
    else:
        assert ocr.call_count == 2
        assert "OCR replacement" in result["text"]
        assert first not in result["text"] and second not in result["text"]
        assert result["pdf_processing"]["ocr_pages"] == [1, 2]


def test_native_only_reads_first_100_pages(pdf_input):
    part, pages, ocr = pdf_input([f"native-{i:03d}" for i in range(1, 102)])
    with PDFProcessor("native_only") as processor:
        result = processor.process(part)[0]
    assert "native-001" in result["text"] and "native-100" in result["text"]
    assert "native-101" not in result["text"]
    assert all(page.get_text.call_count == 1 for page in pages[:100])
    pages[100].get_text.assert_not_called()
    ocr.assert_not_called()


def test_fallback_scans_100_native_pages_but_ocrs_only_first_15(pdf_input):
    # Text after page 100 must not prevent fallback in an otherwise empty PDF.
    part, pages, ocr = pdf_input([""] * 100 + ["late native text" * 40])
    ocr.side_effect = [f"recognized-{i:03d}" for i in range(1, 16)]
    with PDFProcessor("native_then_ocr") as processor:
        result = processor.process(part)[0]
    assert all(page.get_text.call_count == 1 for page in pages[:100])
    pages[100].get_text.assert_not_called()
    assert ocr.call_count == 15
    assert "recognized-015" in result["text"]
    assert "late native text" not in result["text"]
    assert result["pdf_processing"]["ocr_pages"] == list(range(1, 16))


def test_ocr_only_reads_100_pages_without_native_extraction(pdf_input):
    part, pages, ocr = pdf_input(["native content must not be used"] * 101)
    ocr.side_effect = [f"recognized-{i:03d}" for i in range(1, 101)]
    with PDFProcessor("ocr_pages") as processor:
        result = processor.process(part)[0]
    assert ocr.call_count == 100
    assert "recognized-001" in result["text"] and "recognized-100" in result["text"]
    assert "native content" not in result["text"]
    assert all(page.get_text.call_count == 0 for page in pages)
    pages[100].get_pixmap.assert_not_called()
    assert result["pdf_processing"]["ocr_pages"] == list(range(1, 101))


@pytest.mark.parametrize("policy", ["native_then_ocr", "ocr_pages"])
def test_empty_ocr_does_not_restore_native_text(pdf_input, policy):
    part, pages, ocr = pdf_input(["short native text"], recognized=" \n ")
    with PDFProcessor(policy) as processor:
        with pytest.raises(ValueError, match="no readable text"):
            processor.process(part)
    ocr.assert_called_once()


@pytest.mark.parametrize("policy,image_count", [("native_only", 101), ("native_then_ocr", 16)])
def test_page_images_can_extend_beyond_text_limit(pdf_input, tmp_path, policy, image_count):
    texts = [f"native-{i:03d}" for i in range(1, image_count + 1)]
    part, pages, ocr = pdf_input(texts)
    with PDFProcessor(policy, page_images=image_count, cache_dir=tmp_path / "cache") as processor:
        result = processor.process(part)
    images = result[1:]
    assert len(images) == image_count
    assert [image["pdf_page"] for image in images] == list(range(1, image_count + 1))
    assert all(Path(image["path"]).is_file() for image in images)
    assert result[0]["pdf_processing"]["rendered_pages"] == image_count
    if policy == "native_only":
        assert "native-100" in result[0]["text"]
        assert "native-101" not in result[0]["text"]
        pages[100].get_text.assert_not_called()
    else:
        assert ocr.call_count == 15
        assert result[0]["pdf_processing"]["ocr_pages"] == list(range(1, 16))
