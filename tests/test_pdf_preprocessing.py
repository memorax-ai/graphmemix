"""PDF ingestion boundaries, extraction and existing media consumers."""
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

from mm_memory_bench.preprocessing.pdf import PDFProcessor, check_pdf_checkpoint
from mm_memory_bench.benchmarks.reader import BundleReader
from mm_memory_bench.runner.benchmark import _resolved_memory
from mm_memory_bench.methods.media import openai_content_from_parts, text_from_parts


@pytest.fixture
def processor():
    with ExitStack() as stack:
        def create(*args, **kwargs):
            return stack.enter_context(PDFProcessor(*args, **kwargs))
        yield create


@pytest.fixture
def pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "lesson.pdf"
    with fitz.open() as doc:
        for text in ["Flat White: thin foam.", "Cappuccino: equal milk and foam."]:
            page = doc.new_page()
            page.insert_text((72, 72), text)
        doc.save(path)
    return path


def part(pdf):
    return {"type": "document", "asset_id": "pdf1", "path": str(pdf)}


def test_default_does_not_open_documents(processor, tmp_path):
    value = part(tmp_path / "missing.pdf")
    assert processor().process(value) == [value]


def test_native_pages_cache_and_rendered_reader_images(processor, pdf, tmp_path):
    pdf_processor = processor("native_only", page_images=1, cache_dir=tmp_path / "cache")
    source = part(pdf)
    result = pdf_processor.process(source)
    assert "text" not in source
    assert "page 1" in result[0]["text"] and "page 2" in result[0]["text"]
    assert "equal milk and foam" in text_from_parts(result)
    assert result[0]["pdf_processing"]["page_count"] == 2
    assert len(result) == 2
    assert Path(result[1]["path"]).is_file()
    with patch.object(pdf_processor, "_extract", side_effect=AssertionError("reparsed")):
        assert pdf_processor.process(source) == result
    messages = openai_content_from_parts(result)
    assert messages[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_no_gold_leak_or_memory_id_change(pdf, tmp_path):
    (tmp_path / "manifest.json").write_text('{}')
    (tmp_path / "assets.jsonl").write_text(json.dumps({
        "asset_id": "pdf1", "path": str(pdf), "mime_type": "application/pdf",
        "metadata": {"gold": "GOLD_SECRET"}}) + '\n')
    memory = {"memory_id": "m1", "context_id": "c1", "content": [
        {"type": "document", "asset_id": "pdf1"}], "answer": "GOLD_SECRET"}
    with BundleReader(tmp_path, pdf_policy="native_only", pdf_page_images=1) as reader:
        resolved = _resolved_memory(reader, memory)
    assert resolved["memory_id"] == "m1"
    assert "GOLD_SECRET" not in json.dumps(resolved)
    assert "Flat White" in resolved["content"][0]["text"]
    assert resolved["content"][1]["type"] == "image"


def test_scanned_page_requires_ocr_or_images(processor, tmp_path):
    fitz = pytest.importorskip("fitz")
    pytest.importorskip("pytesseract")
    path = tmp_path / "scan.pdf"
    with fitz.open() as doc:
        doc.new_page()
        doc.save(path)
    with pytest.raises(ValueError, match="no readable text"):
        processor("native_only").process(part(path))
    with patch("pytesseract.image_to_string", return_value="Scanned evidence") as ocr:
        value = processor("native_then_ocr").process(part(path))[0]
        assert "Scanned evidence" in value["text"]
        assert value["pdf_processing"]["ocr_pages"] == [1]
        ocr.assert_called_once()


def test_native_text_does_not_call_ocr(processor, pdf):
    with patch.dict("sys.modules", {"pytesseract": None}):
        assert "Flat White" in processor("native_only").process(part(pdf))[0]["text"]


def test_sparse_text_triggers_ocr(processor, pdf):
    pytest.importorskip("pytesseract")
    with patch("pytesseract.image_to_string", return_value="Full page contents") as ocr:
        result = processor("native_then_ocr").process(part(pdf))[0]
        assert ocr.call_count == 2
        assert result["pdf_processing"]["ocr_pages"] == [1, 2]


def test_policy_checkpoint_cannot_mix_inputs(tmp_path):
    check_pdf_checkpoint(tmp_path, policy="native_only", page_images=0)
    check_pdf_checkpoint(tmp_path, policy="native_only", page_images=0)
    for policy, count in [("off", 0), ("native_only", 1), ("ocr_pages", 0)]:
        with pytest.raises(ValueError, match="differs"):
            check_pdf_checkpoint(tmp_path, policy=policy, page_images=count)


def test_old_checkpoint_rejected_for_pdf(tmp_path):
    (tmp_path / "state.json").write_text('{}')
    with pytest.raises(ValueError, match="fresh checkpoint"):
        check_pdf_checkpoint(tmp_path, policy="native_only", page_images=0)
    check_pdf_checkpoint(tmp_path, policy="off", page_images=0)


def test_pdf_text_reaches_all_five_method_ingestion_paths(processor, pdf):
    from unittest.mock import Mock
    from mm_memory_bench.methods.concrete_amem import ConcreteAMemMethod
    from mm_memory_bench.methods.concrete_memguide import ConcreteMemGuideMethod
    from mm_memory_bench.methods.concrete_lightmem import ConcreteLightMemMethod
    from mm_memory_bench.methods.concrete_universalrag import ConcreteUniversalRAGMethod
    from mm_memory_bench.methods.concrete_vimrag import ConcreteVimRAGMethod

    memory = {"memory_id": "m1", "content": processor("native_only").process(part(pdf))}
    amem = ConcreteAMemMethod.__new__(ConcreteAMemMethod)
    amem.video_frames = 8
    amem._memory_json = Mock(return_value={"context": "coffee", "keywords": [], "tags": []})
    assert "Flat White" in amem.memory_to_notes(memory)[0]["content"]
    assert "Flat White" in str(amem._memory_json.call_args)
    memguide = ConcreteMemGuideMethod.__new__(ConcreteMemGuideMethod)
    memguide.question_mode = "template"
    assert "Flat White" in str(memguide._memory_unit(memory))
    lightmem = ConcreteLightMemMethod.__new__(ConcreteLightMemMethod)
    assert "Flat White" in lightmem._memory_text(memory)
    universal = ConcreteUniversalRAGMethod.__new__(ConcreteUniversalRAGMethod)
    universal.adapter_profile = "default"
    assert "Flat White" in str(universal._memory_units(memory))
    vim = ConcreteVimRAGMethod.__new__(ConcreteVimRAGMethod)
    assert "Flat White" in str(vim._memory_units(memory))
