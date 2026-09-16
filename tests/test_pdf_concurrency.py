"""Concurrent PDF readers share extraction without serializing model calls."""
import json
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from threading import Barrier, Event, Lock
from unittest.mock import patch

import pytest

from mm_memory_bench.benchmarks.reader import BundleReader
from mm_memory_bench.evaluation.oracle import run_oracle_bundle
from mm_memory_bench.preprocessing.pdf import PDFProcessor


def document(path):
    return {"type": "document", "path": str(path), "asset_id": "pdf1"}


def extracted(text="PDF evidence"):
    return {"pages": [{"page": 1, "text": text, "ocr": False}], "images": []}


@pytest.mark.parametrize("fail", [False, True])
def test_concurrent_readers_share_result_or_failure_and_can_retry(tmp_path, fail):
    path = tmp_path / "source.pdf"
    path.write_bytes(b"same content")
    processor = PDFProcessor("native_only")
    readers = 4
    read_together = Barrier(readers)
    entered, release = Event(), Event()
    read_bytes = Path.read_bytes
    error = RuntimeError("temporary extraction failure")

    def read(candidate):
        result = read_bytes(candidate)
        read_together.wait(timeout=5)
        return result

    def extract(*args):
        entered.set()
        assert release.wait(5), "extraction was never released"
        if fail:
            raise error
        return extracted()

    try:
        with patch.object(Path, "read_bytes", read), patch.object(
            processor, "_extract", side_effect=extract
        ) as parse, ThreadPoolExecutor(max_workers=readers) as pool:
            pending = [pool.submit(processor.process, document(path)) for _ in range(readers)]
            try:
                assert entered.wait(5)
                # All callers have read the content and must wait for the blocked parse.
                assert not wait(pending, timeout=0.1).done
            finally:
                release.set()
            if fail:
                for future in pending:
                    with pytest.raises(RuntimeError, match="temporary extraction failure"):
                        future.result(timeout=5)
            else:
                results = [future.result(timeout=5) for future in pending]
                assert all(result == results[0] for result in results)
                assert "PDF evidence" in results[0][0]["text"]
            assert parse.call_count == 1
        if fail:
            with patch.object(processor, "_extract", return_value=extracted("retry succeeded")) as retry:
                assert "retry succeeded" in processor.process(document(path))[0]["text"]
                retry.assert_called_once()
    finally:
        processor.close()


@pytest.mark.parametrize("separate_processors", [False, True])
def test_different_documents_never_parse_simultaneously(tmp_path, separate_processors):
    first, second = tmp_path / "first.pdf", tmp_path / "second.pdf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    processor = PDFProcessor("native_only")
    other = PDFProcessor("native_only") if separate_processors else processor
    first_entered, second_read, release = Event(), Event(), Event()
    overlap = Event()
    guard = Lock()
    active = 0
    read_bytes = Path.read_bytes

    def read(candidate):
        result = read_bytes(candidate)
        if candidate == second:
            second_read.set()
        return result

    def extract(_processor, path, digest):
        nonlocal active
        with guard:
            active += 1
            if active > 1:
                overlap.set()
        try:
            if path == first:
                first_entered.set()
                assert release.wait(5), "first extraction was never released"
            return extracted(path.name)
        finally:
            with guard:
                active -= 1

    try:
        with patch.object(Path, "read_bytes", read), patch.object(
            PDFProcessor, "_extract", extract
        ), ThreadPoolExecutor(max_workers=2) as pool:
            one = pool.submit(processor.process, document(first))
            try:
                assert first_entered.wait(5)
                two = pool.submit(other.process, document(second))
                assert second_read.wait(5)
                assert not overlap.wait(0.1), "PDF parsing overlapped"
            finally:
                release.set()
            assert "first.pdf" in one.result(timeout=5)[0]["text"]
            assert "second.pdf" in two.result(timeout=5)[0]["text"]
    finally:
        for item in {processor, other}:
            item.close()


def test_replacing_pdf_content_invalidates_cached_result(tmp_path):
    path = tmp_path / "changing.pdf"
    path.write_bytes(b"old")
    with PDFProcessor("native_only") as processor, patch.object(
        processor, "_extract", side_effect=[extracted("old text"), extracted("new text")]
    ) as parse:
        old = processor.process(document(path))[0]
        path.write_bytes(b"new")
        new = processor.process(document(path))[0]
        assert "old text" in old["text"] and "new text" in new["text"]
        assert old["pdf_processing"]["sha256"] != new["pdf_processing"]["sha256"]
        assert processor.process(document(path))[0] == new
        assert parse.call_count == 2


def test_close_waits_for_pending_parse_and_rejects_further_pdf_work(tmp_path):
    path = tmp_path / "source.pdf"
    path.write_bytes(b"content")
    processor = PDFProcessor("native_only")
    entered, release, closing = Event(), Event(), Event()

    def extract(*args):
        entered.set()
        assert release.wait(5), "extraction was never released"
        return extracted()

    def close():
        closing.set()
        processor.close()

    with patch.object(processor, "_extract", side_effect=extract), ThreadPoolExecutor(
        max_workers=2
    ) as pool:
        result = pool.submit(processor.process, document(path))
        try:
            assert entered.wait(5)
            closed = pool.submit(close)
            assert closing.wait(5)
            assert not wait([closed], timeout=0.1).done
        finally:
            release.set()
        assert "PDF evidence" in result.result(timeout=5)[0]["text"]
        closed.result(timeout=5)
    processor.close()
    with pytest.raises(RuntimeError, match="closed"):
        processor.process(document(path))


@pytest.mark.parametrize("fail_answer", [False, True])
def test_native_pdf_oracle_answers_overlap_and_reader_closes(tmp_path, fail_answer):
    fitz = pytest.importorskip("pymupdf")
    path = tmp_path / "lesson.pdf"
    with fitz.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Flat White has thin foam.")
        pdf.save(path)
    (tmp_path / "manifest.json").write_text("{}")
    rows = {
        "contexts": [{"context_id": "c1"}],
        "assets": [{"asset_id": "pdf1", "path": path.name, "mime_type": "application/pdf"}],
        "memories": [{"context_id": "c1", "memory_id": "m1", "content": [
            {"type": "document", "asset_id": "pdf1"}
        ]}],
        "questions": [{"context_id": "c1", "question_id": f"q{i}", "prompt": [
            {"type": "text", "text": "What kind of foam?"}
        ], "evidence": [{"memory_id": "m1"}]} for i in range(2)],
    }
    for table, values in rows.items():
        (tmp_path / f"{table}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in values))
    together = Barrier(2)

    class AnswerModel:
        def complete(self, messages, **kwargs):
            assert "Flat White has thin foam" in json.dumps(messages)
            together.wait(timeout=5)
            if fail_answer:
                raise RuntimeError("answer failed")
            return "thin foam"

    reader = BundleReader(tmp_path, pdf_policy="native_only", pdf_page_images=1)
    output = tmp_path / "predictions.jsonl"
    with patch("mm_memory_bench.evaluation.oracle.BundleReader", return_value=reader), patch.object(
        reader.pdf_processor, "_extract", wraps=reader.pdf_processor._extract
    ) as parse:
        if fail_answer:
            with pytest.raises(RuntimeError, match="answer failed"):
                run_oracle_bundle(tmp_path, output, answer_model=AnswerModel(), concurrency=2,
                                  pdf_policy="native_only", pdf_page_images=1)
        else:
            summary = run_oracle_bundle(tmp_path, output, answer_model=AnswerModel(), concurrency=2,
                                        pdf_policy="native_only", pdf_page_images=1)
            assert summary["predictions"] == 2
            assert all(json.loads(line)["prediction"] == "thin foam" for line in output.read_text().splitlines())
        assert parse.call_count == 1
    with pytest.raises(RuntimeError, match="closed"):
        reader.resolve_content([{"type": "document", "asset_id": "pdf1"}])
    images = list((tmp_path / ".pdf_cache").rglob("*.png"))
    assert len(images) == 1
    from PIL import Image

    with Image.open(images[0]) as rendered:
        rendered.verify()


def test_failed_page_image_write_is_not_published_and_can_retry(tmp_path):
    fitz = pytest.importorskip("pymupdf")
    from PIL import Image

    path = tmp_path / "lesson.pdf"
    with fitz.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Evidence for the rendered page.")
        pdf.save(path)
    cache = tmp_path / "cache"

    def interrupted_save(pixmap, target, *args, **kwargs):
        Path(target).write_bytes(b"partial PNG")
        raise OSError("page image write interrupted")

    with PDFProcessor("native_only", page_images=1, cache_dir=cache) as processor:
        with patch.object(fitz.Pixmap, "save", interrupted_save):
            with pytest.raises(OSError, match="page image write interrupted"):
                processor.process(document(path))
        assert not list(cache.rglob("*.png")), "an incomplete page image was published"
        result = processor.process(document(path))
        image = Path(result[1]["path"])
        with Image.open(image) as rendered:
            rendered.verify()
        assert list(cache.rglob("*.png")) == [image]
