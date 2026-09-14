# Shared PDF input processing

PDF processing is opt-in and happens in `BundleReader.resolve_content()` before memory ingestion and Oracle evidence generation. `preprocessing/pdf.py` provides the shared implementation. It reads only attached files; it never reads questions, answers or gold evidence to choose content.

The original `document` block receives extracted text with filename and page labels. Optional rendered pages become additional image blocks inside the **same memory**. Memory IDs, evidence IDs and the source bundle are unchanged. All methods use their existing text/image ingestion paths: caption-based methods can describe pages; visual methods can retrieve page images. This does not force every retrieved memory's pages into the final answer.

## Installation and invocation

```sh
python -m pip install -e '.[pdf]'
# Required only for OCR; install the Tesseract executable separately:
# macOS: brew install tesseract
# Debian/Ubuntu: apt-get install tesseract-ocr
```

Append these options to an existing `run-method` or `run-oracle` command:

```sh
--pdf-policy native_then_ocr
```

To additionally expose the first three pages of each PDF as images:

```sh
--pdf-policy native_then_ocr --pdf-page-images 3
```

Use a **new method checkpoint directory and prediction output** when enabling/changing PDF input. Method runs reject checkpoints with an incompatible recorded PDF policy, and enabling PDF processing rejects nonempty legacy checkpoint directories. Do not resume predictions made with another policy. A policy marker is stored as `pdf-input.json` in the method checkpoint directory; summaries record the policy and image limit.

| Policy | Behavior |
|---|---|
| `off` (default) | Previous behavior, no PDF parsing dependency needed |
| `native_only` | Extract all pages' native text with PyMuPDF |
| `native_then_ocr` | Extract native text; supplement pages with fewer than 400 characters using Tesseract at 150 DPI |
| `ocr_pages` | Run OCR on every page, retaining native text when present |

OCR currently uses Tesseract's default English language configuration. Install/configure appropriate language support before applying it to non-English scans. OCR is not a guarantee of correct table structure or visual interpretation. Unreadable files, missing OCR dependencies, or a PDF with neither readable text nor requested page images cause explicit errors instead of silent path-only fallback.

`--pdf-page-images N` renders the first N pages at 144 DPI, not question-selected or gold-selected pages. This is a deliberate cap: later pages are not visually available unless N is increased. Text extraction covers all pages without truncation; long PDFs can therefore increase indexing and prompt costs. Raw-page images may also exceed a model/provider's image limits. PDF figures can remain invisible under text-only input, even when extraction succeeds.

Page images are cached under `<bundle>/.pdf_cache/<content-sha256>/`; extracted text is cached within each reader instance. Restarting a run repeats text/OCR extraction. The bundle directory must be writable when rendering pages. This layer does not modify model configurations or implement method-specific reranking, stopping logic or question-directed page selection.

## Relationship to M³Exam's baselines

Inspired by the official baseline utility [pdf_session_text.py](https://github.com/EverM0re/M-3-Exam/blob/main/baselines/_runtime/multimodel_common/pdf_session_text.py): native text extraction, OCR fallback and optional page rendering. This implementation is independent and is not an exact reproduction of its evaluation protocol. In particular, fallback is checked per page (the official helper checks total extracted characters), and PDF text remains attached to its source memory instead of being copied into all rounds of a session. M³Proctor's two-stage answerer is not added to other methods.

## Validation scope

Tests cover default-off compatibility, native extraction, OCR branching (mock OCR), rendering and image-message encoding, public-input filtering and preservation of memory IDs, checkpoint-policy mismatch rejection, and extracted text reaching all five method ingestion paths. Full model inference and Judge evaluation must still be run separately; these tests do not establish answer accuracy.
