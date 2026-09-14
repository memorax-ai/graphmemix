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

Use a **new method checkpoint directory and prediction output** when enabling/changing PDF input. Method runs reject checkpoints with an incompatible recorded PDF policy, and enabling PDF processing rejects nonempty legacy checkpoint directories. The extraction rules below use policy marker version 2, so PDF checkpoints from version 1 must also be rebuilt. Do not resume predictions made with another policy or extraction version. A policy marker is stored as `pdf-input.json` in the method checkpoint directory; summaries record the policy and image limit.

| Policy | Behavior |
|---|---|
| `off` (default) | Previous behavior, no PDF parsing dependency needed |
| `native_only` | Extract native text from the first 100 pages with PyMuPDF |
| `native_then_ocr` | Extract native text from the first 100 pages. If their combined stripped text has fewer than 400 characters, replace it with OCR text from the first 15 pages at 150 DPI |
| `ocr_pages` | Use only OCR text from the first 100 pages at 150 DPI |

These are upper limits; shorter PDFs use all available pages. The fallback threshold applies to the combined native text, so a sparse page does not trigger OCR when the total reaches 400 characters. When fallback runs, only its OCR output is used, even if it is empty; native text is not retained. `pdf_processing.page_count` records the number of pages used for text extraction.

OCR currently uses Tesseract's default English language configuration. Install/configure appropriate language support before applying it to non-English scans. OCR is not a guarantee of correct table structure or visual interpretation. Unreadable files, missing OCR dependencies, or a PDF with neither readable text nor requested page images cause explicit errors instead of silent path-only fallback.

`--pdf-page-images N` renders the first N pages **per PDF** at 144 DPI, independently of the text extraction page limit, without using questions or gold evidence to select pages. Later pages are not visually available unless N is increased. Within the text page limits above, this shared layer does not truncate characters; indexing and prompt costs still depend on document length. Raw-page images may also exceed a model/provider's image limits. PDF figures can remain invisible under text-only input, even when extraction succeeds.

Page images are cached under `<bundle>/.pdf_cache/<content-sha256>/`; extracted text is cached within each reader instance. Restarting a run repeats text/OCR extraction. The bundle directory must be writable when rendering pages. This layer does not modify model configurations or implement method-specific reranking, stopping logic or question-directed page selection.

## Relationship to M³Exam's baselines

The three enabled text policies follow the defaults of the official baseline utility's [`extract_pdf_pages_for_eval`](https://github.com/EverM0re/M-3-Exam/blob/main/baselines/_runtime/multimodel_common/pdf_session_text.py): the 400-character combined threshold, 100-page native/OCR-only limit, 15-page fallback limit, and replacement of native text by OCR on fallback.

This aligns text selection rules, not the full evaluation protocol. PDF text remains attached to its source memory; the official UniversalRAG baseline also injects session PDF text into dialogue rounds and answer context. The helper's separate indexing and answer-context character budgets are not applied inside this shared parser. Page images retain Graphmemix's per-PDF budget and 144 DPI rendering; the official vision helper instead shares a total page budget across an answer's PDFs and defaults to 120 DPI. PDF structure normalization and M³Proctor's two-stage answerer are not included.

## Validation scope

Tests cover default-off compatibility, native extraction, combined OCR thresholds and page limits (mock OCR), rendering and image-message encoding, public-input filtering and preservation of memory IDs, checkpoint-policy/version mismatch rejection, and extracted text reaching all five method ingestion paths. Full model inference and Judge evaluation must still be run separately; these tests do not establish answer accuracy.
