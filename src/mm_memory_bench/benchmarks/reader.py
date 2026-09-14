from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any, Iterator, Mapping

from .bundle import index_by, iter_jsonl, read_json, resolve_asset_path


@dataclass(frozen=True)
class ContextBatch:
    """One memory ingestion unit and all questions evaluated against it."""

    context: dict[str, Any]
    memories: list[dict[str, Any]]
    questions: list[dict[str, Any]]


def _context_groups(path: Path) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    rows = iter_jsonl(path)
    for context_id, group in groupby(rows, key=lambda row: str(row["context_id"])):
        yield context_id, list(group)


class BundleReader:
    """Streaming reader that lets a memory system ingest each context once.

    Canonical converters guarantee that memories and questions are contiguous in
    the same context order as ``contexts.jsonl``. This reader therefore avoids
    loading long per-query histories for every question or building a second index.
    """

    def __init__(self, root: Path, *, caption_sidecar: Path | None = None,
                 pdf_policy: str = "off", pdf_page_images: int = 0) -> None:
        from ..preprocessing.pdf import PDFProcessor

        self.root = root.resolve()
        self.pdf_processor = PDFProcessor(
            pdf_policy, page_images=pdf_page_images, cache_dir=self.root / ".pdf_cache"
        )
        self.manifest = read_json(self.root / "manifest.json")
        table_names = self.manifest.get("tables", {})
        self.table_paths = {
            table: self.root / str(table_names.get(table, f"{table}.jsonl"))
            for table in ("contexts", "memories", "assets", "questions")
        }
        self.assets = index_by(iter_jsonl(self.table_paths["assets"]), "asset_id")
        self.asset_captions: dict[str, str] = {}
        if caption_sidecar is not None:
            sidecar_path = caption_sidecar.resolve()
            for row in iter_jsonl(sidecar_path):
                asset_id = str(row.get("asset_id", ""))
                caption = str(row.get("caption", "")).strip()
                if not asset_id or not caption:
                    raise ValueError(
                        f"{sidecar_path}: caption rows need non-empty asset_id and caption"
                    )
                if asset_id not in self.assets:
                    raise ValueError(
                        f"{sidecar_path}: caption references unknown asset_id={asset_id}"
                    )
                previous = self.asset_captions.get(asset_id)
                if previous is not None and previous != caption:
                    raise ValueError(
                        f"{sidecar_path}: conflicting captions for asset_id={asset_id}"
                    )
                self.asset_captions[asset_id] = caption

    def close(self) -> None:
        self.pdf_processor.close()

    def __enter__(self) -> BundleReader:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def captions_for_content(self, content: list[dict[str, Any]]) -> list[str]:
        """Return sidecar captions in content order for referenced media assets."""
        return [
            self.asset_captions[str(part["asset_id"])]
            for part in content
            if part.get("type") in {"image", "video"}
            and str(part.get("asset_id", "")) in self.asset_captions
        ]

    def resolve_content(self, content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return content blocks with an absolute ``path`` for media blocks."""
        resolved = []
        for part in content:
            value = dict(part)
            asset_id = value.get("asset_id")
            if asset_id:
                asset = self.assets[str(asset_id)]
                value["path"] = str(resolve_asset_path(self.root, asset))
                value["asset"] = dict(asset)
            resolved.extend(self.pdf_processor.process(value))
        return resolved

    def iter_context_batches(
        self,
        *,
        subset: str | None = None,
        split: str | None = None,
        task_subcategory: str | None = None,
        question_ids: set[str] | None = None,
    ) -> Iterator[ContextBatch]:
        memory_groups = iter(_context_groups(self.table_paths["memories"]))
        question_groups = iter(_context_groups(self.table_paths["questions"]))
        next_memories = next(memory_groups, None)
        next_questions = next(question_groups, None)

        for context in iter_jsonl(self.table_paths["contexts"]):
            context_id = str(context["context_id"])
            memories: list[dict[str, Any]] = []
            questions: list[dict[str, Any]] = []
            if next_memories and next_memories[0] == context_id:
                memories = next_memories[1]
                next_memories = next(memory_groups, None)
            if next_questions and next_questions[0] == context_id:
                questions = next_questions[1]
                next_questions = next(question_groups, None)

            if subset is not None:
                questions = [q for q in questions if q.get("subset") == subset]
            if split is not None:
                questions = [q for q in questions if q.get("split") == split]
            if task_subcategory is not None:
                questions = [
                    q
                    for q in questions
                    if isinstance(q.get("task"), Mapping)
                    and q["task"].get("subcategory") == task_subcategory
                ]
            if question_ids is not None:
                questions = [
                    q for q in questions if str(q.get("question_id")) in question_ids
                ]
            if questions:
                yield ContextBatch(context=context, memories=memories, questions=questions)

        if next_memories is not None or next_questions is not None:
            dangling = (next_memories or next_questions or ("unknown", []))[0]
            raise ValueError(
                "table context order does not match contexts.jsonl; "
                f"first unconsumed context_id={dangling}"
            )
