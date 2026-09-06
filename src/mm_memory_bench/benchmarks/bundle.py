from __future__ import annotations

import json
import mimetypes
import os
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .. import SCHEMA_VERSION


TABLES = ("contexts", "memories", "assets", "questions")
CONTENT_TYPES = {"text", "image", "video", "audio", "document", "table"}
RESPONSE_TYPES = {"text", "choice", "structured_json"}
TOOL_SENSITIVE_KEYS = frozenset(
    {
        "answer",
        "answers",
        "correct_answer",
        "evaluation_private",
        "evaluation_variants",
        "evidence",
        "gold",
        "gold_answer",
        "ground_truth",
        "metadata",
        "provenance",
        "reference_answer",
        "solution",
    }
)
MEMORY_KINDS = {
    "dialogue_message",
    "email",
    "media",
    "profile",
    "document",
    "table",
    "artifact",
    "state",
}


class SchemaError(ValueError):
    """Raised when a canonical record violates the MMMB contract."""


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise SchemaError(f"{path}:{line_number}: each JSONL row must be an object")
            yield value


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def stable_id(*parts: Any) -> str:
    """Create a readable, stable namespaced identifier."""
    cleaned = []
    for part in parts:
        text = str(part).strip().replace("\\", "/")
        text = text.replace(":", "_").replace(" ", "_")
        cleaned.append(text.strip("/"))
    return ":".join(part for part in cleaned if part)


def provenance(source_file: Path, raw_root: Path, json_pointer: str = "") -> dict[str, str]:
    try:
        relative = source_file.resolve().relative_to(raw_root.resolve())
        source = relative.as_posix()
    except ValueError:
        source = os.path.relpath(source_file.resolve(), raw_root.resolve())
    result = {"source_file": source}
    if json_pointer:
        result["json_pointer"] = json_pointer
    return result


def content_text(text: Any, **annotations: Any) -> dict[str, Any]:
    part: dict[str, Any] = {"type": "text", "text": "" if text is None else str(text)}
    if annotations:
        part["annotations"] = annotations
    return part


def content_asset(media_type: str, asset_id: str, **annotations: Any) -> dict[str, Any]:
    if media_type not in CONTENT_TYPES - {"text"}:
        raise SchemaError(f"unsupported asset content type: {media_type}")
    part: dict[str, Any] = {"type": media_type, "asset_id": asset_id}
    if annotations:
        part["annotations"] = annotations
    return part


def guess_media_type(path: Path) -> tuple[str, str | None]:
    mime, _ = mimetypes.guess_type(path.name)
    if mime:
        media_type = mime.split("/", 1)[0]
        if media_type in {"image", "video", "audio"}:
            return media_type, mime
        if mime in {"application/pdf", "text/plain", "text/html"}:
            return "document", mime
    return "document", mime


def _require_string(record: Mapping[str, Any], field: str, table: str) -> None:
    if not isinstance(record.get(field), str) or not record[field].strip():
        raise SchemaError(f"{table}.{field} must be a non-empty string")


def _validate_content(content: Any, table: str) -> None:
    if not isinstance(content, list) or not content:
        raise SchemaError(f"{table}.content must be a non-empty list")
    for index, part in enumerate(content):
        if not isinstance(part, dict) or part.get("type") not in CONTENT_TYPES:
            raise SchemaError(f"{table}.content[{index}] has an invalid type")
        if part["type"] == "text":
            if not isinstance(part.get("text"), str):
                raise SchemaError(f"{table}.content[{index}].text must be a string")
        elif not isinstance(part.get("asset_id"), str) or not part["asset_id"]:
            raise SchemaError(f"{table}.content[{index}].asset_id must be a string")


def _validate_choices(choices: Any) -> None:
    if not isinstance(choices, list):
        raise SchemaError("questions.choices must be a list")
    for index, choice in enumerate(choices):
        if not isinstance(choice, Mapping):
            raise SchemaError(f"questions.choices[{index}] must be an object")
        if not isinstance(choice.get("choice_id"), str) or not choice["choice_id"].strip():
            raise SchemaError(
                f"questions.choices[{index}].choice_id must be a non-empty string"
            )
        if not isinstance(choice.get("text"), str):
            raise SchemaError(f"questions.choices[{index}].text must be a string")
        if "content" in choice:
            _validate_content(choice["content"], f"questions.choices[{index}]")


def normalized_tool_key(key: str) -> str:
    """Normalize a tool key before applying the model-visibility denylist."""
    return key.strip().casefold().replace("-", "_").replace(" ", "_")


def _validate_tool_value(value: Any, path: str) -> None:
    """Validate recursively JSON-compatible, gold-free public tool declarations."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_tool_value(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaError(f"{path}: tool object keys must be strings")
            if normalized_tool_key(key) in TOOL_SENSITIVE_KEYS:
                raise SchemaError(f"{path}.{key}: sensitive key is forbidden in questions.tools")
            _validate_tool_value(item, f"{path}.{key}")
        return
    raise SchemaError(f"{path}: tools must contain only JSON-compatible values")


def _validate_tools(tools: Any) -> None:
    if not isinstance(tools, list):
        raise SchemaError("questions.tools must be a list")
    for index, tool in enumerate(tools):
        if not isinstance(tool, Mapping):
            raise SchemaError(f"questions.tools[{index}] must be an object")
        _validate_tool_value(tool, f"questions.tools[{index}]")


def validate_record(table: str, record: Mapping[str, Any]) -> None:
    if table not in TABLES:
        raise SchemaError(f"unknown table: {table}")
    id_field = {
        "contexts": "context_id",
        "memories": "memory_id",
        "assets": "asset_id",
        "questions": "question_id",
    }[table]
    _require_string(record, id_field, table)

    if table != "contexts":
        if table != "assets":
            _require_string(record, "context_id", table)

    if table == "contexts":
        _require_string(record, "benchmark", table)
    elif table == "memories":
        if record.get("kind") not in MEMORY_KINDS:
            raise SchemaError(f"memories.kind is invalid: {record.get('kind')!r}")
        if not isinstance(record.get("sequence"), int) or record["sequence"] < 0:
            raise SchemaError("memories.sequence must be a non-negative integer")
        _validate_content(record.get("content"), table)
    elif table == "assets":
        if record.get("media_type") not in CONTENT_TYPES - {"text"}:
            raise SchemaError(f"assets.media_type is invalid: {record.get('media_type')!r}")
        _require_string(record, "path", table)
    elif table == "questions":
        if not isinstance(record.get("prompt"), list) or not record["prompt"]:
            raise SchemaError("questions.prompt must be a non-empty content list")
        _validate_content(record["prompt"], "questions.prompt")
        if "instruction" in record and not isinstance(record["instruction"], str):
            raise SchemaError("questions.instruction must be a string")
        if "tools" in record:
            _validate_tools(record["tools"])
        if "choices" in record:
            _validate_choices(record["choices"])
        task = record.get("task")
        if task is not None:
            if not isinstance(task, Mapping):
                raise SchemaError("questions.task must be an object")
            response_type = task.get("response_type")
            if response_type is not None and response_type not in RESPONSE_TYPES:
                allowed = ", ".join(sorted(RESPONSE_TYPES))
                raise SchemaError(
                    f"questions.task.response_type must be one of: {allowed}"
                )
        answer = record.get("answer")
        if not isinstance(answer, dict) or not isinstance(answer.get("text"), str):
            raise SchemaError("questions.answer.text must be a string")
        for evidence_field in ("evidence", "misleading_evidence"):
            if evidence_field in record:
                evidence_rows = record[evidence_field]
                if not isinstance(evidence_rows, list) or any(
                    not isinstance(evidence, Mapping) for evidence in evidence_rows
                ):
                    raise SchemaError(
                        f"questions.{evidence_field} must be a list of objects"
                    )
        scope = record.get("memory_scope")
        if not isinstance(scope, dict) or scope.get("mode") not in {"all", "prefix", "ids"}:
            raise SchemaError("questions.memory_scope.mode must be all, prefix, or ids")
        if scope["mode"] == "prefix" and not isinstance(scope.get("max_sequence"), int):
            raise SchemaError("questions.memory_scope.max_sequence must be an integer for prefix mode")
        if scope["mode"] == "ids":
            memory_ids = scope.get("memory_ids")
            if not isinstance(memory_ids, list) or any(
                not isinstance(memory_id, str) or not memory_id for memory_id in memory_ids
            ):
                raise SchemaError("questions.memory_scope.memory_ids must be a list of strings for ids mode")


class BundleWriter(AbstractContextManager["BundleWriter"]):
    """Streaming writer for one canonical benchmark bundle."""

    def __init__(
        self,
        root: Path,
        manifest: Mapping[str, Any],
        *,
        overwrite: bool = False,
    ) -> None:
        self.root = root
        self.manifest = dict(manifest)
        self.overwrite = overwrite
        self._handles: dict[str, Any] = {}
        self._ids: dict[str, set[str]] = {name: set() for name in TABLES}
        self._counts: dict[str, int] = {name: 0 for name in TABLES}

    def __enter__(self) -> "BundleWriter":
        self.root.mkdir(parents=True, exist_ok=True)
        paths = [self.root / f"{name}.jsonl" for name in TABLES]
        paths.append(self.root / "manifest.json")
        existing = [path for path in paths if path.exists()]
        if existing and not self.overwrite:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(f"bundle already contains generated files: {names}")
        for table in TABLES:
            self._handles[table] = (self.root / f"{table}.jsonl").open(
                "w", encoding="utf-8"
            )
        return self

    def add(self, table: str, record: Mapping[str, Any]) -> None:
        validate_record(table, record)
        id_field = {
            "contexts": "context_id",
            "memories": "memory_id",
            "assets": "asset_id",
            "questions": "question_id",
        }[table]
        record_id = str(record[id_field])
        if record_id in self._ids[table]:
            raise SchemaError(f"duplicate {table} identifier: {record_id}")
        self._ids[table].add(record_id)
        self._counts[table] += 1
        self._handles[table].write(json.dumps(dict(record), ensure_ascii=False) + "\n")

    def add_context(self, record: Mapping[str, Any]) -> None:
        self.add("contexts", record)

    def add_memory(self, record: Mapping[str, Any]) -> None:
        self.add("memories", record)

    def add_asset(self, record: Mapping[str, Any]) -> None:
        self.add("assets", record)

    def add_question(self, record: Mapping[str, Any]) -> None:
        self.add("questions", record)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        for handle in self._handles.values():
            handle.close()
        if exc_type is None:
            payload = {
                **self.manifest,
                "schema_version": SCHEMA_VERSION,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "tables": {name: f"{name}.jsonl" for name in TABLES},
                "counts": self._counts,
            }
            write_json(self.root / "manifest.json", payload)
        return False


def load_bundle(root: Path) -> dict[str, list[dict[str, Any]]]:
    return {table: list(iter_jsonl(root / f"{table}.jsonl")) for table in TABLES}


def resolve_asset_path(bundle_root: Path, asset: Mapping[str, Any]) -> Path:
    path = Path(str(asset["path"]))
    return path if path.is_absolute() else (bundle_root / path).resolve()


def validate_bundle(root: Path, *, check_assets: bool = False) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise SchemaError(f"missing manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(
            f"unsupported schema_version={manifest.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )

    ids: dict[str, set[str]] = {name: set() for name in TABLES}
    counts: dict[str, int] = {name: 0 for name in TABLES}
    id_fields = {
        "contexts": "context_id",
        "memories": "memory_id",
        "assets": "asset_id",
        "questions": "question_id",
    }
    missing_assets: list[str] = []
    context_order: list[str] = []
    for table in TABLES:
        path = root / str(manifest.get("tables", {}).get(table, f"{table}.jsonl"))
        if not path.exists():
            raise SchemaError(f"missing table: {path}")
        for record in iter_jsonl(path):
            validate_record(table, record)
            record_id = record[id_fields[table]]
            if record_id in ids[table]:
                raise SchemaError(f"duplicate {table} identifier: {record_id}")
            ids[table].add(record_id)
            counts[table] += 1
            if table == "contexts":
                context_order.append(str(record_id))
            if table == "assets" and check_assets:
                if not resolve_asset_path(root, record).exists():
                    missing_assets.append(record_id)

    context_position = {
        context_id: position for position, context_id in enumerate(context_order)
    }
    memory_context: dict[str, str] = {}
    memory_sequences: dict[str, set[int]] = {}
    closed_memory_contexts: set[str] = set()
    current_memory_context: str | None = None
    last_memory_context_position = -1
    for record in iter_jsonl(
        root / str(manifest.get("tables", {}).get("memories", "memories.jsonl"))
    ):
        context_id = str(record["context_id"])
        if context_id not in ids["contexts"]:
            raise SchemaError(
                f"{record['memory_id']} references missing "
                f"context {context_id}"
            )
        if context_id != current_memory_context:
            if context_id in closed_memory_contexts:
                raise SchemaError(
                    f"memories context {context_id} is split into non-contiguous groups"
                )
            position = context_position[context_id]
            if position <= last_memory_context_position:
                raise SchemaError(
                    "memories context groups do not follow contexts.jsonl order: "
                    f"{context_id}"
                )
            if current_memory_context is not None:
                closed_memory_contexts.add(current_memory_context)
            current_memory_context = context_id
            last_memory_context_position = position
        sequence = int(record["sequence"])
        sequences = memory_sequences.setdefault(context_id, set())
        if sequence in sequences:
            raise SchemaError(
                f"memories context {context_id} has duplicate sequence {sequence}"
            )
        sequences.add(sequence)
        memory_context[str(record["memory_id"])] = context_id
        for part in record["content"]:
            asset_id = part.get("asset_id")
            if asset_id and asset_id not in ids["assets"]:
                raise SchemaError(f"{record['memory_id']} references missing asset {asset_id}")

    closed_question_contexts: set[str] = set()
    current_question_context: str | None = None
    last_question_context_position = -1
    for record in iter_jsonl(
        root / str(manifest.get("tables", {}).get("questions", "questions.jsonl"))
    ):
        context_id = str(record["context_id"])
        if context_id not in ids["contexts"]:
            raise SchemaError(
                f"{record['question_id']} references missing context {context_id}"
            )
        if context_id != current_question_context:
            if context_id in closed_question_contexts:
                raise SchemaError(
                    f"questions context {context_id} is split into non-contiguous groups"
                )
            position = context_position[context_id]
            if position <= last_question_context_position:
                raise SchemaError(
                    "questions context groups do not follow contexts.jsonl order: "
                    f"{context_id}"
                )
            if current_question_context is not None:
                closed_question_contexts.add(current_question_context)
            current_question_context = context_id
            last_question_context_position = position
        for part in record["prompt"]:
            asset_id = part.get("asset_id")
            if asset_id and asset_id not in ids["assets"]:
                raise SchemaError(f"{record['question_id']} references missing asset {asset_id}")
        for choice in record.get("choices", []):
            for part in choice.get("content", []):
                asset_id = part.get("asset_id")
                if asset_id and asset_id not in ids["assets"]:
                    raise SchemaError(
                        f"{record['question_id']} choice references missing asset {asset_id}"
                    )
        for evidence_field in ("evidence", "misleading_evidence"):
            for evidence in record.get(evidence_field, []):
                memory_id = evidence.get("memory_id")
                asset_id = evidence.get("asset_id")
                if memory_id and memory_id not in ids["memories"]:
                    raise SchemaError(
                        f"{record['question_id']} references missing memory {memory_id}"
                    )
                if memory_id and memory_context[str(memory_id)] != context_id:
                    raise SchemaError(
                        f"{record['question_id']} {evidence_field} references memory "
                        f"{memory_id} from another context"
                    )
                if asset_id and asset_id not in ids["assets"]:
                    raise SchemaError(
                        f"{record['question_id']} references missing asset {asset_id}"
                    )
        scope = record["memory_scope"]
        if scope["mode"] == "ids":
            missing_scope_ids = [
                memory_id for memory_id in scope["memory_ids"] if memory_id not in ids["memories"]
            ]
            if missing_scope_ids:
                raise SchemaError(
                    f"{record['question_id']} memory_scope references missing memories: "
                    + ", ".join(missing_scope_ids[:5])
                )
            cross_context_scope_ids = [
                memory_id
                for memory_id in scope["memory_ids"]
                if memory_id in memory_context
                and memory_context[memory_id] != context_id
            ]
            if cross_context_scope_ids:
                raise SchemaError(
                    f"{record['question_id']} memory_scope references memories from "
                    "another context: " + ", ".join(cross_context_scope_ids[:5])
                )

    expected = manifest.get("counts", {})
    if expected and expected != counts:
        raise SchemaError(f"manifest counts do not match tables: expected={expected}, actual={counts}")
    return {"counts": counts, "missing_asset_count": len(missing_assets), "missing_assets": missing_assets}


def index_by(records: Iterable[Mapping[str, Any]], key: str) -> dict[str, Mapping[str, Any]]:
    return {str(record[key]): record for record in records}
