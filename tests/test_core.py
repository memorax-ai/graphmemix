import json
import tempfile
import unittest
from pathlib import Path

from mm_memory_bench.core import BundleWriter, SchemaError, content_text, validate_bundle
from mm_memory_bench.reader import BundleReader


class BundleWriterTest(unittest.TestCase):
    def test_minimal_bundle_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            with BundleWriter(root, {"benchmark": "fixture"}) as writer:
                writer.add_context({"context_id": "fixture:c0", "benchmark": "fixture"})
                writer.add_memory(
                    {
                        "memory_id": "fixture:m0",
                        "context_id": "fixture:c0",
                        "sequence": 0,
                        "kind": "artifact",
                        "content": [content_text("memory")],
                    }
                )
                writer.add_question(
                    {
                        "question_id": "fixture:q0",
                        "context_id": "fixture:c0",
                        "prompt": [content_text("question")],
                        "answer": {"text": "answer"},
                        "memory_scope": {"mode": "all"},
                    }
                )
            report = validate_bundle(root)
            self.assertEqual(report["counts"]["contexts"], 1)
            self.assertEqual(report["counts"]["memories"], 1)
            self.assertEqual(report["counts"]["questions"], 1)
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "mmmb-1.0")
            batches = list(BundleReader(root).iter_context_batches())
            self.assertEqual(len(batches), 1)
            self.assertEqual(batches[0].memories[0]["memory_id"], "fixture:m0")
            self.assertEqual(batches[0].questions[0]["question_id"], "fixture:q0")

    def test_missing_context_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            with BundleWriter(root, {"benchmark": "fixture"}) as writer:
                writer.add_question(
                    {
                        "question_id": "fixture:q0",
                        "context_id": "fixture:missing",
                        "prompt": [content_text("question")],
                        "answer": {"text": "answer"},
                        "memory_scope": {"mode": "all"},
                    }
                )
            with self.assertRaises(SchemaError):
                validate_bundle(root)

    def test_question_tools_accept_common_schemas_and_reject_sensitive_keys(self) -> None:
        valid_tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a value.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer", "default": 5},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "function_name": "search",
                "default_arguments": {"limit": 5},
                "function_comment": "Search records.",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "valid"
            with BundleWriter(root, {"benchmark": "fixture"}) as writer:
                writer.add_context({"context_id": "fixture:c0", "benchmark": "fixture"})
                writer.add_question(
                    {
                        "question_id": "fixture:q0",
                        "context_id": "fixture:c0",
                        "prompt": [content_text("question")],
                        "tools": valid_tools,
                        "answer": {"text": "answer"},
                        "memory_scope": {"mode": "all"},
                    }
                )
            self.assertEqual(validate_bundle(root)["counts"]["questions"], 1)

            invalid_root = Path(directory) / "invalid"
            with self.assertRaisesRegex(SchemaError, "sensitive key"):
                with BundleWriter(invalid_root, {"benchmark": "fixture"}) as writer:
                    writer.add_context(
                        {"context_id": "fixture:c0", "benchmark": "fixture"}
                    )
                    writer.add_question(
                        {
                            "question_id": "fixture:q0",
                            "context_id": "fixture:c0",
                            "prompt": [content_text("question")],
                            "tools": [
                                {
                                    "name": "lookup",
                                    "parameters": {
                                        "type": "object",
                                        "evidence": "SECRET_TOOL_EVIDENCE",
                                    },
                                }
                            ],
                            "answer": {"text": "answer"},
                            "memory_scope": {"mode": "all"},
                        }
                    )

    def test_question_response_type_and_choice_contracts_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            invalid_response_root = Path(directory) / "invalid-response"
            with self.assertRaisesRegex(SchemaError, "response_type"):
                with BundleWriter(
                    invalid_response_root, {"benchmark": "fixture"}
                ) as writer:
                    writer.add_context(
                        {"context_id": "fixture:c0", "benchmark": "fixture"}
                    )
                    writer.add_question(
                        {
                            "question_id": "fixture:q0",
                            "context_id": "fixture:c0",
                            "prompt": [content_text("question")],
                            "task": {"response_type": "multiple_choice"},
                            "choices": [],
                            "answer": {"text": "answer"},
                            "memory_scope": {"mode": "all"},
                        }
                    )

            invalid_choice_root = Path(directory) / "invalid-choice"
            with self.assertRaisesRegex(SchemaError, r"choices\[0\] must be an object"):
                with BundleWriter(
                    invalid_choice_root, {"benchmark": "fixture"}
                ) as writer:
                    writer.add_context(
                        {"context_id": "fixture:c0", "benchmark": "fixture"}
                    )
                    writer.add_question(
                        {
                            "question_id": "fixture:q0",
                            "context_id": "fixture:c0",
                            "prompt": [content_text("question")],
                            "task": {"response_type": "choice"},
                            "choices": ["raw string choice"],
                            "answer": {"text": "answer"},
                            "memory_scope": {"mode": "all"},
                        }
                    )

            missing_choice_asset_root = Path(directory) / "missing-choice-asset"
            with BundleWriter(
                missing_choice_asset_root, {"benchmark": "fixture"}
            ) as writer:
                writer.add_context(
                    {"context_id": "fixture:c0", "benchmark": "fixture"}
                )
                writer.add_question(
                    {
                        "question_id": "fixture:q0",
                        "context_id": "fixture:c0",
                        "prompt": [content_text("question")],
                        "task": {"response_type": "choice"},
                        "choices": [
                            {
                                "choice_id": "A",
                                "text": "visual choice",
                                "content": [
                                    {
                                        "type": "image",
                                        "asset_id": "fixture:missing-asset",
                                    }
                                ],
                            }
                        ],
                        "answer": {"text": "A"},
                        "memory_scope": {"mode": "all"},
                    }
                )
            with self.assertRaisesRegex(SchemaError, "choice references missing asset"):
                validate_bundle(missing_choice_asset_root)

    def test_context_group_sequence_and_cross_context_relations_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split_root = Path(directory) / "split-context"
            with BundleWriter(split_root, {"benchmark": "fixture"}) as writer:
                for context_id in ("fixture:c0", "fixture:c1"):
                    writer.add_context({"context_id": context_id, "benchmark": "fixture"})
                for memory_id, context_id, sequence in (
                    ("fixture:m0", "fixture:c0", 0),
                    ("fixture:m1", "fixture:c1", 0),
                    ("fixture:m2", "fixture:c0", 1),
                ):
                    writer.add_memory(
                        {
                            "memory_id": memory_id,
                            "context_id": context_id,
                            "sequence": sequence,
                            "kind": "artifact",
                            "content": [content_text("memory")],
                        }
                    )
            with self.assertRaisesRegex(SchemaError, "non-contiguous groups"):
                validate_bundle(split_root)

            duplicate_root = Path(directory) / "duplicate-sequence"
            with BundleWriter(duplicate_root, {"benchmark": "fixture"}) as writer:
                writer.add_context({"context_id": "fixture:c0", "benchmark": "fixture"})
                for memory_id in ("fixture:m0", "fixture:m1"):
                    writer.add_memory(
                        {
                            "memory_id": memory_id,
                            "context_id": "fixture:c0",
                            "sequence": 0,
                            "kind": "artifact",
                            "content": [content_text("memory")],
                        }
                    )
            with self.assertRaisesRegex(SchemaError, "duplicate sequence"):
                validate_bundle(duplicate_root)

            cross_root = Path(directory) / "cross-context"
            with BundleWriter(cross_root, {"benchmark": "fixture"}) as writer:
                for context_id in ("fixture:c0", "fixture:c1"):
                    writer.add_context({"context_id": context_id, "benchmark": "fixture"})
                writer.add_memory(
                    {
                        "memory_id": "fixture:m1",
                        "context_id": "fixture:c1",
                        "sequence": 0,
                        "kind": "artifact",
                        "content": [content_text("memory")],
                    }
                )
                writer.add_question(
                    {
                        "question_id": "fixture:q0",
                        "context_id": "fixture:c0",
                        "prompt": [content_text("question")],
                        "answer": {"text": "answer"},
                        "evidence": [{"memory_id": "fixture:m1"}],
                        "memory_scope": {"mode": "ids", "memory_ids": ["fixture:m1"]},
                    }
                )
            with self.assertRaisesRegex(SchemaError, "from another context"):
                validate_bundle(cross_root)


if __name__ == "__main__":
    unittest.main()
