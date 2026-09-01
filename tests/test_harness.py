import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from mm_memory_bench.core import BundleWriter, content_asset, content_text
from mm_memory_bench.harness import digest_bundle, run_bundle


class RecordingMethod:
    def __init__(self) -> None:
        self.begin_calls = 0
        self.end_calls = 0
        self.contexts: list[dict[str, Any]] = []
        self.ingested: list[dict[str, Any]] = []
        self.questions: dict[str, dict[str, Any]] = {}
        self.answer_memory_ids: dict[str, list[str]] = {}
        self._current_ids: list[str] = []

    def begin_context(self, context: Mapping[str, Any]) -> None:
        self.begin_calls += 1
        self.contexts.append(dict(context))
        self._current_ids = []

    def ingest(self, memory: Mapping[str, Any]) -> None:
        value = dict(memory)
        self.ingested.append(value)
        self._current_ids.append(str(value["memory_id"]))

    def answer(self, question: Mapping[str, Any]) -> str | Mapping[str, Any]:
        value = dict(question)
        question_id = str(value["question_id"])
        self.questions[question_id] = value
        self.answer_memory_ids[question_id] = list(self._current_ids)
        prediction = "seen:" + ",".join(self._current_ids)
        if question_id == "fixture:q-all":
            return {
                "prediction": prediction,
                "retrieved_memory_ids": [self._current_ids[-1]],
                "token_usage": {"input": 12, "output": 3},
            }
        return prediction

    def end_context(self) -> None:
        self.end_calls += 1


class HarnessTest(unittest.TestCase):
    def _bundle(self, root: Path) -> Path:
        bundle_root = root / "bundle"
        media_path = bundle_root / "media" / "pixel.jpg"
        media_path.parent.mkdir(parents=True, exist_ok=True)
        media_path.write_bytes(b"fixture-image")

        with BundleWriter(bundle_root, {"benchmark": "fixture"}) as writer:
            writer.add_context(
                {
                    "context_id": "fixture:c0",
                    "benchmark": "fixture",
                    "metadata": {
                        "private": "SECRET_CONTEXT_METADATA",
                        "agent_visible": {"locale": "en"},
                    },
                }
            )
            writer.add_asset(
                {
                    "asset_id": "fixture:a0",
                    "media_type": "image",
                    "path": "media/pixel.jpg",
                    "metadata": {"private": "SECRET_ASSET_METADATA"},
                }
            )

            # Deliberately write out of sequence; the harness must ingest in
            # canonical sequence order before applying prefix cutoffs.
            writer.add_memory(
                {
                    "memory_id": "fixture:m2",
                    "context_id": "fixture:c0",
                    "sequence": 2,
                    "kind": "artifact",
                    "content": [content_text("memory two")],
                }
            )
            writer.add_memory(
                {
                    "memory_id": "fixture:m0",
                    "context_id": "fixture:c0",
                    "sequence": 0,
                    "kind": "artifact",
                    "content": [content_text("memory zero")],
                }
            )
            writer.add_memory(
                {
                    "memory_id": "fixture:m1",
                    "context_id": "fixture:c0",
                    "sequence": 1,
                    "kind": "media",
                    "content": [
                        content_text("original user text"),
                        content_asset("image", "fixture:a0"),
                    ],
                    "provenance": {"source_file": "SECRET_MEMORY_PROVENANCE"},
                    "metadata": {
                        "derived": {
                            "short_summary": "PUBLIC_SHORT_SUMMARY_708777",
                            "summary": "PUBLIC_LONG_SUMMARY_708777",
                            "caption": "SECRET_DERIVED_CAPTION",
                            "unsupported_private_field": "MUST_NOT_BE_VISIBLE",
                        },
                        "agent_visible": {"label": "public image"},
                    },
                }
            )

            private_question_metadata = {
                "evaluation_private": {"notes": "SECRET_PRIVATE_NOTES"},
                "evaluation_variants": {"niah": "SECRET_EVALUATION_VARIANT"},
                "native": {"raw": "SECRET_NATIVE_METADATA"},
                "agent_visible": {"public_tag": "visible"},
            }
            writer.add_question(
                {
                    "question_id": "fixture:q-all",
                    "semantic_question_id": "fixture:semantic-all",
                    "context_id": "fixture:c0",
                    "subset": "default",
                    "split": "test",
                    "task": {
                        "category": "fixture",
                        "subcategory": "selected_task",
                        "response_type": "text",
                    },
                    "prompt": [content_text("What is visible after all memories?")],
                    "instruction": "Use the supplied lookup tool.",
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "description": "Look up a public value.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"query": {"type": "string"}},
                                    "required": ["query"],
                                },
                            },
                        }
                    ],
                    "answer": {"text": "SECRET_GROUND_TRUTH"},
                    "evidence": [
                        {
                            "memory_id": "fixture:m2",
                            "relation": "supports",
                            "metadata": {"private": "SECRET_EVIDENCE_METADATA"},
                        }
                    ],
                    "memory_scope": {"mode": "all"},
                    "query_at": {
                        "timestamp": "2026-01-02",
                        "session_ids": ["SECRET_TARGET_SESSION"],
                    },
                    "provenance": {"source_file": "SECRET_QUESTION_PROVENANCE"},
                    "metadata": private_question_metadata,
                }
            )
            writer.add_question(
                {
                    "question_id": "fixture:q-prefix",
                    "context_id": "fixture:c0",
                    "subset": "default",
                    "split": "test",
                    "task": {
                        "category": "fixture",
                        "subcategory": "other_task",
                        "response_type": "text",
                    },
                    "prompt": [content_text("What is visible at sequence one?")],
                    "answer": {"text": "SECRET_PREFIX_GROUND_TRUTH"},
                    "evidence": [{"memory_id": "fixture:m1", "relation": "supports"}],
                    "memory_scope": {"mode": "prefix", "max_sequence": 1},
                    "metadata": private_question_metadata,
                }
            )
        # Defense-in-depth check: a hand-edited/unvalidated bundle may contain
        # evaluator-only keys nested in a tool declaration. The harness must
        # strip them recursively before calling the method.
        questions_path = bundle_root / "questions.jsonl"
        questions = [
            json.loads(line)
            for line in questions_path.read_text(encoding="utf-8").splitlines()
        ]
        questions[0]["tools"][0]["metadata"] = {
            "answer": "SECRET_TOOL_METADATA_ANSWER"
        }
        questions[0]["tools"][0]["function"]["parameters"]["evidence"] = {
            "memory_id": "SECRET_TOOL_EVIDENCE"
        }
        questions_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in questions),
            encoding="utf-8",
        )
        return bundle_root

    def test_ingest_failure_uses_abort_and_preserves_primary_error(self):
        class FailingMethod(RecordingMethod):
            def __init__(self):
                super().__init__()
                self.abort_calls = 0

            def ingest(self, memory):
                raise ValueError("primary ingest failure")

            def abort_context(self):
                self.abort_calls += 1

            def end_context(self):
                raise RuntimeError("normal end must not run after failure")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            method = FailingMethod()
            with self.assertRaisesRegex(ValueError, "primary ingest failure"):
                run_bundle(method, self._bundle(root), root / "predictions.jsonl")
            self.assertEqual(method.abort_calls, 1)

    def test_all_and_prefix_scopes_media_resolution_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_root = self._bundle(root)
            output_path = root / "runs" / "predictions.jsonl"
            method = RecordingMethod()

            report = run_bundle(method, bundle_root, output_path)

            self.assertEqual(report["predictions"], 2)
            self.assertEqual(report["memory_ingest_calls"], 3)
            self.assertGreaterEqual(report["digest_seconds"], 0)
            self.assertGreaterEqual(report["answer_seconds"], 0)
            self.assertGreaterEqual(report["overhead_seconds"], 0)
            self.assertEqual(method.begin_calls, 1)
            self.assertEqual(method.end_calls, 1)
            self.assertEqual(
                method.answer_memory_ids["fixture:q-prefix"],
                ["fixture:m0", "fixture:m1"],
            )
            self.assertEqual(
                method.answer_memory_ids["fixture:q-all"],
                ["fixture:m0", "fixture:m1", "fixture:m2"],
            )
            self.assertEqual(
                [memory["memory_id"] for memory in method.ingested],
                ["fixture:m0", "fixture:m1", "fixture:m2"],
            )

            media = next(row for row in method.ingested if row["memory_id"] == "fixture:m1")
            media_part = next(part for part in media["content"] if part["type"] == "image")
            self.assertEqual(media_part["asset_id"], "fixture:a0")
            self.assertEqual(
                Path(media_part["path"]),
                (bundle_root / "media" / "pixel.jpg").resolve(),
            )
            self.assertNotIn("asset", media_part)
            self.assertEqual(media["metadata"], {"label": "public image"})

            derived_method = RecordingMethod()
            run_bundle(
                derived_method,
                bundle_root,
                root / "runs" / "derived.jsonl",
                memory_view="derived",
            )
            derived_media = next(
                row for row in derived_method.ingested if row["memory_id"] == "fixture:m1"
            )
            self.assertTrue(
                any(part.get("text") == "original user text" for part in derived_media["content"])
            )
            self.assertTrue(
                any("SECRET_DERIVED_CAPTION" in part.get("text", "") for part in derived_media["content"])
            )
            self.assertTrue(
                any("PUBLIC_SHORT_SUMMARY_708777" in part.get("text", "") for part in derived_media["content"])
            )
            self.assertTrue(
                any("PUBLIC_LONG_SUMMARY_708777" in part.get("text", "") for part in derived_media["content"])
            )
            self.assertFalse(any(part.get("type") == "image" for part in derived_media["content"]))

            hybrid_method = RecordingMethod()
            run_bundle(
                hybrid_method,
                bundle_root,
                root / "runs" / "raw_derived.jsonl",
                memory_view="raw_derived",
            )
            hybrid_media = next(
                row for row in hybrid_method.ingested if row["memory_id"] == "fixture:m1"
            )
            self.assertTrue(any(part.get("type") == "image" for part in hybrid_media["content"]))
            self.assertEqual(
                hybrid_media["metadata"]["derived"]["caption"],
                "SECRET_DERIVED_CAPTION",
            )
            self.assertEqual(
                hybrid_media["metadata"]["derived"]["short_summary"],
                "PUBLIC_SHORT_SUMMARY_708777",
            )
            self.assertEqual(
                hybrid_media["metadata"]["derived"]["summary"],
                "PUBLIC_LONG_SUMMARY_708777",
            )
            self.assertNotIn(
                "unsupported_private_field", hybrid_media["metadata"]["derived"]
            )

            sidecar_path = root / "captions.jsonl"
            sidecar_path.write_text(
                json.dumps(
                    {"asset_id": "fixture:a0", "caption": "SHARED_PUBLIC_CAPTION"}
                )
                + "\n",
                encoding="utf-8",
            )
            sidecar_method = RecordingMethod()
            run_bundle(
                sidecar_method,
                bundle_root,
                root / "runs" / "sidecar.jsonl",
                memory_view="raw_derived",
                caption_sidecar=sidecar_path,
            )
            sidecar_media = next(
                row for row in sidecar_method.ingested if row["memory_id"] == "fixture:m1"
            )
            self.assertEqual(
                sidecar_media["metadata"]["derived"]["image_captions"],
                ["SHARED_PUBLIC_CAPTION"],
            )

            model_visible = json.dumps(
                {
                    "contexts": method.contexts,
                    "memories": method.ingested,
                    "questions": method.questions,
                },
                ensure_ascii=False,
            )
            for secret in (
                "SECRET_CONTEXT_METADATA",
                "SECRET_ASSET_METADATA",
                "SECRET_MEMORY_PROVENANCE",
                "SECRET_DERIVED_CAPTION",
                "SECRET_GROUND_TRUTH",
                "SECRET_PREFIX_GROUND_TRUTH",
                "SECRET_PRIVATE_NOTES",
                "SECRET_EVALUATION_VARIANT",
                "SECRET_NATIVE_METADATA",
                "SECRET_EVIDENCE_METADATA",
                "SECRET_QUESTION_PROVENANCE",
                "SECRET_TARGET_SESSION",
                "SECRET_TOOL_METADATA_ANSWER",
                "SECRET_TOOL_EVIDENCE",
            ):
                self.assertNotIn(secret, model_visible)
            self.assertEqual(method.contexts[0]["metadata"], {"locale": "en"})
            self.assertEqual(
                method.questions["fixture:q-all"]["metadata"],
                {"public_tag": "visible"},
            )
            self.assertNotIn("answer", method.questions["fixture:q-all"])
            self.assertNotIn("evidence", method.questions["fixture:q-all"])
            self.assertNotIn("provenance", method.questions["fixture:q-all"])
            self.assertEqual(
                method.questions["fixture:q-all"]["query_at"],
                {"timestamp": "2026-01-02"},
            )
            self.assertEqual(
                method.questions["fixture:q-all"]["instruction"],
                "Use the supplied lookup tool.",
            )
            self.assertEqual(
                method.questions["fixture:q-all"]["tools"],
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Look up a public value.",
                            "parameters": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            },
                        },
                    }
                ],
            )

            predictions = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(len(predictions), 2)
            by_id = {row["question_id"]: row for row in predictions}
            expected_fields = {
                "question_id",
                "semantic_question_id",
                "context_id",
                "subset",
                "prediction",
                "retrieved_memory_ids",
                "latency_seconds",
                "metadata",
            }
            self.assertEqual(set(by_id["fixture:q-all"]), expected_fields)
            self.assertEqual(
                by_id["fixture:q-all"]["semantic_question_id"],
                "fixture:semantic-all",
            )
            self.assertEqual(
                by_id["fixture:q-prefix"]["semantic_question_id"],
                "fixture:q-prefix",
            )
            self.assertEqual(by_id["fixture:q-all"]["subset"], "default")
            self.assertEqual(
                by_id["fixture:q-all"]["retrieved_memory_ids"], ["fixture:m2"]
            )
            self.assertEqual(
                by_id["fixture:q-all"]["metadata"],
                {"token_usage": {"input": 12, "output": 3}},
            )
            self.assertEqual(by_id["fixture:q-prefix"]["retrieved_memory_ids"], [])
            self.assertEqual(by_id["fixture:q-prefix"]["metadata"], {})
            self.assertTrue(
                by_id["fixture:q-all"]["prediction"].endswith(
                    "fixture:m0,fixture:m1,fixture:m2"
                )
            )
            self.assertGreaterEqual(by_id["fixture:q-all"]["latency_seconds"], 0.0)

            selected_method = RecordingMethod()
            selected_result = run_bundle(
                selected_method,
                bundle_root,
                root / "runs" / "selected.jsonl",
                task_subcategory="selected_task",
            )
            self.assertEqual(selected_result["predictions"], 1)
            self.assertEqual(selected_result["memory_ingest_calls"], 3)
            self.assertEqual(set(selected_method.questions), {"fixture:q-all"})

    def test_digest_bundle_ingests_all_memories_without_answering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            method = RecordingMethod()

            report = digest_bundle(method, self._bundle(root), memory_view="raw")

            self.assertEqual(report["protocol"], "mmmb-cold-digest-1.0")
            self.assertEqual(report["contexts"], 1)
            self.assertEqual(report["memory_ingest_calls"], 3)
            self.assertEqual(report["context_reports"][0]["memories"], 3)
            self.assertEqual(method.begin_calls, 1)
            self.assertEqual(method.end_calls, 1)
            self.assertEqual(method.questions, {})
            self.assertEqual(
                [memory["memory_id"] for memory in method.ingested],
                ["fixture:m0", "fixture:m1", "fixture:m2"],
            )

            selected = root / "memory_ids.txt"
            selected.write_text("fixture:m2\nfixture:m0\n", encoding="utf-8")
            selected_method = RecordingMethod()
            selected_report = digest_bundle(
                selected_method,
                self._bundle(root / "selected"),
                memory_ids_path=selected,
            )
            self.assertEqual(selected_report["memory_ingest_calls"], 2)
            self.assertEqual(
                [memory["memory_id"] for memory in selected_method.ingested],
                ["fixture:m0", "fixture:m2"],
            )

    def test_prediction_resume_and_per_query_error_isolation(self) -> None:
        class FailsAllQuestion(RecordingMethod):
            def answer(self, question):
                if question["question_id"] == "fixture:q-all":
                    raise RuntimeError("fixture planner parse failure")
                return super().answer(question)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self._bundle(root)
            output = root / "predictions.jsonl"
            output.write_text(
                json.dumps({
                    "question_id": "fixture:q-prefix",
                    "semantic_question_id": "fixture:q-prefix",
                    "context_id": "fixture:c0",
                    "subset": "default",
                    "prediction": "already complete",
                    "retrieved_memory_ids": [],
                    "latency_seconds": 0.1,
                    "metadata": {},
                }) + "\n",
                encoding="utf-8",
            )
            report = run_bundle(
                FailsAllQuestion(),
                bundle,
                output,
                resume_predictions=True,
                continue_on_query_error=True,
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(report["predictions"], 2)
            self.assertEqual(report["resumed_predictions"], 1)
            self.assertEqual(report["new_predictions"], 1)
            self.assertEqual([row["question_id"] for row in rows], [
                "fixture:q-prefix", "fixture:q-all"
            ])
            self.assertEqual(rows[1]["prediction"], "")
            self.assertIn("fixture planner parse failure", rows[1]["metadata"]["method_error"])


if __name__ == "__main__":
    unittest.main()
