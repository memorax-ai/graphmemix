"""Data-only protocol tests; no model/API calls."""

import base64
import csv
import copy
import json
import tempfile
import unittest
from pathlib import Path

from mm_memory_bench.benchmarks.bundle import SchemaError, iter_jsonl, load_bundle
from mm_memory_bench.benchmarks.registry import convert
from mm_memory_bench.benchmarks.reader import BundleReader
from mm_memory_bench.runner.benchmark import _resolved_memory, _resolved_question, run_bundle


class AdditionalConverters(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = self.root / "raw"
        self.out = self.root / "bundles"

    def write(self, name, value, jsonl=False):
        p = self.raw / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            "\n".join(json.dumps(x) for x in value) if jsonl else json.dumps(value)
        )
        return p

    def rows(self, bench, table):
        return list(iter_jsonl(self.out / bench / f"{table}.jsonl"))

    def test_m3_round_expansion_and_document_asset(self):
        self.write(
            "m3exam/example_set/P/sessions.json",
            [
                {
                    "session_id": "D1",
                    "date": "2020-01-01",
                    "dialogues": [
                        {
                            "round": "D1:1",
                            "user": "read file",
                            "assistant": "okay",
                            "pdf_file": ["x.pdf"],
                        }
                    ],
                }
            ],
        )
        self.write(
            "m3exam/example_set/P/question.json",
            [
                {
                    "question": "what?",
                    "answer": ["x", "y"],
                    "supporting_facts": ["D1:1"],
                    "type": "ss",
                }
            ],
        )
        p = self.raw / "m3exam/example_set/P/pdfs/x.pdf"
        p.parent.mkdir()
        p.write_bytes(b"%PDF fixture")
        result = convert("m3exam", self.raw, self.out)
        self.assertEqual(result["counts"]["memories"], 2)
        q = self.rows("m3exam", "questions")[0]
        self.assertEqual(len(q["evidence"]), 2)
        self.assertEqual(q["answer"]["accepted_answers"], ["x", "y"])
        self.assertEqual(self.rows("m3exam", "assets")[0]["media_type"], "document")

    def test_persona_mme_alignment_and_private_profile(self):
        self.write(
            "persona_mme/Persona-MME/samples/50/0/user_data.json",
            {
                "profile": "PRIVATE_PROFILE",
                "imgs": ["./samples/50/0/x.png"],
                "sessions": [[{"user": "look <img>", "assistant": "yes"}]],
            },
        )
        p = self.raw / "persona_mme/Persona-MME/samples/50/0/x.png"
        p.write_bytes(b"image")
        self.write(
            "persona_mme/Persona-MME/Persona-MME.json",
            [
                {
                    "data_path": "Persona-MME/samples/50/0/user_data.json",
                    "query": "Q",
                    "answer": "(a)",
                    "choices": {"(a)": "X", "(b)": "Y"},
                    "type": "Intent",
                    "question_type": "*Implicit",
                    "alignment": {"chosen": "good", "rejected": "bad"},
                }
            ],
        )
        result = convert("persona_mme", self.raw, self.out)
        self.assertEqual(result["counts"]["questions"], 3)
        self.assertNotIn("PRIVATE_PROFILE", str(self.rows("persona_mme", "memories")))
        qs = self.rows("persona_mme", "questions")
        self.assertEqual([q["answer"]["choice_id"] for q in qs], ["(a)", "(a)", "(b)"])
        self.assertTrue(all(q["evidence"] == [] for q in qs))

    def test_omni_session_evidence_and_annotation_not_input(self):
        self.write(
            "mobilemem_omni/omni/data.jsonl",
            [
                {
                    "uuid": 0,
                    "Basic_Profile": "HIDDEN",
                    "sessions": [
                        {
                            "session_id": "s",
                            "dialogue": [
                                {"role": "user", "content": "A"},
                                {"role": "assistant", "content": "B"},
                            ],
                        }
                    ],
                }
            ],
            True,
        )
        q = {
            "question_id": "q",
            "question": "Q",
            "answer": "A",
            "question_type": "single_hop",
            "evidence": [{"session_id": "s", "explanation": "GOLD EXPLANATION"}],
            "image_refs": ["gold.png"],
        }
        for file in ("questions", "filtered_questions"):
            self.write(
                f"mobilemem_omni/omni/{file}.jsonl",
                [{"uuid": 0, "questions": [q]}],
                True,
            )
        convert("mobilemem_omni", self.raw, self.out)
        row = self.rows("mobilemem_omni", "questions")[0]
        self.assertEqual(len(row["evidence"]), 2)
        self.assertEqual(row["subset"], "filtered")
        self.assertNotIn("gold.png", str(row["prompt"]))
        self.assertNotIn("HIDDEN", str(self.rows("mobilemem_omni", "memories")))

    def test_personamem_variants_stable_choices_and_snippet(self):
        snippet = [{"role": "user", "content": "I like blue."}]
        history = {
            "chat_history": [{"role": "system", "content": "Native profile"}, *snippet]
        }
        for size in ("32k", "128k"):
            self.write(f"personamem_v2/{size}.json", history)
        row = {
            "persona_id": "1",
            "chat_history_32k_link": "32k.json",
            "chat_history_128k_link": "128k.json",
            "user_query": repr({"role": "user", "content": "Which color?"}),
            "correct_answer": "Blue",
            "incorrect_answers": json.dumps(["Red", "Green"]),
            "related_conversation_snippet": json.dumps(snippet),
            "preference": "DO_NOT_INJECT",
        }
        for mode in ("text", "multimodal"):
            p = self.raw / f"personamem_v2/benchmark/{mode}/benchmark.csv"
            p.parent.mkdir(parents=True)
            with p.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
                w.writerow(row)
        convert("personamem_v2", self.raw, self.out)
        qs = self.rows("personamem_v2", "questions")
        self.assertEqual(len(qs), 4)
        self.assertEqual(qs[0]["semantic_question_id"], qs[1]["semantic_question_id"])
        self.assertEqual(qs[0]["choices"], qs[1]["choices"])
        self.assertTrue(all(len(q["evidence"]) == 1 for q in qs))
        self.assertNotIn("DO_NOT_INJECT", str(self.rows("personamem_v2", "memories")))

    def test_smm_zero_based_assignment_and_misleading_separation(self):
        self.write(
            "smmbench/Samples/cluster_1/group_chat_x.json",
            {
                "conversation": [
                    {
                        "content": "false",
                        "timestamp": "2",
                        "conversation_name": "group_chat_x",
                    },
                    {
                        "content": "true",
                        "timestamp": "1",
                        "conversation_name": "group_chat_x",
                    },
                ]
            },
        )
        self.write(
            "smmbench/Samples/cluster_1/QA_sample.json",
            [
                {
                    "id": "Q",
                    "category": "Conflict_Resolution_QA",
                    "domain": "D",
                    "question": "Q?",
                    "answer": "true",
                    "multi_choice_QA": {
                        "multi_choice_QA_answer": 1,
                        "multi_choice_QA_options": ["false", "true"],
                    },
                    "evidence_assignment": {
                        "text_evidence_assignment": [
                            {
                                "conversation_name": "group_chat_x",
                                "insert_conversation_turn": 1,
                            }
                        ],
                        "mis_text_evidence_assignment": [
                            {
                                "conversation_name": "group_chat_x",
                                "insert_conversation_turn": 0,
                            }
                        ],
                    },
                }
            ],
        )
        convert("smmbench", self.raw, self.out)
        q = self.rows("smmbench", "questions")[0]
        memories = self.rows("smmbench", "memories")
        self.assertEqual(q["evidence"][0]["memory_id"], memories[0]["memory_id"])
        self.assertEqual(
            q["misleading_evidence"][0]["memory_id"], memories[1]["memory_id"]
        )
        self.assertEqual(q["answer"]["choice_id"], "1")

    def test_omni_normalizes_directories_without_changing_person_names(self):
        from mm_memory_bench.benchmarks.converters.mobilemem_omni import (
            resolve_image_path,
        )

        expected = self.raw / "omni/image/uid1/chat_records/Alex Bell.png"
        expected.parent.mkdir(parents=True)
        expected.write_bytes(b"fixture")
        self.assertEqual(
            resolve_image_path(self.raw, "image/uid1/chat records/Alex Bell.png"),
            expected,
        )

    def test_smm_embedded_image_and_function_plan(self):
        self.write(
            "smmbench/Samples/cluster_1/group_chat_x.json",
            {
                "conversation": [
                    {
                        "content": [
                            {"type": "image", "content": {"image_path": "x.png"}}
                        ],
                        "caption": "Fig. abcdef12\nprivate descriptive caption",
                        "timestamp": "1",
                        "conversation_name": "group_chat_x",
                    }
                ]
            },
        )
        image = self.raw / "smmbench/Images/x.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"fixture")
        tools = [
            {
                "function_name": "lookup",
                "default_arguments": {},
                "function_comment": "Find a record",
            }
        ]
        self.write("smmbench/candidate_tools.json", tools)
        self.write(
            "smmbench/Samples/cluster_1/QA_sample.json",
            [
                {
                    "id": "Q",
                    "category": "Function_Call",
                    "domain": "D",
                    "question": "Find it",
                    "answer": [
                        {"step": 1, "calls": [{"name": "lookup", "arguments": {}}]}
                    ],
                    "evidence_assignment": {},
                }
            ],
        )
        convert("smmbench", self.raw, self.out)
        q = self.rows("smmbench", "questions")[0]
        self.assertEqual(q["tools"], tools)
        self.assertEqual(q["task"]["response_type"], "structured_json")
        memory = self.rows("smmbench", "memories")[0]
        self.assertIn("Fig. abcdef12", str(memory["content"]))
        self.assertNotIn("private descriptive", str(memory["content"]))
        self.assertTrue(any(p["type"] == "image" for p in memory["content"]))

    def test_embedded_base64_assets_are_deduplicated(self):
        from mm_memory_bench.benchmarks.bundle import BundleWriter
        from mm_memory_bench.benchmarks.converters._shared import Assets

        output = self.out / "asset_fixture"
        with BundleWriter(output, {"benchmark": "fixture"}) as writer:
            assets = Assets(writer, self.raw, output, "fixture")
            data = "data:image/png;base64," + base64.b64encode(b"png-fixture").decode()
            first = assets.content([{"type": "image_url", "image_url": {"url": data}}])
            second = assets.content([{"type": "image_url", "image_url": {"url": data}}])
            self.assertEqual(first, second)
            with self.assertRaises(SchemaError):
                assets.content(
                    [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://unfetched.example/image.png"},
                        }
                    ]
                )
        self.assertEqual(len(self.rows("asset_fixture", "assets")), 1)

    def test_missing_image_fails_instead_of_silent_text_fallback(self):
        self.write(
            "mobilemem_omni/omni/data.jsonl",
            [
                {
                    "uuid": 0,
                    "sessions": [
                        {
                            "session_id": "s",
                            "dialogue": [
                                {"role": "user", "image_inline": "missing.png"}
                            ],
                        }
                    ],
                }
            ],
            True,
        )
        for file in ("questions", "filtered_questions"):
            self.write(
                f"mobilemem_omni/omni/{file}.jsonl",
                [{"uuid": 0, "questions": []}],
                True,
            )
        with self.assertRaises(FileNotFoundError):
            convert("mobilemem_omni", self.raw, self.out)


class DownloadIntegrity(unittest.TestCase):
    def test_omni_gbk_zip_names_decode_without_changing_content(self):
        import zipfile

        from scripts.download_additional_benchmarks import extract_omni_images

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "images.zip"
            with zipfile.ZipFile(archive, "w") as writer:
                writer.writestr("uid0/xx.png", b"image-bytes")
            archive.write_bytes(
                archive.read_bytes().replace(b"xx.png", "乔.png".encode("gbk"))
            )
            extract_omni_images(archive, root / "image")
            self.assertEqual((root / "image/uid0/乔.png").read_bytes(), b"image-bytes")

    def test_range_download_rejects_wrong_content_range(self):
        import io
        from unittest.mock import patch

        from scripts.download_additional_benchmarks import fetch_ranges

        def response(*args, **kwargs):
            result = io.BytesIO(b"0123456789")
            result.status = 206
            result.headers = {"Content-Range": "bytes 10-19/20"}
            return result

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.zip"
            with (
                patch("urllib.request.urlopen", side_effect=response),
                patch("time.sleep"),
                self.assertRaisesRegex(ValueError, "requested byte range"),
            ):
                fetch_ranges("https://example.invalid/file", path, 10, 2)
            self.assertFalse(path.exists())

    def test_incomplete_download_manifest_blocks_conversion(self):
        from mm_memory_bench.benchmarks.converters._shared import source_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "download-manifest.json").write_text(
                json.dumps({"errors": [{"file": "missing"}]})
            )
            with self.assertRaisesRegex(SchemaError, "unresolved errors"):
                source_manifest(root)


def trajectory(person_id="person-1"):
    return {
        "person": {"id": person_id, "profile": "PRIVATE_PROFILE"},
        "graphs": {"gold": "PRIVATE_GRAPH"},
        "old_question_type_toolbook": {"question_types": [{"qa_pairs": [{"id": "discarded"}]}]},
        "sessions": [{"id": "s1", "event_id": "e1", "messages": [
            {"id": "m2", "role": "assistant", "name": "Assistant", "content": "", "timestamp": "2025-01-02 10:00:00"},
            {"id": "m1", "role": "system", "name": "Calendar", "content": "Appointment Tuesday", "timestamp": "2025-01-01 10:00:00", "side_note": "PRIVATE_NOTE"},
        ]}],
        "question_type_toolbook": {"question_types": [{"name": "temporal-reasoning", "qa_pairs": [{
            "id": "q1", "question": "Which day? A. Tuesday B. Friday", "question_form": "single_choice",
            "question_type": "temporal-reasoning", "golden_answers": ["A. Tuesday", "Tuesday"],
            "effective_timestamp": "2024-01-01 00:00:00", "side_note": "PRIVATE_QA",
            "source_evidences": [{"id": "m1"}, {"id": "m1"}],
        }]}]},
    }


class MobileMemConverterTest(unittest.TestCase):
    def write_source(self, root, data):
        source = root / "raw/mobilemem/text/mobilemem_data.json"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps(data), encoding="utf-8")

    def test_registration_ids_order_references_and_public_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_source(root, [trajectory(), trajectory("person-2")])
            report = convert("mobilemem", root / "raw", root / "unified")
            self.assertEqual(report["counts"], {"contexts": 2, "memories": 4, "assets": 0, "questions": 2})
            bundle_root = root / "unified/mobilemem"
            data = load_bundle(bundle_root)
            q = data["questions"][0]
            self.assertEqual(q["memory_scope"], {"mode": "all"})
            self.assertEqual(q["answer"]["accepted_answers"], ["A. Tuesday", "Tuesday"])
            self.assertEqual(len(q["evidence"]), 1)
            self.assertEqual(q["evidence"][0]["memory_id"], data["memories"][0]["memory_id"])
            self.assertEqual([m["source_id"] for m in data["memories"][:2]], ["m1", "m2"])
            reader = BundleReader(bundle_root)
            visible = _resolved_question(reader, q)
            self.assertNotIn("answer", visible)
            self.assertNotIn("evidence", visible)
            self.assertNotIn("subcategory", visible["task"])
            public = json.dumps([visible, *[_resolved_memory(reader, m) for m in data["memories"]]])
            self.assertNotIn("PRIVATE_", public)
            self.assertIn("Calendar", public)
            with self.assertRaises(FileExistsError):
                convert("mobilemem", root / "raw", root / "unified")

    def test_rejects_unknown_evidence_and_duplicate_native_messages_before_writing(self):
        for error in ("reference", "duplicate"):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                item = trajectory()
                if error == "reference":
                    item["question_type_toolbook"]["question_types"][0]["qa_pairs"][0]["source_evidences"] = [{"id": "missing"}]
                else:
                    item["sessions"][0]["messages"].append(copy.deepcopy(item["sessions"][0]["messages"][0]))
                self.write_source(root, [item])
                with self.assertRaises(SchemaError):
                    convert("mobilemem", root / "raw", root / "unified")
                self.assertFalse((root / "unified/mobilemem/manifest.json").exists())

    def test_keeps_evidence_free_questions_and_unknown_native_capabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = trajectory()
            q = item["question_type_toolbook"]["question_types"][0]["qa_pairs"][0]
            q.update(source_evidences=[], question_type="new-native-capability", question_form="multiple_choice")
            self.write_source(root, [item])
            convert("mobilemem", root / "raw", root / "unified")
            result = load_bundle(root / "unified/mobilemem")["questions"][0]
            self.assertEqual(result["evidence"], [])
            self.assertEqual(result["task"]["subcategory"], "new-native-capability")
            self.assertEqual(result["prompt"][0]["text"], q["question"])

    def test_existing_runner_ingests_all_messages_without_gold_inputs(self):
        class Method:
            def begin_context(self, context):
                self.memories = []

            def ingest(self, memory):
                self.memories.append(memory)

            def answer(self, question):
                assert len(self.memories) == 2
                assert "answer" not in question and "evidence" not in question
                return "Tuesday"

            def end_context(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_source(root, [trajectory()])
            convert("mobilemem", root / "raw", root / "unified")
            output = root / "predictions.jsonl"
            run_bundle(Method(), root / "unified/mobilemem", output)
            rows = list(iter_jsonl(output))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["prediction"], "Tuesday")


if __name__ == "__main__":
    unittest.main()
