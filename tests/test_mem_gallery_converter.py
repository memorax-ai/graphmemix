import json
import tempfile
import unittest
from pathlib import Path

from mm_memory_bench.benchmarks.converters.mem_gallery import convert as convert_mem_gallery
from mm_memory_bench.benchmarks.bundle import load_bundle, validate_bundle
from mm_memory_bench.runner.benchmark import _resolved_memory
from mm_memory_bench.benchmarks.reader import BundleReader


class ConverterFixtureTest(unittest.TestCase):
    def _write_json(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def _mem_gallery_fixture(self, root: Path) -> Path:
        snapshot = root / "mem_gallery"
        scenario = "Fixture_Scenario"
        image_dir = snapshot / "data/image" / scenario
        image_dir.mkdir(parents=True, exist_ok=True)
        for name in ("h1.jpg", "h2.jpg", "h3.jpg", "query.jpg"):
            (image_dir / name).write_bytes(b"jpg")
        self._write_json(
            snapshot / "data/dialog" / f"{scenario}.json",
            {
                "character_profile": {
                    "name": "Julian",
                    "persona_summary": "PRIVATE PROFILE SUMMARY",
                    "traits": ["analytical"],
                    "conversation_style": "careful",
                },
                "multi_session_dialogues": [
                    {
                        "session_id": "D1",
                        "date": "2024-06-17",
                        "dialogues": [
                            {
                                "round": "D1:1",
                                "user": "Here are two images.",
                                "assistant": "I can see them.",
                                "input_image": [
                                    f"../image/{scenario}/h1.jpg",
                                    f"../image/{scenario}/h2.jpg",
                                ],
                                "image_id": ["D1:IMG_001", "D1:IMG_002"],
                                "image_caption": ["caption one", "caption two"],
                            }
                        ],
                    },
                    {
                        "session_id": "D2",
                        "date": "2024-06-18",
                        "dialogues": [
                            {
                                "round": "D2:1",
                                "user": "Remember the earlier images.",
                                "assistant": "I remember.",
                                "input_image": [f"../image/{scenario}/h3.jpg"],
                                "image_id": ["D2:IMG_001"],
                                "image_caption": ["caption three"],
                            }
                        ],
                    },
                ],
                "human-annotated QAs": [
                    {
                        "point": "MR",
                        "question": "What connects the images?",
                        "answer": "A shared theme.",
                        "session_id": ["D1", "D2"],
                        "clue": ["D1:1", "D2:1"],
                        "question_image": f"../image/{scenario}/query.jpg",
                        "image_caption": "query caption",
                    },
                    {
                        "point": "AR",
                        "question": "What exact camera setting was used?",
                        "answer": "Not mentioned.",
                        "session_id": ["D2"],
                        "clue": [],
                    },
                ],
            },
        )
        return snapshot

    def test_mem_gallery_multi_image_roles_query_scope_and_round_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self._mem_gallery_fixture(root)
            bundle_root = root / "unified/mem_gallery"
            result = convert_mem_gallery(snapshot, bundle_root)

            self.assertEqual(
                result["counts"],
                {"contexts": 1, "memories": 5, "assets": 4, "questions": 2},
            )
            self.assertEqual(result["unresolved_clue_count"], 0)
            self.assertEqual(
                validate_bundle(bundle_root, check_assets=True)["missing_asset_count"], 0
            )
            bundle = load_bundle(bundle_root)

            profile = next(row for row in bundle["memories"] if row["kind"] == "profile")
            self.assertEqual(profile["content"][0]["text"], "Julian")
            self.assertNotIn("PRIVATE PROFILE SUMMARY", profile["content"][0]["text"])

            d1_user = next(
                row
                for row in bundle["memories"]
                if row.get("role") == "user"
                and row["metadata"]["native_round_id"] == "D1:1"
            )
            d1_assistant = next(
                row
                for row in bundle["memories"]
                if row.get("role") == "assistant"
                and row["metadata"]["native_round_id"] == "D1:1"
            )
            d2_user = next(
                row
                for row in bundle["memories"]
                if row.get("role") == "user"
                and row["metadata"]["native_round_id"] == "D2:1"
            )
            self.assertEqual(d2_user["source_id"], "D2:IMG_001")
            history_images = [
                part for part in d1_user["content"] if part["type"] == "image"
            ]
            self.assertEqual(len(history_images), 2)
            self.assertEqual(
                d1_user["metadata"]["derived"]["image_captions"],
                ["caption one", "caption two"],
            )
            self.assertEqual(
                d1_user["metadata"]["agent_visible"]["image_ids"],
                ["D1:IMG_001", "D1:IMG_002"],
            )
            reader = BundleReader(bundle_root)
            harness_memory = _resolved_memory(reader, d1_user)
            self.assertEqual(
                harness_memory["metadata"],
                {"image_ids": ["D1:IMG_001", "D1:IMG_002"]},
            )
            self.assertNotIn("caption", json.dumps(harness_memory, ensure_ascii=False))
            for part in history_images:
                self.assertEqual(part["annotations"]["native_scope"], "dialogue_round")
                self.assertEqual(
                    part["annotations"]["role_assignment"],
                    "inferred_from_input_image",
                )
            self.assertFalse(
                any(part["type"] == "image" for part in d1_assistant["content"])
            )
            self.assertEqual(d1_assistant["metadata"]["native_round_id"], "D1:1")

            mr_question = next(
                row for row in bundle["questions"] if row["task"]["subcategory"] == "MR"
            )
            query_image = next(
                part for part in mr_question["prompt"] if part["type"] == "image"
            )
            self.assertEqual(query_image["annotations"]["native_scope"], "question")
            self.assertEqual(query_image["annotations"]["role_assignment"], "query")
            self.assertEqual(mr_question["query_at"]["session_ids"], ["D1", "D2"])
            self.assertEqual(len(mr_question["evidence"]), 4)
            self.assertEqual(
                [(item["native_id"], item["role"]) for item in mr_question["evidence"]],
                [
                    ("D1:1", "user"),
                    ("D1:1", "assistant"),
                    ("D2:1", "user"),
                    ("D2:1", "assistant"),
                ],
            )
            self.assertTrue(
                all(row["memory_scope"] == {"mode": "all"} for row in bundle["questions"])
            )
            self.assertTrue(
                all(
                    row["task"]["response_type"] == "text"
                    for row in bundle["questions"]
                )
            )
            ar_question = next(
                row for row in bundle["questions"] if row["task"]["subcategory"] == "AR"
            )
            self.assertTrue(ar_question["answer"]["unanswerable"])
            self.assertEqual(
                ar_question["instruction"],
                ar_question["metadata"]["official_format_constraint"],
            )
            self.assertIn("Not mentioned", ar_question["instruction"])

            manifest = json.loads(
                (bundle_root / "manifest.json").read_text(encoding="utf-8")
            )
            notes = " ".join(manifest["conversion_notes"])
            self.assertIn("does not explicitly assign", notes)
            self.assertIn("role_assignment=inferred_from_input_image", notes)


if __name__ == "__main__":
    unittest.main()
