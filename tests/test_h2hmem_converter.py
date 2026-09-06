import json
import tempfile
import unittest
from pathlib import Path

from mm_memory_bench.benchmarks.converters.h2hmem import convert
from mm_memory_bench.benchmarks.bundle import load_bundle, validate_bundle


class H2HMemConverterTest(unittest.TestCase):
    def _write(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_dialogue_context_session0_question_image_and_participant_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "h2hmem"
            revision = "3" * 40
            (snapshot / ".git/refs/heads").mkdir(parents=True)
            (snapshot / ".git/HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
            (snapshot / ".git/refs/heads/main").write_text(revision + "\n", encoding="utf-8")

            for interaction_type in ("dyadic", "multi-party"):
                session = snapshot / interaction_type / "dialogue1/scenes/session1"
                self._write(
                    session / "session.json",
                    {
                        "session_id": "native-1",
                        "session_title": "Fixture",
                        "theme": "test",
                        "timeline_date": "2026-01-01",
                        "dialogue": [
                            {"role": "Alice", "content": {"text": "Blue object", "image": "1.png"}},
                            {"role": "Bob", "content": {"text": "I saw it", "image": ""}},
                        ],
                    },
                )
                (session / "image").mkdir()
                (session / "image/1.png").write_bytes(b"fixture")
                self._write(
                    session / "questions.json",
                    {
                        "questions": [
                            {
                                "question": {"text": "What color?", "image": "1.png"},
                                "original_answer": "Blue",
                                "answer_session": (
                                    ["S1-1"] if interaction_type == "multi-party" else ["session1"]
                                ),
                                "answer_dialogue": "dialogue1",
                                "question_type": {
                                    "main_type": "Memory Recall",
                                    "sub_type": "Cross-modal Related Retrieval",
                                },
                                "difficulty": "easy",
                                "validated": True,
                            }
                        ]
                    },
                )
                self._write(
                    snapshot / interaction_type / "dialogue1/scenes/session0/questions.json",
                    {
                        "questions": [
                            {
                                "original_question_id": "CSQ001",
                                "question": {"text": "Recall the image", "image": "session1/1.png"},
                                "original_answer": "Blue",
                                "answer_session": ["session0"],
                                "answer_dialogue": "dialogue1",
                                "question_type": {
                                    "main_type": "Memory Reasoning",
                                    "sub_type": "Multimodal Causal Inference",
                                },
                                "difficulty": "medium",
                                "validated": True,
                            }
                        ]
                    },
                )

            bundle_root = root / "unified/h2hmem"
            result = convert(
                snapshot,
                bundle_root,
                strict_release_counts=False,
            )
            self.assertEqual(
                result["counts"],
                {"contexts": 2, "memories": 4, "assets": 2, "questions": 4},
            )
            self.assertEqual(validate_bundle(bundle_root, check_assets=True)["missing_asset_count"], 0)
            bundle = load_bundle(bundle_root)
            self.assertEqual({row["speaker"] for row in bundle["memories"]}, {"Alice", "Bob"})
            self.assertTrue(all(row["role"] == "participant" for row in bundle["memories"]))
            image_memories = [
                row
                for row in bundle["memories"]
                if any(part["type"] == "image" for part in row["content"])
            ]
            self.assertTrue(image_memories)
            self.assertTrue(
                all(row["source_id"] == "session1/1.png" for row in image_memories)
            )
            cross = [row for row in bundle["questions"] if row["metadata"]["native_question_session"] == "session0"]
            self.assertEqual(len(cross), 2)
            self.assertTrue(all(part["type"] == "image" for row in cross for part in row["prompt"][1:]))
            ordinary = [row for row in bundle["questions"] if row not in cross]
            self.assertTrue(all(row["evidence"][0]["memory_ids"] for row in ordinary))
            multi_party = next(
                row for row in ordinary if "multi-party" in row["context_id"]
            )
            self.assertEqual(
                multi_party["evidence"][0]["normalized_session_id"], "session1"
            )


if __name__ == "__main__":
    unittest.main()
