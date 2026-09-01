import json
import tempfile
import unittest
from pathlib import Path

from mm_memory_bench.converters.memeye import convert
from mm_memory_bench.core import load_bundle, validate_bundle
from mm_memory_bench.harness import _visible_context
from mm_memory_bench.reader import BundleReader


class MemEyeConverterTest(unittest.TestCase):
    def _write_json(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def _fixture(self, root: Path) -> Path:
        snapshot = root / "memeye"
        fixture_revision = "1" * 40
        (snapshot / ".git/refs/heads").mkdir(parents=True)
        (snapshot / ".git/HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (snapshot / ".git/refs/heads/main").write_text(fixture_revision + "\n", encoding="utf-8")
        base = {
            "character_profile": {
                "name": "Fixture_Task",
                "task_family": "visual_fixture",
                "persona_summary": "SECRET PROFILE SUMMARY",
                "traits": ["SECRET TRAIT"],
            },
            "multi_session_dialogues": [
                {
                    "session_id": "D1",
                    "date": "2026-01-01",
                    "network": "fixture-network",
                    "title": "Fixture session",
                    "dialogues": [
                        {
                            "round": "D1:1",
                            "user": "Remember both pictures.",
                            "assistant": "I will.",
                            "input_image": [
                                "Fixture_Task/one.jpg",
                                "Fixture_Task/two.png",
                            ],
                            "image_id": ["IMG-1", "IMG-2"],
                            "image_caption": ["DERIVED ONE", "DERIVED TWO"],
                        }
                    ],
                }
            ],
        }
        rotations = [
            {"A": "red", "B": "blue", "C": "green", "D": "yellow", "answer": "A"},
            {"A": "yellow", "B": "red", "C": "blue", "D": "green", "answer": "B"},
            {"A": "green", "B": "yellow", "C": "red", "D": "blue", "answer": "C"},
            {"A": "blue", "B": "green", "C": "yellow", "D": "red", "answer": "D"},
        ]
        mcq = {
            **base,
            "human-annotated QAs": [
                {
                    "question_id": "Q1",
                    "question": "Which colour was requested?",
                    "answer": "D",
                    "point": [["X4"], ["Y2"]],
                    "session_id": ["D1"],
                    "clue": ["D1:1"],
                    "question_image": "Fixture_Task/query.png",
                    "explanation": "PRIVATE GOLD EXPLANATION",
                    "latest_clue_round": "D1:1",
                    "stale_clue_rounds": [],
                    "options": rotations,
                }
            ],
        }
        open_base = json.loads(json.dumps(base))
        open_base["multi_session_dialogues"][0]["dialogues"][0]["user"] = (
            "Open-specific history wording."
        )
        opened = {
            **open_base,
            "human-annotated QAs": [
                {
                    "question_id": "Q1",
                    "question": "Which colour was requested?",
                    "answer": "red",
                    "point": [["X4"], ["Y2"]],
                    "session_id": ["D1"],
                    "clue": ["D1:1"],
                    "question_image": "Fixture_Task/query.png",
                    "visual_state_probe_passed": True,
                }
            ],
        }
        self._write_json(snapshot / "data/dialog/Fixture_Task.json", mcq)
        self._write_json(snapshot / "data/dialog/Fixture_Task_Open.json", opened)
        # A malformed derived concatenation proves that it is never parsed.
        self._write_json(snapshot / "data/dialog/concat_fixture.json", ["ignore me"])
        image_dir = snapshot / "data/image/Fixture_Task"
        image_dir.mkdir(parents=True)
        for name in ("one.jpg", "two.png", "query.png"):
            (image_dir / name).write_bytes(name.encode())
        return snapshot

    def test_mirrored_variants_and_round_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self._fixture(root)
            bundle_root = root / "unified" / "memeye"
            result = convert(snapshot, bundle_root)

            self.assertEqual(result["semantic_questions"], 1)
            self.assertEqual(result["physical_question_rows"], 5)
            self.assertEqual(
                result["counts"],
                {"contexts": 2, "memories": 4, "assets": 3, "questions": 5},
            )
            self.assertEqual(validate_bundle(bundle_root, check_assets=True)["missing_asset_count"], 0)
            bundle = load_bundle(bundle_root)
            batches = list(BundleReader(bundle_root).iter_context_batches())
            self.assertEqual([len(batch.questions) for batch in batches], [4, 1])
            self.assertTrue(all(len(batch.memories) == 2 for batch in batches))
            self.assertTrue(
                all(
                    context["profile"] == {"name": "Fixture_Task"}
                    for context in bundle["contexts"]
                )
            )
            visible_contexts = [_visible_context(context) for context in bundle["contexts"]]
            self.assertNotIn("SECRET", json.dumps(visible_contexts))
            self.assertEqual(visible_contexts[0]["profile"], {"name": "Fixture_Task"})
            self.assertEqual(
                {context["metadata"]["evaluation_mode"] for context in bundle["contexts"]},
                {"mcq", "open"},
            )
            self.assertTrue(
                all(
                    context["metadata"]["native"]["sessions"][0]["network"]
                    == "fixture-network"
                    for context in bundle["contexts"]
                )
            )
            self.assertEqual(
                bundle["contexts"][0]["metadata"]["native"]["character_profile"][
                    "persona_summary"
                ],
                "SECRET PROFILE SUMMARY",
            )

            user = next(row for row in bundle["memories"] if row["role"] == "user")
            image_parts = [part for part in user["content"] if part["type"] == "image"]
            self.assertEqual(len(image_parts), 2)
            self.assertTrue(all(part["annotations"]["role_assignment_inferred"] for part in image_parts))
            self.assertTrue(all(part["annotations"]["assigned_role"] == "user" for part in image_parts))
            self.assertEqual(user["metadata"]["derived"]["image_captions"][0], "DERIVED ONE")

            semantic_ids = {row["semantic_question_id"] for row in bundle["questions"]}
            self.assertEqual(len(semantic_ids), 1)
            mcq = [row for row in bundle["questions"] if row["variant"]["response_format"] == "multiple_choice"]
            opened = [row for row in bundle["questions"] if row["variant"]["response_format"] == "open"]
            self.assertEqual([row["answer"]["text"] for row in mcq], ["A", "B", "C", "D"])
            self.assertEqual(opened[0]["answer"]["text"], "red")
            self.assertTrue(all(row["subset"] == "mcq" for row in mcq))
            self.assertEqual(opened[0]["subset"], "open")
            self.assertTrue(all(row["context_id"].endswith(":mcq") for row in mcq))
            self.assertTrue(opened[0]["context_id"].endswith(":open"))
            self.assertEqual(
                mcq[0]["metadata"]["native"]["explanation"],
                "PRIVATE GOLD EXPLANATION",
            )
            self.assertTrue(opened[0]["metadata"]["native"]["visual_state_probe_passed"])
            mode_user_text = {
                row["metadata"]["evaluation_mode"]: row["content"][0]["text"]
                for row in bundle["memories"]
                if row["role"] == "user"
            }
            self.assertEqual(mode_user_text["open"], "Open-specific history wording.")
            self.assertEqual(mode_user_text["mcq"], "Remember both pictures.")
            for row in bundle["questions"]:
                self.assertEqual(row["memory_scope"], {"mode": "all"})
                self.assertEqual(row["tags"], ["X4", "Y2"])
                self.assertEqual(len(row["evidence"]), 2, "a clue round maps to both messages")
                self.assertEqual(len([p for p in row["prompt"] if p["type"] == "image"]), 1)

            manifest = json.loads((bundle_root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["release_counts"]["semantic_questions"], 1)
            self.assertEqual(manifest["release_counts"]["physical_question_rows"], 5)
            self.assertEqual(manifest["subsets"], ["mcq", "open"])
            self.assertEqual(manifest["raw_snapshot"]["resolved_data_revision"], "1" * 40)
            self.assertIn("concat_", manifest["normalization"]["ignored_files"])
            self.assertIn("upstream_asset_risk", manifest["license"])


if __name__ == "__main__":
    unittest.main()
