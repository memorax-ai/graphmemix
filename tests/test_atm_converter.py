import json
import tempfile
import unittest
from pathlib import Path

from mm_memory_bench.benchmarks.converters.atm import OFFICIAL_DATA_REVISION, convert
from mm_memory_bench.benchmarks.bundle import load_bundle, validate_bundle


class ATMConverterTest(unittest.TestCase):
    def _write_json(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def _fixture(self, root: Path) -> Path:
        snapshot = root / "atm_bench"
        image_id = "20240102_030405"
        video_id = "20240103_040506"
        email_id = "email202401010001"

        self._write_json(
            snapshot / "data/raw_memory/email/emails.json",
            [
                {
                    "id": email_id,
                    "timestamp": "2024-01-01 01:02:03",
                    "short_summary": "Derived one-line email summary",
                    "detail": "Date: 2024-01-01\nSubject: Receipt\n\nContent: Paid €62.8.",
                }
            ],
        )
        self._write_json(
            snapshot / "data/processed_memory/image_batch_results.json",
            [
                {
                    "image_path": f"data/raw_memory/image/{image_id}.jpg",
                    "file_size": 3,
                    "file_modified": "2024-01-02 03:04:05",
                    "timestamp": "2024-01-02 03:04:05",
                    "location": [1.0, 2.0],
                    "location_name": "Derived Place",
                    "city": "Derived City",
                    "camera_settings": {"iso": 100},
                    "caption": "SECRET DERIVED IMAGE CAPTION",
                    "short_caption": "Derived image summary",
                    "tags": ["receipt"],
                    "entities": [{"entity": "receipt", "type": "object"}],
                    "ocr_text": "TOTAL 62.8",
                    "safety_content": "safe",
                    "processed_at": "2026-03-01 00:00:00",
                    "processing_version": "1.1",
                    "model_used": "fixture-vlm",
                }
            ],
        )
        self._write_json(
            snapshot / "data/processed_memory/video_batch_results.json",
            [
                {
                    "video_path": f"data/raw_memory/video/{video_id}.mp4",
                    "file_size": 4,
                    "file_modified": "2024-01-03 04:05:06",
                    "timestamp": "2024-01-03 04:05:06+00:00",
                    "location": [1.0, 2.0],
                    "location_name": "Derived Place",
                    "city": "Derived City",
                    "has_gps": True,
                    "duration": 3.5,
                    "duration_formatted": "3.5s",
                    "width": 720,
                    "height": 480,
                    "rotation": 0,
                    "codec": "h264",
                    "device": "",
                    "caption": "SECRET DERIVED VIDEO CAPTION",
                    "short_caption": "Derived video summary",
                    "tags": ["bridge"],
                    "entities": [],
                    "ocr_text": "",
                    "safety_content": "safe",
                    "processed_at": "2026-03-01 00:00:00",
                    "processing_version": "1.0",
                    "model_used": "fixture-vlm",
                    "num_frames_analyzed": 8,
                }
            ],
        )

        default_question = {
            "id": "default-q",
            "question": "How much did I pay?",
            "answer": "€62.8",
            "notes": "",
            "evidence_ids": [email_id],
            "qtype": "number",
        }
        hard_question = {
            "id": "hard-q",
            "question": "Which media recorded the event?",
            "answer": f"{image_id}, {video_id}",
            "notes": "PRIVATE: use the receipt and both media IDs.",
            "evidence_ids": [image_id, video_id],
            "qtype": "list_recall",
        }
        self._write_json(snapshot / "data/atm-bench/atm-bench.json", [default_question])
        self._write_json(snapshot / "data/atm-bench/atm-bench-hard.json", [hard_question])
        self._write_json(
            snapshot / "data/atm-bench/niah/atm-bench-hard-niah3.json",
            [{**hard_question, "niah_evidence_ids": [email_id, image_id, video_id]}],
        )

        image = snapshot / f"data/raw_memory/image/{image_id}.jpg"
        video = snapshot / f"data/raw_memory/video/{video_id}.mp4"
        image.parent.mkdir(parents=True, exist_ok=True)
        video.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"jpg")
        video.write_bytes(b"mp4!")
        return snapshot

    def test_loss_preserving_shared_context_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self._fixture(root)
            bundle_root = root / "unified" / "atm_bench"
            result = convert(snapshot, bundle_root)

            self.assertEqual(
                result["counts"],
                {"contexts": 1, "memories": 3, "assets": 2, "questions": 2},
            )
            self.assertEqual(
                validate_bundle(bundle_root, check_assets=True)["missing_asset_count"], 0
            )
            bundle = load_bundle(bundle_root)
            context_id = bundle["contexts"][0]["context_id"]
            self.assertTrue(all(row["context_id"] == context_id for row in bundle["memories"]))
            self.assertTrue(all(row["context_id"] == context_id for row in bundle["questions"]))

            memories = {row["metadata"]["source_type"]: row for row in bundle["memories"]}
            self.assertEqual(memories["email"]["source_id"], "email202401010001")
            self.assertEqual(memories["image"]["source_id"], "20240102_030405")
            self.assertEqual(memories["video"]["source_id"], "20240103_040506")
            for media_type in ("image", "video"):
                memory = memories[media_type]
                self.assertEqual(memory["content"][0]["type"], media_type)
                self.assertNotIn("text", memory["content"][0])
                self.assertIn("caption", memory["metadata"]["derived"])
                self.assertNotIn("SECRET DERIVED", json.dumps(memory["content"]))
            self.assertEqual(
                memories["email"]["content"][0]["text"].splitlines()[0],
                "Date: 2024-01-01",
            )
            self.assertEqual(
                memories["email"]["metadata"]["derived"]["short_summary"],
                "Derived one-line email summary",
            )

            questions = {row["subset"]: row for row in bundle["questions"]}
            self.assertEqual(questions["default"]["answer"]["text"], "€62.8")
            self.assertEqual(
                questions["hard"]["answer"]["text"],
                "20240102_030405, 20240103_040506",
            )
            self.assertEqual(
                questions["hard"]["metadata"]["evaluation_private"]["notes"],
                "PRIVATE: use the receipt and both media IDs.",
            )
            self.assertEqual(questions["default"]["memory_scope"], {"mode": "all"})
            self.assertEqual(questions["hard"]["memory_scope"], {"mode": "all"})
            self.assertIn("source_id", questions["hard"]["instruction"])
            for evidence in questions["hard"]["evidence"]:
                self.assertTrue(evidence["memory_id"].startswith("atm_bench:memory:"))
                self.assertIn("native_id", evidence)

            variants = questions["hard"]["metadata"]["evaluation_variants"]["niah"]
            self.assertEqual(len(variants), 1)
            self.assertEqual(variants[0]["candidate_count"], 3)
            self.assertEqual(len(variants[0]["evidence"]), 3)
            self.assertEqual(len(bundle["questions"]), 2, "NIAH must not duplicate questions")

            manifest = json.loads((bundle_root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["raw_snapshot"]["git_revision"], OFFICIAL_DATA_REVISION)
            self.assertEqual(manifest["license"]["id"], "CC-BY-NC-4.0")
            self.assertEqual(
                manifest["evaluation"]["private_metadata"]["policy"],
                "never_expose_to_agent",
            )
            self.assertEqual(manifest["paper_release_difference"]["paper"]["questions"]["hard"], 25)
            self.assertEqual(
                manifest["paper_release_difference"]["public_release"]["questions"]["hard"],
                31,
            )


if __name__ == "__main__":
    unittest.main()
