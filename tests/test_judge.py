import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from mm_memory_bench.core import BundleWriter, content_text
from mm_memory_bench.judge import judge_predictions


class FakeJudge:
    model = "fixture-judge"

    def __init__(self) -> None:
        self.items: list[Mapping[str, Any]] = []

    def judge(self, item: Mapping[str, Any]) -> Mapping[str, Any]:
        self.items.append(item)
        reference = str(item["reference_answer"]["text"])
        correct = str(item["prediction"]) == reference
        return {"correct": correct}


class JudgeTest(unittest.TestCase):
    def test_question_id_allowlist_limits_reporting_track(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            with BundleWriter(bundle, {"benchmark": "fixture"}) as writer:
                writer.add_context({"context_id": "c", "benchmark": "fixture"})
                for question_id in ("q1", "q2"):
                    writer.add_question({
                        "question_id": question_id,
                        "context_id": "c",
                        "subset": "default",
                        "prompt": [content_text("question")],
                        "task": {"response_type": "text"},
                        "answer": {"text": question_id},
                        "evidence": [],
                        "memory_scope": {"mode": "all"},
                    })
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps({"question_id": "q1", "prediction": "q1"}) + "\n"
                + json.dumps({"question_id": "q2", "prediction": "wrong"}) + "\n",
                encoding="utf-8",
            )
            allowlist = root / "ids.txt"
            allowlist.write_text("q1\n", encoding="utf-8")
            backend = FakeJudge()
            summary = judge_predictions(
                backend,
                bundle,
                predictions,
                root / "judgments.jsonl",
                question_ids_path=allowlist,
            )
            self.assertEqual(summary["total_predictions"], 1)
            self.assertEqual(summary["reporting_track_size"], 1)
            self.assertEqual(summary["accuracy_conservative"], 1.0)
            self.assertEqual(len(backend.items), 1)

    def test_empty_and_failed_predictions_are_false_without_calling_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            with BundleWriter(bundle, {"benchmark": "fixture"}) as writer:
                writer.add_context({"context_id": "c", "benchmark": "fixture"})
                for question_id in ("q-empty", "q-error"):
                    writer.add_question({
                        "question_id": question_id,
                        "context_id": "c",
                        "subset": "default",
                        "prompt": [content_text("question")],
                        "task": {"response_type": "text"},
                        "answer": {"text": "Not mentioned."},
                        "evidence": [],
                        "memory_scope": {"mode": "all"},
                    })
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps({"question_id": "q-empty", "prediction": ""}) + "\n"
                + json.dumps({
                    "question_id": "q-error",
                    "prediction": "Not mentioned.",
                    "metadata": {"status": "error", "error_type": "runtime"},
                }) + "\n",
                encoding="utf-8",
            )
            backend = FakeJudge()
            summary = judge_predictions(
                backend, bundle, predictions, root / "judgments.jsonl"
            )
            self.assertEqual(summary["accuracy_conservative"], 0.0)
            self.assertEqual(backend.items, [])

    def test_uniform_judge_records_and_semantic_macro(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            with BundleWriter(bundle, {"benchmark": "fixture"}) as writer:
                writer.add_context({"context_id": "fixture:c0", "benchmark": "fixture"})
                for index, semantic_id, subset, answer in (
                    (1, "fixture:s1", "mcq", "A"),
                    (2, "fixture:s1", "mcq", "B"),
                    (3, "fixture:s2", "open", "blue"),
                ):
                    writer.add_question(
                        {
                            "question_id": f"fixture:q{index}",
                            "semantic_question_id": semantic_id,
                            "context_id": "fixture:c0",
                            "subset": subset,
                            "prompt": [content_text(f"question {index}")],
                            "task": {
                                "response_type": "choice" if subset == "mcq" else "text"
                            },
                            "choices": (
                                [
                                    {"choice_id": "A", "text": "alpha"},
                                    {"choice_id": "B", "text": "beta"},
                                ]
                                if subset == "mcq"
                                else []
                            ),
                            "answer": {"text": answer},
                            "evidence": [{"native_id": "PRIVATE_EVIDENCE"}],
                            "memory_scope": {"mode": "all"},
                            "metadata": {"evaluation_private": {"notes": "SECRET"}},
                        }
                    )
            predictions = root / "predictions.jsonl"
            rows = [
                {"question_id": "fixture:q1", "prediction": "A"},
                {"question_id": "fixture:q2", "prediction": "A"},
                {"question_id": "fixture:q3", "prediction": "blue"},
            ]
            predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            output = root / "judgments.jsonl"
            backend = FakeJudge()
            summary = judge_predictions(
                backend, bundle, predictions, output, concurrency=2
            )

            self.assertEqual(summary["valid_judgments"], 3)
            self.assertEqual(summary["concurrency"], 2)
            self.assertAlmostEqual(summary["accuracy_conservative"], 2 / 3)
            self.assertAlmostEqual(summary["semantic_question_macro_accuracy"], 0.75)
            self.assertEqual(summary["by_subset"]["mcq"]["count"], 2)
            self.assertEqual(len(list(output.read_text(encoding="utf-8").splitlines())), 3)
            self.assertTrue(Path(summary["summary_path"]).is_file())
            self.assertTrue(all("evidence" not in item for item in backend.items))
            self.assertTrue(all("metadata" not in item for item in backend.items))
            judgment = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertNotIn("explanation", judgment)
            self.assertNotIn("judge_response", judgment)

            # A resumed run keeps successful rows and does not call the judge again.
            resumed = FakeJudge()
            second = judge_predictions(resumed, bundle, predictions, output, resume=True)
            self.assertEqual(second["total_predictions"], 3)
            self.assertEqual(resumed.items, [])
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 3)

            # Changing a prediction invalidates only its cached judgment.
            rows[0]["prediction"] = "B"
            predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            changed = FakeJudge()
            third = judge_predictions(changed, bundle, predictions, output, resume=True)
            self.assertEqual(third["total_predictions"], 3)
            self.assertEqual(len(changed.items), 1)
            self.assertEqual(changed.items[0]["prediction"], "B")


if __name__ == "__main__":
    unittest.main()
