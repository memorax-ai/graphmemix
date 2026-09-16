import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mm_memory_bench.benchmarks.bundle import BundleWriter, content_text
from mm_memory_bench.evaluation.native_runner import OpenAICompatibleJudge, judge_predictions
from mm_memory_bench.evaluation.native import personamem_v2_judge as native


def question(qid="q1", preference="I like blue"):
    return {
        "question_id": qid, "context_id": "c", "semantic_question_id": qid,
        "subset": "multimodal_32k_generative", "prompt": [content_text("MODIFIED READER QUERY")],
        "task": {"response_type": "text"}, "answer": {"text": "PRIVATE_REFERENCE"},
        "evidence": [], "memory_scope": {"mode": "all"},
        "metadata": {"native_eval_mode": "generative", "native_user_query": "What color?",
                     "preference": preference},
    }


class FakePreferenceJudge:
    model = "fixture"
    scoring_protocol = "personamem_v2_open"

    def __init__(self):
        self.items = []

    def judge(self, item):
        self.items.append(item)
        return native.parse_response("Reasoning. \\boxed{0.75}")


class PersonaJudgeTests(unittest.TestCase):
    def test_pinned_official_prompt_hash(self):
        self.assertEqual(native.RUBRIC_SHA256, "b2ca516e66ffa8571ed5e5300c2506a2ac241f1e5b3c67014c18ea86ea838c7a")

    def test_exact_payload_and_prompt_polarity(self):
        for pref, template in [("I like blue", native.JUDGE_PROMPT_NARROW_POSITIVE),
                               ("Do not use my age", native.JUDGE_PROMPT_NARROW_NEGATIVE),
                               (" do not use my age", native.JUDGE_PROMPT_NARROW_POSITIVE)]:
            item = native.judge_item(question(preference=pref), "answer")
            self.assertEqual(item, {"user_query": "What color?", "preference": pref, "model_response": "answer"})
            body = native.request_body("model", item)
            self.assertEqual(body["messages"], [{"role": "user", "content": template.format(**item)}])
            self.assertNotIn("response_format", body)
            self.assertNotIn("PRIVATE_REFERENCE", str(body))
            self.assertNotIn("MODIFIED READER QUERY", str(body))

    def test_native_numeric_rules_and_invalid_response(self):
        for text, expected in [(r"reason \boxed{0.75}", .75), (r"\boxed {0.2}", .2),
                               (r"\boxed{1.5}", 1), ("Score: 0.3", .3),
                               ("rating 0.4", .4), ("0.7 / 1.0", .7), (r"\boxed{0}", 0)]:
            self.assertEqual(native.parse_response(text)["score"], expected)
        for text in ["not a score", "", '{"correct": true}']:
            self.assertEqual(native.parse_response(text)["score"], 0.0)
        for value in [True, -1, 2, float("nan"), float("inf"), "0.5"]:
            with self.assertRaises(ValueError):
                native.normalize({"score": value})

    def test_missing_original_query_and_mcq_rejected(self):
        q = question()
        del q["metadata"]["native_user_query"]
        with self.assertRaisesRegex(ValueError, "regenerate"):
            native.judge_item(q, "answer")
        q = question()
        q["subset"] = "multimodal_32k"
        with self.assertRaises(ValueError):
            native.judge_item(q, "answer")

    def test_transport_uses_native_prompt_without_json_constraint(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                return json.dumps({"choices": [{"message": {"content": "Reason \\boxed{0.6}"}}]}).encode()
        backend = OpenAICompatibleJudge(model="fixture", scoring_protocol="personamem_v2_open")
        with patch("urllib.request.urlopen", return_value=Response()) as request:
            result = backend.judge(native.judge_item(question(), "answer"))
        self.assertEqual(result["score"], .6)
        body = json.loads(request.call_args.args[0].data)
        self.assertNotIn("response_format", body)
        self.assertIn("Ground truth user preference: I like blue", body["messages"][0]["content"])

    def test_shared_runner_resume_and_preference_change(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bundle = root / "bundle"
            with BundleWriter(bundle, {"benchmark": "PersonaMem-v2"}) as w:
                w.add_context({"context_id": "c", "benchmark": "personamem_v2"})
                w.add_question(question())
                w.add_question(question("q2", "Do not use age"))
            predictions = root / "predictions.jsonl"
            predictions.write_text(''.join(json.dumps({"question_id": qid, "prediction": "answer"}) + '\n' for qid in ["q1", "q2"]))
            output = root / "judgments.jsonl"
            def run(backend):
                return judge_predictions(backend, bundle, predictions, output,
                                         scoring_protocol="personamem_v2_open", concurrency=2)
            backend = FakePreferenceJudge()
            result = run(backend)
            self.assertEqual(result["mean_score_valid_only"], .75)
            self.assertEqual(result["by_preference_kind"]["negative"]["count"], 1)
            self.assertNotIn("accuracy_conservative", result)
            rows = [json.loads(l) for l in output.read_text().splitlines()]
            self.assertTrue(all("correct" not in row and row["judge_response"] for row in rows))
            resumed = FakePreferenceJudge()
            run(resumed)
            self.assertFalse(resumed.items)
            path = bundle / "questions.jsonl"
            qs = [json.loads(l) for l in path.read_text().splitlines()]
            qs[0]["metadata"]["preference"] = "I like red"
            path.write_text(''.join(json.dumps(q) + '\n' for q in qs))
            changed = FakePreferenceJudge()
            run(changed)
            self.assertEqual(len(changed.items), 1)
            self.assertEqual(changed.items[0]["preference"], "I like red")

    def test_failure_summary_does_not_call_scores_accuracy(self):
        summary = native.summarize([
            {"status": "ok", "score": .8, "subset": "32k", "preference_kind": "positive"},
            {"status": "error", "score": 0, "subset": "32k", "preference_kind": "negative"},
        ])
        self.assertEqual(summary["failed_judgments"], 1)
        self.assertEqual(summary["mean_score_conservative"], .4)
        self.assertEqual(summary["mean_score_valid_only"], .8)
        self.assertIsNone(summary["by_preference_kind"]["negative"]["mean_score_valid_only"])

    def test_binary_normalization_is_unchanged(self):
        from mm_memory_bench.evaluation.judge import _normalized_judgment
        self.assertEqual(_normalized_judgment({"correct": True}), (True, 1.0))
        with self.assertRaises(ValueError):
            _normalized_judgment({"score": .8})

    def test_mismatched_backend_rejected_before_reading_bundle(self):
        with self.assertRaisesRegex(ValueError, "protocols differ"):
            judge_predictions(FakePreferenceJudge(), Path("missing"), Path("missing"), Path("missing"), scoring_protocol="m3exam")

    def test_cli_protocol_option(self):
        from mm_memory_bench.cli import _parser
        args = _parser().parse_args(["judge", "bundle", "predictions", "--output", "out", "--model", "m", "--scoring-protocol", "personamem_v2_open"])
        self.assertEqual(args.scoring_protocol, "personamem_v2_open")
