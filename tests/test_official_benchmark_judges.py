"""Pinned prompt contracts and protocol/runner boundaries; no API calls."""
import json
from unittest.mock import patch

import pytest

from mm_memory_bench.evaluation.native_runner import OpenAICompatibleJudge, judge_predictions
from mm_memory_bench.evaluation.native import m3exam_judge as m3
from mm_memory_bench.evaluation.native import mobilemem_omni_judge as omni


def q(bench="m3exam", kind="mr"):
    return {"question_id": bench + ":q", "context_id": "c", "subset": "test",
            "prompt": [{"type": "text", "text": "Question?"}],
            "task": {"response_type": "text", "subcategory": kind},
            "answer": {"text": "first", "accepted_answers": ["first", "second"],
                       "native_ordered_answers": ["first", "second"]},
            "metadata": {"native_label": "specific label", "native_evidence": []}}


def test_pinned_prompts():
    assert m3.RUBRIC_SHA256 == "f390f4c9682b842289da077352373f1148be325523fe3d47fd11c7c0089b4a8c"
    assert omni.RUBRIC_SHA256 == "45aa76b94578c3623d55d3c79b2e02915eb9655fe855656eb06c5440432c79fb"


@pytest.mark.parametrize("raw,score", [("0", 0), ("0.25", .25), ("0.5", .5), ("0.75", .75),
    ("1", 1), ("score=.62", .5), ("0.625", .5), ("nonsense", 0), ("", 0),
    ("9.8", 0), ("Score 0.88", 1), ("0.75 then 0", .75), ("1.000", 1)])
def test_m3_official_parse(raw, score):
    assert m3.parse_response(raw)["score"] == score


def test_m3_reference_and_request():
    question = q()
    item = m3.judge_item(question, "second")
    assert item["gold"] == "specific label"
    question["metadata"]["native_label"] = ""
    assert m3.judge_item(question, "second")["gold"] == "first"
    body = m3.request_body("model", item)
    assert body == {"model": "model", "temperature": 0.0, "max_tokens": 16,
                    "messages": [{"role": "user", "content": m3.JUDGE_PROMPT_TEMPLATE.format(**item)}]}
    for kind in ("fm", "fj", "unknown"):
        with pytest.raises(ValueError): m3.judge_item(q(kind=kind), "answer")
    for score in (.1, True, float("nan")):
        with pytest.raises(ValueError): m3.normalize({"score": score})


def test_m3_summary_native_rounding_denominator():
    rows = [{"native_type": "mr", "status": "ok", "score": 1, "prediction": "yes"},
            {"native_type": "mr", "status": "ok", "score": .25, "prediction": "maybe"},
            {"native_type": "tr", "status": "ok", "score": 0, "prediction": ""}]
    summary = m3.summarize(rows)
    assert summary["total"] == {"count": 3, "answered": 2, "llm_score": .4167}
    assert summary["per_type"]["mr"]["llm_score"] == .625
    assert "accuracy_conservative" not in summary


def test_omni_gold_evidence_and_placeholder_substitution():
    question = q("mobilemem_omni", "multi_hop")
    question["metadata"]["native_evidence"] = [{"session_id": "SECRET_ID", "explanation": "evidence"},
                                                {"session_id": "NO_EXPLANATION"}, "plain evidence"]
    item = omni.judge_item(question, "second")
    assert item["gold"] == "second"
    assert item["evidence"] == ["evidence", "plain evidence"]
    assert item["category"] == "Multi-hop"
    item["question"] = "Keep [Gold Answer] as data"
    body = omni.request_body("m", item)
    content = body["messages"][0]["content"]
    assert "Question: Keep [Gold Answer] as data" in content
    assert 'Evidence: ["evidence", "plain evidence"]' in content
    assert "Gold answer: second" in content
    assert "SECRET_ID" not in content
    assert "response_format" not in body
    assert "f1" not in item and "bleu1" not in item


@pytest.mark.parametrize("raw,label", [('Reason. {"label":"CORRECT"}', "CORRECT"),
    ('```json\n{"label":"WRONG"}\n```', "WRONG")])
def test_omni_local_parser(raw, label):
    result = omni.parse_response(raw)
    assert result["label"] == label
    assert result["score"] == (label == "CORRECT")


@pytest.mark.parametrize("raw", ['CORRECT', '{"correct":true}', '{"label":"correct"}',
    '{"label":"CORRECT"} {"label":"WRONG"}', '{"label":"CORRECT or WRONG"}', ''])
def test_omni_invalid_local_parse_is_resumable_error(raw):
    with pytest.raises(ValueError): omni.parse_response(raw)


def test_omni_summary_excludes_null_labels():
    rows = [{"status": "ok", "label": "CORRECT", "native_category": "Multi-hop"},
            {"status": "ok", "label": "WRONG", "native_category": "Multi-hop"},
            {"status": "ok", "label": None, "native_category": "Multi-hop"},
            {"status": "error", "label": None, "native_category": "Single-hop"}]
    summary = omni.summarize(rows)
    assert summary["overall"] == {"LLM_JUDGE": .5}
    assert summary["valid_judgments"] == 2
    assert summary["failed_judgments"] == 1
    assert summary["skipped_judgments"] == 1
    assert summary["by_category"]["Multi-hop"]["count"] == 3


@pytest.mark.parametrize("bench,protocol,raw", [("m3exam", "m3exam", "0.75"),
                                               ("mobilemem_omni", "mobilemem_omni", '{"label":"CORRECT"}')])
def test_actual_transport_request(tmp_path, bench, protocol, raw):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({"choices": [{"message": {"content": raw}}]}).encode()
    backend = OpenAICompatibleJudge(model="fixture", scoring_protocol=protocol)
    item = backend.protocol.judge_item(q(bench), "prediction")
    with patch("urllib.request.urlopen", return_value=Response()) as call:
        backend.judge(item)
    assert json.loads(call.call_args.args[0].data) == backend.protocol.request_body("fixture", item)


def test_empty_prediction_policy_and_wrong_bundle(tmp_path):
    question = q("mobilemem_omni")
    (tmp_path/"manifest.json").write_text(json.dumps({"benchmark": "MobileMem-Omni"}))
    qp = tmp_path/"questions.jsonl"
    qp.write_text(json.dumps(question)+'\n')
    pp = tmp_path/"predictions.jsonl"
    pp.write_text(json.dumps({"question_id": question["question_id"], "prediction": ""})+'\n')
    class Backend:
        model = "fake"
        scoring_protocol = "mobilemem_omni"
        calls = 0
        def judge(self, item):
            self.calls += 1
            return {"label": "WRONG"}
    backend = Backend()
    out = tmp_path/"out.jsonl"
    summary = judge_predictions(backend, tmp_path, pp, out, scoring_protocol="mobilemem_omni")
    assert backend.calls == 1  # Official evaluator still calls Judge on empty prediction.
    assert summary["overall"]["LLM_JUDGE"] == 0
    question["answer"] = {"text": ""}
    qp.write_text(json.dumps(question)+'\n')
    summary = judge_predictions(backend, tmp_path, pp, out, scoring_protocol="mobilemem_omni")
    assert backend.calls == 1
    assert summary["skipped_judgments"] == 1
    (tmp_path/"manifest.json").write_text(json.dumps({"benchmark":"M3Exam"}))
    with pytest.raises(ValueError, match="requires"):
        judge_predictions(backend, tmp_path, pp, out, scoring_protocol="mobilemem_omni")


@pytest.mark.parametrize("content,finish", [("", "length"), ("1", "length"),
                                          ("", "stop"), (None, "stop")])
def test_invalid_transport_is_error_and_keeps_usage(tmp_path, content, finish):
    from mm_memory_bench.evaluation.native_runner import JudgeResponseError
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({"choices": [{"message": {"content": content},
                                           "finish_reason": finish}],
                               "usage": {"completion_tokens": 16}}).encode()
    backend = OpenAICompatibleJudge(model="fixture", scoring_protocol="m3exam", max_tokens=512)
    question = q()
    (tmp_path / "manifest.json").write_text(json.dumps({"benchmark": "m3exam"}))
    (tmp_path / "questions.jsonl").write_text(json.dumps(question) + "\n")
    pred = tmp_path / "predictions.jsonl"
    pred.write_text(json.dumps({"question_id": question["question_id"], "prediction": "answer"}) + "\n")
    out = tmp_path / "out.jsonl"
    with patch("urllib.request.urlopen", return_value=Response()) as call:
        result = judge_predictions(backend, tmp_path, pred, out, scoring_protocol="m3exam")
    row = json.loads(out.read_text())
    assert result["failed_judgments"] == 1
    assert row["status"] == "error"
    assert row["judge_response_metadata"]["finish_reason"] == finish
    assert row["judge_response_metadata"]["usage"]["completion_tokens"] == 16
    assert json.loads(call.call_args.args[0].data)["max_tokens"] == 512
    # A valid zero must remain a successful score, and a failed cache must retry.
    with patch("urllib.request.urlopen", return_value=Response()):
        with pytest.raises(JudgeResponseError): backend.judge(m3.judge_item(question, "answer"))


def test_request_budget_change_invalidates_cached_judgment(tmp_path):
    question = q()
    (tmp_path / "manifest.json").write_text(json.dumps({"benchmark": "m3exam"}))
    (tmp_path / "questions.jsonl").write_text(json.dumps(question) + "\n")
    pred = tmp_path / "predictions.jsonl"
    pred.write_text(json.dumps({"question_id": question["question_id"], "prediction": "answer"}) + "\n")
    out = tmp_path / "out.jsonl"
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps({"choices": [{"message": {"content": "0"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 74}}).encode()
    backend = OpenAICompatibleJudge(model="fixture", scoring_protocol="m3exam", max_tokens=512)
    with patch("urllib.request.urlopen", return_value=Response()) as call:
        for _ in range(2): judge_predictions(backend, tmp_path, pred, out, scoring_protocol="m3exam")
        assert call.call_count == 1
        assert json.loads(out.read_text())["status"] == "ok"
        assert json.loads(out.read_text())["score"] == 0
        old = json.loads(out.read_text()); old.pop("judge_request_config")
        out.write_text(json.dumps(old) + "\n")
        judge_predictions(backend, tmp_path, pred, out, scoring_protocol="m3exam")
        assert call.call_count == 2
        backend = OpenAICompatibleJudge(model="fixture", scoring_protocol="m3exam", max_tokens=1024)
        judge_predictions(backend, tmp_path, pred, out, scoring_protocol="m3exam")
        assert call.call_count == 3
