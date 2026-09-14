import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from mm_memory_bench.cli import main
from mm_memory_bench.evaluation import dispatcher as dispatch


def question(benchmark, name, *, response="text", kind="mr"):
    return {"question_id": f"{benchmark}:{name}", "context_id": "c",
            "subset": "test", "prompt": [{"type": "text", "text": name}],
            "task": {"response_type": response, "subcategory": kind},
            "answer": {"text": "yes"}, "metadata": {"native_evidence": []}}


def mcq(benchmark, name="mcq"):
    q = question(benchmark, name, response="choice")
    label = "(a)" if benchmark == "persona_mme" else "A"
    q.update(choices=[{"choice_id": label, "text": "yes"}])
    q["answer"].update(choice_id=label, native_label=label)
    return q


def bundle_files(tmp_path, benchmark, questions):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"benchmark": benchmark}))
    (bundle / "questions.jsonl").write_text(''.join(json.dumps(q) + '\n' for q in questions))
    pred = tmp_path / "predictions.jsonl"
    pred.write_text(''.join(json.dumps({"question_id": q["question_id"],
                                      "prediction": "Answer: (A)" if q["task"]["response_type"] == "choice" else "yes"}) + '\n'
                            for q in questions))
    return bundle, pred, tmp_path / "results" / "judgments.jsonl"


class FakeJudge:
    model = "fixture"

    def __init__(self, protocol):
        self.scoring_protocol = protocol
        self.calls = []
        self.fail = False

    def judge(self, item):
        self.calls.append(item)
        if self.fail:
            raise RuntimeError("test service unavailable")
        if self.scoring_protocol in {"personamem_v2_open", "m3exam"}:
            return {"score": .75}
        if self.scoring_protocol == "mobilemem_omni":
            return {"label": "CORRECT"}
        return {"correct": True}


@pytest.fixture
def clients(monkeypatch):
    clients = {}

    def factory(**kwargs):
        route = kwargs["scoring_protocol"]
        return clients.setdefault(route, FakeJudge(route))

    monkeypatch.setattr(dispatch, "backend_from_env", factory)
    return clients


def test_script_only_cli_needs_no_model_or_api(tmp_path, monkeypatch):
    b, p, out = bundle_files(tmp_path, "SMMBench", [mcq("smmbench")])
    factory = Mock(side_effect=AssertionError("must not create LLM client"))
    monkeypatch.setattr(dispatch, "backend_from_env", factory)
    assert main(["judge", str(b), str(p), "--output", str(out),
                 "--scoring-protocol", "benchmark"]) == 0
    factory.assert_not_called()
    assert not out.exists()
    rows = [json.loads(l) for l in (out.parent / "judgments.native.jsonl").read_text().splitlines()]
    assert rows[0]["metrics"] == {"choice_accuracy": 1}


def test_m3_mixed_routes_and_resume(tmp_path, clients):
    qs = [question("m3exam", k, kind=k) for k in ["fj", "fm", "mr"]]
    b, p, out = bundle_files(tmp_path, "M3Exam", qs)
    result = dispatch.score_benchmark(b, p, out, model="fixture")
    assert result["selected_questions"] == 3
    assert result["routes"]["script"]["completed"] == 2
    assert result["routes"]["m3exam"]["completed"] == 1
    assert len(clients["m3exam"].calls) == 1
    assert clients["m3exam"].calls[0]["question"] == "mr"
    dispatch.score_benchmark(b, p, out, model="fixture")
    assert len(clients["m3exam"].calls) == 1
    dispatch.score_benchmark(b, p, out, model="fixture", resume=False)
    assert len(clients["m3exam"].calls) == 2


@pytest.mark.parametrize("mixed", [False, True])
def test_distinct_outputs_keep_each_runs_script_results(tmp_path, clients, mixed):
    benchmark = "m3exam" if mixed else "smmbench"
    qs = [question(benchmark, "script", kind="fj"), question(benchmark, "llm")] if mixed else [mcq(benchmark)]
    b, first_predictions, first_output = bundle_files(tmp_path, benchmark, qs)
    first = dispatch.score_benchmark(b, first_predictions, first_output, model="fixture")
    first_script = Path(first["routes"]["script"]["output"])
    saved_records = first_script.read_bytes()
    saved_summary = first_script.with_suffix(".summary.json").read_bytes()
    assert first["routes"]["script"]["summary"]["total"]["metrics"][
        "em" if mixed else "choice_accuracy"
    ]["mean"] == 1

    second_predictions = tmp_path / "second_predictions.jsonl"
    second_predictions.write_text(''.join(
        json.dumps({"question_id": q["question_id"], "prediction": "wrong"}) + '\n'
        for q in qs
    ))
    second_output = first_output.with_name("second.judgments.jsonl")
    second = dispatch.score_benchmark(b, second_predictions, second_output, model="fixture")
    assert first_script.read_bytes() == saved_records
    assert first_script.with_suffix(".summary.json").read_bytes() == saved_summary
    assert second["routes"]["script"]["output"] != str(first_script)
    assert first_script == first_output.parent / "judgments.native.jsonl"
    assert second["routes"]["script"]["summary"]["total"]["metrics"][
        "em" if mixed else "choice_accuracy"
    ]["mean"] == 0


def test_persona_mixed_keeps_preference_summary(tmp_path, clients):
    opened = question("personamem_v2", "open")
    opened.update(subset="multimodal_32k_generative", metadata={
        "native_eval_mode": "generative", "native_user_query": "Color?", "preference": "I like blue"})
    b, p, out = bundle_files(tmp_path, "PersonaMem-v2", [mcq("personamem_v2"), opened])
    result = dispatch.score_benchmark(b, p, out, model="fixture")
    assert set(result["routes"]) == {"script", "personamem_v2_open"}
    summary = result["routes"]["personamem_v2_open"]["summary"]
    assert summary["mean_score_valid_only"] == .75
    assert "accuracy_conservative" not in summary
    assert "qa" not in clients


def test_omni_failure_retry_and_prediction_change(tmp_path, clients):
    b, p, out = bundle_files(tmp_path, "MobileMem-Omni", [question("mobilemem_omni", "q")])
    client = clients.setdefault("mobilemem_omni", FakeJudge("mobilemem_omni"))
    client.fail = True
    assert dispatch.score_benchmark(b, p, out, model="fixture")["failed_judgments"] == 1
    client.fail = False
    assert dispatch.score_benchmark(b, p, out, model="fixture")["failed_judgments"] == 0
    assert len(client.calls) == 2
    row = json.loads(p.read_text()); row["prediction"] = "changed"
    p.write_text(json.dumps(row) + '\n')
    dispatch.score_benchmark(b, p, out, model="fixture")
    assert len(client.calls) == 3


@pytest.mark.parametrize("problem", ["missing", "duplicate", "unknown", "bad_task", "bad_gold", "no_model"])
def test_preflight_no_calls_or_output(tmp_path, monkeypatch, problem):
    qs = [question("m3exam", "llm", kind="mr"), question("m3exam", "script", kind="fj")]
    if problem == "bad_task":
        qs[1]["task"] = {"response_type": "unsupported"}
    if problem == "bad_gold":
        qs[1]["answer"]["accepted_answers"] = []
    b, p, out = bundle_files(tmp_path, "M3Exam", qs)
    if problem == "missing": p.write_text(p.read_text().splitlines()[0] + '\n')
    if problem == "duplicate": p.write_text(p.read_text() * 2)
    if problem == "unknown":
        p.write_text(p.read_text() + json.dumps({"question_id": "unknown", "prediction": "x"}) + '\n')
    factory = Mock(side_effect=AssertionError("preflight must finish first"))
    monkeypatch.setattr(dispatch, "backend_from_env", factory)
    with pytest.raises(ValueError):
        dispatch.score_benchmark(b, p, out, model=None if problem == "no_model" else "fixture")
    factory.assert_not_called()
    assert not out.parent.exists()


def test_explicit_subset_max_items_and_in_memory_selection(tmp_path, clients):
    qs = [question("mobilemem_omni", str(i)) for i in range(3)]
    b, p, out = bundle_files(tmp_path, "MobileMem-Omni", qs)
    ids = tmp_path / "ids.txt"
    ids.write_text(qs[2]["question_id"] + '\n' + qs[1]["question_id"] + '\n')
    result = dispatch.score_benchmark(b, p, out, model="fixture", question_ids_path=ids, max_items=1)
    assert result["selected_questions"] == 1
    assert json.loads(out.read_text())["question_id"] == qs[2]["question_id"]
    assert result["routes"]["mobilemem_omni"]["summary"]["reporting_track_size"] == 1


def test_output_collision_and_old_qa_requires_model(tmp_path, clients):
    b, p, out = bundle_files(tmp_path, "M3Exam", [question("m3exam", "q", kind="fj"), question("m3exam", "llm")])
    colliding_predictions = tmp_path / "judgments.native.summary.json"
    colliding_predictions.write_bytes(p.read_bytes())
    with pytest.raises(ValueError, match="collide"):
        dispatch.score_benchmark(b, colliding_predictions, tmp_path / "judgments.jsonl", model="fixture")
    assert colliding_predictions.read_bytes() == p.read_bytes()
    with pytest.raises(ValueError, match="collide"):
        dispatch.score_benchmark(b, p, p, model="fixture")
    assert not clients
    assert main(["judge", str(b), str(p), "--output", str(out)]) == 2


def test_smm_plan_and_persona_routes():
    tool = question("smmbench", "tool", response="structured_json")
    tool["tool_mode"] = "plan"
    assert dispatch.scoring_route("smmbench", tool) == "script"
    assert dispatch.scoring_route("persona_mme", mcq("persona_mme")) == "script"
    with pytest.raises(ValueError):
        dispatch.scoring_route("unknown", tool)
