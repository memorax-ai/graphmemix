import json
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mm_memory_bench.methods.concrete_universalrag import ConcreteUniversalRAGMethod as UR
from mm_memory_bench.methods.media import image_content, table_text
from mm_memory_bench.benchmarks.converters.smmbench import _parts
from mm_memory_bench.evaluation.native.script_judge import score_question, score_predictions
from mm_memory_bench.evaluation.prediction_status import method_failure
from mm_memory_bench.evaluation.native_runner import judge_predictions


def choice():
    return dict(question_id='smmbench:q', context_id='c', subset='mcq',
                task={'response_type': 'choice'},
                choices=[{'choice_id':'0','text':'Corsica'}, {'choice_id':'1','text':'Sardinia'}],
                answer={'choice_id':'0','text':'Corsica'})



@pytest.mark.parametrize('prediction,score', [
    ('0: Corsica', 1), ('1: Sardinia', 0), ('0: Sardinia', 0),
    ('0: Corsica or 1: Sardinia', 0), ('0: Corsica\nBut choose 1', 0), ('0', 1)])
def test_legacy_numeric_option_renderings(prediction, score):
    assert score_question('smmbench', choice(), prediction)[0]['choice_accuracy'] == score



def test_table_marks_only_actual_structure_and_preserves_ordinary_text():
    value={'title':'Career', 'table_header':['Year'], 'table_rows':[['2020']]}
    for raw in (value, json.dumps(value)):
        parts=_parts(raw, None, None)
        assert parts[0]['annotations']=={'format':'table'}
        assert json.loads(table_text(parts[0]))==value
    assert _parts('None', None, None)==[{'type':'text','text':'None'}]
    assert table_text({'type':'text','text':'Look at Table. 12345678'})==''
    assert json.loads(table_text({'type':'text','text':json.dumps(value)}))==value



def method():
    m=UR.__new__(UR)
    m.adapter_profile='official'
    m.units={k:[] for k in ['paragraph','document','table','image','clip','video']}
    m._flush_pending=Mock()
    m.generation=SimpleNamespace(top_k=10)
    m.answer_model=SimpleNamespace(complete=Mock(return_value='img_51'))
    return m



def test_structured_table_enters_real_table_corpus():
    m=method()
    text=json.dumps({'table_header':['Division'], 'table_rows':[['Championship']]})
    units=m._memory_units({'memory_id':'m','source_id':'stream','content':[{'type':'text','text':text}]})
    assert 'Championship' in units['table'][0]['text']
    assert units['document']  # Existing document coverage is retained.


def test_runner_preserves_only_table_format_before_universalrag_ingest(tmp_path):
    from mm_memory_bench.benchmarks.reader import BundleReader, ContextBatch
    from mm_memory_bench.runner.benchmark import run_context

    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "assets.jsonl").write_text("")
    table = json.dumps({"table_header": ["Division"], "table_rows": [["Championship"]]})
    contents = [
        _parts(table, None, None)[0],
        {"type": "text", "text": table},  # Previously converted bundle.
        {"type": "text", "text": "Year | Division\n2020 | Championship",
         "annotations": {"format": "table"}},  # Explicit marker needs no JSON inference.
        {"type": "text", "text": "Look at Table. 12345678",
         "annotations": {"format": "text"}},
    ]
    for part in contents:
        part.setdefault("annotations", {})["private_answer"] = "SECRET"
    memories = [dict(memory_id=f"m{i}", context_id="c", sequence=i, content=[part])
                for i, part in enumerate(contents)]
    received, units = [], []
    ur = method()

    class Probe:
        def begin_context(self, context): pass
        def ingest(self, memory):
            received.append(memory)
            units.append(ur._memory_units(memory))
        def answer(self, question): return "unused"
        def end_context(self): pass

    with BundleReader(tmp_path) as reader:
        run_context(Probe(), reader, ContextBatch({"context_id": "c"}, memories, []))
    for index in (0, 2):
        assert received[index]["content"][0]["annotations"] == {"format": "table"}
    for index in (1, 3):
        assert "annotations" not in received[index]["content"][0]
    for original, visible, corpora in zip(contents, received, units):
        assert visible["content"][0]["type"] == "text"
        assert visible["content"][0]["text"] == original["text"]
        assert corpora["paragraph"] and corpora["document"]
    assert all(value["table"] for value in units[:3])
    assert not units[3]["table"]
    assert "SECRET" not in json.dumps([received, units])


def test_base_runner_and_smoke_execute_without_optional_methods_dependencies(tmp_path):
    rows = {
        "contexts": [{"context_id": "c", "benchmark": "test"}],
        "memories": [{"memory_id": "m", "context_id": "c", "sequence": 0,
                      "content": [{"type": "text", "text": "blue"}]}],
        "questions": [{"question_id": "q", "context_id": "c", "subset": "qa",
                       "prompt": [{"type": "text", "text": "Color?"}],
                       "answer": {"text": "blue"}}],
        "assets": [],
    }
    (tmp_path / "manifest.json").write_text("{}")
    for table, values in rows.items():
        (tmp_path / f"{table}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in values))
    code = textwrap.dedent("""
        import sys
        from pathlib import Path
        sys.path.insert(0, sys.argv[1])
        from mm_memory_bench.runner.benchmark import run_context
        from mm_memory_bench.runner.smoke import run_diagnostic_smoke
        class Probe:
            def begin_context(self, context): pass
            def ingest(self, memory): self.text = memory['content'][0]['text']
            def answer(self, question):
                assert 'answer' not in question
                return self.text
            def end_context(self): pass
        root = Path(sys.argv[2])
        report = run_diagnostic_smoke(Probe(), root, root / 'report.json')
        assert report['prediction']['prediction'] == 'blue'
        assert 'numpy' not in sys.modules
        assert 'mm_memory_bench.methods' not in sys.modules
    """)
    result = subprocess.run(
        [sys.executable, "-S", "-c", code, str(Path(__file__).parents[1] / "src"), str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr



def test_image_ids_are_public_and_preserved_per_image(monkeypatch):
    monkeypatch.setattr('mm_memory_bench.methods.media.data_url', lambda path:'data:image/png;base64,AA==')
    m=method()
    parts=[{'type':'image','path':f'/renamed/{i}.png','source_id':f'img_{i}.jpg'} for i in (51,52)]
    units=m._memory_units({'memory_id':'same','content':parts,'metadata':{'derived':{}}})
    selected=m._merge_corpus_candidates([units['image']],10)
    assert len(selected)==2
    m.generate_answer({'prompt':[{'type':'text','text':'Return image ID'}], 'answer':{'text':'GOLD_SECRET'}},selected)
    content=m.answer_model.complete.call_args.args[0][0]['content']
    text=' '.join(p.get('text','') for p in content)
    assert 'img_51.jpg' in text and 'img_52.jpg' in text
    assert 'GOLD_SECRET' not in text
    assert sum(p['type']=='image_url' for p in content)==2
    assert len(m._merge_corpus_candidates([units['image'],units['image']],10))==2
    assert len(image_content({'path':'/a.png'}))==1



def test_empty_evidence_does_not_claim_router_chose_no():
    m=method();m.generate_answer({'prompt':[]},[])
    text=m.answer_model.complete.call_args.args[0][0]['content'][0]['text']
    assert 'No memory evidence' in text
    assert 'retrieval is unnecessary' not in text



def test_m3_converter_preserves_individual_image_names(tmp_path, monkeypatch):
    from mm_memory_bench.benchmarks.converters.m3exam import convert
    from mm_memory_bench.benchmarks.reader import BundleReader
    persona=tmp_path/'raw/example_set/person'
    (persona/'images').mkdir(parents=True)
    for name in ['img_51.jpg','img_52.jpg']:
        (persona/'images'/name).write_bytes(b'image fixture')
    (persona/'sessions.json').write_text(json.dumps([{'session_id':'D1','date':'2020-01-01',
        'dialogues':[{'round':'D1:1','user':'Two images','img_file':['img_51.jpg','img_52.jpg']}]}]))
    (persona/'question.json').write_text(json.dumps([{'question':'Find image', 'answer':['img_52.jpg'],
                                                    'type':'fm','supporting_facts':'D1:1'}]))
    out=tmp_path/'bundle';convert(tmp_path/'raw',out)
    with BundleReader(out) as reader:
        batch=next(reader.iter_context_batches())
        parts=reader.resolve_content(batch.memories[0]['content'])
        assert [p['source_id'] for p in parts if p['type']=='image']==['img_51.jpg','img_52.jpg']
        # Exercise the actual runner boundary, not just BundleReader resolution.
        from mm_memory_bench.runner.benchmark import run_context
        monkeypatch.setattr('mm_memory_bench.methods.media.data_url', lambda path:'data:image/jpeg;base64,AA==')
        ur = method()
        received = []
        class Probe:
            def begin_context(self, context): pass
            def ingest(self, memory):
                received.append(memory)
                for key, values in ur._memory_units(memory).items():
                    ur.units[key].extend(values)
            def answer(self, question):
                assert 'answer' not in question and 'evidence' not in question
                return ur.generate_answer(question, ur.units['image'])
            def end_context(self): pass
        for part in batch.memories[0]['content']:
            part['annotations'] = {'private_answer': 'DO_NOT_EXPOSE'}
        run_context(Probe(), reader, batch)
        assert len(ur.units['image']) == 2
        assert [u['media_source_id'] for u in ur.units['image']] == ['img_51.jpg','img_52.jpg']
        assert 'DO_NOT_EXPOSE' not in json.dumps(received)
        content = ur.answer_model.complete.call_args.args[0][0]['content']
        labels = [p['text'] for p in content if p.get('type')=='text' and p.get('text','').startswith('Image source_id:')]
        assert labels == ['Image source_id: img_51.jpg','Image source_id: img_52.jpg']
        for i, part in enumerate(content):
            if part['type']=='image_url':
                assert content[i-1]['text'].startswith('Image source_id: img_')



@pytest.mark.parametrize('metadata', [
    {'method_error':'TimeoutError: timeout'},
    {'status':'error','error':'context too large'},
    {'error_type':'invalid_router_output','error':'bad route'},
])
def test_method_failure_overrides_residual_correct_script_answer(tmp_path,metadata):
    q=choice();(tmp_path/'questions.jsonl').write_text(json.dumps(q)+'\n')
    prediction={'question_id':q['question_id'],'prediction':'0: Corsica','metadata':metadata}
    assert method_failure(prediction)
    pred=tmp_path/'pred.jsonl';pred.write_text(json.dumps(prediction)+'\n')
    out=tmp_path/'score.jsonl'
    result=score_predictions(tmp_path,pred,out,benchmark='smmbench')
    row=json.loads(out.read_text())
    assert row['status']=='method_error' and row['metrics']['choice_accuracy']==0
    assert row['details']['error']
    assert result['total']['count']==1



def test_dedicated_judge_does_not_reuse_false_success_for_method_error(tmp_path):
    q={'question_id':'m3exam:q','context_id':'c','prompt':[{'type':'text','text':'Q'}],
       'task':{'subcategory':'ii','response_type':'text'},'answer':{'text':'A'}}
    (tmp_path/'manifest.json').write_text(json.dumps({'benchmark':'M3Exam'}))
    (tmp_path/'questions.jsonl').write_text(json.dumps(q)+'\n')
    pred=tmp_path/'pred.jsonl';pred.write_text(json.dumps({'question_id':q['question_id'],'prediction':'A','metadata':{'method_error':'timeout'}})+'\n')
    backend=SimpleNamespace(model='fake',judge=Mock(side_effect=AssertionError('must not call judge')))
    out=tmp_path/'scores.jsonl'
    for _ in range(2):
        summary=judge_predictions(backend,tmp_path,pred,out,scoring_protocol='m3exam')
        row=json.loads(out.read_text())
        assert row['status']=='method_error' and row['score']==0
        assert summary['method_failures']==1 and summary['failed_judgments']==0
        assert summary['valid_judgments']==0
        # Simulate a cached false success produced by the old runner using the
        # same input hashes; failure detection must override that cached row.
        out.write_text(json.dumps({**row, 'status':'ok', 'score':1.0})+'\n')
    backend.judge.assert_not_called()



def test_dispatch_counts_method_error_and_keeps_denominator(tmp_path):
    from mm_memory_bench.evaluation.dispatcher import score_benchmark
    q=choice()
    (tmp_path/'manifest.json').write_text(json.dumps({'benchmark':'SMMBench'}))
    (tmp_path/'questions.jsonl').write_text(json.dumps(q)+'\n')
    pred=tmp_path/'pred.jsonl'
    pred.write_text(json.dumps({'question_id':q['question_id'], 'prediction':'0: Corsica',
                               'metadata':{'method_error':'timeout'}})+'\n')
    result=score_benchmark(tmp_path,pred,tmp_path/'judgments.jsonl')
    assert result['routes']['script']['method_failures']==1
    assert result['routes']['script']['summary']['total']['metrics']['choice_accuracy']=={'count':1,'mean':0}



def test_dedicated_summaries_separate_failures_without_dropping_zero():
    from mm_memory_bench.evaluation.native import mobilemem_omni_judge as omni, personamem_v2_judge as persona
    rows=[{'status':'ok','score':1.,'label':'CORRECT','native_category':'Single-hop'},
          {'status':'method_error','score':0.,'label':'WRONG','native_category':'Single-hop'},
          {'status':'error','score':None,'label':None,'native_category':'Single-hop'}]
    summary=omni.summarize(rows)
    assert summary['overall']['LLM_JUDGE']==.5
    assert summary['method_failures']==1 and summary['failed_judgments']==1
    assert summary['valid_judgments']==1
    rows=[dict(r,subset='test',preference_kind='normal') for r in rows]
    summary=persona.summarize(rows)
    assert summary['mean_score_conservative']==1/3
    assert summary['by_subset']['test']['method_failures']==1
    assert summary['by_subset']['test']['failed_judgments']==1
