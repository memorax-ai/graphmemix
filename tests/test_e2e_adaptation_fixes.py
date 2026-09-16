import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mm_memory_bench.methods.concrete_universalrag import ConcreteUniversalRAGMethod as UR
from mm_memory_bench.methods.media import image_content, table_text
from mm_memory_bench.benchmarks.converters.smmbench import _parts
from mm_memory_bench.evaluation.native.script_judge import score_question


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
