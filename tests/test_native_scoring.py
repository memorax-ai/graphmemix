import json
from pathlib import Path
import pytest
from mm_memory_bench.evaluation.native import score_question, score_predictions
from mm_memory_bench.evaluation.native import script_judge
from mm_memory_bench.evaluation.native.script_judge import plan_score, text_em, image_em


def q(benchmark='smmbench'):
    return dict(question_id=benchmark+':q', subset='mcq', task={'response_type':'choice'},
                answer={'choice_id':'0','text':'yes'}, choices=[{'choice_id':'0','text':'yes'},{'choice_id':'1','text':'no'}])


@pytest.mark.parametrize('response,expected', [('A',1),('(A)',1),('Answer: (A)',1),('B',0),('yes',0),('Answer: (A) or (B)',1),('0',1),('',0)])
def test_choice(response, expected):
    assert score_question('smmbench', q(), response)[0]['choice_accuracy'] == expected


def test_plan_protocol():
    call = {'name':'LOOKUP', 'arguments':{'ID':'Abc'}}
    expected = [{'step':1,'calls':[call]}]
    tools = [{'function_name':'lookup','default_arguments':{'limit':10}}]
    predicted = [{'step':99,'calls':[{'name':'lookup','arguments':{'id':'abc','limit':10}}, {'name':'extra'}]}]
    assert plan_score(json.dumps(predicted), expected, tools) == (1.,'ok')
    assert plan_score('not json', expected, tools)[1] == 'invalid_prediction'
    assert plan_score(predicted*2, expected, tools)[0] == 0
    assert plan_score(predicted, [{'calls':[call, call]}], tools)[0] == 0
    assert plan_score('```json\n'+json.dumps(predicted)+'\n```', expected, tools)[0] == 1
    with pytest.raises(ValueError):
        plan_score(predicted, 'invalid reference', tools)


def test_em_boundaries():
    assert text_em('A, B!', ['a b']) == 1
    assert image_em('img_02 and img_99', ['img2.png','img_3']) == 1
    assert image_em('img_9', ['img_2']) == 0


def test_m3_applicability():
    item={'answer':{'text':'a','accepted_answers':['a','b']},'task':{'subcategory':'fj'}}
    assert score_question('m3exam', item, 'b')[0] == {'em':1}
    item['task']['subcategory']='qa'
    item['metadata']={'native_label':'a b'}
    result=score_question('m3exam',item,'a')[0]
    assert result == {'em': 1}


def test_omni_has_no_binary_script_metric(tmp_path):
    item={'answer':{'text':'a'},'task':{}}
    with pytest.raises(ValueError, match='unsupported benchmark'):
        score_question('mobilemem_omni', item, 'a')
    with pytest.raises(ValueError, match='unsupported benchmark'):
        score_predictions(tmp_path, tmp_path/'pred.jsonl', tmp_path/'scores.jsonl',
                          benchmark='mobilemem_omni')


def test_runner_integrity(tmp_path):
    bundle=tmp_path/'bundle';bundle.mkdir()
    item=q()
    (bundle/'questions.jsonl').write_text(json.dumps(item)+'\n')
    pred=tmp_path/'pred.jsonl';pred.write_text(json.dumps({'question_id':item['question_id'],'prediction':'(A)'})+'\n')
    output=tmp_path/'native.jsonl'
    result=score_predictions(bundle,pred,output,benchmark='smmbench')
    assert result['total']['metrics']['choice_accuracy']=={'count':1,'mean':1}
    assert output.with_suffix('.summary.json').exists()
    with pytest.raises(ValueError):score_predictions(bundle,pred,pred,benchmark='smmbench')
    pred.write_text('')
    with pytest.raises(ValueError,match='lack predictions'):score_predictions(bundle,pred,output,benchmark='smmbench')
    pred.write_text(json.dumps({'question_id':'unknown','prediction':'(A)'})+'\n')
    with pytest.raises(ValueError,match='unknown'):score_predictions(bundle,pred,output,benchmark='smmbench')


def test_empty_prediction_not_dropped(tmp_path):
    (tmp_path/'questions.jsonl').write_text(json.dumps(q())+'\n')
    pred=tmp_path/'pred.jsonl';pred.write_text(json.dumps({'question_id':'smmbench:q','prediction':''})+'\n')
    result=score_predictions(tmp_path,pred,tmp_path/'score.jsonl',benchmark='smmbench')
    assert result['total']['statuses']=={'empty_prediction':1}
    assert result['total']['metrics']['choice_accuracy']['mean']==0


@pytest.mark.parametrize('collision', ['summary_input', 'summary_questions', 'output_alias', 'manifest', 'table'])
def test_output_collisions_leave_inputs_and_outputs_unchanged(tmp_path, collision):
    bundle = tmp_path/'bundle'
    bundle.mkdir()
    questions = bundle/'questions.jsonl'
    questions.write_text(json.dumps(q())+'\n')
    pred = tmp_path/('scores.summary.json' if collision == 'summary_input' else 'pred.jsonl')
    pred.write_text(json.dumps({'question_id':'smmbench:q','prediction':'(A)'})+'\n')
    output = tmp_path/'scores.jsonl'
    summary = output.with_suffix('.summary.json')
    protected = [questions, pred]
    if collision == 'summary_questions':
        summary.symlink_to(questions)
    elif collision == 'output_alias':
        output.write_text('previous scores\n')
        summary.symlink_to(output)
        protected.append(output)
    elif collision in {'manifest', 'table'}:
        manifest = bundle/'manifest.json'
        table = bundle/'custom-memories.jsonl'
        manifest.write_text(json.dumps({'tables': {'memories': table.name}}))
        table.write_text('previous memories\n')
        summary.symlink_to(manifest if collision == 'manifest' else table)
        protected.extend([manifest, table])
    before = {path: path.read_bytes() for path in protected}
    with pytest.raises(ValueError, match='collide|overwrite'):
        score_predictions(bundle, pred, output, benchmark='smmbench')
    assert {path: path.read_bytes() for path in protected} == before
    if collision != 'output_alias':
        assert not output.exists()

@pytest.mark.parametrize('response,expected', [
    ('(d): Fallen leaves', 1), ('(d)', 1), ('D', 1),
    ('(a): Berry bushes', 0), ('', 0), (')', 0)])
def test_persona_mme_native_parser(response, expected):
    item=q('persona_mme')
    item['choices']=[{'choice_id':'(a)','text':'Berry bushes'}, {'choice_id':'(d)','text':'Fallen leaves'}]
    item['answer']={'choice_id':'(d)','native_label':'(d)','text':'Fallen leaves'}
    assert score_question('persona_mme',item,response)[0]['choice_accuracy']==expected


@pytest.mark.parametrize('response,expected', [
    ('A',1), ('(A)',0), ('A.',1), ('The answer is A',1),
    ('Final answer: A',1), ('Answer: A. Final answer: B',0),
    ('Answer: A or B',1), ('',0)])
def test_personamem_v2_scoring_adapter(response,expected):
    item=q('personamem_v2')
    item['choices']=[{'choice_id':'A','text':'yes'},{'choice_id':'B','text':'no'}]
    item['answer']={'choice_id':'A','text':'yes'}
    assert score_question('personamem_v2',item,response)[0]['choice_accuracy']==expected


def test_official_parsers_still_reject_bundle_only_labels():
    assert not script_judge.check_answer_match_multiple_choice('(A)', '0')
    assert script_judge._personamem_v2_extract_final_answer('A') == ''


@pytest.mark.parametrize('gold', range(4))
def test_smm_labels_score_by_id_not_choice_order(gold):
    item = q()
    item['choices'] = [{'choice_id': str(i), 'text': f'option {i}'} for i in (2, 0, 3, 1)]
    item['answer'] = {'choice_id': str(gold), 'native_label': str(gold), 'text': f'option {gold}'}
    for predicted in range(4):
        response = f' \n{predicted}\t'
        assert script_judge.adapt_choice_prediction('smmbench', item['choices'], response) == f'({chr(65 + predicted)})'
        assert script_judge.choice_score(item, response) == (float(predicted == gold), 'ok')


@pytest.mark.parametrize('gold', ['A', 'B'])
def test_personamem_label_adaptation_is_independent_of_gold(gold):
    item = q('personamem_v2')
    item['choices'] = [{'choice_id': 'B', 'text': 'second'}, {'choice_id': 'A', 'text': 'first'}]
    item['answer'] = {'choice_id': gold, 'text': 'first' if gold == 'A' else 'second'}
    for predicted in ('A', 'B'):
        response = f' {predicted.lower()}\n'
        assert script_judge.adapt_choice_prediction('personamem_v2', item['choices'], response) == f'Final Answer: {predicted}'
        assert score_question('personamem_v2', item, response)[0] == {'choice_accuracy': float(predicted == gold)}


@pytest.mark.parametrize('benchmark,labels,responses', [
    ('smmbench', ['0', '1'], ['2', '4', '-1', '00', '0 or 1', 'Answer is 0', '0.', '(A)', 'Answer: (A) or (B)', '']),
    ('smmbench', ['(A)', '(B)'], ['0', '(A)']),
    ('personamem_v2', ['A', 'B'], ['C', '0', '(A)', 'A or B', 'A.', 'Final Answer: A', 'Answer: A or B', 'ａ', '']),
    ('persona_mme', ['A', '0'], ['A', '0', ' (a) ']),
    ('m3exam', ['A', '0'], ['A', '0']),
])
def test_adapter_preserves_nonlabels_and_other_benchmarks(benchmark, labels, responses):
    choices = [{'choice_id': label, 'text': label} for label in labels]
    for response in responses:
        assert script_judge.adapt_choice_prediction(benchmark, choices, response) == response


@pytest.mark.parametrize('benchmark,label', [('smmbench', '0'), ('personamem_v2', 'A')])
def test_adapted_scoring_preserves_raw_input_files(tmp_path, benchmark, label):
    item = q(benchmark)
    if benchmark == 'personamem_v2':
        item['choices'] = [{'choice_id': 'A', 'text': 'yes'}, {'choice_id': 'B', 'text': 'no'}]
        item['answer'] = {'choice_id': 'A', 'text': 'yes'}
    questions = tmp_path/'questions.jsonl'
    questions.write_text(json.dumps(item)+'\n')
    predictions = tmp_path/'predictions.jsonl'
    predictions.write_text(json.dumps({'question_id': item['question_id'], 'prediction': f' {label}\n'})+'\n')
    before = {path: path.read_bytes() for path in (questions, predictions)}
    output = tmp_path/'scores.jsonl'
    summary = score_predictions(tmp_path, predictions, output, benchmark=benchmark)
    assert summary['total']['metrics']['choice_accuracy']['mean'] == 1
    assert summary['protocol'] == 'mmmb-native-scripts-2.2'
    assert json.loads(output.read_text())['protocol'] == summary['protocol']
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize('metadata', [{'status':'error'}, {'error_type':'TimeoutError'}])
def test_failed_prediction_not_scored_from_residual_text(tmp_path,metadata):
    (tmp_path/'questions.jsonl').write_text(json.dumps(q())+'\n')
    pred=tmp_path/'pred.jsonl'
    pred.write_text(json.dumps({'question_id':'smmbench:q','prediction':'(A)','metadata':metadata})+'\n')
    result=score_predictions(tmp_path,pred,tmp_path/'score.jsonl',benchmark='smmbench')
    assert result['total']['statuses']=={'method_error':1}
    assert result['total']['metrics']['choice_accuracy']['mean']==0


@pytest.mark.parametrize('benchmark,metric', [('m3exam','em')])
def test_native_empty_reference_boundary(tmp_path,benchmark,metric):
    row={'question_id':benchmark+':q','task':{'subcategory':'fj'},'answer':{'text':''}}
    (tmp_path/'questions.jsonl').write_text(json.dumps(row)+'\n')
    pred=tmp_path/'pred.jsonl';pred.write_text(json.dumps({'question_id':row['question_id'],'prediction':''})+'\n')
    result=score_predictions(tmp_path,pred,tmp_path/'score.jsonl',benchmark=benchmark)
    assert result['total']['metrics'][metric]['mean']==1


def test_mixed_m3_binary_metrics(tmp_path):
    rows=[];predictions=[]
    for kind in ['fj','fm','qa']:
        row={'question_id':'m3exam:'+kind,'task':{'subcategory':kind},'answer':{'text':'img_2'}}
        rows.append(json.dumps(row));predictions.append(json.dumps({'question_id':row['question_id'],'prediction':'img_2'}))
    (tmp_path/'questions.jsonl').write_text('\n'.join(rows))
    pred=tmp_path/'pred.jsonl';pred.write_text('\n'.join(predictions))
    result=score_predictions(tmp_path,pred,tmp_path/'scores.jsonl',benchmark='m3exam')
    assert result['total']['metrics']['em']['count']==3
    assert set(result['total']['metrics']) == {'em'}
    scored = [json.loads(line) for line in (tmp_path/'scores.jsonl').read_text().splitlines()]
    assert all(row['metrics'] == {'em': 1} for row in scored)
