import pytest

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
