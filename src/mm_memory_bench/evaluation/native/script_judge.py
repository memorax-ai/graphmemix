"""Binary native script scoring: official rules, dispatch, I/O and summaries.

Per-question metrics are 0 or 1; aggregate means may be fractional.
See docs/native_scoring.md for supported benchmarks and protocol boundaries.
"""
import hashlib
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path


def text_em(prediction, references):
    def norm(x):
        return ' '.join(str(x).lower().translate(str.maketrans('', '', string.punctuation)).split())
    return float(any(norm(prediction) == norm(r) for r in references))


def image_em(prediction, references):
    def ids(x):
        return {int(n) for n in re.findall(r'\bimg[_-]?(\d+)(?:\.[a-zA-Z0-9]+)?\b', x, re.I)}
    gold = set().union(*(ids(r) for r in references))
    return float(bool(ids(prediction) & gold)) if gold else text_em(prediction, references)


def payload(value):
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    candidates = [value.strip()]
    for block in value.split('```')[1::2]:
        block = block.strip()
        if '\n' in block and block.split('\n', 1)[0].strip().lower() == 'json':
            block = block.split('\n', 1)[1].strip()
        candidates.append(block)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    # Upstream searches arrays before objects, including fenced JSON / surrounding prose.
    for opening in ('[', '{'):
        for match in re.finditer(re.escape(opening), value):
            try:
                return json.JSONDecoder().raw_decode(value[match.start():])[0]
            except ValueError:
                continue
    return None


def normalize_plan(value, tools):
    if isinstance(value, dict):
        if isinstance(value.get('answer'), list):
            value = value['answer']
        elif isinstance(value.get('steps'), list):
            value = value['steps']
        elif isinstance(value.get('calls'), list):
            value = [value]
        else:
            return None
    if not isinstance(value, list):
        return None
    def lower(v):
        if isinstance(v, dict):
            return {str(k).lower(): lower(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]).lower())}
        if isinstance(v, list):
            return [lower(x) for x in v]
        return v.lower() if isinstance(v, str) else v
    result = []
    for step in value:
        if not isinstance(step, dict) or not isinstance(step.get('calls'), list):
            return None
        calls = []
        for call in step['calls']:
            if not isinstance(call, dict):
                return None
            name, args = call.get('name'), call.get('arguments', {})
            args = {} if args is None else args
            if not isinstance(name, str) or not name.strip() or not isinstance(args, dict):
                return None
            args = dict(args)
            for tool in tools:
                if isinstance(tool, dict) and str(tool.get('function_name', '')).strip().lower() == name.strip().lower():
                    defaults = tool.get('default_arguments', {})
                    if isinstance(defaults, dict):
                        for k, v in defaults.items():
                            if str(k).lower() not in {str(x).lower() for x in args}:
                                args[k] = v
                    break
            calls.append({'name': name.strip().lower(), 'arguments': lower(args)})
        result.append(calls)
    return result


def plan_score(prediction, reference, tools):
    gold = normalize_plan(reference, tools)
    if gold is None:
        raise ValueError('invalid native reference plan')
    pred = normalize_plan(payload(prediction), tools)
    if pred is None:
        return 0., 'invalid_prediction'
    if len(pred) != len(gold):
        return 0., 'ok'
    for expected, actual in zip(gold, pred):
        pool = list(actual)
        for call in expected:
            if call not in pool:
                return 0., 'ok'
            pool.remove(call)
    return 1., 'ok'


# Native MCQ parsers, preserving each upstream implementation's matching rules.
# Sources: SMMBench evaluation/utils.py; PersonaVLM eval.py;
# PersonaMem-v2 inference.py. See docs/native_scoring.md for snapshots.
def equal_answer_with_or_without_parentheses(answer: str, raw_response: str) -> bool:
    predicted_answer = raw_response.replace('(', '').replace(')', '').lower().strip()
    ground_truth = answer.replace('(', '').replace(')', '').lower().strip()
    return predicted_answer == ground_truth

def check_answer_match_multiple_choice(answer: str, raw_response: str) -> bool:
    try:
        if answer is None or raw_response is None:
            return False
        predicted_answer = raw_response.lower().strip()
        ground_truth = answer.lower().strip()
        if equal_answer_with_or_without_parentheses(ground_truth, predicted_answer):
            return True
        elif ground_truth in predicted_answer:
            return True
        elif '<|begin_of_box|>' in predicted_answer:
            predicted_answer = predicted_answer.replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '')
            if equal_answer_with_or_without_parentheses(ground_truth, predicted_answer):
                return True
            else:
                return False
        else:
            return False
    except Exception:
        return False

def _persona_mme_check_result(correct_answer, response):
    if not response:
        return False
    if correct_answer[1].lower() == response.split(')')[0].lower()[-1]:
        return True
    return False

def _personamem_v2_extract_final_answer(response: str) -> str:
    """Extract final answer letter from MCQ response."""
    if not response:
        return ''
    import re
    patterns = ['\\$\\\\boxed\\{([A-Z])\\}\\$', '\\\\boxed\\{([A-Z])\\}', 'Final Answer:\\s*([A-Z])', 'final answer:\\s*([A-Z])', 'Answer:\\s*([A-Z])', 'answer:\\s*([A-Z])', 'final answer is\\s*\\$?\\\\boxed\\{([A-Z])\\}\\$?', 'final answer is\\s*([A-Z])', 'the answer is\\s*\\$?\\\\boxed\\{([A-Z])\\}\\$?', 'the answer is\\s*([A-Z])', '\\b([A-Z])\\.\\s*$']
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return ''

def adapt_choice_prediction(benchmark, choices, prediction):
    """Translate complete public option labels without consulting reference answers."""
    label = prediction.strip()
    labels = {str(choice['choice_id']).upper() for choice in choices}
    if label.upper() not in labels:
        return prediction
    if benchmark == 'smmbench' and label in ('0', '1', '2', '3'):
        return f'({chr(65 + int(label))})'
    if benchmark == 'personamem_v2' and re.fullmatch('[A-Za-z]', label):
        return f'Final Answer: {label.upper()}'
    return prediction


def choice_score(question, prediction):
    """Adapt public option labels, then dispatch to the unchanged native parser."""
    valid = {str(c['choice_id']).upper(): c['text'] for c in question['choices']}
    gold = str(question['answer']['choice_id']).upper()
    if gold not in valid:
        raise ValueError('reference choice missing from choices')
    benchmark = question.get('_benchmark', 'smmbench')
    prediction = adapt_choice_prediction(benchmark, question['choices'], prediction)
    if benchmark == 'smmbench':
        native_gold = str(question['answer'].get('native_label', question['answer']['choice_id']))
        # Official evaluate_single_qa_answer maps the native zero-based answer
        # index to (A)..(D), matching the public-label adapter above.
        if native_gold in ('0', '1', '2', '3'):
            native_gold = ('(A)', '(B)', '(C)', '(D)')[int(native_gold)]
        return float(check_answer_match_multiple_choice(native_gold, prediction)), 'ok'
    if benchmark == 'persona_mme':
        native_gold = str(question['answer'].get('native_label', question['answer']['choice_id']))
        if len(native_gold) < 3 or not native_gold.startswith('('):
            raise ValueError('Persona-MME requires its native parenthesized answer label')
        # Upstream accesses the last character of the prefix before ')'.
        # Malformed responses that would raise are invalid, not scorer crashes.
        try:
            correct = _persona_mme_check_result(native_gold, prediction)
        except IndexError:
            return 0., 'invalid_prediction'
        return float(correct), 'ok'
    if benchmark == 'personamem_v2':
        choice = _personamem_v2_extract_final_answer(prediction)
        if not choice:
            return 0., 'invalid_prediction'
        return float(valid.get(choice, '') == valid[gold]), 'ok'
    raise ValueError(f'unsupported MCQ benchmark: {benchmark}')


BENCHMARKS = ('smmbench', 'persona_mme', 'personamem_v2', 'm3exam')
PROTOCOL = 'mmmb-native-scripts-2.1'


def score_question(benchmark, question, prediction):
    if benchmark not in BENCHMARKS:
        raise ValueError(f'unsupported benchmark: {benchmark}')
    task = question.get('task', {})
    answer = question['answer']
    details = {}
    if benchmark == 'smmbench' and question.get('tool_mode') == 'plan':
        tools = question.get('tools')
        if not isinstance(tools, list):
            raise ValueError('candidate tools list required for default argument normalization')
        score, status = plan_score(prediction, question['metadata']['native_answer'], tools)
        return {'plan_accuracy': score}, status, details
    if benchmark in ('smmbench', 'persona_mme', 'personamem_v2'):
        if task.get('response_type') != 'choice':
            raise ValueError('only MCQ track supported for this benchmark')
        score, status = choice_score(dict(question, _benchmark=benchmark), prediction)
        return {'choice_accuracy': score}, status, details
    references = answer.get('accepted_answers', [answer['text']])
    if not isinstance(references, list) or not references or not all(isinstance(r, str) for r in references):
        raise ValueError('nonempty reference string list required')
    if benchmark == 'm3exam':
        kind = task['subcategory']
        scores = {'em': image_em(prediction, references) if kind == 'fm' else text_em(prediction, references)}
        return scores, 'ok', details
    raise ValueError(f'unsupported benchmark: {benchmark}')


def _load(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = row['question_id']
        if qid in result:
            raise ValueError(f'duplicate question_id in {path}: {qid}')
        result[qid] = row
    return result


def score_predictions(bundle, predictions, output, *, benchmark, question_ids=None):
    if benchmark not in BENCHMARKS:
        raise ValueError(f'unsupported benchmark: {benchmark}')
    questions = _load(Path(bundle) / 'questions.jsonl')
    if any(not qid.startswith((benchmark + ':', benchmark + '_')) for qid in questions):
        raise ValueError('benchmark does not match question ID namespace')
    predictions_path = Path(predictions)
    predictions = _load(predictions_path)
    if set(predictions) - set(questions):
        raise ValueError('predictions contain unknown question IDs')
    selected = set(questions) if question_ids is None else set(question_ids)
    if not selected or selected - set(questions):
        raise ValueError('empty selection or unknown question IDs')
    missing = selected - set(predictions)
    if missing:
        raise ValueError(f'{len(missing)} selected questions lack predictions; use an explicit question ID file for a subset')
    rows = []
    for qid, q in questions.items():
        if qid not in selected:
            continue
        value = predictions[qid].get('prediction', '')
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        if not isinstance(value, str):
            raise ValueError(f'{qid}: prediction must be text or JSON object/list')
        scores, status, details = score_question(benchmark, q, value)
        metadata = predictions[qid].get('metadata')
        failed = isinstance(metadata, dict) and (
            metadata.get('status') == 'error' or bool(metadata.get('error_type')))
        if failed:
            scores = {key: 0. for key in scores}
            status = 'method_error'
        elif not value.strip():
            # Retain native metric semantics, including empty-reference boundaries.
            status = 'empty_prediction'
        rows.append(dict(question_id=qid, benchmark=benchmark, protocol=PROTOCOL,
                         backend='script', subset=q.get('subset', ''),
                         category=q.get('task', {}).get('category', ''),
                         subcategory=q.get('task', {}).get('subcategory', ''),
                         status=status, metrics=scores, details=details))
    def summarize(items):
        metrics = defaultdict(list)
        for row in items:
            for key, value in row['metrics'].items():
                metrics[key].append(value)
        return {'count': len(items), 'statuses': dict(Counter(r['status'] for r in items)),
                'metrics': {k: {'count': len(v), 'mean': round(sum(v)/len(v), 4) if benchmark == 'm3exam' else sum(v)/len(v)} for k, v in metrics.items()}}
    groups = {}
    for field in ('subset', 'category', 'subcategory'):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row[field]].append(row)
        groups[field] = {key: summarize(items) for key, items in grouped.items()}
    digest = hashlib.sha256(json.dumps({'questions': [questions[r['question_id']] for r in rows],
                                       'predictions': [predictions[r['question_id']] for r in rows]}, sort_keys=True).encode()).hexdigest()
    summary = dict(protocol=PROTOCOL, benchmark=benchmark, input_sha256=digest,
                   scope='all' if question_ids is None else 'explicit_subset',
                   total=summarize(rows), groups=groups)
    output = Path(output)
    summary_path = output.with_suffix('.summary.json')
    destinations = [output.resolve(), summary_path.resolve()]
    manifest_path = Path(bundle)/'manifest.json'
    protected = {predictions_path.resolve(), (Path(bundle)/'questions.jsonl').resolve(),
                 manifest_path.resolve()}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        protected.update((Path(bundle)/name).resolve()
                         for name in manifest.get('tables', {}).values())
    if len(set(destinations)) != len(destinations) or set(destinations) & protected:
        raise ValueError('output paths collide with each other or with inputs')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    return summary
