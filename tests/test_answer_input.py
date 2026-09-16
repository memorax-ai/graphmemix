import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from mm_memory_bench.evaluation.oracle import OracleEvidenceGenerator
from mm_memory_bench.methods.answer_input import build_answer_task
from mm_memory_bench.methods.base import GenerationConfig
from mm_memory_bench.methods.concrete_amem import ConcreteAMemMethod
from mm_memory_bench.methods.concrete_lightmem import ConcreteLightMemMethod
from mm_memory_bench.methods.concrete_memguide import ConcreteMemGuideMethod
from mm_memory_bench.methods.concrete_memix import ConcreteMemixMethod
from mm_memory_bench.methods.concrete_universalrag import ConcreteUniversalRAGMethod
from mm_memory_bench.methods.concrete_vimrag import ConcreteVimRAGMethod
from mm_memory_bench.methods.media import question_text


def question(plan=True):
    q = {
        'prompt': [{'type': 'text', 'text': '如何安排出行？'}],
        'choices': [{'choice_id': 'A', 'text': '火车'}],
        'instruction': 'Return a JSON plan; do not execute tools.',
        'answer': {'text': 'GOLD_SECRET'},
        'evidence': ['EVIDENCE_SECRET'],
        'metadata': {'private': 'METADATA_SECRET'},
    }
    if plan:
        q.update(tool_mode='plan', tools=[{
            'type': 'function', 'function': {
                'name': 'book_train', 'description': '预订火车',
                'parameters': {'type': 'object', 'properties': {
                    'destination': {'type': 'string'}}, 'required': ['destination']},
            },
        }])
    return q


def request_text(model):
    parts = []
    for message in model.complete.call_args.args[0]:
        content = message['content']
        parts.append(content if isinstance(content, str) else '\n'.join(
            p['text'] for p in content if p['type'] == 'text'))
    return '\n'.join(parts)


def make_method(cls):
    method = cls.__new__(cls)
    method.generation = GenerationConfig()
    method.answer_model = SimpleNamespace(complete=Mock(return_value='[]'))
    method._flush_pending = Mock()
    return method


def answer_with_evidence(cls, q):
    """Exercise each reader's real message assembly with a captured model call."""
    memory = {'memory_id': 'm1', 'source_id': 's1',
              'content': [{'type': 'text', 'text': 'retrieved note'}]}
    if cls is ConcreteMemixMethod:
        method = make_method(cls)
        method._record_by_id = {'m1': memory}
        method._reader_text = Mock(return_value='retrieved note')
        method._generate_answer(q, ['m1'])
        method._reader_text.assert_called_once_with(memory, query=question_text(q))
    else:
        method = cls(answer_model=SimpleNamespace(complete=Mock(return_value='[]')))
        method.answer(q, [memory])
    return method.answer_model


class AnswerInputTest(unittest.TestCase):
    def test_memix_oracle_keep_system_plan_candidates_once(self):
        from mm_memory_bench.benchmarks.converters.smmbench import function_plan_instruction

        q = question()
        q['tools'] = [{'function_name': 'book_train', 'function_comment': 'Book a train.'}]
        q.update(instruction=function_plan_instruction(q['tools']),
                 instruction_role='system', instruction_includes_tools=True)
        original = copy.deepcopy(q)
        for cls in [ConcreteMemixMethod, OracleEvidenceGenerator]:
            with self.subTest(reader=cls.__name__):
                model = answer_with_evidence(cls, q)
                messages = model.complete.call_args.args[0]
                self.assertEqual([message['role'] for message in messages], ['system', 'user'])
                self.assertEqual(messages[0]['content'], q['instruction'])
                text = request_text(model)
                self.assertEqual(text.count('Book a train.'), 1)
                self.assertIn('retrieved note', text)
                self.assertIn('A: 火车', text)
                for secret in ['GOLD_SECRET', 'EVIDENCE_SECRET', 'METADATA_SECRET']:
                    self.assertNotIn(secret, text)
                self.assertIsNone(model.complete.call_args.kwargs['tools'])
                self.assertEqual(q, original)

    def test_memix_oracle_preserve_ordinary_and_legacy_plan_messages(self):
        for cls in [ConcreteMemixMethod, OracleEvidenceGenerator]:
            for plan in [False, True]:
                with self.subTest(reader=cls.__name__, plan=plan):
                    q = question(plan)
                    model = answer_with_evidence(cls, q)
                    prefix = q['instruction'] + '\nQuestion: ' + question_text(q)
                    if plan:
                        prefix += '\nCandidate tools:\n' + json.dumps(q['tools'], ensure_ascii=False)
                    if cls is ConcreteMemixMethod:
                        expected = [
                            {'type': 'text', 'text': prefix + '\n\nUse only the following Memix evidence packet.'},
                            {'type': 'text', 'text': 'Evidence 1; memory_id=m1; source_id=s1; modality=unknown:\nretrieved note'},
                        ]
                    else:
                        expected = [
                            {'type': 'text', 'text': prefix + '\nThe following memories were selected by an oracle. Answer using only these memories. Do not assume that every memory is independently sufficient.'},
                            {'type': 'text', 'text': 'Oracle memory 1: {"memory_id": "m1", "source_id": "s1"}\nPublic annotations: {}'},
                            {'type': 'text', 'text': 'retrieved note'},
                        ]
                    self.assertEqual(model.complete.call_args.args[0], [{'role': 'user', 'content': expected}])
                    self.assertIsNone(model.complete.call_args.kwargs['tools'])

    def test_memix_oracle_forward_executable_tools_separately(self):
        q = question()
        q.pop('tool_mode')
        for cls in [ConcreteMemixMethod, OracleEvidenceGenerator]:
            with self.subTest(reader=cls.__name__):
                model = answer_with_evidence(cls, q)
                self.assertEqual(model.complete.call_args.kwargs['tools'], q['tools'])
                self.assertNotIn('book_train', request_text(model))

    def test_plan_is_data_not_api_tools_and_excludes_private_fields(self):
        q = question()
        original = copy.deepcopy(q)
        task = build_answer_task(q)
        self.assertIn('A: 火车', task.text)
        self.assertIn(q['instruction'], task.text)
        self.assertEqual(json.loads(task.text.split('Candidate tools:\n')[1]), q['tools'])
        self.assertIsNone(task.api_tools)
        self.assertTrue(task.is_tool_plan)
        for secret in ['GOLD_SECRET', 'EVIDENCE_SECRET', 'METADATA_SECRET']:
            self.assertNotIn(secret, task.text)
        self.assertEqual(q, original)
        self.assertEqual(question_text(q), '如何安排出行？\nChoices:\nA: 火车')

    def test_plain_task_preserves_existing_reader_prefix(self):
        q = question(False)
        task = build_answer_task(q)
        self.assertEqual(task.text, q['instruction'] + '\nQuestion: ' + question_text(q))
        self.assertIsNone(task.api_tools)

    def test_agent_layout_preserves_plain_question_bytes(self):
        for instruction in [None, "", "  Return JSON.  "]:
            q = question(False)
            if instruction is None:
                q.pop('instruction')
            else:
                q['instruction'] = instruction
            expected = question_text(q)
            if instruction and instruction.strip():
                expected += "\n\nAnswer requirements: " + instruction.strip()
            self.assertEqual(build_answer_task(q, question_first=True).text, expected)

    def test_api_tools_remain_separate_and_unsupported_consumers_fail(self):
        q = question()
        q.pop('tool_mode')
        task = build_answer_task(q)
        self.assertEqual(task.api_tools, q['tools'])
        self.assertNotIn('book_train', task.text)
        with self.assertRaises(NotImplementedError):
            build_answer_task(q, supports_api_tools=False)

    def test_four_reader_adapters_forward_the_same_public_task(self):
        for cls in [ConcreteAMemMethod, ConcreteMemGuideMethod,
                    ConcreteUniversalRAGMethod, ConcreteLightMemMethod]:
            for plan in [False, True]:
                with self.subTest(method=cls.__name__, plan=plan):
                    q = question(plan)
                    method = make_method(cls)
                    if cls is ConcreteAMemMethod:
                        method.generate_answer(q, [{'text': 'retrieved note'}])
                    elif cls is ConcreteMemGuideMethod:
                        method._generate_answer(q, [{'memory_id': 'm1', 'text': 'retrieved note'}])
                    elif cls is ConcreteUniversalRAGMethod:
                        method.generate_answer(q, [{'memory_id': 'm1', 'text': 'retrieved note'}])
                    else:
                        method.backend = SimpleNamespace(retrieve=Mock(return_value=['retrieved note']))
                        method._answer(q)
                        method.backend.retrieve.assert_called_once_with(question_text(q), limit=10)
                    text = request_text(method.answer_model)
                    self.assertIn(build_answer_task(q).text, text)
                    self.assertIn('retrieved note', text)
                    self.assertEqual(text.count('Candidate tools:'), int(plan))
                    self.assertIsNone(method.answer_model.complete.call_args.kwargs['tools'])
                    for secret in ['GOLD_SECRET', 'EVIDENCE_SECRET', 'METADATA_SECRET']:
                        self.assertNotIn(secret, text)

    def test_smmbench_requirements_reach_readers_without_changing_query(self):
        from mm_memory_bench.benchmarks.converters.smmbench import function_plan_instruction
        q = question()
        retrieval_query = question_text(q)
        q['tools'] = [{'function_name': 'book_train', 'function_comment': 'Book a train.'}]
        q['instruction'] = function_plan_instruction(q['tools'])
        q['instruction_role'] = 'system'
        q['instruction_includes_tools'] = True
        self.assertEqual(question_text(q), retrieval_query)
        self.assertIn('only one step', q['instruction'].lower())
        for cls in [ConcreteAMemMethod, ConcreteMemGuideMethod,
                    ConcreteUniversalRAGMethod, ConcreteLightMemMethod]:
            with self.subTest(method=cls.__name__):
                method = make_method(cls)
                if cls is ConcreteAMemMethod:
                    method.generate_answer(q, [])
                elif cls is ConcreteMemGuideMethod:
                    method._generate_answer(q, [])
                elif cls is ConcreteUniversalRAGMethod:
                    method.generate_answer(q, [])
                else:
                    method.backend = SimpleNamespace(retrieve=Mock(return_value=[]))
                    method._answer(q)
                    method.backend.retrieve.assert_called_once_with(retrieval_query, limit=10)
                expected_system = q['instruction']
                if cls is ConcreteAMemMethod:
                    expected_system = 'Answer only from retrieved memory evidence.\n\n' + expected_system
                self.assertEqual(method.answer_model.complete.call_args.args[0][0],
                                 {'role': 'system', 'content': expected_system})
                self.assertEqual(request_text(method.answer_model).count('Book a train.'), 1)
                for secret in ['GOLD_SECRET', 'EVIDENCE_SECRET', 'METADATA_SECRET']:
                    self.assertNotIn(secret, request_text(method.answer_model))
                self.assertIsNone(method.answer_model.complete.call_args.kwargs['tools'])

    def test_method_system_rules_are_retained_with_benchmark_requirements(self):
        from mm_memory_bench.methods.answer_input import AnswerTask
        task = AnswerTask(text='Question', system_text='Return a plan.')
        self.assertEqual(task.messages('evidence', default_system='Use memory only.'), [
            {'role': 'system', 'content': 'Use memory only.\n\nReturn a plan.'},
            {'role': 'user', 'content': 'evidence'},
        ])
        self.assertEqual(task.messages('evidence', default_system='Return a plan.')[0]['content'],
                         'Return a plan.')
        self.assertEqual(AnswerTask(text='Question').messages('evidence', default_system='Original'), [
            {'role': 'system', 'content': 'Original'}, {'role': 'user', 'content': 'evidence'},
        ])

    def test_system_role_without_embedded_tools_keeps_candidates(self):
        for flag in [None, False, 'true']:
            with self.subTest(flag=flag):
                q = question()
                q['instruction_role'] = 'system'
                if flag is not None:
                    q['instruction_includes_tools'] = flag
                task = build_answer_task(q)
                self.assertEqual(json.loads(task.text.split('Candidate tools:\n')[1]), q['tools'])
                self.assertEqual(task.system_text, q['instruction'])
                self.assertIsNone(task.api_tools)
                self.assertTrue(task.is_tool_plan)
                self.assertIn('book_train', task.agent_text)
        q['instruction_includes_tools'] = True
        q['instruction'] = ''
        self.assertIn('book_train', build_answer_task(q).text)

    def test_api_tools_forwarding_for_reader_adapters(self):
        q = question()
        q.pop('tool_mode')
        for cls in [ConcreteAMemMethod, ConcreteMemGuideMethod,
                    ConcreteUniversalRAGMethod, ConcreteLightMemMethod]:
            with self.subTest(method=cls.__name__):
                method = make_method(cls)
                if cls is ConcreteAMemMethod:
                    method.generate_answer(q, [])
                elif cls is ConcreteMemGuideMethod:
                    method._generate_answer(q, [])
                elif cls is ConcreteUniversalRAGMethod:
                    method.generate_answer(q, [])
                else:
                    method.backend = SimpleNamespace(retrieve=Mock(return_value=[]))
                    method._answer(q)
                self.assertEqual(method.answer_model.complete.call_args.kwargs['tools'], q['tools'])

    def test_memguide_single_and_batch_paths_use_public_input(self):
        method = make_method(ConcreteMemGuideMethod)
        method._active = True
        method.units = []
        method._missing_information_filter = Mock(return_value=[])
        q = question()
        method._answer(q)
        self.assertIn('Candidate tools:', request_text(method.answer_model))
        method.answer_many([q], concurrency=1)
        method._missing_information_filter.assert_called_with(question_text(q), [])
        self.assertIn('Candidate tools:', request_text(method.answer_model))

    def test_vimrag_gets_planning_data_without_changing_internal_tools(self):
        method = make_method(ConcreteVimRAGMethod)
        method.official_repo = Path('.')
        method.max_steps = 20
        method.video_frames = 8
        method._query_media = Mock(return_value=[])
        method._search = Mock()
        plan_answer = {'calls': [{'name': 'book_train', 'arguments': {'destination': '北京'}}]}
        agent = SimpleNamespace(run=Mock(side_effect=lambda sample: iter([
            {'event': 'answer', 'content': plan_answer if ('Candidate tools:' in sample['query'] or '## Candidate Tools' in sample['query']) else '[]', 'sample': sample}])))
        method.agent_factory = Mock(return_value=agent)
        for plan in [False, True]:
            q = question(plan)
            if plan:
                from mm_memory_bench.benchmarks.converters.smmbench import function_plan_instruction
                q['tools'] = [{'function_name': 'book_train', 'function_comment': 'Book a train.'}]
                q['instruction'] = function_plan_instruction(q['tools'])
                q['instruction_role'] = 'system'
                q['instruction_includes_tools'] = True
            result = method._answer(q)
            self.assertEqual(json.loads(result.prediction), plan_answer if plan else [])
            task_text = agent.run.call_args.args[0]['query']
            if plan:
                self.assertTrue(task_text.startswith(build_answer_task(q, question_first=True).agent_text))
                self.assertIn('not executable VimRAG actions', task_text)
                self.assertIn('add_answer_node', task_text)
                self.assertIn('JSON-encoded string', task_text)
            else:
                self.assertEqual(task_text, build_answer_task(q, question_first=True).text)
            self.assertIs(method.agent_factory.call_args.kwargs['search'], method._search)
        q = question()
        q.pop('tool_mode')
        method.agent_factory.reset_mock()
        with self.assertRaises(NotImplementedError):
            method._answer(q)
        method.agent_factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
