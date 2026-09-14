from types import SimpleNamespace
from mm_memory_bench.methods.concrete_lightmem import _OfficialLightMemBackend


def test_official_backend_preserves_provenance_after_source_resolution():
    inserted = []
    retriever = SimpleNamespace(insert=lambda **kwargs: inserted.append(kwargs))
    seen = []
    def add(messages, **kwargs):
        seen.extend(messages)
        # Upstream resolves source_id to speaker metadata before inserting facts.
        # Reverse source order to ensure this is not positional attribution.
        for message in reversed(messages):
            retriever.insert(vectors=[[0.25]], ids=[message['speaker_id']], payloads=[{
                'speaker_id': message['speaker_id'], 'speaker_name': message['speaker_name'],
                'memory': 'extracted fact',
            }])
        return {'api_call_nums': 1}
    backend = _OfficialLightMemBackend(SimpleNamespace(embedding_retriever=retriever, add_memory=add))
    messages = [dict(role='user', content='source text', speaker_id='same-speaker',
                     speaker_name='Alice', canonical_memory_id=mid, ingest_batch_id='batch')
                for mid in ['m1', 'm2']]
    backend.add_memory(messages)
    assert messages[0]['speaker_id'] == 'same-speaker'
    assert seen[0]['content'] == 'source text' and seen[0]['speaker_name'] == 'Alice'
    for call, mid in zip(inserted, ['m2', 'm1']):
        payload = call['payloads'][0]
        assert call['vectors'] == [[0.25]]
        assert payload['source_memory_id'] == mid
        assert payload['ingest_batch_id'] == 'batch'
        assert payload['speaker_id'] == 'same-speaker'
        assert payload['memory'] == f'[memory_id={mid}]\nextracted fact'
