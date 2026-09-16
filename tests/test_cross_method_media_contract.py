import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mm_memory_bench.methods.concrete_amem import ConcreteAMemMethod
from mm_memory_bench.methods.concrete_memguide import ConcreteMemGuideMethod
from mm_memory_bench.methods.concrete_lightmem import ConcreteLightMemMethod
from mm_memory_bench.methods.concrete_universalrag import ConcreteUniversalRAGMethod
from mm_memory_bench.methods.concrete_vimrag import ConcreteVimRAGMethod, official_vimrag_agent_factory
from mm_memory_bench.methods.media import openai_content_from_parts


CLASSES=[ConcreteAMemMethod, ConcreteMemGuideMethod, ConcreteLightMemMethod,
         ConcreteUniversalRAGMethod, ConcreteVimRAGMethod]


def memory():
    return {'memory_id':'m1','content':[
        {'type':'text','text':'Actual table data: 93 C','annotations':{'format':'table'}},
        {'type':'image','source_id':'img_51.jpg','path':'/opaque/a.jpg'},
        {'type':'image','source_id':'img_52.jpg','path':'/opaque/b.jpg'}]}


def configured(cls):
    m=cls.__new__(cls)
    m.adapter_profile='official';m.question_mode='fixed';m.caption_model=object()
    m._caption=Mock(side_effect=lambda part:'beans' if part['path'].endswith('a.jpg') else 'grinder')
    m._memory_json=Mock(return_value={'context':'summary','keywords':[],'tags':[]})
    m.video_frames=1
    return m


def stored(m, data):
    if isinstance(m, ConcreteAMemMethod):return m.memory_to_notes(data)
    if isinstance(m, ConcreteMemGuideMethod):return m._memory_unit(data)
    if isinstance(m, ConcreteLightMemMethod):return m._memory_text(data)
    return m._memory_units(data)


@pytest.mark.parametrize('cls',CLASSES)
def test_table_and_each_image_identity_survive_method_ingest_input(cls):
    data=memory();original=copy.deepcopy(data)
    value=stored(configured(cls),data)
    text=json.dumps(value,ensure_ascii=False)
    assert 'Actual table data: 93 C' in text
    assert 'img_51.jpg' in text and 'img_52.jpg' in text
    if cls in [ConcreteAMemMethod,ConcreteMemGuideMethod,ConcreteLightMemMethod]:
        assert 'img_51.jpg\\nbeans' in text
        assert 'img_52.jpg\\ngrinder' in text
    assert data==original


@pytest.mark.parametrize('cls',[ConcreteAMemMethod,ConcreteMemGuideMethod,ConcreteLightMemMethod])
def test_unlabelled_caption_inputs_keep_previous_text(cls):
    data=memory()
    for part in data['content']:part.pop('source_id',None)
    text=json.dumps(stored(configured(cls),data))
    assert 'beans' in text and 'grinder' in text
    assert 'Image source_id' not in text


def test_oracle_shared_renderer_binds_labels_to_pixels(monkeypatch):
    monkeypatch.setattr('mm_memory_bench.methods.media.data_url',lambda path:path)
    parts=openai_content_from_parts(memory()['content'])
    for i,p in enumerate(parts):
        if p['type']=='image_url':
            label='img_51.jpg' if p['image_url']['url'].endswith('a.jpg') else 'img_52.jpg'
            assert label in parts[i-1]['text']


def test_vimrag_adapter_keeps_agent_picture_ids_and_adds_original_ids(monkeypatch):
    class Upstream:
        def __init__(self,**kwargs):pass
        def format_search_results(self,result,add_vision_ids=False):
            content=[{'type':'text','text':'Picture 1:'},{'type':'image','image':'/opaque/a.jpg'}]
            return (content,{'Picture 1':'/opaque/a.jpg'}) if add_vision_ids else content
    # Exercise the installed upstream formatter when its source is available,
    # without constructing its API client or loading the rest of the agent.
    source = Path(__file__).parents[1]/'sources/vimrag/demo/vimrag_agent.py'
    if source.is_file():
        import ast
        module = ast.parse(source.read_text())
        cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'VimRAG')
        function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'format_search_results')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        Upstream.format_search_results = namespace['format_search_results']
        Upstream.max_pixels_image = 1024
    monkeypatch.setattr('mm_memory_bench.methods.concrete_vimrag._load_official_demo',
                        lambda root:(SimpleNamespace(VimRAG=Upstream),None,None))
    agent=official_vimrag_agent_factory(official_repo=Path('.'),answer_model=Mock(),
        generation=SimpleNamespace(base_url='unused',model='test'),search=Mock(),query_media=[],
        search_top_k=10,memory_top_k=10,max_steps=20,video_frames=1)
    data={'data':[{'type':'image','file_path':'/opaque/a.jpg','media_source_id':'img_51.jpg'}]}
    for flag in [True,False]:
        result=agent.format_search_results(data,add_vision_ids=flag)
        content,ids=result if flag else (result,None)
        index=next(i for i,p in enumerate(content) if p.get('type')=='image')
        assert content[index-1]['text']=='Image source_id: img_51.jpg'
        assert content[index]['image']=='/opaque/a.jpg'
        if flag:
            assert any(p.get('text')=='Picture 1:' for p in content)
            assert ids=={'Picture 1':'/opaque/a.jpg'}


@pytest.mark.parametrize("cls", [ConcreteMemGuideMethod, ConcreteLightMemMethod])
def test_existing_caption_explicit_asset_mapping_is_not_positional(cls):
    data = memory()
    data["content"][1]["asset_id"] = "a"
    data["content"][2]["asset_id"] = "b"
    data["metadata"] = {"derived": {"image_captions": [
        {"asset_id": "b", "caption": "grinder", "secret": "GOLD_SECRET"},
        {"asset_id": "a", "caption": "beans"}]}}
    text = str(stored(configured(cls), data))
    assert "img_52.jpg" + chr(10) + "grinder" in text or "img_52.jpg\\ngrinder" in text
    assert "img_51.jpg" + chr(10) + "beans" in text or "img_51.jpg\\nbeans" in text
    assert "GOLD_SECRET" not in text


def test_ambiguous_multi_image_captions_are_not_zipped():
    from mm_memory_bench.methods.media import caption_metadata_with_sources
    data = memory()
    captions = {"image_captions": ["grinder", "beans"]}
    assert caption_metadata_with_sources(data, captions) == captions
    for cls in (ConcreteMemGuideMethod, ConcreteLightMemMethod):
        data["metadata"] = {"derived": captions}
        text = str(stored(configured(cls), data))
        assert "img_51.jpg" in text and "img_52.jpg" in text
        assert "no caption ordering implied" in text


@pytest.mark.parametrize("view", ["derived", "raw_derived"])
def test_partial_sidecar_is_bound_to_its_asset(view):
    from mm_memory_bench.runner.benchmark import _resolved_memory
    class Reader:
        def resolve_content(self, parts): return parts
        def captions_for_content(self, parts):
            return ["grinder" for p in parts if p.get("asset_id") == "b"]
    data = memory()
    data["content"][1]["asset_id"] = "a"
    data["content"][2]["asset_id"] = "b"
    result = _resolved_memory(Reader(), data, memory_view=view)
    text = json.dumps(result)
    assert "img_52.jpg\\ngrinder" in text
    assert "img_51.jpg\\ngrinder" not in text
    assert "img_51.jpg" in text


@pytest.mark.parametrize("view", ["derived", "raw_derived"])
def test_explicit_and_sidecar_captions_can_coexist_without_annotations(view):
    from mm_memory_bench.runner.benchmark import _resolved_memory
    class Reader:
        def resolve_content(self, parts): return parts
        def captions_for_content(self, parts):
            return ["grinder" for p in parts if p.get("asset_id") == "b"]
    data = memory()
    data["content"][1]["asset_id"] = "a"
    data["content"][2]["asset_id"] = "b"
    data["metadata"] = {"derived": {"image_captions": [
        {"asset_id": "a", "caption": "beans", "private_answer": "SECRET"},
        {"asset_id": "unknown", "caption": "unmatched"}]}}
    result = _resolved_memory(Reader(), data, memory_view=view)
    text = json.dumps(result)
    assert "img_51.jpg\\nbeans" in text and "img_52.jpg\\ngrinder" in text
    assert "SECRET" not in text
    assert "unmatched" in text and "unknown" not in text


def test_single_image_caption_does_not_duplicate_identity():
    from mm_memory_bench.methods.media import caption_metadata_with_sources
    data = memory(); data["content"] = data["content"][:2]
    result = caption_metadata_with_sources(data, {"caption": "beans"})
    assert result["caption"] == "Image source_id: img_51.jpg\nbeans"
    assert caption_metadata_with_sources(data, result) == result
