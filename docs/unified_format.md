# MMMB 统一格式 v1

`mmmb-1.0` 是面向多模态 agent 长期记忆评测的关系式 JSONL bundle。它的目标是统一“方法输入接口”，而不是篡改各基准的官方评分语义。

每个 bundle 有五个文件：

- `manifest.json`：来源、版本、许可、subset、推荐评分器和各表行数。
- `contexts.jsonl`：一个用户、persona、全局记忆池或独立对话上下文一行。
- `memories.jsonl`：可按顺序摄入记忆系统的最小项目，例如对话消息、邮件、图片、视频、文档或 profile。
- `assets.jsonl`：媒体注册表。媒体不复制，`path` 相对 bundle 指向 `data/raw` 中的官方快照。
- `questions.jsonl`：问题、答案、选项、问题时点、任务标签和 evidence 引用。

## 核心记录

### memory

```json
{
  "memory_id": "memeye:task_a:D1:1:user",
  "context_id": "memeye:task_a",
  "session_id": "D1",
  "source_id": "optional-native-source-or-evidence-id",
  "sequence": 0,
  "timestamp": "2026-03-01",
  "kind": "dialogue_message",
  "role": "user",
  "speaker": "user",
  "content": [
    {"type": "text", "text": "..."},
    {"type": "image", "asset_id": "memeye:asset:..."}
  ],
  "provenance": {"source_file": "data/dialog/task_a.json", "json_pointer": "/multi_session_dialogues/0/dialogues/0"},
  "metadata": {}
}
```

### question

```json
{
  "question_id": "memeye:task_a:q0",
  "context_id": "memeye:task_a",
  "subset": "default",
  "split": "test",
  "prompt": [{"type": "text", "text": "..."}],
  "instruction": "",
  "tools": [{"type": "function", "function": {"name": "...", "parameters": {"type": "object"}}}],
  "task": {"category": "visual_memory", "subcategory": "...", "response_type": "text"},
  "choices": [],
  "answer": {"text": "...", "choice_ids": [], "unanswerable": false},
  "evidence": [{"memory_id": "memeye:task_a:D1:1:user", "relation": "supports"}],
  "memory_scope": {"mode": "all"},
  "query_at": {"session_ids": ["D1"]},
  "provenance": {"source_file": "data/dialog/task_a.json", "json_pointer": "/human-annotated QAs/0"},
  "metadata": {}
}
```

## 设计约束

1. `question_id`、`memory_id`、`asset_id` 和 `context_id` 在各自表内唯一且稳定。
2. 对话的 user/assistant 消息拆成两条 memory，并共享原生 round id；媒体留在原来所属的消息上。
3. `content` 是有序的。若原数据规定图片占位符的消费顺序，转换器必须保持该顺序。
4. 预生成 caption、OCR、摘要是资产或 memory 的 annotation，不能冒充原始视觉输入。
5. `answer.text` 永远保留官方原始答案。choice、number、list 等结构化结果只作为附加字段，不能替换原值。
6. 官方 evidence 能映射时使用 canonical id；不能无歧义映射时保留 `native_id` 并在转换报告中告警。
7. subset 表示官方变体（例如 `hard`、`32k`），split 表示 train/validation/test；二者不可混用。
8. `provenance` 必须可回到原始快照文件和 JSON Pointer。未进入公共字段的原生注释保存在 `metadata.native`。
9. 各表按 `contexts.jsonl` 的 context 顺序连续分组，以便 loader 流式读取超长上下文。
10. 每个问题必须给出 `memory_scope`；in-situ 问题不得看到查询时点之后的 future memories。
11. `tools` 只用于官方 model-visible 的候选工具定义，必须是 JSON 对象数组；任意嵌套层的 answer、evidence、provenance、metadata、gold/ground-truth、solution 和 evaluator-private 字段均禁止进入该公共结构。
12. `task.response_type` 的公共枚举只有 `text`、`choice`、`structured_json`；binary、abstention、image-id list 等输出约束写入公开 `instruction`，未来若加入媒体生成再扩展枚举。
13. `choices` 必须是 `{choice_id, text}` 对象数组，可选 `content` 承载多模态选项；不接受裸字符串 choice。
14. `memory_id` 是 bundle 内稳定关系键；若 benchmark 另有要求模型读取或输出的原生标识，统一放在 model-visible 的 `source_id`，不能只藏在私有 metadata 或 canonical ID 前缀里。

## 适配器接口

通用框架只需完成三步：

1. 按 `context_id` 读取并按 `sequence` 排序 memories；
2. 将 `content` 映射成模型支持的 text/image/video/document block；
3. 对同一 `context_id` 的 questions 推理，并按 manifest 指向的官方协议评分。

统一数据层不强制统一评分器。尤其 LLM-as-a-Judge、MCQ accuracy、检索 Recall@K 和 action prediction 的语义不同，强行用一个分数会破坏可比性。
