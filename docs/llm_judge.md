# 统一 LLM Judge 评估

当前正式统一评分协议为 `mmmb-llm-judge-1.1`。它统一“答案是否正确”的判断入口，但不会删除各数据集的原始 gold、官方指标或 scorer 信息，后续仍可重跑专属评估。

## 协议

- judge 输入只有 question、公开 instruction、choices、response type、reference answer 和 prediction；不使用 evidence、私有 notes 或 memory 内容。
- 固定 rubric、`temperature=0`，只允许返回 `{correct: boolean}` JSON；逐题记录不保存解释或原始模型回复。
- 数字检查数值/单位/币种，列表检查完整性和多余项，choice 接受 ID 或无歧义文本，函数调用检查工具名、参数、依赖和 step 顺序。
- 每条 judgment 保存 judge model、协议版本与 rubric SHA-256，便于锁定复现实验。
- 同时报告物理记录 accuracy 和 `semantic_question_id` macro accuracy。后者避免 MemEye 的四个 MCQ rotation 获得不成比例的权重。
- API/judge 失败会写成 `status=error` 并在 conservative accuracy 中按 0 计；再次执行默认只保留成功项并重试失败项。

## 运行

先用统一 harness 生成预测，然后调用任何 OpenAI-compatible chat endpoint：

```bash
export OPENAI_API_KEY="..."

PYTHONPATH=src python -m mm_memory_bench.cli judge \
  data/unified/memeye \
  runs/my_method/memeye.jsonl \
  --output runs/my_method/memeye.judgments.jsonl \
  --model YOUR_LOCKED_JUDGE_MODEL \
  --concurrency 8
```

本地兼容服务：

```bash
PYTHONPATH=src python -m mm_memory_bench.cli judge \
  data/unified/atm_bench \
  runs/my_method/atm.jsonl \
  --output runs/my_method/atm.judgments.jsonl \
  --model local-judge \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key-env LOCAL_API_KEY
```

输出包括逐题 `*.judgments.jsonl` 和聚合 `*.judgments.summary.json`。正式对比时必须固定并报告 judge model 的精确版本、endpoint/provider、协议版本和 rubric hash；不同 judge 版本的分数不可直接混合。

`--concurrency` 只影响吞吐，不改变 rubric；应按 provider 限流设置。中断后直接执行同一命令会复用同模型、同协议、同 rubric 的成功 judgment，并重试失败项。
