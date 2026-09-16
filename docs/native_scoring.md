# Benchmark 原生评分与输入适配

本文描述当前脚本评分、专用 LLM Judge，以及它们使用的公开输入约定。原始 bundle 和 predictions 是评分输入，评分过程不改写它们；通用 QA Judge 仍通过 `--scoring-protocol qa` 单独使用。

## 运行入口

推荐按题型自动分发：

```bash
mmmb judge data/unified/smmbench runs/smmbench/predictions.jsonl \
  --scoring-protocol benchmark \
  --output runs/smmbench/scoring/judgments.jsonl
```

SMMBench、Persona-MME 只需脚本，不需要模型、API 或 GPU。有 LLM 题时增加模型配置：

```bash
mmmb judge data/unified/m3exam runs/m3exam/predictions.jsonl \
  --scoring-protocol benchmark \
  --output runs/m3exam/scoring/judgments.jsonl \
  --model "$JUDGE_MODEL" --base-url "$JUDGE_URL" \
  --api-key-env JUDGE_API_KEY --concurrency 8
```

| Benchmark / 题型 | 分发规则 |
|---|---|
| SMMBench MCQ / 工具规划 | 选项匹配 / 工具计划脚本 |
| Persona-MME | 选择题脚本，主问题与 alignment 按 subset 分开 |
| PersonaMem-v2 MCQ | 选择题脚本，历史条件分别汇总 |
| PersonaMem-v2 多模态开放轨道 | 窄偏好 LLM Judge，保留连续偏好分 |
| M³Exam fj / fm | 文本 EM / 图片 ID 命中脚本 |
| M³Exam mr、tr、ms、ss、th、ii | 五档 LLM Judge：0 / 0.25 / 0.5 / 0.75 / 1 |
| MobileMem-Omni | 官方公开 prompt 与本仓库请求、解析实现 |

每题只分配一个评分器，不将 EM、准确率与偏好分混成总分。分发协议为 `mmmb-benchmark-dispatch-2.2`，逐题结果还记录实际使用的脚本或 LLM 协议。

- benchmark 入口默认要求完整预测。小样本使用 `--question-ids ids.txt`；`--max-items N` 按 bundle 顺序（有 allowlist 时按列表顺序）取前 N 题，并明确标记子集。
- 选题、参考数据、输出路径和模型需求在写结果或调用模型前检查。重复 ID、未知 ID、缺预测或未知题型直接报错，不自动退回 QA。
- 脚本输出在指定扩展名前加 `.native`：`judgments.jsonl` 对应 `judgments.native.jsonl` 和 `judgments.native.summary.json`。
- LLM 输出使用指定路径和对应的 `.summary.json`；只有脚本题时不生成 LLM 结果文件。
- 额外生成 `<output-stem>.dispatch.summary.json`，记录每路分配数、完成数、方法失败数及各自汇总。
- 可直接选择 `personamem_v2_open`、`m3exam`、`mobilemem_omni` 专用 LLM 协议，均要求模型配置；直接使用 M³Exam 协议时需通过题目列表排除 fj/fm。完整性等统一运行检查以 benchmark 分发入口为准。

只计算脚本指标时也可使用独立入口：

```bash
PYTHONPATH=src python scripts/score_script.py smmbench \
  data/unified/smmbench runs/smmbench/predictions.jsonl \
  --output runs/smmbench/native_scores.jsonl
```

其他名称为 `persona_mme`、`personamem_v2`、`m3exam`。此入口直接使用指定输出路径，默认检查完整覆盖；小样本需指定 `--question-ids` 或使用小样本 bundle。主输出及派生 summary 不得覆盖预测、题目、manifest、数据表或彼此。

## 选择题格式与脚本规则

**SMMBench converter 保留数字标签 0–3；PersonaMem-v2 converter 保留字母标签。格式适配只发生在评分入口。**

- SMMBench 接受当前题目完整有效的数字标签，或完整的 `数字: 对应公开选项文字`，再映射到官方 `(A)–(D)`。映射按标签所代表的原始编号进行，不按 choices 列表位置，也不读取标准答案。
- PersonaMem-v2 将完整有效的单个 ASCII 字母包装为 `Final Answer: X`，再调用原生解析规则比较选项文本；单字母忽略大小写。
- 适配仅去除外侧空白，不从解释或多选表达中猜标签。`0: Sardinia` 不会匹配文字为 Corsica 的选项 0；`0 or 1` 不会被转换成 `(A)`。
- 已有原生格式及其余文本原样交给原生解析器。原生解析器自身的宽松匹配行为保持不变。
- 旧预测必须与生成它的原 bundle 配对。评分不会把旧预测自动迁移到选项标签已改变的新 bundle。

| 脚本任务 | 规则 |
|---|---|
| SMMBench MCQ | 适配公开标签后，使用官方选项匹配规则 |
| SMMBench 工具计划 | 规范化大小写、补候选工具默认参数；步骤数量相同，按列表顺序比较；各步覆盖标准调用，可有额外调用；重复标准调用逐个匹配，不比较 step 编号 |
| Persona-MME | 使用官方 check_result 解析，如 `(d): Fallen leaves`；会触发原生 IndexError 的畸形输出记 invalid_prediction=0 |
| PersonaMem-v2 MCQ | 使用官方 extract_final_answer 的正则与优先级，比较对应选项文本 |
| M³Exam | 文本题使用 accepted_answers 计算 EM；fm 按图片 ID 任意命中，无图片 ID 时退回文本 EM；均值按官方规则保留四位 |

脚本协议为 `mmmb-native-scripts-2.2`，逐题指标为 0/1，按 subset/category/subcategory 分组，每个指标有自己的样本数。独立脚本可计算其他 M³Exam 文本题的 EM；benchmark 分发入口只将 fj/fm 交给脚本。F1、BLEU-1 不作为报告指标，MobileMem-Omni 不属于脚本评分范围。

每次脚本评分重新计算，汇总保留输入 SHA256。该协议表示“Graphmemix 回答格式适配＋原生评分规则”，不代表直接调用外部官方运行器，也不代表完整复现官方生成流程。

## 方法失败、评分失败与续跑

脚本、专用 LLM Judge 和分发汇总共用 `evaluation/prediction_status.py`。它识别 metadata 或预测顶层的 `method_error`、`error_type`，以及 `status=error/failed`。

- 对可评分题，方法异常记 `method_error`，保留原因、计零并计入相应分母，残留答案不参与评分。
- **Omni 无参考题始终不进入原生准确率分母。** 即使方法失败，标签仍为 null；该记录同时计入方法失败数和跳过数。这两个计数可以重叠，不能相加当作题目总数。
- 正常空回答遵循各协议规则，例如 M³Exam 的空回答对空参考可能 EM=1；这与方法运行失败不同。
- Judge 空回复、异常结束（包括 `finish_reason=length`）和需要抛错的解析失败记 `status=error`，计入 `failed_judgments`，可在评分续跑时重试。保留 `judge_response_metadata` 中的结束原因、usage 和请求额度。
- 非空、正常结束的回答仍按原生解析规则处理；M³Exam、PersonaMem-v2 原生解析器无法解析时返回 0 的行为保留。

专用 LLM 评分缓存匹配模型、协议、rubric、题目内容、预测及请求配置；`--no-resume` 强制重评。修改 Judge 输出预算、使用新传输实现或升级相应协议后，不复用不兼容缓存。

这些规则适用于原生评分入口；通用 `evaluation/judge.py` 保留自己的既有行为。评分续跑只重试 Judge，不会重新调用记忆方法生成回答；已有方法失败需先补跑作答。

## 模型额度

`mmmb judge ... --scoring-protocol benchmark --judge-max-tokens 512` 可覆盖专用 LLM Judge 的输出额度，直接调用专用协议时也支持。未指定时保留各协议默认值；QA 不接受这个参数。M³Exam 默认 16 tokens，推理模型可能需要更大额度，512 只是实验配置而非普遍保证。

Reader 使用 `mmmb run-method ... --reader-max-model-len 65536` 指定输入与输出的总窗口。服务端须另行支持该容量；客户端不会调整服务，也不会静默截断证据。容量变化使用独立预测目录，记录实验配置，详见 [model_context.md](model_context.md)。

## SMMBench 工具规划输入

新转换的工具题使用[官方固定提示词](https://github.com/FatCatCHC/SMMBench/blob/c52cf9d2b6b800784b097d6c055b0e9d8d105842/evaluation/agents/prompt.py)，要求单步 `{"calls": [...]}` JSON。converter 将完整候选工具写入指令，以 `instruction_role="system"` 和 `instruction_includes_tools=true` 声明消息角色和候选工具已被渲染。

公共 `AnswerTask` 将该指令与方法自身 system 约束合并；A-Mem、MemGuide、LightMem、UniversalRAG、Memix 和 Oracle 共用这一入口，候选工具只呈现一次。VimRAG 将相同要求作为 agent 任务文本，保留其内部 system 提示和检索工具。候选工具只用于生成计划，不实际执行。旧 bundle 没有“已包含工具”的声明时，公共入口仍单独呈现 tools 字段。

部分官方参考包含空的第一步，与单步提示词冲突。保留原题与原评分并单独报告该冲突；不根据 gold 选择提示词，不删除参考步骤或放宽评分。提示词对齐不等于复现官方模型、检索器和证据组织。

## 图片、caption 与表格

M³Exam converter 将附件原始文件名写入每个 content part 的公开 `source_id`。图片渲染将编号与像素一起呈现。UniversalRAG 按证据单元保留同一记忆中的不同图片，重复检索同一资产仍去重，top-k 仍表示证据单元数。

公共 caption 提取和图片关联位于不依赖具体方法的 `preprocessing/captions.py`，runner 与直接调用方法的入口复用同一幂等实现。显式引用按资产关联，无引用的多图 caption 不按位置猜配对。生成 caption 缓存保留原描述，使用时再加图片编号。A-Mem、MemGuide、LightMem 使用这些公共函数；VimRAG 在原生搜索结果的图片旁增加 source_id，保留其 Picture 编号；Oracle 复用公共图片渲染。

SMMBench 结构化表格保留 `type=text` 和 JSON 内容，额外标记 `annotations.format=table`。runner 只保留该明确标记，其他私有 annotations 继续过滤。UniversalRAG 对已标记表格直接建库；旧未标记的 header/rows JSON 保留精确识别，原 paragraph/document 覆盖不变。

## 迁移与结果配对

| 变更目的 | 需要做什么 |
|---|---|
| 修正旧选择题标签评分、重新识别方法失败 | 使用原 bundle＋原 predictions 重评分；不为评分修复重新转换或调用回答模型 |
| 更新 Omni 的无参考题统计 | 使用原输入重评；新协议 `mmmb-omni-published-prompt-1.1` 不复用旧协议缓存 |
| 启用新的工具规划指令 | 重新转换相关 bundle 并生成新 predictions；若记忆输入未变，可复用兼容记忆索引或冻结的检索证据 |
| 使用新增图片身份或改变记忆媒体输入 | 重新转换缺少相应字段的 bundle，使用新 checkpoint 建库并重新作答，保留原实验文件 |
| 仅调整 Reader 容量 | 服务端确认容量后，使用独立预测目录重新作答；兼容记忆索引可复用 |

A-Mem note prompt v3、MemGuide/LightMem checkpoint v3、UniversalRAG checkpoint v3、VimRAG checkpoint v2 对应已更新的媒体输入。旧索引被拒绝时使用新目录，不将旧索引视为已包含新图片身份或表格单元。公共函数搬迁本身不另行升级这些版本。

曾使用字母选项版本 converter 的实验，应保留其字母 bundle 与预测一起归档和重评；恢复数字 converter 后，不把两组输入条件当成同一实验追加结果。

## 原生 LLM 协议与来源

### M³Exam

协议 `mmmb-m3exam-five-point-1.0`，固定官方 commit `1dbe10441a043d86043ad13283e1ee03fbe154b6`：

- [SharedLLM.judge](https://github.com/EverM0re/M-3-Exam/blob/1dbe10441a043d86043ad13283e1ee03fbe154b6/baselines/_runtime/extended_common/llm_client.py)：保留提示词、temperature=0、默认 max_tokens=16，以及五档解析、平局取较小档、无法解析记 0。
- [score_record](https://github.com/EverM0re/M-3-Exam/blob/1dbe10441a043d86043ad13283e1ee03fbe154b6/baselines/run.py)：参考取 native_label，空值退回首个原生有序答案，不按预测挑选参考。
- [aggregate](https://github.com/EverM0re/M-3-Exam/blob/1dbe10441a043d86043ad13283e1ee03fbe154b6/m3proctor/evaluation/metrics.py)：按非 fj/fm 题目平均并保留四位，另按原生题型汇总。

逐题保留 score、judge_response、native_type、source_revision，不伪造 correct。HTTP 请求与失败续跑由本仓库实现，未复刻官方 SDK 自动重试或 temperature 不支持时的兼容重发。

### MobileMem-Omni

协议 `mmmb-omni-published-prompt-1.1`，固定官方 commit `919e0f545722030898cee03b263f08c8092f2ebb`：

- [公开 Judge prompt](https://github.com/zjunlp/MobileMem/blob/919e0f545722030898cee03b263f08c8092f2ebb/omni/eval/eval/question_answering_and_judge_prompts.txt)：保留全部规则与占位符。
- [Evaluator](https://github.com/zjunlp/MobileMem/blob/919e0f545722030898cee03b263f08c8092f2ebb/omni/eval/eval/evaluator.py)：按 F1＋BLEU-1 最大选择一个参考，平局保留首个；这两个值只用于内部选参考，不报告为指标。
- [Raw2Locomo](https://github.com/zjunlp/MobileMem/blob/919e0f545722030898cee03b263f08c8092f2ebb/omni/eval/eval/Raw2Locomo.py)：native_evidence 字典取 explanation，字符串保留原文。适配支持当前已转换的文本查询轨道。

公开仓库缺少导入的 `eval/llm_judge.py`。本地请求采用 user 文本、temperature=0、JSON 序列化证据；解析唯一包含 label 的 JSON 对象，只接受 CORRECT/WRONG。这些请求与解析选择属于本仓库实现，逐题标记 `implementation_scope=published_prompt_local_parser`，不宣称完整官方实现等价。1.1 修正无参考题与方法失败同时发生时的统计，公开提示词保持不变。

### PersonaMem-v2

协议 `mmmb-personamem-v2-narrow-1.1`。保留原有窄偏好提示词和官方 extract_judge_decision 数值规则，包括无法解析返回 0；连续偏好分不报告为二元准确率。四种 MCQ 历史条件与开放回答轨道使用各自的评分路径。

### 脚本参考实现与验证

- [SMMBench utils](https://github.com/FatCatCHC/SMMBench/blob/c52cf9d2b6b800784b097d6c055b0e9d8d105842/evaluation/utils.py)：工具计划和选项匹配。
- [Persona-MME eval](https://github.com/MiG-NJU/PersonaVLM/blob/main/eval.py)：选择题及 alignment。
- [PersonaMem-v2 inference](https://github.com/bowen-upenn/PersonaMem-v2/blob/main/inference.py)：选项文本比较。
- [M³Exam metrics](https://github.com/EverM0re/M-3-Exam/blob/1dbe10441a043d86043ad13283e1ee03fbe154b6/m3proctor/evaluation/metrics.py)：文本 EM、图片 ID 命中。

仓库测试覆盖固定提示词哈希、请求与解析、评分分发、缓存失效、失败统计、输入隔离，以及跨方法的图片/caption/表格输入。历史官方快照差分材料位于 `tmp/native-scoring/`、`tmp/native-scoring-audit/`、`tmp/official-judge-validation/`；它们不是每次修改后的完整回归记录，也不证明真实模型成绩或缺失的 Omni 请求/解析实现等价。
