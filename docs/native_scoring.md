# 新增 benchmark 的脚本评分

只读取统一 bundle 的问题与已有 predictions，不调用模型、不需要 GPU，也不修改记忆或预测。只保留逐题结果为 0/1 的脚本指标，范围为 SMMBench、Persona-MME、PersonaMem-v2 MCQ 和 M³Exam。F1、BLEU-1 已移除，因此 MobileMem-Omni 不再属于脚本评分范围；本次不修改任何 LLM Judge。

## 运行

```bash
PYTHONPATH=src python scripts/score_script.py smmbench \
  data/unified/smmbench runs/smmbench/predictions.jsonl \
  --output runs/smmbench/native_scores.jsonl
```

其他名称：`persona_mme`、`personamem_v2`、`m3exam`。
小样本必须指定 `--question-ids path/to/ids.txt`（每行一个 ID），或传入对应的小样本 bundle。默认要求覆盖整个 bundle，禁止悄悄按成功预测的交集报告。重复 ID、未知 ID、缺预测会报错；方法失败保留并计零；空答案按各官方指标规则计分。

生成逐题 `native_scores.jsonl` 和 `native_scores.summary.json`。逐题指标值仅为 0 或 1，汇总均值可为小数，按 subset/category/subcategory 分组，每个指标记录自己的样本数；没有跨不同指标混合的总分。协议 `mmmb-native-scripts-2.1`，不称为 Judge Acc。重跑重新计算，不复用旧缓存；汇总保留输入 SHA256。

## 规则与边界

| 数据集 | 本次规则 |
|---|---|
| SMMBench MCQ | 评分入口将完整、有效的预测标签 0/1/2/3 映射为 (A)/(B)/(C)/(D)；标准答案沿用官方序号映射，再调用官方匹配规则 |
| SMMBench 工具计划 | 对齐官方 `evaluate_function_call_response`：规范化大小写、补候选工具默认参数；步骤数量相同，按列表顺序比较，各步覆盖标准调用，可有额外调用；重复标准调用需逐个匹配，不比较 step 编号 |
| Persona-MME | 使用官方 check_result 的选项解析；接受如 `(d): Fallen leaves`；按 subset 分开主问题与 alignment |
| PersonaMem-v2 | 评分入口将完整、有效的字母预测标签包装为 `Final Answer: X`，再使用官方 extract_final_answer 的正则及优先级比较选项文本；四种历史条件分别汇总 |
| M³Exam | fj 和其他文本题型使用 accepted_answers 计算文本 EM；fm 为任意标准图片 ID 命中，无图片 ID 时退回文本 EM。汇总均值仍按官方 aggregate 四舍五入到四位 |

**评分入口先适配统一 bundle 的完整标签，再调用各 benchmark 原有的官方解析器。**原始 PersonaMem-v2 解析函数不接受独立 `A`，但接受 `Final Answer: A`；适配只发生在调用它之前。SMMBench 官方允许包含匹配，Persona-MME 按第一个右括号前的末字符判断。这些解析行为保持不变，包括官方对部分多选式表达的宽松之处。Persona-MME 对官方会抛 IndexError 的畸形输出记 invalid_prediction=0。

converter 保留原来的统一输入：SMMBench 展示数字选项，PersonaMem-v2 展示字母选项，均要求返回标签。评分适配只去除标签外侧空白并检查它是否属于当前题目的选项；PersonaMem-v2 的单个 ASCII 字母忽略大小写。SMMBench 按标签本身代表的原始编号映射，不按 `choices` 列表当前位置映射；PersonaMem-v2 使用当前题目打乱后的标签。映射不读取标准答案，不改变选项内容或顺序。

已符合原生格式的回答及其余文本原样交给官方解析器；适配层不从解释文字或多选表达中猜标签，例如 `0 or 1` 不会被转换成 `(A)`。转换仅存在于评分调用中，不写回 bundle 或 predictions。旧数字/字母标签预测可以直接重评分，无需为本次格式适配重新转换数据或调用回答模型；已有原生格式预测也仍可评分。通用 QA Judge、工具计划和开放题评分不受此适配影响。

协议 2.1 表示“Graphmemix 回答格式适配 + 官方评分规则”，不是将原始响应直接交给官方运行器，也不代表完整复现官方生成协议。脚本每次重新计算，旧 2.0 分数应重评后按新协议报告；本次没有增加外部官方运行器的导出或调用功能。

方法标记为失败（metadata.status=error 或 error_type 非空）时记 method_error、所有指标为零，残留答案不参与计分。这是框架的运行失败统计约定。正常空响应保留官方边界：例如 M³Exam 空响应对空参考的 EM 可能为 1。

评分参考字段只供评分器读取，不进入回答模型。invalid_prediction 是模型输出格式问题，不等于评分脚本故障；无效标准数据直接报错。独立脚本在写入前同时检查主输出及派生 summary 的路径，拒绝覆盖预测、题目、manifest 或声明的数据表，也拒绝两个输出指向同一文件。

## 代码组织与原入口

- [score_script.py](../scripts/score_script.py)：单 bundle 命令行入口。
- [evaluation/native/__init__.py](../src/mm_memory_bench/evaluation/native/__init__.py)：仅导出脚本评分公共入口。
- [evaluation/native/script_judge.py](../src/mm_memory_bench/evaluation/native/script_judge.py)：官方二元规则、题型分发、覆盖检查、逐题输出和分组汇总集中在一个文件。
- [evaluation/native/personamem_v2_judge.py](../src/mm_memory_bench/evaluation/native/personamem_v2_judge.py)：独立的 PersonaMem-v2 开放题 LLM 协议，由独立 native_runner.py 调用，不受脚本指标裁剪影响。
- [test_native_scoring.py](../tests/test_native_scoring.py)：计划、选项、EM 规则及输出完整性测试。

原 `scripts/compute_native_metrics.py` 和 `evaluation/judge.py` 不在本次修改范围。独立脚本命令继续可用；也可使用下方 benchmark 分发入口，按题型复用脚本和 LLM 评分流程。

## 参考实现

- [SMMBench utils](https://github.com/FatCatCHC/SMMBench/blob/main/evaluation/utils.py)：工具计划规范化与覆盖比较。
- [Persona-MME eval](https://github.com/MiG-NJU/PersonaVLM/blob/main/eval.py)：选择题及 alignment 分组。
- [PersonaMem-v2 inference](https://github.com/bowen-upenn/PersonaMem-v2/blob/main/inference.py)：选择题选项文本比较。
- [M³Exam metrics](https://github.com/EverM0re/M-3-Exam/blob/main/m3proctor/evaluation/metrics.py)：文本 EM、图片 ID 命中。

本次核对使用本地官方源码快照，具体文件 SHA256 与差分结果保存在 `tmp/native-scoring/verification.json`。真实模型预测重评分不需要重新运行 method，不代表扩大了模型测试样本。

## 扩大验证（官方入口对照）

历史协议 1.2 的扩大核查（包含现已移除的 F1、BLEU-1）：`tmp/native-scoring-audit/full/verify.py` 与 `verification.json`。覆盖现有 34,749 条题目记录的构造回答、60 条已有真实预测、全部 108 道 SMMBench 工具题及实际官方默认参数；记录官方源码与本地实现 SHA256。单题与汇总分别核查。该记录是合并文件及裁剪指标之前的验证结果，不能直接当作当前脚本协议的测试记录；不包含 PersonaMem-v2 开放回答评分、任何 LLM Judge 或方法生成协议的完整复现。

## 统一 benchmark 评分入口

先运行 method 或 Oracle 得到 predictions，再执行：

```bash
mmmb judge data/unified/smmbench runs/smmbench/predictions.jsonl \
  --scoring-protocol benchmark \
  --output runs/smmbench/scoring/judgments.jsonl
```

SMMBench 和 Persona-MME 只需脚本，不需要模型、API 或 GPU。有 LLM 题时增加配置：

```bash
mmmb judge data/unified/m3exam runs/m3exam/predictions.jsonl \
  --scoring-protocol benchmark \
  --output runs/m3exam/scoring/judgments.jsonl \
  --model "$JUDGE_MODEL" --base-url "$JUDGE_URL" \
  --api-key-env JUDGE_API_KEY --concurrency 8
```

| Benchmark / 题型 | 分发规则 |
|---|---|
| SMMBench 普通选择题 / 工具规划题 | 原生选项匹配 / 工具计划脚本 |
| Persona-MME | 原生选择题脚本 |
| PersonaMem-v2 MCQ | 原生选择题脚本 |
| PersonaMem-v2 多模态开放轨道 | 已有窄偏好 LLM Judge，保留连续偏好分 |
| M³Exam fj / fm | 文本 EM / 图片 ID 命中脚本 |
| M³Exam mr、tr、ms、ss、th、ii | 官方基线五档 Judge，0 / 0.25 / 0.5 / 0.75 / 1 |
| MobileMem-Omni | 官方公开 prompt 适配，CORRECT / WRONG；请求与 JSON 解析为本仓库实现，官方 llm_judge.py 未发布 |

此组合协议版本为 `mmmb-benchmark-dispatch-2.1`，脚本路由包含上述标签适配。M³Exam 使用固定官方快照的提示词、请求参数、五档解析和均值规则；Omni 使用官方公开提示词、参考答案选择及汇总规则，但不能声称缺失的请求/解析实现完全等价。版本 1.0 使用的通用 QA Judge 缓存不会被新协议复用；本次 2.1 调整不改变各专用 LLM Judge 的协议或缓存。每题只分配一个评分器，不再额外对 M³Exam 其他题型计算 EM。独立脚本入口仍允许单独计算这些题型的 EM。

- 默认要求所有 bundle 问题都有预测；小样本使用 `--question-ids ids.txt`。`--max-items N` 按 bundle 问题顺序（有 allowlist 时按列表顺序）取前 N 题，并明确标记子集。
- 所有选中题目、标准答案和模型需求先检查，再开始 API 调用或写结果。未知 benchmark、未知题型、重复 ID 和缺预测明确报错，不自动退回通用 Judge。
- 脚本输出在 `--output` 的扩展名前加 `.native`：例如指定 `method_a.judgments.jsonl`，生成 `method_a.judgments.native.jsonl` 和 `method_a.judgments.native.summary.json`。不同输出文件名拥有独立的脚本结果；LLM 输出使用 `--output` 和相应 summary。只有脚本题时不生成 LLM 结果文件。独立 `score_script.py` 仍直接使用其 `--output` 路径。
- 额外生成 `<output-stem>.dispatch.summary.json`，记录协议、各路分配数、完成数、方法失败数、评分状态及各自汇总；不把 EM、QA 准确率与偏好分混成总分。每次运行建议使用独立结果目录，避免其他任务的旧结果混淆。
- LLM 复用既有模型、协议、题目及预测哈希的续跑检查；`--no-resume` 强制重评。脚本每次重新计算，不调用模型。
- 原 `--scoring-protocol qa` 和 `personamem_v2_open` 行为保留，仍要求 `--model`。也可显式选择 `m3exam` 或 `mobilemem_omni` 专用协议；M³Exam 直接调用需用 `--question-ids` 排除 fj/fm。benchmark 模式仅在分配到 LLM 题目时要求模型配置；密钥通过指定环境变量读取，也可使用无需认证的本地端点。

分发实现位于 `evaluation/dispatcher.py`；`evaluation/native_runner.py` 接收问题 ID 列表并执行专用 LLM 评分，避免为每个分组生成临时 allowlist 文件。`evaluation/judge.py` 已恢复为原提交版本，`qa` 命令仍调用它。converter、method、Reader 不因该入口而改变。

## 专用 LLM 协议与官方对齐边界

```text
evaluation/
├── dispatcher.py                 # 统一分发
├── judge.py                      # 原有通用 QA Judge，保持原样
├── native_runner.py              # 专用 LLM Judge 的 API、并发与续跑
└── native/
    ├── __init__.py
    ├── script_judge.py
    ├── personamem_v2_judge.py
    ├── m3exam_judge.py
    └── mobilemem_omni_judge.py
```

### M³Exam

固定官方 commit `1dbe10441a043d86043ad13283e1ee03fbe154b6`：

- [SharedLLM.judge 及提示词、解析函数](https://github.com/EverM0re/M-3-Exam/blob/1dbe10441a043d86043ad13283e1ee03fbe154b6/baselines/_runtime/extended_common/llm_client.py)。提示词逐字保留；使用 baselines/config.yaml 的 temperature=0，max_tokens 按官方下限设为 16。
- [score_record](https://github.com/EverM0re/M-3-Exam/blob/1dbe10441a043d86043ad13283e1ee03fbe154b6/baselines/run.py)：参考答案取 native_label，空值退回首个原生有序答案，不根据预测选择另一个答案。fj/fm 不调用 LLM。
- [_parse_judge_score](https://github.com/EverM0re/M-3-Exam/blob/1dbe10441a043d86043ad13283e1ee03fbe154b6/baselines/_runtime/extended_common/llm_client.py)：保留五档就近取值、平局选择较小档、无法解析记 0；不额外转换为二元正确性。
- [aggregate](https://github.com/EverM0re/M-3-Exam/blob/1dbe10441a043d86043ad13283e1ee03fbe154b6/m3proctor/evaluation/metrics.py)：llm_score 按非 fj/fm 题目平均并保留四位小数，另按原生题型汇总。这里只输出已选 LLM 题目的统计，不恢复已移除的 F1/BLEU-1。

成功结果保留 `score`、`judge_response`、`native_type`、`source_revision`，不伪造 `correct`。协议 `mmmb-m3exam-five-point-1.0`。HTTP 调用与失败续跑复用本仓库；未复刻官方 OpenAI SDK 的自动重试与 temperature 不支持时的兼容重发。方法失败和 API 异常按框架记录，需区分成功评分与运行失败。

### MobileMem-Omni

固定官方 commit `919e0f545722030898cee03b263f08c8092f2ebb`：

- [官方公开 Judge prompt](https://github.com/zjunlp/MobileMem/blob/919e0f545722030898cee03b263f08c8092f2ebb/omni/eval/eval/question_answering_and_judge_prompts.txt)：原样保留全部评判规则及占位符结构。
- [Evaluator](https://github.com/zjunlp/MobileMem/blob/919e0f545722030898cee03b263f08c8092f2ebb/omni/eval/eval/evaluator.py)：按 F1＋BLEU-1 最大选择单个参考答案，平局保留首个。这两个数值仅用于内部选参考，不写入评分结果。空预测仍可进入 Judge；选择出的参考为空时不调用 Judge。
- [Raw2Locomo](https://github.com/zjunlp/MobileMem/blob/919e0f545722030898cee03b263f08c8092f2ebb/omni/eval/eval/Raw2Locomo.py)：native_evidence 中字典提取 explanation，字符串原样保留；不把 memory_id 或整段 session 当作官方 Judge 的证据输入。当前转换器没有查询图片，适配仅支持这一文本查询轨道。
- 汇总 `LLM_JUDGE` 的分母是有非空标签的记录，按官方分类名称分组；同时报告 failed_judgments、skipped_judgments，防止只看准确率忽略缺失评分。

**公开仓库缺少其导入的 eval/llm_judge.py，不能验证完整官方请求与解析流程。** 本地明确采用：user 文本 prompt、temperature=0、证据列表 JSON 序列化；提取唯一包含 label 的 JSON 对象，只接受 CORRECT/WRONG，解析失败记录为可重跑的评分错误。上述请求拼装与解析属于本仓库选择，而非已核实的官方实现。

协议特意命名为 `mmmb-omni-published-prompt-1.0`，逐题记录 `implementation_scope=published_prompt_local_parser`。结果含 label、correct、score 和 judge_response；无参考且正常跳过的标签为 null，不进入标签准确率分母；方法失败记录为 method_error、WRONG 和 0 分，并保留在分母中。仍可显式使用 `qa` 得到原通用 Judge 分数。

### 验证范围

`tests/test_official_benchmark_judges.py` 固定提示词哈希并验证请求、解析、参考与证据映射、空答案行为和分类汇总；分发测试验证新协议及缓存隔离。独立差分脚本 `tmp/official-judge-validation/verify.py` 直接执行下载的官方函数：2,000 个 M³Exam 解析案例、100 组 M³Exam 汇总、1,000 组 Omni 参考选择、100 组 Omni 汇总，以及现有 595 条 M³Exam LLM 题和 9,308 条 Omni 题的输入映射。源码 SHA256 和结果在同目录 results.json。未调用真实模型，不宣称模型输出与官方成绩一致，也不把缺失的 Omni 解析器列为差分通过项。

### 三个专用 Judge 与原通用 QA 的回归核查

PersonaMem-v2 使用官方 `extract_judge_decision` 的数值规则，包括无法解析时返回 0；省略官方函数仅用于控制台的警告输出。协议更新为 `mmmb-personamem-v2-narrow-1.1`，旧版本评分缓存会重新计算。原 `qa` 的提示词、正确性判断和准确率公式保持不变；`judge.py` 已完整恢复为原提交版本。专用评分的选题、输入哈希、调用和续跑放在 `native_runner.py`，各 native 模块直接暴露函数，不再使用包装协议类。

`tmp/official-judge-validation/verify_persona_and_qa.py` 对照官方两个 PersonaMem-v2 prompt、3,005 个解析案例与三种偏好输入的实际提示词调用；另直接加载当前 HEAD 中的原 judge.py，对正确、错误、空回答、API 失败、方法失败五种记录核对原字段和汇总值。差分无不一致。Omni JSON 解析使用本仓库实现，其官方提示词、参考选择、证据映射及可见汇总规则分别验证，不将缺失源码的解析部分冒充官方对照通过。

## Dedicated Judge output budget

Use `mmmb judge ... --scoring-protocol benchmark --judge-max-tokens 512`
to override the dedicated LLM Judge output budget (also supported for individual
native LLM protocols). Without the option, each official protocol retains its
default. The QA Judge is unchanged. Reasoning models may require a larger budget
than the M³Exam official 16-token setting; 512 is an experiment setting, not a
guarantee for every question.

Empty responses and abnormal termination (including `finish_reason=length`) are
recorded as `status=error`, counted in `failed_judgments`, and retried on resume.
They are not successful zero judgments. Records preserve `judge_response_metadata`
with finish reason, usage and requested budget. Existing native metric aggregation
is unchanged; check failure counts before interpreting aggregate scores.
The request configuration is stored with each judgment, so old transport results
or results from a different budget are not silently reused.


### 端到端适配修复（脚本协议 2.2）

- 新转换的 SMMBench MCQ 使用官方 `(A)`–`(D)` 选项标签，并保留原始答案索引。旧 bundle 的纯数字以及完整的 `数字: 对应公开选项文字` 输出会转换为字母标签后调用原生评分；不解析任意解释文本，不根据标准答案进行转换。工具计划规则不变。
- 脚本评分、专用 Judge 与分发汇总统一识别 runner 的 `metadata.method_error`、`status=error`、`error_type`。方法异常保留原因、记零并留在分母；与 Judge 请求失败分开计数。原 `evaluation/judge.py` 未修改。
- SMMBench JSON 表格保留原文本并增加公共表格格式标记；UniversalRAG 同时识别旧 bundle 的原生 header/rows JSON 结构。
- M³Exam 每个附件 content part 保留原始 `source_id`；公共图片输入函数将其与图片一起呈现。UniversalRAG 保留同一记忆中的不同图片；同一资产的重复检索仍去重，top-k 保持为证据单元数。
- 无证据时不再声称路由器选择了“不检索”。

媒体与表格单元格式已更新为 UniversalRAG checkpoint v3。旧 checkpoint 会被明确拒绝；请重新转换相关 bundle 并使用新的 checkpoint 目录建库，不能将旧索引视为包含新的表格与图片标识。已有预测仍可直接用于重新评分，无需为修复评分而重跑方法。


### 跨方法媒体输入回归

原始资产标识由 converter 提供；公共媒体函数负责与图片或生成 caption 绑定。A-Mem、MemGuide、LightMem 接入生成 caption 标识，A-Mem 的笔记身份数据额外保留图片编号列表；VimRAG 在官方工具结果格式化后追加编号，不替换其 Picture 标识或搜索工具；Oracle 复用公共图片渲染。

以上覆盖本次原始媒体与生成 caption 路径，不等同于真实模型长期记忆提炼后的正确率保证。新增跨方法测试覆盖表格文本保留、每图标识、无标识媒体兼容和 VimRAG 官方格式化函数。A-Mem 笔记版本、MemGuide/LightMem/VimRAG checkpoint 版本随输入变化更新；后续模型验证需使用新索引目录。

### SMMBench planning instructions

Newly converted tool questions render the official planner system prompt from
SMMBench `c52cf9d2b6b800784b097d6c055b0e9d8d105842`, including its candidate-tool
format and single-step `{"calls": [...]}` output. `instruction_role="system"`
marks this self-contained instruction; `instruction_includes_tools=true`
explicitly declares that the candidate list is already rendered. A system role
alone never suppresses the separate candidate list. Older bundles without this
flag may repeat the candidates; reconvert to obtain the explicit declaration.
The shared answer-input layer preserves that role for Reader adapters and joins
benchmark requirements with any existing method system constraints, rather than
replacing them. Thus the official template is preserved, but a method's combined
system message may contain its own additional constraints. VimRAG receives the same requirements as agent
task text, retaining its own system prompt and executable retrieval tools.
Other questions retain their original message layout and retrieval queries.

Published references with an empty first step still conflict with the official
single-step instruction. Keep those questions and their original scores, and
report that conflict separately: do not choose the prompt by inspecting gold,
remove empty steps, or relax scoring. Prompt alignment does not imply reproducing
the official baseline's model, retriever, or evidence organization.

Existing bundles need reconversion to receive the new instructions. A controlled
prompt comparison may reuse the same memory index and frozen retrieved evidence,
but must generate fresh predictions and retain the old artifacts separately.
