# GraphMemix benchmark and release guide

This document describes the frozen four-benchmark suite, the reusable artifact
layout, the resumable execution stages, and the separate ATM-Bench-Hard
extension.

## Benchmark suite

| key | benchmark | evaluated questions |
|---|---|---:|
| `atm` | ATM-Bench, default + hard | 1,044 |
| `mem_gallery` | Mem-Gallery | 1,711 |
| `memeye` | MemEye | 1,855 |
| `h2hmem` | H2HMem evidence-supported track | 1,982 |

Two complete backbone configurations are available. A configuration controls
the node verifier, ECV, and reader together:

- `qwen3vl8b`: `Qwen/Qwen3-VL-8B-Instruct`
- `gemma4_12b`: `google/gemma-4-12B-it`

The frozen execution contract is stored in
`configs/release/graphmemix.json`. It contains candidate-generation, graph,
solver, context, and evaluation parameters.

## Artifact layout

Large datasets, model weights, checkpoints, captions, and predictions are not
committed. Place reusable artifacts below one root, by default
`artifacts/graphmemix/`:

```text
artifacts/graphmemix/
├── captions/h2hmem.jsonl
├── checkpoints/{atm,mem_gallery,memeye,h2hmem}/
├── relation_store/{atm,mem_gallery,memeye,h2hmem}.jsonl
├── source_priors/{atm,mem_gallery,memeye,h2hmem}.jsonl
└── vendor/memix-core/
```

The checkpoint, source-prior, and RelationStore stages create their own
missing outputs. The frozen calibration artifact is committed at
`configs/release/calibration/shared_dev.json` and should not be refit during a
benchmark run.

## Commands

Validate the configuration:

```bash
python scripts/run_graphmemix_release.py validate-config
```

Inspect one complete cell without calling a model:

```bash
python scripts/run_graphmemix_release.py benchmark \
  --dataset atm --backbone qwen3vl8b \
  --base-url http://127.0.0.1:8000/v1 \
  --judge-base-url https://YOUR-ENDPOINT/v1 \
  --dry-run
```

Run a cell or resume it from a named stage:

```bash
python scripts/run_graphmemix_release.py benchmark \
  --dataset memeye --backbone gemma4_12b \
  --base-url http://127.0.0.1:8000/v1 \
  --judge-base-url https://YOUR-ENDPOINT/v1 \
  --from-stage node --to-stage judge
```

The stages are `checkpoint`, `priors`, `relations`, `node`, `ecv`, `select`,
`prune`, `reader`, and `judge`. Model-stage artifacts record their input
contracts and refuse to resume from incompatible rows.

## ATM-Bench-Hard evidence-graph extension

The separate ATM-Hard pipeline uses Pi and DeepSeek V4 Flash to build a
query-conditioned evidence graph. It projects the agent's unrestricted graph
onto memory nodes and applies the GraphMemix deterministic 1-swap selector,
whose forest for each proposed node set is solved exactly by Kruskal. Its v3
node prize combines graph confidence, inverse-hub-size structural coverage,
a bounded semantic source-prior rank bonus, and a relation-agnostic edge-
incidence term for list membership. The only routing signal is the
canonical task type (`list_recall`, `number`, or `open_end`), which selects
frozen budgets and prize weights. The selector never reads question text,
answers, gold evidence, or scenario names. For numeric graphs only, it can
supply a bounded timeline for unsupported temporal aggregates and an
organizational ledger when same-role temporal structures occupy distinct graph
branches. Explicit answer/claim nodes are excluded, although graph-generated
component labels and time ranges remain model-derived organizational evidence.
GPT-5.6 Sol Medium is used as one uniform representation-aware reader.

- `configs/release/atm_hard_graph_proof.json`
- `configs/release/prompts/atm_hard_evidence_graph_v3_compact.txt`

Inspect the extension without API calls:

```bash
python scripts/run_atm_hard_graph_proof_release.py \
  --atm-agent-root vendor/ATM-Bench \
  --checkpoint-root artifacts/graphmemix/checkpoints/atm \
  --source-priors artifacts/graphmemix/source_priors/atm.jsonl \
  --memix-repo artifacts/graphmemix/vendor/memix-core \
  --dry-run
```

The same command without `--dry-run` executes graph construction, GraphMemix
forest selection, and final answering. Add the `judge` and `score` stages plus
an OpenAI-compatible GPT-5-mini endpoint and the official ATM ground-truth
file to produce the final QS from that same prediction file:

```bash
python scripts/run_atm_hard_graph_proof_release.py \
  --atm-agent-root vendor/ATM-Bench \
  --checkpoint-root artifacts/graphmemix/checkpoints/atm \
  --source-priors artifacts/graphmemix/source_priors/atm.jsonl \
  --memix-repo artifacts/graphmemix/vendor/memix-core \
  --judge-base-url https://YOUR-JUDGE-ENDPOINT/v1 \
  --official-ground-truth /path/to/atm_hard_ground_truth.json \
  --to-stage score
```

Reader resume is accepted only when the complete retrieval row hash matches,
including evidence order and representation actions. Judge resume is likewise
bound to the complete prediction row. Legacy ATM-Hard scores combined
incompatible reader contracts or used the retired rule router; they are not
attributed to this protocol. The current hash-bound v3 full rerun scored 57.95
QS (list recall 74.70, number 66.67, open end 38.46) with GPT-5-mini and the
official open-ended judge prompt. Earlier 53.80/57.02 runs are retained only as
variance context. Because this extension was developed on the same 31 public hard
questions, the score is a development-set result, not an independent held-out
measurement.

## Evaluation and repository integrity

- GPT-5-mini is an evaluator and is excluded from method inference cost.
- Raw benchmark assets follow their upstream licenses and are not redistributed.
- API keys must be supplied through environment variables.
- Complete runs require the expected number of unique question IDs, non-empty
  predictions, aligned evidence, and compatible model/judge contracts.
- Run `pytest -q tests` rather than unrestricted `pytest`; vendored projects
  may contain unrelated dependency-heavy tests.

The distributable tree is defined by `release/repository_allowlist.txt` and can
be checked with:

```bash
python scripts/audit_graphmemix_release.py
```
