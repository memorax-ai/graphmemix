#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EVAL_ROOT="${AGENT_EVAL_ROOT:-${ROOT}/runs/official/ATM-Bench-main/agent_systems/eval_root_sgm}"
TAG="${DEEPSEEK_GRAPH_RUN_TAG:-atm-hard-deepseek-evidence-graph-pi-v3-compact-v1}"
MODEL_TAG="openai-compatible_${DEEPSEEK_MODEL:-deepseek-v4-flash}"
RUN_ROOT="${AGENT_ANSWERS_ROOT:-${EVAL_ROOT}/runs/${TAG}/pi/${MODEL_TAG}}"
WORKERS="${GRAPH_WORKERS:-2}"

run_one() {
  local qid="$1"
  local manifest="${RUN_ROOT}/${qid}/workspace-output/graph_manifest.json"
  if [[ -s "${manifest}" ]]; then
    echo "SKIP ${qid}"
    return 0
  fi
  bash "${ROOT}/scripts/run_atm_hard_deepseek_evidence_graph_pi_one.sh" "${qid}"
}
export -f run_one
export ROOT RUN_ROOT

mapfile -t QIDS < "${EVAL_ROOT}/question_ids.txt"
printf '%s\n' "${QIDS[@]}" | xargs -r -n 1 -P "${WORKERS}" bash -c 'run_one "$1"' _

python3 - "${RUN_ROOT}" "${EVAL_ROOT}/question_ids.txt" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
qids = [line.strip() for line in Path(sys.argv[2]).read_text().splitlines() if line.strip()]
issues = []
totals = {"nodes": 0, "edges": 0}
for qid in qids:
    graph = root / qid / "workspace-output"
    try:
        manifest = json.loads((graph / "graph_manifest.json").read_text(encoding="utf-8"))
        nodes = [json.loads(line) for line in (graph / "graph_nodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        edges = [json.loads(line) for line in (graph / "graph_edges.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        node_ids = {str(row["node_id"]) for row in nodes}
        if str(manifest.get("question_id")) != qid:
            raise ValueError("question_id mismatch")
        if int(manifest.get("node_count", -1)) != len(nodes) or int(manifest.get("edge_count", -1)) != len(edges):
            raise ValueError("manifest count mismatch")
        if any(str(row.get("source")) not in node_ids or str(row.get("target")) not in node_ids for row in edges):
            raise ValueError("unknown edge endpoint")
        totals["nodes"] += len(nodes)
        totals["edges"] += len(edges)
    except Exception as exc:
        issues.append({"question_id": qid, "error": f"{type(exc).__name__}: {exc}"})
summary = {"questions": len(qids), "valid": len(qids) - len(issues), **totals, "issues": issues}
print(json.dumps(summary, ensure_ascii=False, indent=2))
if issues:
    raise SystemExit(1)
PY
