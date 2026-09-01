#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ATM_ROOT="${ATM_AGENT_ROOT:-${WORKSPACE_ROOT}/runs/official/ATM-Bench-main}"
AUTH_ROOT="${DEEPSEEK_AUTH_ROOT:-/tmp/atm-deepseek-anchor-auth}"
QID="${1:?usage: $0 QUESTION_ID}"

if [[ -z "${OPENAI_COMPATIBLE_API_KEY:-}" ]]; then
  export OPENAI_COMPATIBLE_API_KEY="$(python3 - "${AUTH_ROOT}/data/opencode/auth.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))["openai-compatible"]
key = value.get("key") if isinstance(value, dict) else None
if not key:
    raise SystemExit("missing openai-compatible key")
print(key, end="")
PY
)"
fi

export AGSYS_RUN_TAG="${DEEPSEEK_GRAPH_RUN_TAG:-atm-hard-deepseek-evidence-graph-pi-v3-compact-v1}"
export AGSYS_MEMORY_MODE="sgm"
export AGSYS_SKIP_EVAL="1"
export AGSYS_PI_GRAPH_MODE="1"
export AGSYS_SYSTEM_PROMPT="${DEEPSEEK_GRAPH_SYSTEM_PROMPT:-${WORKSPACE_ROOT}/configs/release/prompts/atm_hard_evidence_graph_v3_compact.txt}"
export AGSYS_PI_BIN="${DEEPSEEK_PI_BIN:-/tmp/atm-pi-runner/node_modules/.bin/pi}"
export AGSYS_PI_TIMEOUT_S="${DEEPSEEK_PI_TIMEOUT:-1800}"
export AGSYS_PI_SANDBOX="bwrap"
export AGSYS_PI_OFFLINE="1"
export AGSYS_PI_TOOLS="${DEEPSEEK_PI_TOOLS:-read,bash,grep,find,ls}"
export AGSYS_PI_THINKING="${DEEPSEEK_PI_THINKING:-medium}"
export PI_OPENAI_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com/v1}"
export PI_OPENAI_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash}"
export PI_OPENAI_CONTEXT_WINDOW="${DEEPSEEK_CONTEXT_WINDOW:-128000}"
export PI_OPENAI_MAX_TOKENS="${DEEPSEEK_MAX_TOKENS:-16384}"

cd "${ATM_ROOT}"
exec bash agent_systems/scripts/pi/run_pi_openai_compatible.sh "${QID}"
