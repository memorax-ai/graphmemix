#!/usr/bin/env python3
"""Run the public ATM-Hard graph-proof extension without gold-answer inputs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("graph", "select", "reader", "judge", "score")


def run(command: list[str], *, dry_run: bool, env: dict[str, str] | None = None, check: bool = True) -> int:
    print("+ " + shlex.join(command), flush=True)
    if dry_run:
        return 0
    merged = os.environ.copy()
    merged["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"), merged.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    if env:
        merged.update(env)
    return subprocess.run(command, cwd=ROOT, env=merged, check=check).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atm-agent-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=ROOT / "data/unified/atm_bench")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/release/atm_hard_graph_proof.json"
    )
    parser.add_argument(
        "--run-root", type=Path,
        default=ROOT / "runs/release/atm_hard_graphmemix_forest_v3",
    )
    parser.add_argument("--graph-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--memix-repo", type=Path, required=True)
    parser.add_argument("--source-priors", type=Path, required=True)
    parser.add_argument("--deepseek-base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--deepseek-model", default="deepseek-v4-flash")
    parser.add_argument("--reader-base-url", default="http://127.0.0.1:18956/v1")
    parser.add_argument("--reader-model", default="gpt-5.6-sol")
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--judge-concurrency", type=int, default=2)
    parser.add_argument("--official-repo", type=Path)
    parser.add_argument("--official-ground-truth", type=Path)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--from-stage", choices=STAGES, default="graph")
    parser.add_argument("--to-stage", choices=STAGES, default="reader")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    selector = config["selector"]
    budgets = selector["budgets"]
    media_limits = selector["high_media_limits"]
    reader = config["reader"]

    eval_root = args.atm_agent_root / "agent_systems/eval_root_sgm"
    graph_root = args.graph_root or (
        eval_root / "runs/atm-hard-deepseek-evidence-graph-pi-v3-compact-v1/pi"
        / f"openai-compatible_{args.deepseek_model}"
    )
    canonical_questions = args.bundle / "questions.jsonl"
    retrieval = args.run_root / "retrieval.jsonl"
    predictions = args.run_root / "predictions.jsonl"
    judgments = args.run_root / "judgments.jsonl"
    score = args.run_root / "official_qs.json"
    start, stop = STAGES.index(args.from_stage), STAGES.index(args.to_stage)

    def enabled(stage: str) -> bool:
        return start <= STAGES.index(stage) <= stop

    if enabled("graph"):
        run(
            ["bash", str(ROOT / "scripts/run_atm_hard_deepseek_evidence_graph_pi_batch.sh")],
            dry_run=args.dry_run,
            env={
                "ATM_AGENT_ROOT": str(args.atm_agent_root),
                "AGENT_EVAL_ROOT": str(eval_root),
                "DEEPSEEK_BASE_URL": args.deepseek_base_url,
                "DEEPSEEK_MODEL": args.deepseek_model,
                "DEEPSEEK_GRAPH_SYSTEM_PROMPT": str(
                    ROOT / "configs/release/prompts/atm_hard_evidence_graph_v3_compact.txt"
                ),
                "GRAPH_WORKERS": str(args.concurrency),
            },
        )

    if enabled("select"):
        run([
            sys.executable, str(ROOT / "scripts/select_atm_hard_graphmemix_forest.py"),
            "--graph-root", str(graph_root), "--questions-jsonl", str(canonical_questions),
            "--bundle", str(args.bundle),
            "--source-priors", str(args.source_priors),
            "--output", str(retrieval), "--audit-output", str(args.run_root / "selection_audit.json"),
            "--list-k", str(budgets["list_recall"]),
            "--number-k", str(budgets["number"]),
            "--open-k", str(budgets["open_end"]),
            "--list-high-media", str(media_limits["list_recall"]),
            "--number-high-media", str(media_limits["number"]),
            "--open-high-media", str(media_limits["open_end"]),
            "--list-prior-limit", str(selector["source_prior_limits"]["list_recall"]),
            "--number-prior-limit", str(selector["source_prior_limits"]["number"]),
            "--open-prior-limit", str(selector["source_prior_limits"]["open_end"]),
            "--list-prior-weight", str(selector["source_prior_weights"]["list_recall"]),
            "--number-prior-weight", str(selector["source_prior_weights"]["number"]),
            "--open-prior-weight", str(selector["source_prior_weights"]["open_end"]),
            "--list-coverage-weight", str(selector["structural_coverage_weights"]["list_recall"]),
            "--number-coverage-weight", str(selector["structural_coverage_weights"]["number"]),
            "--open-coverage-weight", str(selector["structural_coverage_weights"]["open_end"]),
            "--list-incidence-weight", str(selector["edge_incidence_weights"]["list_recall"]),
            "--number-incidence-weight", str(selector["edge_incidence_weights"]["number"]),
            "--open-incidence-weight", str(selector["edge_incidence_weights"]["open_end"]),
            "--edge-weight", str(selector["edge_weight"]),
            "--root-cost", str(selector["root_cost"]),
        ], dry_run=args.dry_run)

    if enabled("reader"):
        run([
            sys.executable, str(ROOT / "scripts/run_memix_answers_from_retrieval.py"),
            "--bundle", str(args.bundle), "--retrieval", str(retrieval),
            "--checkpoint-root", str(args.checkpoint_root), "--memix-repo", str(args.memix_repo),
            "--output", str(predictions), "--base-url", args.reader_base_url,
            "--model", args.reader_model, "--concurrency", str(args.concurrency),
            "--max-model-len", str(reader["max_model_len"]),
            "--max-output-tokens", str(reader["max_output_tokens"]),
            "--reasoning-effort", str(reader["reasoning_effort"]),
            "--allow-variable-retrieval", "--respect-selected-actions",
            "--video-frames-override", str(reader["video_frames"]),
            "--reader-ocr-budget-chars", str(reader["ocr_budget_chars"]),
            "--resume", "--retry-errors",
            "--record-errors", "--progress-interval", "10",
        ], dry_run=args.dry_run)

    if enabled("judge"):
        if not args.judge_base_url or args.official_ground_truth is None:
            raise ValueError(
                "--judge-base-url and --official-ground-truth are required for the judge stage"
            )
        run([
            sys.executable, str(ROOT / "scripts/judge_atm_official_prompt.py"),
            "--official-repo", str(args.official_repo or args.atm_agent_root),
            "--ground-truth", str(args.official_ground_truth),
            "--predictions", str(predictions), "--output", str(judgments),
            "--model", args.judge_model or str(config["judge"]["model"]),
            "--base-url", args.judge_base_url,
            "--api-key-env", args.judge_api_key_env,
            "--concurrency", str(args.judge_concurrency),
        ], dry_run=args.dry_run)

    if enabled("score"):
        if args.official_ground_truth is None:
            raise ValueError("--official-ground-truth is required for the score stage")
        run([
            sys.executable, str(ROOT / "scripts/score_atm_hard_official_qs_from_frozen.py"),
            "--official-repo", str(args.official_repo or args.atm_agent_root),
            "--ground-truth", str(args.official_ground_truth),
            "--predictions", str(predictions),
            "--open-judgments", str(judgments),
            "--open-judge-label", args.judge_model or str(config["judge"]["model"]),
            "--require-bound-judgments",
            "--output", str(score),
        ], dry_run=args.dry_run)

    print(json.dumps({
        "status": "planned" if args.dry_run else "complete",
        "track": str(config["track"]), "run_root": str(args.run_root),
        "graph_root": str(graph_root), "predictions": str(predictions),
        "judgments": str(judgments), "score": str(score),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
