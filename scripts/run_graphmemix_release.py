#!/usr/bin/env python3
"""Portable orchestration for the frozen GraphMemix benchmark suite.

The runner delegates each stage to a focused script, removes machine-specific
paths, and makes the artifact contract explicit. Use ``--dry-run`` to inspect
every command.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/release/graphmemix.json"
STAGES = ("checkpoint", "priors", "relations", "node", "ecv", "select", "prune", "reader", "judge")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_jsonl(path: Path, expected: int, *, predictions: bool = False) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    ids = [str(row["question_id"]) for row in rows]
    if len(rows) != expected or len(set(ids)) != expected:
        raise RuntimeError(f"{path}: expected {expected} unique questions, found {len(rows)}/{len(set(ids))}")
    if predictions:
        invalid = [
            row["question_id"] for row in rows
            if not str(row.get("prediction", "")).strip()
            or (row.get("metadata") or {}).get("status") == "error"
        ]
        if invalid:
            raise RuntimeError(f"{path}: {len(invalid)} empty/error predictions")


def command_text(command: list[str]) -> str:
    return shlex.join(command)


class Runner:
    def __init__(self, *, dry_run: bool) -> None:
        self.dry_run = dry_run

    def run(self, command: list[str], *, env: dict[str, str] | None = None) -> None:
        print(f"+ {command_text(command)}", flush=True)
        if self.dry_run:
            return
        merged = os.environ.copy()
        merged["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT / "src"), str(ROOT / "scripts"), merged.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        if env:
            merged.update(env)
        subprocess.run(command, cwd=ROOT, env=merged, check=True)


def script(name: str) -> str:
    return str(ROOT / "scripts" / name)


def artifact_path(layout: dict[str, str], key: str, root: Path, dataset: str) -> Path:
    return root / layout[key].format(dataset=dataset)


def benchmark_commands(args: argparse.Namespace, config: dict[str, Any]) -> int:
    suite = config["benchmark_suite"]
    dataset_cfg = suite["datasets"][args.dataset]
    backbone_cfg = suite["backbones"][args.backbone]
    expected = int(dataset_cfg["expected_questions"])
    bundle = args.data_root / dataset_cfg["bundle"]
    layout = config["artifact_layout"]
    checkpoint = artifact_path(layout, "checkpoint", args.artifact_root, args.dataset)
    priors = artifact_path(layout, "source_priors", args.artifact_root, args.dataset)
    relations = artifact_path(layout, "relation_store", args.artifact_root, args.dataset)
    calibration = ROOT / suite["calibration_file"]
    caption_sidecar = artifact_path(layout, "caption_sidecar", args.artifact_root, args.dataset)
    memix_repo = args.memix_repo or artifact_path(layout, "memix_repo", args.artifact_root, args.dataset)
    run_root = args.run_root / args.backbone / args.dataset
    node = run_root / "node_scores.jsonl"
    ecv = run_root / "ecv_scores.jsonl"
    fixed = run_root / "selection/original_ecv_edge_p1.jsonl"
    pruned = run_root / "pruning/root0.12_node0.jsonl"
    predictions = run_root / "reader/predictions.jsonl"
    judgments = run_root / "judge/judgments.jsonl"
    model = args.model or backbone_cfg["model"]
    runner = Runner(dry_run=args.dry_run)

    start = STAGES.index(args.from_stage)
    stop = STAGES.index(args.to_stage)
    if start > stop:
        raise ValueError("--from-stage must not come after --to-stage")

    def enabled(name: str) -> bool:
        return start <= STAGES.index(name) <= stop

    if enabled("checkpoint"):
        command = [
            sys.executable, script("build_memix_checkpoints.py"),
            "--bundle", str(bundle), "--checkpoint-root", str(checkpoint),
            "--embedding-model", suite["embedding_model"], "--memix-repo", str(memix_repo),
            "--memory-view", "derived", "--video-frames", "8",
        ]
        if args.dataset == "h2hmem":
            command.extend(["--caption-sidecar", str(caption_sidecar)])
        runner.run(command)

    if enabled("priors"):
        runner.run([
            sys.executable, script("run_memix_main_track_retrieval.py"),
            "--bundle", str(bundle), "--checkpoint-root", str(checkpoint),
            "--embedding-model", suite["embedding_model"], "--memix-repo", str(memix_repo),
            "--output", str(run_root / "atomic_retrieval.jsonl"),
            "--source-priors", str(priors), "--source-top-k", str(suite["source_top_k"]),
            "--processes", str(args.local_processes),
        ])
        if not args.dry_run:
            validate_jsonl(priors, expected)

    if enabled("relations"):
        runner.run([
            sys.executable, script("build_graphmemix_relation_store.py"),
            "--bundle", str(bundle), "--checkpoint-root", str(checkpoint),
            "--output", str(relations), "--max-semantic-k", str(suite["semantic_k"]),
        ])

    if enabled("node"):
        runner.run([
            sys.executable, script("score_graphmemix_verifier.py"),
            "--bundle", str(bundle), "--source-priors", str(priors),
            "--relation-store", str(relations), "--graph-mode", "full",
            "--source-top-l", str(suite["source_top_l"]),
            "--candidate-limit", str(suite["candidate_limit"]),
            "--semantic-k", str(suite["semantic_k"]), "--output", str(node),
            "--base-url", args.base_url, "--model", model,
            "--concurrency", str(args.concurrency), "--all-questions",
            "--max-model-len", str(suite["max_model_len"]),
            "--reasoning-effort", str(backbone_cfg["reasoning_effort"]),
            "--snippet-chars", str(suite["snippet_chars"]),
            "--ocr-chars", str(suite["verifier_ocr_chars"]),
        ])
        if not args.dry_run:
            validate_jsonl(node, expected)

    if enabled("ecv"):
        runner.run([
            sys.executable, script("score_graphmemix_ecv.py"),
            "--bundle", str(bundle), "--candidate-cache", str(node),
            "--relation-store", str(relations), "--source-priors", str(priors),
            "--calibration", str(calibration), "--output", str(ecv),
            "--base-url", args.base_url, "--model", model,
            "--concurrency", str(args.concurrency),
            "--anchor-count", str(suite["anchor_count"]),
            "--snippet-chars", str(suite["snippet_chars"]),
            "--ocr-chars", str(suite["verifier_ocr_chars"]),
            "--max-model-len", str(suite["max_model_len"]),
            "--max-output-tokens", str(suite["verifier_max_output_tokens"]),
            "--reasoning-effort", str(backbone_cfg["reasoning_effort"]),
            "--candidate-batch-size", "0", "--edge-only", "--split", "all",
        ])
        if not args.dry_run:
            validate_jsonl(ecv, expected)

    if enabled("select"):
        runner.run([
            sys.executable, script("evaluate_graphmemix_ecv.py"),
            "--bundle", str(bundle), "--ecv-scores", str(ecv),
            "--original-verifier", str(node), "--source-priors", str(priors),
            "--relation-store", str(relations), "--calibration", str(calibration),
            "--output-dir", str(run_root / "selection"),
            "--edge-weight", str(suite["edge_weight"]),
            "--root-cost", str(suite["proposal_root_cost"]),
            "--edge-power", str(suite["edge_power"]), "--k", str(suite["reader_k"]),
        ])
        if not args.dry_run:
            validate_jsonl(fixed, expected)

    if enabled("prune"):
        runner.run([
            sys.executable, script("evaluate_graphmemix_exact_pruning.py"),
            "--bundle", str(bundle), "--scores", str(node), "--ecv-scores", str(ecv),
            "--source-priors", str(priors), "--retrieval", str(fixed),
            "--calibration", str(calibration), "--output-dir", str(run_root / "pruning"),
            "--edge-weight", str(suite["edge_weight"]),
            "--root-cost", str(suite["pruning_root_cost"]),
            "--edge-power", str(suite["edge_power"]),
        ])
        if not args.dry_run:
            validate_jsonl(pruned, expected)

    if enabled("reader"):
        runner.run([
            sys.executable, script("run_memix_answers_from_retrieval.py"),
            "--bundle", str(bundle), "--retrieval", str(pruned),
            "--checkpoint-root", str(checkpoint), "--memix-repo", str(memix_repo),
            "--output", str(predictions), "--base-url", args.base_url, "--model", model,
            "--concurrency", str(args.concurrency), "--max-model-len", str(suite["max_model_len"]),
            "--max-output-tokens", str(suite["reader_max_output_tokens"]),
            "--reasoning-effort", str(backbone_cfg["reasoning_effort"]),
            "--overflow-image-max-edge", str(suite["overflow_image_max_edge"]),
            "--reader-ocr-budget-chars", str(suite["reader_ocr_chars"]),
            "--allow-variable-retrieval", "--resume", "--retry-errors",
            "--record-errors", "--progress-interval", "100",
        ])
        if not args.dry_run:
            validate_jsonl(predictions, expected, predictions=True)

    if enabled("judge"):
        if not args.judge_base_url:
            raise ValueError("--judge-base-url is required for the judge stage")
        runner.run([
            sys.executable, "-m", "mm_memory_bench.cli", "judge",
            str(bundle), str(predictions), "--output", str(judgments),
            "--model", args.judge_model or suite["judge"]["model"],
            "--base-url", args.judge_base_url,
            "--api-key-env", args.judge_api_key_env,
            "--timeout-seconds", "1200", "--concurrency", str(args.judge_concurrency),
        ])
        if not args.dry_run:
            validate_jsonl(judgments, expected)

    print(json.dumps({
        "status": "planned" if args.dry_run else "complete",
        "dataset": args.dataset, "backbone": args.backbone,
        "model": model, "run_root": str(run_root),
    }, indent=2))
    return 0


def validate_config(config: dict[str, Any]) -> int:
    suite = config["benchmark_suite"]
    assert set(suite["datasets"]) == {"atm", "mem_gallery", "memeye", "h2hmem"}
    assert set(suite["backbones"]) == {"qwen3vl8b", "gemma4_12b"}
    assert suite["source_top_l"] <= suite["candidate_limit"]
    assert suite["reader_k"] <= suite["candidate_limit"]
    calibration = ROOT / suite["calibration_file"]
    assert calibration.is_file(), calibration
    print(json.dumps({
        "status": "valid", "config": str(DEFAULT_CONFIG),
        "datasets": suite["datasets"], "backbones": suite["backbones"],
    }, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-config")
    benchmark = sub.add_parser("benchmark", help="Run one dataset/backbone benchmark cell.")
    benchmark.add_argument("--dataset", required=True, choices=("atm", "mem_gallery", "memeye", "h2hmem"))
    benchmark.add_argument("--backbone", required=True, choices=("qwen3vl8b", "gemma4_12b"))
    benchmark.add_argument("--data-root", type=Path, default=ROOT / "data/unified")
    benchmark.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts/graphmemix")
    benchmark.add_argument("--run-root", type=Path, default=ROOT / "runs/release/graphmemix")
    benchmark.add_argument("--memix-repo", type=Path)
    benchmark.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    benchmark.add_argument("--model")
    benchmark.add_argument("--concurrency", type=int, default=4)
    benchmark.add_argument("--local-processes", type=int, default=1)
    benchmark.add_argument("--judge-base-url")
    benchmark.add_argument("--judge-model")
    benchmark.add_argument("--judge-api-key-env", default="OPENAI_API_KEY")
    benchmark.add_argument("--judge-concurrency", type=int, default=2)
    benchmark.add_argument("--from-stage", choices=STAGES, default="checkpoint")
    benchmark.add_argument("--to-stage", choices=STAGES, default="judge")
    benchmark.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_json(args.config)
    if args.command == "validate-config":
        return validate_config(config)
    return benchmark_commands(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
