#!/usr/bin/env python3
"""Compute four benchmark-native metrics for one prediction set."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path, key: str) -> dict[str, dict]:
    return {
        str(row[key]): row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def answer_text(row: dict) -> str:
    answer = row.get("answer", {})
    return str(answer.get("text", "")) if isinstance(answer, dict) else str(answer)


def prompt_text(row: dict) -> str:
    return "\n".join(
        str(part.get("text", ""))
        for part in row.get("prompt", [])
        if part.get("type") == "text"
    )


def mean_percent(values: list[float]) -> float:
    return 100.0 * statistics.fmean(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reader", required=True)
    parser.add_argument(
        "--prediction-relative-path",
        type=Path,
        default=Path("predictions.jsonl"),
        help="prediction path below each benchmark directory",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "sources/atm_bench"))
    from memqa.utils.evaluator.evaluate_qa import deterministic_accuracy

    memeye_eval = load_module(
        "memeye_evaluator", ROOT / "sources/memeye/benchmark/evaluator.py"
    )
    h2h_eval = load_module(
        "h2h_lexical",
        ROOT / "sources/h2hmem/evaluate_metrics/Lexical_metrics/Lexical_metrics.py",
    )

    bundles = {
        "atm": ROOT / "data/unified/atm_bench",
        "mem_gallery": ROOT / "data/unified/mem_gallery",
        "memeye": ROOT / "data/unified/memeye",
        "h2hmem": ROOT / "data/unified/h2hmem",
    }
    questions = {
        key: load_jsonl(path / "questions.jsonl", "question_id")
        for key, path in bundles.items()
    }
    h2h_allowlist = {
        line.strip()
        for line in (
            ROOT / "data/derived/h2hmem/evidence_supported_v1/question_ids.txt"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    questions["h2hmem"] = {
        qid: row for qid, row in questions["h2hmem"].items()
        if qid in h2h_allowlist
    }
    predictions = {
        key: load_jsonl(
            args.run_root / key / args.prediction_relative_path, "question_id"
        )
        for key in bundles
    }
    for key in bundles:
        if not set(questions[key]).issubset(predictions[key]):
            raise RuntimeError(
                f"{key}: predictions do not cover reporting questions: "
                f"{len(predictions[key])} vs {len(questions[key])}"
            )

    atm_scores = []
    for qid, question in questions["atm"].items():
        prediction = str(predictions["atm"][qid].get("prediction", ""))
        atm_scores.append(float(deterministic_accuracy(
            answer_text(question), prediction, prompt_text(question)
        )))

    gallery_em = []
    gallery_f1 = []
    gallery_bleu1 = []
    gallery_bleu2 = []
    gallery_bleu4 = []
    for qid, question in questions["mem_gallery"].items():
        prediction = str(predictions["mem_gallery"][qid].get("prediction", ""))
        reference = answer_text(question)
        exact, _ = memeye_eval.score_open(prediction, reference)
        gallery_em.append(float(exact))
        gallery_f1.append(memeye_eval.f1_score(prediction, reference))
        gallery_bleu1.append(memeye_eval.bleu_score(
            prediction, reference, weights=(1.0, 0.0, 0.0, 0.0)
        ))
        gallery_bleu2.append(memeye_eval.bleu_score(
            prediction, reference, weights=(0.5, 0.5, 0.0, 0.0)
        ))
        gallery_bleu4.append(memeye_eval.bleu_score(
            prediction, reference, weights=(0.25, 0.25, 0.25, 0.25)
        ))

    memeye_mcq = []
    for qid, question in questions["memeye"].items():
        if question.get("subset") != "mcq":
            continue
        prediction = str(predictions["memeye"][qid].get("prediction", ""))
        valid = {str(choice["choice_id"]).upper() for choice in question["choices"]}
        memeye_mcq.append(float(
            memeye_eval.extract_choice(prediction, valid) == answer_text(question).upper()
        ))

    calculator = h2h_eval.EvaluationMetricsCalculator(
        h2h_eval.EnglishTextProcessor()
    )
    h2h_rows = []
    for qid, question in questions["h2hmem"].items():
        prediction = str(predictions["h2hmem"][qid].get("prediction", ""))
        h2h_rows.append(calculator.calculate_single_pair(
            prediction, answer_text(question)
        ))

    result = {
        "protocol": "graphmemix-native-metrics-1.0",
        "reader": args.reader,
        "run_root": str(args.run_root),
        "questions": {key: len(value) for key, value in questions.items()},
        "results_percent": {
            "atm_em": mean_percent(atm_scores),
            "gallery_em": mean_percent(gallery_em),
            "gallery_f1": mean_percent(gallery_f1),
            "gallery_bleu1": mean_percent(gallery_bleu1),
            "gallery_bleu2": mean_percent(gallery_bleu2),
            "gallery_bleu4": mean_percent(gallery_bleu4),
            "memeye_mcq_em": mean_percent(memeye_mcq),
            "h2h_precision": mean_percent([row["precision"] for row in h2h_rows]),
            "h2h_recall": mean_percent([row["recall"] for row in h2h_rows]),
            "h2h_f1": mean_percent([row["f1"] for row in h2h_rows]),
            "h2h_bleu1": mean_percent([row["bleu1"] for row in h2h_rows]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
