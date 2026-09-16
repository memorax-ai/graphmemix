#!/usr/bin/env python3
"""Score one additional benchmark bundle using deterministic rules only."""
import argparse
import json
from pathlib import Path
from mm_memory_bench.evaluation.native import BENCHMARKS, score_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', choices=BENCHMARKS)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--question-ids', type=Path)
    args = parser.parse_args()
    ids = args.question_ids.read_text().splitlines() if args.question_ids else None
    if ids is not None:
        ids = [x.strip() for x in ids if x.strip()]
    print(json.dumps(score_predictions(args.bundle, args.predictions, args.output,
                                      benchmark=args.benchmark, question_ids=ids), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
