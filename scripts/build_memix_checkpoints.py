#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mm_memory_bench.runner.benchmark import _resolved_memory, _visible_context
from mm_memory_bench.methods import (
    ConcreteMemixMethod,
    FaissFlatIPIndex,
    GMEQwen2VLEmbedder,
    GenerationConfig,
)
from mm_memory_bench.benchmarks.reader import BundleReader


class UnusedAnswerModel:
    def complete(self, messages, *, tools=None, response_format=None):
        raise RuntimeError("checkpoint-only run must not call the answer model")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or resume Memix context checkpoints without answering questions."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--memix-repo", type=Path, required=True)
    parser.add_argument("--caption-sidecar", type=Path)
    parser.add_argument("--memory-view", default="raw_derived")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--video-frames", type=int, default=8)
    args = parser.parse_args()

    reader = BundleReader(args.bundle, caption_sidecar=args.caption_sidecar)
    embedder = GMEQwen2VLEmbedder(args.embedding_model)
    method = ConcreteMemixMethod(
        GenerationConfig(top_k=10),
        answer_model=UnusedAnswerModel(),
        embedder=embedder,
        index=FaissFlatIPIndex(embedder.dimension),
        memix_repo=args.memix_repo,
        checkpoint_dir=args.checkpoint_root,
        checkpoint_interval=args.checkpoint_interval,
        ingest_batch_size=args.batch_size,
        source_top_k=200,
        video_frames=args.video_frames,
    )
    started = time.perf_counter()
    batches = list(reader.iter_context_batches())
    for index, batch in enumerate(batches, 1):
        context_started = time.perf_counter()
        method.begin_context(_visible_context(batch.context))
        try:
            for memory in batch.memories:
                method.ingest(
                    _resolved_memory(reader, memory, memory_view=args.memory_view)
                )
        except BaseException:
            method.abort_context()
            raise
        else:
            method.end_context()
        print(json.dumps({
            "event": "context_checkpointed",
            "context": index,
            "contexts": len(batches),
            "context_id": batch.context["context_id"],
            "memories": len(batch.memories),
            "elapsed_seconds": time.perf_counter() - context_started,
        }), flush=True)
    print(json.dumps({
        "event": "complete",
        "contexts": len(batches),
        "checkpoint_root": str(args.checkpoint_root),
        "elapsed_seconds": time.perf_counter() - started,
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
