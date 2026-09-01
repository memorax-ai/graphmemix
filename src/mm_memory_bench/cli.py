from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import SchemaError, validate_bundle
from .registry import CONVERTERS, convert


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mmmb")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="normalize one official snapshot")
    convert_parser.add_argument("benchmark", choices=sorted(CONVERTERS))
    convert_parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    convert_parser.add_argument("--output-root", type=Path, default=Path("data/unified"))
    convert_parser.add_argument("--overwrite", action="store_true")

    all_parser = subparsers.add_parser("convert-all", help="normalize every available snapshot")
    all_parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    all_parser.add_argument("--output-root", type=Path, default=Path("data/unified"))
    all_parser.add_argument("--overwrite", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="validate one canonical bundle")
    validate_parser.add_argument("bundle", type=Path)
    validate_parser.add_argument("--check-assets", action="store_true")

    run_parser = subparsers.add_parser(
        "run-method", help="run one concrete memory method on a canonical bundle"
    )
    run_parser.add_argument(
        "method",
        choices=["amem", "lightmem", "memguide", "memix", "universalrag", "vimrag"],
    )
    run_parser.add_argument("bundle", type=Path)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--subset")
    run_parser.add_argument("--split")
    run_parser.add_argument("--task-subcategory")
    run_parser.add_argument("--question-ids", type=Path)
    run_parser.add_argument(
        "--memory-ids",
        type=Path,
        help="newline-delimited canonical memory-ID allowlist for --digest-only",
    )
    run_parser.add_argument("--base-url", default="http://127.0.0.1:8091/v1")
    run_parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    run_parser.add_argument("--memory-base-url")
    run_parser.add_argument("--memory-model")
    run_parser.add_argument("--caption-base-url")
    run_parser.add_argument("--caption-model")
    run_parser.add_argument("--caption-cache-dir", type=Path)
    run_parser.add_argument(
        "--caption-sidecar",
        type=Path,
        help="benchmark-level asset caption JSONL merged by the harness",
    )
    run_parser.add_argument("--memory-max-output-tokens", type=int, default=1000)
    run_parser.add_argument("--memory-workers", type=int, default=1)
    run_parser.add_argument("--checkpoint-dir", type=Path)
    run_parser.add_argument("--checkpoint-interval", type=int, default=100)
    run_parser.add_argument("--memory-retry-count", type=int, default=3)
    run_parser.add_argument("--memory-retry-backoff", type=float, default=5.0)
    run_parser.add_argument("--max-consecutive-fallbacks", type=int, default=10)
    run_parser.add_argument(
        "--memory-view", choices=["raw", "derived", "raw_derived"], default="raw"
    )
    run_parser.add_argument("--embedding-model", type=Path)
    run_parser.add_argument("--memguide-embedding-model", default="nvidia/NV-Embed-v2")
    run_parser.add_argument(
        "--memguide-question-mode", choices=["fixed", "llm"], default="fixed"
    )
    run_parser.add_argument("--memguide-question-workers", type=int, default=8)
    run_parser.add_argument("--memguide-question-max-output-tokens", type=int, default=64)
    run_parser.add_argument("--lightmem-official-repo", type=Path, default=Path("sources/lightmem"))
    run_parser.add_argument("--lightmem-llmlingua-model", default="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank")
    run_parser.add_argument("--lightmem-embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    run_parser.add_argument("--lightmem-device", default="cuda")
    run_parser.add_argument("--vimrag-official-repo", type=Path, default=Path("sources/vimrag"))
    run_parser.add_argument(
        "--vimrag-embedding-backend", choices=["qwen3vl", "gme"], default="qwen3vl"
    )
    run_parser.add_argument("--vimrag-max-steps", type=int, default=20)
    run_parser.add_argument("--vimrag-video-frames", type=int, default=8)
    run_parser.add_argument("--memix-repo", type=Path, default=Path("vendor/memix-core"))
    run_parser.add_argument("--memix-embedding-model", default="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct")
    run_parser.add_argument("--memix-source-top-k", type=int, default=200)
    run_parser.add_argument("--memix-video-frames", type=int, default=8)
    run_parser.add_argument(
        "--memix-media-atomic-limit",
        type=int,
        default=0,
        help="append up to this many text-field units for each image/video source",
    )
    run_parser.add_argument("--memix-use-llm-verify", action="store_true")
    run_parser.add_argument("--memix-drop-reader-ocr", action="store_true")
    run_parser.add_argument(
        "--memix-verifier-input-mode",
        choices=["current", "query_caption", "full_vl"],
        default="query_caption",
        help="query_caption is the formal setting; other modes are ablations",
    )
    run_parser.add_argument("--memix-query-caption-cache-dir", type=Path)
    run_parser.add_argument("--unit-batch-size", type=int, default=16)
    run_parser.add_argument("--router-base-url")
    run_parser.add_argument("--router-model")
    run_parser.add_argument("--universal-text-model", default="Qwen/Qwen3-Embedding-4B")
    run_parser.add_argument("--universal-visual-model", default="VLM2Vec/VLM2Vec-V2.0")
    run_parser.add_argument("--universal-official-repo", type=Path, default=Path("sources/universalrag"))
    run_parser.add_argument(
        "--universal-profile",
        choices=["official", "mem-gallery"],
        default="official",
        help="UniversalRAG corpus/router adaptation profile",
    )
    run_parser.add_argument("--top-k", type=int, default=10)
    run_parser.add_argument("--query-concurrency", type=int, default=1)
    run_parser.add_argument("--resume-predictions", action="store_true")
    run_parser.add_argument("--continue-on-query-error", action="store_true")
    run_parser.add_argument(
        "--digest-only",
        action="store_true",
        help=(
            "build memory state without answering questions; --output receives "
            "a JSON timing summary instead of predictions"
        ),
    )

    smoke_parser = subparsers.add_parser(
        "smoke-method", help="run one diagnostic query with a tiny evidence-selected memory slice"
    )
    smoke_parser.add_argument(
        "method", choices=["amem", "lightmem", "memguide", "memix", "vimrag"]
    )
    smoke_parser.add_argument("bundle", type=Path)
    smoke_parser.add_argument("--output", type=Path, required=True)
    smoke_parser.add_argument("--subset")
    smoke_parser.add_argument("--max-memories", type=int, default=1)
    smoke_parser.add_argument("--question-id")
    smoke_parser.add_argument(
        "--memory-view", choices=["raw", "derived", "raw_derived"], default="raw"
    )
    smoke_parser.add_argument("--base-url", default="http://127.0.0.1:8091/v1")
    smoke_parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    smoke_parser.add_argument("--embedding-model", type=Path)
    smoke_parser.add_argument("--vimrag-official-repo", type=Path, default=Path("sources/vimrag"))
    smoke_parser.add_argument(
        "--vimrag-embedding-backend", choices=["qwen3vl", "gme"], default="qwen3vl"
    )
    smoke_parser.add_argument("--vimrag-max-steps", type=int, default=20)
    smoke_parser.add_argument("--vimrag-video-frames", type=int, default=8)
    smoke_parser.add_argument("--checkpoint-dir", type=Path)
    smoke_parser.add_argument("--memix-repo", type=Path, default=Path("vendor/memix-core"))
    smoke_parser.add_argument("--memix-embedding-model", default="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct")
    smoke_parser.add_argument("--memix-source-top-k", type=int, default=200)
    smoke_parser.add_argument("--memix-media-atomic-limit", type=int, default=0)
    smoke_parser.add_argument(
        "--memix-verifier-input-mode",
        choices=["current", "query_caption", "full_vl"],
        default="query_caption",
    )
    smoke_parser.add_argument("--memix-query-caption-cache-dir", type=Path)

    oracle_parser = subparsers.add_parser(
        "run-oracle", help="run the gold-evidence generation upper bound"
    )
    oracle_parser.add_argument("bundle", type=Path)
    oracle_parser.add_argument("--output", type=Path, required=True)
    oracle_parser.add_argument("--base-url", default="http://127.0.0.1:8091/v1")
    oracle_parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    oracle_parser.add_argument(
        "--memory-view", choices=["raw", "derived", "raw_derived"], default="raw_derived"
    )
    oracle_parser.add_argument("--caption-sidecar", type=Path)
    oracle_parser.add_argument(
        "--question-ids",
        type=Path,
        help="optional newline-delimited question-id allowlist",
    )
    oracle_parser.add_argument("--concurrency", type=int, default=8)

    judge_parser = subparsers.add_parser(
        "judge", help="score canonical predictions with one OpenAI-compatible LLM judge"
    )
    judge_parser.add_argument("bundle", type=Path)
    judge_parser.add_argument("predictions", type=Path)
    judge_parser.add_argument("--output", type=Path, required=True)
    judge_parser.add_argument("--model", required=True)
    judge_parser.add_argument("--base-url", default="https://api.openai.com/v1")
    judge_parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    judge_parser.add_argument("--timeout-seconds", type=float, default=120.0)
    judge_parser.add_argument("--max-items", type=int)
    judge_parser.add_argument(
        "--question-ids",
        type=Path,
        help="optional newline-delimited question-id allowlist for a reporting track",
    )
    judge_parser.add_argument("--concurrency", type=int, default=1)
    judge_parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "convert":
            result = convert(
                args.benchmark,
                args.raw_root,
                args.output_root,
                overwrite=args.overwrite,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "convert-all":
            results = {}
            failures = {}
            for name in CONVERTERS:
                try:
                    results[name] = convert(
                        name,
                        args.raw_root,
                        args.output_root,
                        overwrite=args.overwrite,
                    )
                except (FileNotFoundError, SchemaError, ValueError) as exc:
                    failures[name] = str(exc)
            print(json.dumps({"converted": results, "failed": failures}, ensure_ascii=False, indent=2))
            return 1 if failures else 0
        if args.command == "validate":
            result = validate_bundle(args.bundle, check_assets=args.check_assets)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "run-method":
            from .harness import digest_bundle, run_bundle
            from .methods import (
                ConcreteAMemMethod,
                ConcreteMemGuideMethod,
                ConcreteLightMemMethod,
                FaissFlatIPIndex,
                GMEQwen2VLEmbedder,
                GenerationConfig,
            )
            from .methods.backends import OpenAICompatibleQwenVL

            generation = GenerationConfig(
                model=args.model,
                base_url=args.base_url,
                temperature=0,
                top_k=args.top_k,
                max_model_len=32768,
                overflow_policy="error",
            )
            if args.method == "amem":
                memory_generation = GenerationConfig(
                    model=args.memory_model or args.model,
                    base_url=args.memory_base_url or args.base_url,
                    temperature=0,
                    top_k=args.top_k,
                    max_model_len=32768,
                    max_output_tokens=args.memory_max_output_tokens,
                    overflow_policy="error",
                )
                caption_model = None
                if args.caption_base_url or args.caption_model:
                    caption_generation = GenerationConfig(
                        model=args.caption_model or args.model,
                        base_url=args.caption_base_url or args.base_url,
                        temperature=0,
                        top_k=args.top_k,
                        max_model_len=32768,
                        max_output_tokens=args.memory_max_output_tokens,
                        overflow_policy="error",
                    )
                    caption_model = OpenAICompatibleQwenVL(caption_generation)
                method = ConcreteAMemMethod(
                    generation,
                    memory_model=OpenAICompatibleQwenVL(memory_generation),
                    caption_model=caption_model,
                    caption_cache_dir=(
                        args.caption_cache_dir
                        or ((args.checkpoint_dir / "captions") if args.checkpoint_dir else None)
                    ) if caption_model is not None else None,
                    memory_workers=args.memory_workers,
                    checkpoint_dir=args.checkpoint_dir,
                    checkpoint_interval=args.checkpoint_interval,
                    memory_retry_count=args.memory_retry_count,
                    memory_retry_backoff=args.memory_retry_backoff,
                    max_consecutive_fallbacks=args.max_consecutive_fallbacks,
                )
            elif args.method == "memguide":
                from .methods import NVEmbedV2TextEmbedder
                from .methods.backends import SentenceTransformerEmbedder

                embedder = (
                    SentenceTransformerEmbedder(args.memguide_embedding_model)
                    if args.memguide_embedding_model == "all-MiniLM-L6-v2"
                    else NVEmbedV2TextEmbedder(args.memguide_embedding_model)
                )
                question_model = None
                if args.memguide_question_mode == "llm":
                    question_generation = GenerationConfig(
                        model=args.memory_model or args.model,
                        base_url=args.memory_base_url or args.base_url,
                        temperature=0,
                        top_k=args.top_k,
                        max_model_len=32768,
                        max_output_tokens=args.memguide_question_max_output_tokens,
                        overflow_policy="error",
                    )
                    question_model = OpenAICompatibleQwenVL(question_generation)
                method = ConcreteMemGuideMethod(
                    generation,
                    question_model=question_model,
                    embedder=embedder,
                    index=FaissFlatIPIndex(embedder.dimension),
                    caption_cache_dir=(
                        args.caption_cache_dir
                        or ((args.checkpoint_dir / "captions") if args.checkpoint_dir else None)
                    ),
                    checkpoint_dir=args.checkpoint_dir,
                    checkpoint_interval=args.checkpoint_interval,
                    ingest_batch_size=args.unit_batch_size,
                    question_mode=args.memguide_question_mode,
                    question_workers=args.memguide_question_workers,
                )
            elif args.method == "lightmem":
                lightmem_generation = GenerationConfig(
                    model=args.memory_model or args.model,
                    base_url=args.memory_base_url or args.base_url,
                    temperature=0,
                    top_k=args.top_k,
                    max_model_len=32768,
                    max_output_tokens=args.memory_max_output_tokens,
                    overflow_policy="error",
                )
                method = ConcreteLightMemMethod(
                    generation,
                    memory_generation=lightmem_generation,
                    caption_cache_dir=(
                        args.caption_cache_dir
                        or ((args.checkpoint_dir / "captions") if args.checkpoint_dir else None)
                    ),
                    checkpoint_dir=args.checkpoint_dir,
                    official_repo=args.lightmem_official_repo,
                    llmlingua_model=args.lightmem_llmlingua_model,
                    embedding_model=args.lightmem_embedding_model,
                    device=args.lightmem_device,
                    ingest_batch_size=args.unit_batch_size,
                )
            elif args.method == "vimrag":
                from .methods import ConcreteVimRAGMethod, Qwen3VLVimRAGEmbedder

                if args.vimrag_embedding_backend == "qwen3vl":
                    embedding_name = (
                        str(args.embedding_model.resolve())
                        if args.embedding_model
                        else "Qwen/Qwen3-VL-Embedding-2B"
                    )
                    embedder = Qwen3VLVimRAGEmbedder(
                        embedding_name, official_repo=args.vimrag_official_repo
                    )
                else:
                    embedding_name = (
                        str(args.embedding_model.resolve())
                        if args.embedding_model
                        else "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
                    )
                    embedder = GMEQwen2VLEmbedder(embedding_name)
                method = ConcreteVimRAGMethod(
                    generation,
                    embedder=embedder,
                    index=FaissFlatIPIndex(embedder.dimension),
                    official_repo=args.vimrag_official_repo,
                    checkpoint_dir=args.checkpoint_dir,
                    checkpoint_interval=args.checkpoint_interval,
                    ingest_batch_size=args.unit_batch_size,
                    max_steps=args.vimrag_max_steps,
                    video_frames=args.vimrag_video_frames,
                    embedding_name=embedding_name,
                )
            elif args.method == "memix":
                from .methods import ConcreteMemixMethod

                embedder = GMEQwen2VLEmbedder(args.memix_embedding_model)
                method = ConcreteMemixMethod(
                    generation,
                    embedder=embedder,
                    index=FaissFlatIPIndex(embedder.dimension),
                    memix_repo=args.memix_repo,
                    checkpoint_dir=args.checkpoint_dir,
                    checkpoint_interval=args.checkpoint_interval,
                    ingest_batch_size=args.unit_batch_size,
                    source_top_k=args.memix_source_top_k,
                    video_frames=args.memix_video_frames,
                    use_llm_verify=args.memix_use_llm_verify,
                    drop_reader_ocr=args.memix_drop_reader_ocr,
                    media_atomic_limit=args.memix_media_atomic_limit,
                    verifier_input_mode=args.memix_verifier_input_mode,
                    query_caption_cache_dir=args.memix_query_caption_cache_dir,
                )
            else:
                from .methods import (
                    ConcreteUniversalRAGMethod,
                    OfficialQwen3TextEmbedder,
                    OfficialVLM2VecEmbedder,
                    MemGalleryUniversalRouter,
                    OpenAIUniversalRouter,
                    UNIVERSALRAG_CORPORA,
                )

                router_generation = GenerationConfig(
                    model=args.router_model or args.model,
                    base_url=args.router_base_url or args.base_url,
                    temperature=0,
                    top_k=args.top_k,
                    max_model_len=32768,
                    max_output_tokens=32,
                    overflow_policy="error",
                )
                if args.universal_profile == "mem-gallery":
                    shared_embedder = GMEQwen2VLEmbedder(
                        str(args.embedding_model.resolve())
                        if args.embedding_model
                        else "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
                    )
                    embedders = {
                        corpus: shared_embedder for corpus in UNIVERSALRAG_CORPORA
                    }
                    router = MemGalleryUniversalRouter(
                        OpenAICompatibleQwenVL(router_generation),
                        model_name=router_generation.model,
                    )
                else:
                    text_embedder = OfficialQwen3TextEmbedder(args.universal_text_model)
                    visual_embedder = OfficialVLM2VecEmbedder(
                        args.universal_visual_model,
                        official_repo=args.universal_official_repo,
                    )
                    embedders = {
                        corpus: text_embedder if corpus in {"paragraph", "document", "table"} else visual_embedder
                        for corpus in UNIVERSALRAG_CORPORA
                    }
                    router = OpenAIUniversalRouter(
                        OpenAICompatibleQwenVL(router_generation),
                        model_name=router_generation.model,
                    )
                method = ConcreteUniversalRAGMethod(
                    generation,
                    router=router,
                    corpus_embedders=embedders,
                    ingest_batch_size=args.unit_batch_size,
                    checkpoint_dir=args.checkpoint_dir,
                    checkpoint_interval=args.checkpoint_interval,
                    adapter_profile=args.universal_profile,
                )
            if args.digest_only:
                if args.resume_predictions:
                    raise ValueError("--digest-only cannot use --resume-predictions")
                if args.question_ids:
                    raise ValueError("--digest-only cannot use --question-ids")
                result = digest_bundle(
                    method,
                    args.bundle,
                    subset=args.subset,
                    split=args.split,
                    task_subcategory=args.task_subcategory,
                    memory_view=args.memory_view,
                    caption_sidecar=args.caption_sidecar,
                    memory_ids_path=args.memory_ids,
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            else:
                if args.memory_ids:
                    raise ValueError("--memory-ids requires --digest-only")
                result = run_bundle(
                    method,
                    args.bundle,
                    args.output,
                    subset=args.subset,
                    split=args.split,
                    task_subcategory=args.task_subcategory,
                    memory_view=args.memory_view,
                    query_concurrency=args.query_concurrency,
                    caption_sidecar=args.caption_sidecar,
                    resume_predictions=args.resume_predictions,
                    continue_on_query_error=args.continue_on_query_error,
                    question_ids_path=args.question_ids,
                )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "smoke-method":
            from .methods import (
                ConcreteAMemMethod,
                ConcreteMemGuideMethod,
                ConcreteLightMemMethod,
                ConcreteVimRAGMethod,
                ConcreteMemixMethod,
                GenerationConfig,
            )
            from .smoke import run_diagnostic_smoke

            generation = GenerationConfig(
                model=args.model,
                base_url=args.base_url,
                temperature=0,
                top_k=10,
                max_model_len=32768,
                overflow_policy="error",
            )
            if args.method == "vimrag":
                from .methods import FaissFlatIPIndex, GMEQwen2VLEmbedder, Qwen3VLVimRAGEmbedder

                if args.vimrag_embedding_backend == "qwen3vl":
                    embedding_name = (
                        str(args.embedding_model.resolve())
                        if args.embedding_model
                        else "Qwen/Qwen3-VL-Embedding-2B"
                    )
                    embedder = Qwen3VLVimRAGEmbedder(
                        embedding_name, official_repo=args.vimrag_official_repo
                    )
                else:
                    embedding_name = (
                        str(args.embedding_model.resolve())
                        if args.embedding_model
                        else "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
                    )
                    embedder = GMEQwen2VLEmbedder(embedding_name)
                method = ConcreteVimRAGMethod(
                    generation,
                    embedder=embedder,
                    index=FaissFlatIPIndex(embedder.dimension),
                    official_repo=args.vimrag_official_repo,
                    max_steps=args.vimrag_max_steps,
                    video_frames=args.vimrag_video_frames,
                    embedding_name=embedding_name,
                )
            elif args.method == "memix":
                from .methods import FaissFlatIPIndex, GMEQwen2VLEmbedder

                embedder = GMEQwen2VLEmbedder(args.memix_embedding_model)
                method = ConcreteMemixMethod(
                    generation,
                    embedder=embedder,
                    index=FaissFlatIPIndex(embedder.dimension),
                    memix_repo=args.memix_repo,
                    checkpoint_dir=args.checkpoint_dir,
                    source_top_k=args.memix_source_top_k,
                    media_atomic_limit=args.memix_media_atomic_limit,
                    verifier_input_mode=args.memix_verifier_input_mode,
                    query_caption_cache_dir=args.memix_query_caption_cache_dir,
                )
            else:
                method_class = {
                    "amem": ConcreteAMemMethod,
                    "memguide": ConcreteMemGuideMethod,
                    "lightmem": ConcreteLightMemMethod,
                }[args.method]
                method = method_class(generation)
            result = run_diagnostic_smoke(
                method,
                args.bundle,
                args.output,
                subset=args.subset,
                max_memories=args.max_memories,
                question_id=args.question_id,
                memory_view=args.memory_view,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "run-oracle":
            from .methods import GenerationConfig
            from .oracle import run_oracle_bundle

            generation = GenerationConfig(
                model=args.model,
                base_url=args.base_url,
                temperature=0,
                max_model_len=32768,
                max_output_tokens=1000,
                overflow_policy="error",
            )
            result = run_oracle_bundle(
                args.bundle,
                args.output,
                generation=generation,
                memory_view=args.memory_view,
                concurrency=args.concurrency,
                caption_sidecar=args.caption_sidecar,
                question_ids_path=args.question_ids,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "judge":
            from .judge import backend_from_env, judge_predictions

            backend = backend_from_env(
                model=args.model,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                timeout_seconds=args.timeout_seconds,
            )
            result = judge_predictions(
                backend,
                args.bundle,
                args.predictions,
                args.output,
                resume=not args.no_resume,
                max_items=args.max_items,
                question_ids_path=args.question_ids,
                concurrency=args.concurrency,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1 if result["failed_judgments"] else 0
    except (FileNotFoundError, FileExistsError, KeyError, RuntimeError, SchemaError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
