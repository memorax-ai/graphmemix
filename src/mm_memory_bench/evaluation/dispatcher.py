"""Versioned benchmark routing; existing scorers retain their output contracts."""
from collections import Counter, defaultdict
import json
from pathlib import Path

from ..benchmarks.bundle import read_json, write_json
from .native_runner import backend_from_env, get_scorer, judge_predictions
from .native.script_judge import _load, score_predictions, score_question

DISPATCH_VERSION = "mmmb-benchmark-dispatch-2.1"
BENCHMARKS = {"smmbench", "persona_mme", "personamem_v2", "m3exam", "mobilemem_omni"}


def scoring_route(benchmark, question):
    """Return one scoring route per question; never silently fall back."""
    task = question.get("task", {})
    response = task.get("response_type")
    if benchmark == "smmbench":
        if question.get("tool_mode") == "plan" and response == "structured_json":
            return "script"
        if response == "choice" and question.get("tool_mode") in (None, "none"):
            return "script"
    elif benchmark == "persona_mme" and response == "choice":
        return "script"
    elif benchmark == "personamem_v2":
        if question.get("metadata", {}).get("native_eval_mode") == "generative":
            if response == "text":
                return "personamem_v2_open"
        elif response == "choice":
            return "script"
    elif benchmark == "m3exam":
        kind = task.get("subcategory")
        if kind in ("fj", "fm"):
            return "script"
        if kind in {"mr", "tr", "ms", "ss", "th", "ii"} and response in ("text", "choice"):
            return "m3exam"
    elif benchmark == "mobilemem_omni" and response == "text":
        return "mobilemem_omni"
    raise ValueError(f"unsupported scoring task: {benchmark}/{question.get('question_id')}")


def score_benchmark(bundle, predictions, output, *, model=None,
                    base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY",
                    timeout_seconds=120.0, resume=True, max_items=None,
                    question_ids_path=None, concurrency=1):
    """Preflight all selected tasks, then score disjoint groups without rerunning methods."""
    bundle, predictions, output = Path(bundle), Path(predictions), Path(output)
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    manifest_path = bundle / "manifest.json"
    manifest = read_json(manifest_path)
    benchmark = str(manifest.get("benchmark", "")).lower().replace("-", "_")
    if benchmark not in BENCHMARKS:
        raise ValueError(f"unsupported benchmark: {benchmark}; use qa explicitly if intended")
    question_path = bundle / manifest.get("tables", {}).get("questions", "questions.jsonl")
    questions, predicted = _load(question_path), _load(predictions)
    if set(predicted) - questions.keys():
        raise ValueError("predictions contain unknown question IDs")
    selected = list(questions)
    if question_ids_path is not None:
        selected = [s.strip() for s in Path(question_ids_path).read_text().splitlines() if s.strip()]
        if len(selected) != len(set(selected)) or set(selected) - questions.keys():
            raise ValueError("duplicate or unknown question IDs in allowlist")
    if max_items is not None:
        if max_items < 1:
            raise ValueError("max_items must be positive")
        selected = selected[:max_items]
    if not selected or set(selected) - predicted.keys():
        raise ValueError("empty selection or incomplete predictions; use --question-ids for a subset")

    groups = defaultdict(list)
    for qid in selected:
        q, prediction = questions[qid], predicted[qid].get("prediction", "")
        if isinstance(prediction, (dict, list)):
            prediction = json.dumps(prediction, ensure_ascii=False)
        if not isinstance(prediction, str):
            raise ValueError(f"{qid}: prediction must be text or JSON object/list")
        route = scoring_route(benchmark, q)
        if route == "script":
            score_question(benchmark, q, prediction)  # Validate gold data before any API call/write.
        else:
            q["context_id"]
            get_scorer(route).judge_item(q, prediction)
        groups[route].append(qid)
    if "script" in groups:
        if question_path.resolve() != (bundle / "questions.jsonl").resolve():
            raise ValueError("script scoring requires the canonical questions.jsonl table")
        if any(not qid.startswith((benchmark + ":", benchmark + "_")) for qid in questions):
            raise ValueError("benchmark does not match question ID namespace")
    llm_routes = set(groups) - {"script"}
    if llm_routes and not model:
        raise ValueError("--model is required for selected LLM tasks")

    # Derive script filenames from the requested output so separate runs stay separate.
    paths = {route: (output.with_name(f"{output.stem}.native{output.suffix}") if route == "script" else output)
             for route in groups}
    if len(llm_routes) > 1:
        for route in llm_routes:
            paths[route] = output.with_name(f"{output.stem}.{route}{output.suffix}")
    summary_path = output.with_suffix(".dispatch.summary.json")
    destinations = [summary_path]
    for path in paths.values():
        destinations.extend([path, path.with_suffix(".summary.json")])
    resolved = [p.resolve() for p in destinations]
    protected = {manifest_path.resolve(), question_path.resolve(), predictions.resolve()}
    protected.update((bundle / name).resolve() for name in manifest.get("tables", {}).values())
    if question_ids_path is not None:
        protected.add(Path(question_ids_path).resolve())
    if len(set(resolved)) != len(resolved) or set(resolved) & protected:
        raise ValueError("output paths collide with each other or with inputs")

    # Construct clients only for routes that need a model; constructors do no API I/O.
    backends = {route: backend_from_env(model=model, base_url=base_url,
                api_key_env=api_key_env, timeout_seconds=timeout_seconds,
                scoring_protocol=route) for route in llm_routes}
    results = {}
    for route, ids in groups.items():
        path = paths[route]
        if route == "script":
            result = score_predictions(bundle, predictions, path,
                                       benchmark=benchmark, question_ids=ids)
        else:
            result = judge_predictions(backends[route], bundle, predictions, path,
                       question_ids=ids, scoring_protocol=route, concurrency=concurrency,
                       resume=resume)
        records = _load(path)
        if set(records) != set(ids):
            raise RuntimeError(f"scorer coverage mismatch: {route}")
        results[route] = {"assigned": len(ids), "completed": len(records),
                          "statuses": dict(Counter(r.get("status") for r in records.values())),
                          "method_failures": sum(
                              isinstance(predicted[qid].get("metadata"), dict) and (
                                  predicted[qid]["metadata"].get("status") == "error" or
                                  bool(predicted[qid]["metadata"].get("error_type"))) for qid in ids),
                          "output": str(path), "summary": result}
    summary = {"benchmark": benchmark, "dispatch_protocol": DISPATCH_VERSION,
               "scoring_protocol": "benchmark", "selected_questions": len(selected),
               "scope": "explicit_subset" if question_ids_path or max_items else "all",
               "routes": results,
               "failed_judgments": sum(r["summary"].get("failed_judgments", 0)
                                       for r in results.values())}
    write_json(summary_path, summary)
    return dict(summary, summary_path=str(summary_path))
