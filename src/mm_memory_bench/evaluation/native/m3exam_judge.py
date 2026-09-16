"""M³Exam five-point Judge, ported from the pinned official baseline runtime.

Source: EverM0re/M-3-Exam, commit 1dbe10441a043d86043ad13283e1ee03fbe154b6,
baselines/_runtime/extended_common/llm_client.py and m3proctor/evaluation/metrics.py.
Apache-2.0. Adaptation: canonical bundle inputs and common runner result fields.
"""
import hashlib
from collections import defaultdict

SOURCE_REVISION = "1dbe10441a043d86043ad13283e1ee03fbe154b6"
PROTOCOL_VERSION = "mmmb-m3exam-five-point-1.0"
SUPPORTED_TYPES = {"mr", "tr", "ms", "ss", "th", "ii"}

JUDGE_PROMPT_TEMPLATE = (
    "You are an impartial judge evaluating the memory capabilities of an AI assistant "
    "with the question-answering task. Your task is to compare the Assistant's Answer "
    "against the Ground Truth and assign a score of 0, 0.25, 0.5, 0.75, or 1.\n\n"
    "Question     : {question}\n"
    "Gold Answer  : {gold}\n"
    "Model Answer : {model_answer}\n\n"
    "Scoring Rubric\n"
    "Score 0 (Incorrect / Miss):\n"
    "- The answer contradicts the Ground Truth.\n"
    "- For Yes/No questions: The answer has the wrong polarity (e.g., says \"Yes\" when Ground Truth is \"No\").\n"
    "- For Open-ended questions: The answer provides factually wrong information or hallucinations.\n"
    "- The assistant fails to provide the required information.\n"
    "Score 0.25 (Poor / Tangential):\n"
    "- The answer touches on the topic but misses the core entity or key value required.\n"
    "- The answer contains a mix of minor correct details and significant hallucinations or wrong associations.\n"
    "- The answer is excessively vague to the point of being useless (e.g., answering \"a dog\" instead of \"a golden retriever\").\n"
    "Score 0.5 (Partial / Vague / Excessive):\n"
    "- The answer is technically correct, but lacks confidence or is incomplete.\n"
    "- The answer captures the main entity or concept correctly but misses a part of the required supporting details.\n"
    "- The answer includes the correct information, but is over informative or excessive.\n"
    "- For Yes/No questions: The polarity is correct, but the reasoning is flawed (if have), or the assistant is uncertain (e.g., \"I think it might be Yes\").\n"
    "- For Open-ended questions: The answer is too general or misses key adjectives/details present in the Ground Truth.\n"
    "Score 0.75 (Good / Minor Imperfection):\n"
    "- The answer is largely accurate and captures the core information confidently.\n"
    "- It misses only minor details (e.g., specific adjectives or secondary details) that do not alter the main truth.\n"
    "- The answer contains all the correct information but includes unnecessary \"fluff\" or slight conversational filler that reduces precision.\n"
    "Score 1 (Correct / Exact):\n"
    "- The answer is accurate, precise, and confident.\n"
    "- For Yes/No questions: The polarity matches the Ground Truth perfectly.\n"
    "- For Open-ended questions: The answer contains all the core information and necessary details required by the Ground Truth without hallucinations.\n\n"
    "Reply with ONLY the score digit. No other text."
)

_ALLOWED_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)

def _parse_judge_score(text: str) -> float:
    import re as _re

    s = (text or "").strip()
    for k in ("0.75", "0.25", "0.5", "1.0", "1", "0.0", "0"):
        if s == k:
            v = float(k)
            if v in _ALLOWED_SCORES:
                return v
    m = _re.search(r"(?<![\d.])(0?\.\d+|1(?:\.0+)?|0)(?![\d])", s)
    if m:
        try:
            v = float(m.group(1))
        except ValueError:
            return 0.0
        return min(_ALLOWED_SCORES, key=lambda x: abs(x - v))
    return 0.0

RUBRIC_SHA256 = hashlib.sha256(JUDGE_PROMPT_TEMPLATE.encode()).hexdigest()


def judge_item(question, prediction):
    kind = question.get("task", {}).get("subcategory")
    if kind not in SUPPORTED_TYPES:
        raise ValueError("M3Exam five-point Judge requires a non-fj/fm question")
    # The official score_record uses label, falling back to the first ordered answer.
    answer = question["answer"]
    refs = answer.get("native_ordered_answers", answer.get("accepted_answers", [answer["text"]])) or []
    gold = question.get("metadata", {}).get("native_label") or (refs[0] if refs else "")
    parts = question.get("prompt", [])
    if not parts or any(p.get("type") != "text" for p in parts):
        raise ValueError("M3Exam native question must contain text only")
    return {"question": "\n".join(p["text"] for p in parts), "gold": gold,
            "model_answer": prediction, "type": kind}


def request_body(model, item):
    # Pinned baselines/config.yaml: judge_temperature=0, judge_max_tokens=8;
    # SharedLLM.judge applies max(judge_max_tokens, 16).
    return {"model": model, "temperature": 0.0, "max_tokens": 16,
            "messages": [{"role": "user", "content": JUDGE_PROMPT_TEMPLATE.format(**item)}]}


def parse_response(text):
    raw = (text or "").strip()
    # Preserve official nearest-bin rounding AND its unparseable-response zero.
    return {"score": _parse_judge_score(raw), "judge_response": raw}


def normalize(value):
    score = value.get("score")
    if isinstance(score, bool) or score not in _ALLOWED_SCORES:
        raise ValueError("M3Exam score must be one of 0, 0.25, 0.5, 0.75, 1")
    return {"score": float(score), "judge_response": str(value.get("judge_response", ""))}


def summarize(records):
    def summarize_group(rows):
        vals = [float(r.get("score", 0.0) or 0.0) if r.get("status") == "ok" else 0.0 for r in rows]
        return {"count": len(rows),
                "answered": sum(bool((r.get("prediction") or "").strip()) for r in rows),
                "llm_score": round(sum(vals) / len(vals), 4) if vals else 0.0}
    groups = defaultdict(list)
    for row in records:
        groups[row["native_type"]].append(row)
    failed = sum(r.get("status") != "ok" for r in records)
    return {"metric": "llm_score", "total_predictions": len(records),
            "valid_judgments": len(records) - failed, "failed_judgments": failed,
            "total": summarize_group(records),
            "per_type": {kind: summarize_group(rows) for kind, rows in groups.items()}}


benchmark = "m3exam"

def zero():
    return {"score": 0.0, "judge_response": ""}


def record_fields(item):
    return {"metric": "llm_score", "native_type": item["type"],
            "source_revision": SOURCE_REVISION}
