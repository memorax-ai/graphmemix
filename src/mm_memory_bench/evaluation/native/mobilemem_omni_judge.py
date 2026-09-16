"""Omni published-prompt adapter, NOT a verified port of missing llm_judge.py.

Pinned zjunlp/MobileMem commit 919e0f545722030898cee03b263f08c8092f2ebb:
- eval/question_answering_and_judge_prompts.txt: verbatim Judge prompt;
- eval/evaluator.py: reference selection, normalization and LLM-label denominator;
- eval/Raw2Locomo.py: evidence explanation extraction.
The repository imports eval/llm_judge.py but does not publish it. Request formatting
and JSON parsing here are explicit local choices, not asserted upstream behavior.
"""
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

SOURCE_REVISION = "919e0f545722030898cee03b263f08c8092f2ebb"
PROTOCOL_VERSION = "mmmb-omni-published-prompt-1.1"
CATEGORIES = {
    "multi_hop": "Multi-hop", "temporal_reasoning": "Temporal Reasoning",
    "abstention": "Abstention", "single_hop": "Single-hop",
    "implicit_preference": "Implicit Preference", "visual_reasoning": "Visual Reasoning",
    "knowledge_update": "Knowledge Update",
}

JUDGE_PROMPT = 'Your task is to classify an answer to a question as CORRECT or WRONG. You will be provided with the following information:\n(1) A question (asked by a user to an AI assistant);\n(2) A gold (ground truth) answer;\n(3) A generated answer;\n(4) A set of evidence sentences relevant to the question, which may be used to verify the generated answer.\nThe question focuses on information that an AI assistant should recall about the user based on prior conversations. The gold answer is typically concise and includes all key facts.\n\nGRADING RULES (Must Be Strictly Followed)\n1. COMPLETE COVERAGE REQUIRED: The generated answer must include every key element present in the gold answer. Missing any required fact results in a WRONG classification.\n2. NO REDUNDANT CONTENT: Including excessive or irrelevant details not supported by the evidence or gold answer results in a WRONG classification.\n3. TIME-RELATED QUESTIONS: Different date formats (e.g., May 7th vs. 7 May) are acceptable as long as the same date or time period is referenced.\n\nINSTRUCTIONS FOR THIS TASK\nUse the provided evidence only as supporting context for verification when available. Do not infer facts that are not present in the question, gold answer, or evidence.\nFirst, provide a brief (one-sentence) explanation of your reasoning. Then, output the label CORRECT or WRONG.\nDo not include both labels in your response, as this will break the evaluation script.\nReturn the label in JSON format with the key label.\n\nQUESTION TO EVALUATE\nQuestion: [Question]\nGold answer: [Gold Answer]\nEvidence: [Evidences]\nGenerated answer: [Generated Answer]'
RUBRIC_SHA256 = hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest()


class _ReferenceSelector:
    @staticmethod
    def tokenize(text: str) -> list[str]:
            """Tokenizes string into lowercase words and CJK characters."""
            text = str(text).lower()
            text = re.sub(r'[\u0000-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]', ' ', text)
            return re.findall(r'[\u4e00-\u9fff]|[a-z0-9]+', text)

    def calculate_f1(self, pred: str, ref: str) -> float:
            pt, rt = self.tokenize(pred), self.tokenize(ref)
            if not pt and not rt: return 1.0
            if not pt or not rt: return 0.0
            pc, rc = Counter(pt), Counter(rt)
            overlap = sum((pc & rc).values())
            if overlap == 0: return 0.0
            precision = overlap / len(pt)
            recall = overlap / len(rt)
            return 2 * precision * recall / (precision + recall)

    def calculate_bleu1(self, pred: str, ref: str) -> float:
            pt, rt = self.tokenize(pred), self.tokenize(ref)
            if not pt: return 0.0
            pc, rc = Counter(pt), Counter(rt)
            clipped_overlap = sum(min(pc[w], rc[w]) for w in pc)
            precision = clipped_overlap / len(pt)
            bp = math.exp(1 - len(rt) / len(pt)) if len(pt) < len(rt) else 1.0
            return bp * precision

    def _get_best_metrics(self, pred: str, refs: list[str]) -> tuple[float, float, str]:
            best_f1, best_bleu, best_ref = -1, -1, ""
            for r in refs:
                f1 = self.calculate_f1(pred, r)
                bleu = self.calculate_bleu1(pred, r)
                if (f1 + bleu) > (best_f1 + best_bleu):
                    best_f1, best_bleu, best_ref = f1, bleu, r
            return best_f1, best_bleu, best_ref

    def _norm_refs(self, refs: Any) -> list[str]:
            if refs is None: return [""]
            if isinstance(refs, (str, int, float, bool)): return [str(refs)]
            items = [refs] if isinstance(refs, dict) else refs
            out = []
            for x in items:
                if isinstance(x, dict):
                    val = x.get("text") or x.get("answer") or x.get("value") or ""
                    out.append(str(val))
                else:
                    out.append(str(x))
            return out or [""]


def judge_item(question, prediction):
    if question.get("task", {}).get("response_type") != "text":
        raise ValueError("Omni published-prompt adapter supports the released open-ended track")
    parts = question.get("prompt", [])
    if not parts or any(p.get("type") != "text" for p in parts):
        raise ValueError("Omni query-image transport is not published; text-only questions required")
    metadata = question.get("metadata", {})
    if "native_evidence" not in metadata:
        raise ValueError("Omni native_evidence missing; regenerate the bundle")
    evidence = []
    for ev in metadata["native_evidence"]:
        if isinstance(ev, dict):
            if ev.get("explanation", ""):
                evidence.append(ev["explanation"])
        elif isinstance(ev, str):
            evidence.append(ev)
    selector = _ReferenceSelector()
    answer = question["answer"]
    refs = selector._norm_refs(answer.get("accepted_answers") or answer["text"])
    # Although F1/BLEU-1 are no longer reported, the official evaluator uses
    # their sum to select the ONE gold answer passed to the Judge.
    _, _, gold = selector._get_best_metrics(prediction, refs)
    return {"question": "\n".join(p["text"] for p in parts), "gold": gold,
            "prediction": prediction, "evidence": evidence,
            "category": CATEGORIES.get(question.get("task", {}).get("subcategory"), "0")}


def request_body(model, item):
    values = {"Question": item["question"], "Gold Answer": item["gold"],
              "Evidences": json.dumps(item["evidence"], ensure_ascii=False),
              "Generated Answer": item["prediction"]}
    # One-pass substitution prevents placeholders inside quoted inputs changing.
    prompt = re.sub(r"\[(Question|Gold Answer|Evidences|Generated Answer)\]",
                    lambda m: values[m.group(1)], JUDGE_PROMPT)
    # Local transport choice: plain user prompt, temperature zero, no forced
    # response_format (official prompt permits a reasoning sentence before JSON).
    return {"model": model, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}


def parse_response(text):
    candidates = []
    for match in re.finditer(r"\{", text):
        try:
            value, _ = json.JSONDecoder().raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(value, dict) and "label" in value:
            candidates.append(value)
    if len(candidates) != 1 or candidates[0]["label"] not in ("CORRECT", "WRONG"):
        raise ValueError("Omni Judge must return one JSON label: CORRECT or WRONG")
    label = candidates[0]["label"]
    return {"label": label, "correct": label == "CORRECT",
            "score": float(label == "CORRECT"), "judge_response": text}


def normalize(value):
    label = value.get("label")
    if label not in ("CORRECT", "WRONG"):
        raise ValueError("invalid Omni Judge label")
    return {"label": label, "correct": label == "CORRECT",
            "score": float(label == "CORRECT"),
            "judge_response": str(value.get("judge_response", ""))}


def summarize(records):
    def metric(rows):
        judged = [r for r in rows if r.get("status") in {"ok", "method_error"} and r.get("label") is not None]
        return sum(r["label"] == "CORRECT" for r in judged) / len(judged) if judged else 0.0
    grouped = defaultdict(list)
    for row in records:
        grouped[row["native_category"]].append(row)
    failed = sum(r.get("status") == "error" for r in records)
    return {"metric": "LLM_JUDGE", "total_predictions": len(records),
            "valid_judgments": sum(r.get("status") == "ok" and r.get("label") is not None for r in records),
            "failed_judgments": failed,
            "method_failures": sum(r.get("status") == "method_error" for r in records),
            "skipped_judgments": sum(r.get("status") in {"ok", "method_error"} and r.get("label") is None for r in records),
            "total_questions": len(records), "overall": {"LLM_JUDGE": metric(records)},
            "by_category": {category: {"count": len(rows), "metrics": {"LLM_JUDGE": metric(rows)}}
                            for category, rows in grouped.items()}}


benchmark = "mobilemem_omni"
empty_prediction_is_zero = False

def skip_item(item):
    return not item["gold"]


def zero():
    return {"label": None, "correct": None, "score": None, "judge_response": ""}


def record_fields(item):
    return {"metric": "LLM_JUDGE", "native_category": item["category"],
            "source_revision": SOURCE_REVISION,
            "implementation_scope": "published_prompt_local_parser"}
