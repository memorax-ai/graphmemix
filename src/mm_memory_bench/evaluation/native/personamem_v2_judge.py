"""PersonaMem-v2 narrow preference scoring, separate from binary QA accuracy.

Prompts and numeric matching rules follow official inference_utils.py at
https://github.com/bowen-upenn/PersonaMem-v2/tree/d29d91d016add354e459dfeb0d24af08bc402e2a
No broad-profile Judge or conversion to binary correctness is performed.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from typing import Any, Mapping

JUDGE_PROMPT_NARROW_POSITIVE = r"""You are evaluating an AI assistant's personalization capabilities in open-ended conversations.

User's query:
{user_query}

Ground truth user preference: {preference}

Model's response to evaluate:
{model_response}

Task: Evaluate how well the model personalizes its response according to the user's ground-truth preference. We expect the model to demonstrate awareness of this preference and provide a response that is specifically aligned with or relevant to it, rather than giving a generic answer that could apply to anyone.

A well-personalized response should:
- Acknowledge or incorporate the specific preference mentioned
- Tailor recommendations, advice, or information to align with this preference
- Show that the model remembers and respects the user's stated preference

A poorly personalized response:
- Ignores the preference entirely and gives generic advice
- Explicitly states it doesn't remember or track preferences
- Provides suggestions that contradict the stated preference

Give a score from 0.0 (completely irrelevant to or contradicts the preference) to 1.0 (explicitly and thoughtfully personalized to the preference).

Provide your reasoning first, then give your final score as \boxed{{score}} where score is a number between 0.0 and 1.0.
Your response:"""

JUDGE_PROMPT_NARROW_NEGATIVE = r"""You are evaluating an AI assistant's personalization capabilities in open-ended conversations.

User's query:
{user_query}

Ground truth user preference: {preference}

Model's response to evaluate:
{model_response}

In this case, the user's ground-truth preference is a "do not personalize" or "avoid" instruction about a certain topic or attribute. The user does not want the assistant to tailor its response based on that specific preference, and instead expects a more general or neutral answer that does not rely on or mention that preference.

Task: Evaluate how well the model respects this preference. The model should avoid using or referencing the sensitive preference in a way that personalizes the response, while still providing a helpful and relevant answer.

Give a score from 0.0 (clearly personalizes using the forbidden preference or ignores the instruction) to 1.0 (fully respects the instruction to not personalize while remaining helpful).

Provide your reasoning first, then give your final score as \boxed{{score}} where score is a number between 0.0 and 1.0.
Your response:"""

PROTOCOL_VERSION = "mmmb-personamem-v2-narrow-1.1"
RUBRIC_SHA256 = hashlib.sha256(
    (JUDGE_PROMPT_NARROW_POSITIVE + "\n" + JUDGE_PROMPT_NARROW_NEGATIVE).encode()
).hexdigest()


def judge_item(question: Mapping[str, Any], prediction: str) -> dict[str, Any]:
    metadata = question.get("metadata", {})
    if (metadata.get("native_eval_mode") != "generative"
            or question.get("subset") not in {
                "multimodal_32k_generative", "multimodal_128k_generative"}):
        raise ValueError("personamem_v2_open requires the multimodal generative track")
    query = metadata.get("native_user_query")
    preference = metadata.get("preference")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("native_user_query is missing; regenerate the PersonaMem-v2 bundle")
    if not isinstance(preference, str) or not preference.strip():
        raise ValueError("PersonaMem-v2 native preference is missing")
    return {"user_query": query, "preference": preference, "model_response": prediction}


def judge_kind(item: Mapping[str, Any]) -> str:
    return "negative" if item["preference"].lower().startswith("do not") else "positive"


def request_body(model: str, item: Mapping[str, Any]) -> dict[str, Any]:
    template = (JUDGE_PROMPT_NARROW_NEGATIVE if judge_kind(item) == "negative"
                else JUDGE_PROMPT_NARROW_POSITIVE)
    return {"model": model, "temperature": 0,
            "messages": [{"role": "user", "content": template.format(**item)}]}


def extract_judge_decision(response: str) -> float:
    """Extract numeric score from judge response."""
    if not response:
        return 0.0

    import re

    # Look for boxed format first (most reliable)
    boxed_patterns = [
        r'\\boxed\{([0-9]*\.?[0-9]+)\}',
        r'\$\\boxed\{([0-9]*\.?[0-9]+)\}\$',
        r'\\boxed\s*\{([0-9]*\.?[0-9]+)\}',
    ]

    for pattern in boxed_patterns:
        match = re.search(pattern, response)
        if match:
            try:
                score = float(match.group(1))
                # Clamp score between 0.0 and 1.0
                return max(0.0, min(1.0, score))
            except ValueError:
                continue

    # Fallback: look for standalone decimal number that looks like a score
    score_patterns = [
        r'score[:\s]+([0-9]*\.?[0-9]+)',
        r'rating[:\s]+([0-9]*\.?[0-9]+)',
        r'([0-9]*\.[0-9]+)\s*/\s*1\.?0?',
    ]

    for pattern in score_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                continue

    # Default to 0.0 if unclear
    return 0.0


def parse_response(text: str) -> dict[str, Any]:
    return {"score": extract_judge_decision(text), "judge_response": text}


def normalize(value: Mapping[str, Any]) -> dict[str, Any]:
    score = value.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("preference score must be a finite number in [0, 1]")
    return {"score": float(score), "judge_response": str(value.get("judge_response", ""))}


def summarize(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [r for r in records if r.get("status") == "ok"]
    total_score = sum(r["score"] for r in valid)
    def group(rows, key):
        grouped = defaultdict(list)
        for row in rows:
            grouped[str(row.get(key, "unknown"))].append(row)
        result = {}
        for label, members in sorted(grouped.items()):
            ok = [r for r in members if r.get("status") == "ok"]
            score = sum(r["score"] for r in ok)
            result[label] = {"count": len(members), "valid_judgments": len(ok),
                             "failed_judgments": len(members) - len(ok),
                             "mean_score_conservative": score / len(members),
                             "mean_score_valid_only": score / len(ok) if ok else None}
        return result
    by_subset = group(records, "subset")
    for subset, entry in by_subset.items():
        entry["by_preference_kind"] = group([r for r in records if r.get("subset") == subset], "preference_kind")
    return {"metric": "preference_alignment", "total_predictions": len(records),
            "valid_judgments": len(valid), "failed_judgments": len(records) - len(valid),
            "mean_score_conservative": total_score / len(records) if records else 0.0,
            "mean_score_valid_only": total_score / len(valid) if valid else None,
            "by_subset": by_subset, "by_preference_kind": group(records, "preference_kind")}




def zero():
    return {"score": 0.0}


def record_fields(item):
    return {"metric": "preference_alignment", "preference_kind": judge_kind(item)}
