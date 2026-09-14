"""Shared task input for answering; retrieval queries remain method-owned.

Only public task fields are read. Gold answers, evidence labels and metadata
must never be serialized into the answer model's task input. Media and retrieved
evidence are attached separately by each method.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .media import question_text


@dataclass(frozen=True)
class AnswerTask:
    text: str
    api_tools: Sequence[Mapping[str, Any]] | None = None
    is_tool_plan: bool = False


def build_answer_task(
    question: Mapping[str, Any], *, supports_api_tools: bool = True, question_first: bool = False
) -> AnswerTask:
    """Render task text and keep API tool schemas separate from planning data.

    Plan-mode candidates are descriptions only, never executable API tools.
    Other modes preserve the existing adapters' API-tools behavior; consumers
    without that interface must opt out explicitly instead of dropping tools.
    """
    tools = question.get("tools")
    tools_text = ""
    if tools and question.get("tool_mode") == "plan":
        tools_text = "\nCandidate tools:\n" + json.dumps(tools, ensure_ascii=False)
        tools = None
    elif tools and not supports_api_tools:
        raise NotImplementedError("this answer adapter supports tool planning only, not API tool calls")
    if question_first:
        # Preserve the agent's existing task layout for non-tool questions.
        text = question_text(question)
        instruction = str(question.get("instruction", "")).strip()
        if instruction:
            text += f"\n\nAnswer requirements: {instruction}"
    else:
        text = f"{question.get('instruction', '')}\nQuestion: {question_text(question)}"
    return AnswerTask(text=text + tools_text, api_tools=tools, is_tool_plan=bool(tools_text))
