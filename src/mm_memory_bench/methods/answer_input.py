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
    system_text: str = ""

    def messages(self, user_content: Any, *, default_system: str = "") -> list[dict[str, Any]]:
        """Keep benchmark system instructions separate from method evidence."""
        # Benchmark requirements supplement, rather than replace, method rules.
        system = default_system
        if self.system_text and self.system_text != default_system:
            system = "\n\n".join(value for value in (default_system, self.system_text) if value)
        result = [{"role": "system", "content": system}] if system else []
        return result + [{"role": "user", "content": user_content}]

    @property
    def agent_text(self) -> str:
        # Agent backends own their system messages and executable retrieval tools.
        return f"{self.system_text}\n\n{self.text}" if self.system_text else self.text


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
    instruction = str(question.get("instruction", ""))
    system_text = instruction if question.get("instruction_role") == "system" else ""
    is_plan = bool(tools) and question.get("tool_mode") == "plan"
    if tools and question.get("tool_mode") == "plan":
        # Message role alone says nothing about whether candidates were rendered.
        # Only an explicit converter declaration may suppress the separate list.
        embedded_tools = bool(instruction.strip()) and question.get("instruction_includes_tools") is True
        if not embedded_tools:
            tools_text = "\nCandidate tools:\n" + json.dumps(tools, ensure_ascii=False)
        tools = None
    elif tools and not supports_api_tools:
        raise NotImplementedError("this answer adapter supports tool planning only, not API tool calls")
    if system_text:
        text = f"Question: {question_text(question)}"
    elif question_first:
        # Preserve the agent's existing task layout for non-tool questions.
        text = question_text(question)
        instruction = str(question.get("instruction", "")).strip()
        if instruction:
            text += f"\n\nAnswer requirements: {instruction}"
    else:
        text = f"{question.get('instruction', '')}\nQuestion: {question_text(question)}"
    return AnswerTask(text=text + tools_text, api_tools=tools, is_tool_plan=is_plan, system_text=system_text)
