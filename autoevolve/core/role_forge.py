"""Dynamic role generation. Given a task and a subtask, ask the LLM to design
the ideal expert: title, expertise, success heuristics, allowed tools."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..llm.base import Backend, Message


FORGE_SYSTEM = """You are a Role Forge. Given a task and a specific subtask,
design the ideal expert agent for that subtask. Output STRICT JSON with keys:
{
  "title": string,
  "expertise": string,
  "system_prompt": string,   // first-person system prompt for the agent
  "success_heuristics": [string, ...],
  "allowed_tools": [string, ...]
}
No commentary outside the JSON."""


@dataclass
class RoleSpec:
    title: str
    expertise: str
    system_prompt: str
    success_heuristics: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "expertise": self.expertise,
            "system_prompt": self.system_prompt,
            "success_heuristics": self.success_heuristics,
            "allowed_tools": self.allowed_tools,
        }


class RoleForge:
    def __init__(self, backend: Backend):
        self.backend = backend

    async def forge(self, task_requirements: str, subtask: str, domain_hints: list[str]) -> RoleSpec:
        user = (
            f"TASK: {task_requirements}\n"
            f"DOMAIN HINTS: {', '.join(domain_hints) or 'none'}\n"
            f"SUBTASK: {subtask}\n\n"
            "Design the ideal expert for this subtask."
        )
        resp = await self.backend.complete(
            system=FORGE_SYSTEM, messages=[Message("user", user)]
        )
        data = _extract_json(resp.text) or {}
        return RoleSpec(
            title=data.get("title", "Worker"),
            expertise=data.get("expertise", ""),
            system_prompt=data.get("system_prompt", "You are a helpful expert."),
            success_heuristics=data.get("success_heuristics", []),
            allowed_tools=data.get("allowed_tools", []),
        )


def _extract_json(text: str) -> dict | None:
    """Tolerantly pull a JSON object out of a model response."""
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Strip ``` fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Find first { ... } block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None
