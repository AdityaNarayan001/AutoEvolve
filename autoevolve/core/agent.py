"""Agent runtime: a single LLM identity with a system prompt, tool registry,
and sandbox. Runs a tool-use loop until the model returns a final answer."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..llm.base import Backend, Message
from ..sandbox import Sandbox
from ..tools import ToolRegistry
from .role_forge import RoleSpec


@dataclass
class AgentResult:
    final_text: str
    transcript: list[Message] = field(default_factory=list)
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


TOOL_PROTOCOL = """You can call tools by emitting a JSON block on a single line:
<tool>{"name": "shell", "args": {"cmd": "ls"}}</tool>
After you see the tool result, continue. When done, emit:
<final>your final answer</final>
"""


class Agent:
    def __init__(
        self,
        role: RoleSpec,
        backend: Backend,
        tools: ToolRegistry,
        sandbox: Sandbox,
        max_steps: int = 30,
        should_continue: Callable[[], bool] | None = None,
        get_nudge: Callable[[], str] | None = None,
        validate_final: Callable[["AgentResult"], str | None] | None = None,
    ):
        self.role = role
        self.backend = backend
        self.tools = tools
        self.sandbox = sandbox
        self.max_steps = max_steps
        self.should_continue = should_continue
        self.get_nudge = get_nudge
        # validate_final: if provided, returns None to accept a <final>, or a
        # string explaining why the final is rejected. The agent then sees the
        # rejection as a user message and must keep working.
        self.validate_final = validate_final

    async def run(self, user_prompt: str) -> AgentResult:
        sys_prompt = self.role.system_prompt + "\n\n" + TOOL_PROTOCOL
        if self.role.allowed_tools:
            sys_prompt += f"\nAllowed tools: {', '.join(self.role.allowed_tools)}"
        if self.role.success_heuristics:
            sys_prompt += "\nSuccess heuristics:\n- " + "\n- ".join(
                self.role.success_heuristics
            )

        messages: list[Message] = [Message("user", user_prompt)]
        result = AgentResult(final_text="", transcript=messages)

        for _ in range(self.max_steps):
            if self.should_continue and not self.should_continue():
                result.final_text = "(stopped by orchestrator)"
                return result
            if self.get_nudge:
                nudge = self.get_nudge()
                if nudge:
                    messages.append(Message("user", f"USER NUDGE: {nudge}"))
            resp = await self.backend.complete(system=sys_prompt, messages=messages)
            result.tokens_in += resp.input_tokens
            result.tokens_out += resp.output_tokens
            text = resp.text
            messages.append(Message("assistant", text))

            final = _extract_tag(text, "final")
            if final is not None:
                result.final_text = final
                # Run validation. If the validator rejects, force the agent to
                # keep working instead of letting it short-circuit.
                if self.validate_final:
                    reason = self.validate_final(result)
                    if reason:
                        messages.append(
                            Message(
                                "user",
                                f"REJECTED: you cannot finish yet. {reason}\n"
                                "Do the actual work using your tools, then try again.",
                            )
                        )
                        result.final_text = ""
                        continue
                return result

            tool_call = _extract_tag(text, "tool")
            if not tool_call:
                # No tool call and no final tag. If a validator is set we
                # require explicit progress, so push the agent to use tools.
                if self.validate_final:
                    messages.append(
                        Message(
                            "user",
                            "You must either call a tool with <tool>{...}</tool> "
                            "or finish with <final>...</final>. Free text alone "
                            "does not count as progress. Pick the next concrete "
                            "tool call.",
                        )
                    )
                    continue
                # Permissive mode for plain agents: treat free text as final.
                result.final_text = text
                return result

            try:
                call = json.loads(tool_call)
            except Exception:
                messages.append(
                    Message("user", f"TOOL_ERROR: invalid JSON: {tool_call[:200]}")
                )
                continue

            name = call.get("name", "")
            args = call.get("args", {}) or {}
            if self.role.allowed_tools and name not in self.role.allowed_tools:
                messages.append(
                    Message("user", f"TOOL_ERROR: tool {name} not allowed for this role")
                )
                continue

            tool_result = await self.tools.call(name, self.sandbox, args)
            result.tool_calls += 1
            messages.append(
                Message("user", f"<tool_result name=\"{name}\">{tool_result}</tool_result>")
            )

        result.final_text = "(agent hit max steps)"
        return result


def _extract_tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None
