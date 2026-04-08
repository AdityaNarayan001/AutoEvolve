"""Tool registry. Each tool is an async callable that takes a Sandbox and a
JSON-shaped argument dict and returns a string result the LLM can read."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..llm.base import ToolSpec
from ..sandbox import Sandbox


ToolFn = Callable[[Sandbox, dict[str, Any]], Awaitable[str]]


@dataclass
class Tool:
    spec: ToolSpec
    fn: ToolFn


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    async def call(self, name: str, sandbox: Sandbox, args: dict[str, Any]) -> str:
        t = self._tools.get(name)
        if not t:
            return f"ERROR: unknown tool {name}"
        try:
            return await t.fn(sandbox, args)
        except Exception as e:  # surfaced back to the LLM, never crashes the loop
            return f"ERROR: {type(e).__name__}: {e}"


def build_default_registry(human_ask_fn=None) -> ToolRegistry:
    from .shell import shell_tool
    from .fs import fs_read_tool, fs_write_tool, fs_list_tool
    from .python_exec import python_exec_tool
    from .human_ask import build_human_ask_tool

    reg = ToolRegistry()
    reg.register(shell_tool())
    reg.register(fs_read_tool())
    reg.register(fs_write_tool())
    reg.register(fs_list_tool())
    reg.register(python_exec_tool())
    if human_ask_fn is not None:
        reg.register(build_human_ask_tool(human_ask_fn))
    return reg
