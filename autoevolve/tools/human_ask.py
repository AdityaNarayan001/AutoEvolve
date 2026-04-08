from __future__ import annotations

from typing import Awaitable, Callable

from ..llm.base import ToolSpec
from ..sandbox import Sandbox
from .registry import Tool


HumanAskFn = Callable[[str, str], Awaitable[str]]


def build_human_ask_tool(ask_fn: HumanAskFn) -> Tool:
    async def _run(sandbox: Sandbox, args: dict) -> str:
        q = args.get("question", "")
        urgency = args.get("urgency", "normal")
        return await ask_fn(q, urgency)

    return Tool(
        spec=ToolSpec(
            name="human_ask",
            description="Ask the human user a question via Telegram (or local CLI fallback). Blocks until they reply.",
            input_schema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "urgency": {
                        "type": "string",
                        "enum": ["low", "normal", "high"],
                        "default": "normal",
                    },
                },
                "required": ["question"],
            },
        ),
        fn=_run,
    )
