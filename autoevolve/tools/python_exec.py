from __future__ import annotations

import uuid

from ..llm.base import ToolSpec
from ..sandbox import Sandbox
from .registry import Tool


async def _run(sandbox: Sandbox, args: dict) -> str:
    code = args.get("code", "")
    if not code:
        return "ERROR: code required"
    name = f".scratch_{uuid.uuid4().hex[:8]}.py"
    await sandbox.write_file(name, code)
    r = await sandbox.run(["python", name], timeout=int(args.get("timeout", 600)))
    return f"exit={r.exit_code}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"


def python_exec_tool() -> Tool:
    return Tool(
        spec=ToolSpec(
            name="python_exec",
            description="Execute a Python snippet in the sandbox. Has numpy/scipy/matplotlib if image installed.",
            input_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "timeout": {"type": "integer", "default": 600},
                },
                "required": ["code"],
            },
        ),
        fn=_run,
    )
