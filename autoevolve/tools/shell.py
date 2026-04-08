from __future__ import annotations

from ..llm.base import ToolSpec
from ..sandbox import Sandbox
from .registry import Tool


async def _run(sandbox: Sandbox, args: dict) -> str:
    cmd = args.get("cmd")
    if not cmd:
        return "ERROR: cmd required"
    timeout = int(args.get("timeout", 600))
    parts = cmd if isinstance(cmd, list) else ["bash", "-lc", str(cmd)]
    r = await sandbox.run(parts, timeout=timeout)
    return f"exit={r.exit_code}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"


def shell_tool() -> Tool:
    return Tool(
        spec=ToolSpec(
            name="shell",
            description="Run a shell command inside the sandboxed workspace.",
            input_schema={
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Command to run"},
                    "timeout": {"type": "integer", "default": 600},
                },
                "required": ["cmd"],
            },
        ),
        fn=_run,
    )
