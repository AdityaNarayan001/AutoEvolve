from __future__ import annotations

from ..llm.base import ToolSpec
from ..sandbox import Sandbox
from .registry import Tool


async def _read(sandbox: Sandbox, args: dict) -> str:
    return await sandbox.read_file(args["path"])


async def _write(sandbox: Sandbox, args: dict) -> str:
    await sandbox.write_file(args["path"], args["content"])
    return f"wrote {args['path']}"


async def _list(sandbox: Sandbox, args: dict) -> str:
    r = await sandbox.run(["bash", "-lc", f"ls -la {args.get('path', '.')}"])
    return r.stdout


def fs_read_tool() -> Tool:
    return Tool(
        spec=ToolSpec(
            name="fs_read",
            description="Read a file from the workspace.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        fn=_read,
    )


def fs_write_tool() -> Tool:
    return Tool(
        spec=ToolSpec(
            name="fs_write",
            description="Write a file in the workspace (overwrites).",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        ),
        fn=_write,
    )


def fs_list_tool() -> Tool:
    return Tool(
        spec=ToolSpec(
            name="fs_list",
            description="List files in the workspace.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
            },
        ),
        fn=_list,
    )
