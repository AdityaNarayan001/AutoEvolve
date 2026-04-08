"""Local sandbox: confines subprocesses to the workspace directory.

This is the cross-platform fallback when Docker is unavailable. It is *less*
isolated than a container — agents share the host kernel and Python — but the
working directory is the only writable path the tools expose, and we never let
the agent see absolute host paths."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from .base import ExecResult, Sandbox


class LocalSandbox(Sandbox):
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def run(self, cmd: list[str], timeout: int = 600) -> ExecResult:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self.workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "HOME": str(self.workspace)},
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return ExecResult(124, "", f"timeout after {timeout}s")
        return ExecResult(
            proc.returncode or 0,
            stdout.decode(errors="ignore"),
            stderr.decode(errors="ignore"),
        )

    def _resolve(self, rel_path: str) -> Path:
        p = (self.workspace / rel_path).resolve()
        if not str(p).startswith(str(self.workspace.resolve())):
            raise PermissionError(f"path escapes workspace: {rel_path}")
        return p

    async def write_file(self, rel_path: str, content: str) -> None:
        p = self._resolve(rel_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    async def read_file(self, rel_path: str) -> str:
        return self._resolve(rel_path).read_text()
