"""Docker sandbox. One container per task, workspace bind-mounted, no network."""

from __future__ import annotations

import asyncio
import os
import shlex
import uuid
from pathlib import Path

from ..config import SETTINGS
from .base import ExecResult, Sandbox


class DockerSandbox(Sandbox):
    def __init__(self, workspace: Path, image: str | None = None):
        self.workspace = workspace
        self.image = image or SETTINGS.sandbox_image
        self.container = f"autoevolve-{uuid.uuid4().hex[:10]}"
        self._started = False

    async def _docker(self, *args: str, timeout: int | None = None) -> ExecResult:
        proc = await asyncio.create_subprocess_exec(
            "docker", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout or 600
            )
        except asyncio.TimeoutError:
            proc.kill()
            return ExecResult(124, "", "docker timeout")
        return ExecResult(
            proc.returncode or 0,
            stdout.decode(errors="ignore"),
            stderr.decode(errors="ignore"),
        )

    async def start(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        # Long-lived container; we exec into it for each tool call.
        r = await self._docker(
            "run", "-d", "--rm",
            "--name", self.container,
            "--network", "none",
            "--cpus", SETTINGS.sandbox_cpu,
            "--memory", SETTINGS.sandbox_mem,
            # Use the host user's uid/gid so files in the bind-mounted workspace
            # are readable/writable from both sides. Falls back to root in the
            # container if uid lookup fails (e.g. on non-POSIX hosts).
            "--user", f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else "0:0",
            "-v", f"{self.workspace}:/workspace",
            "-w", "/workspace",
            self.image,
            "sleep", "infinity",
        )
        if r.exit_code != 0:
            raise RuntimeError(f"docker run failed: {r.stderr}")
        self._started = True

    async def stop(self) -> None:
        if self._started:
            await self._docker("kill", self.container)
            self._started = False

    async def run(self, cmd: list[str], timeout: int = 600) -> ExecResult:
        if not self._started:
            await self.start()
        return await self._docker(
            "exec", self.container, "bash", "-lc", shlex.join(cmd), timeout=timeout
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
