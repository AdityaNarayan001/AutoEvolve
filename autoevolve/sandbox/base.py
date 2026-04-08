from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str


class Sandbox(ABC):
    """Abstract sandbox bound to a single workspace directory."""

    workspace: Path

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def run(self, cmd: list[str], timeout: int = 600) -> ExecResult: ...

    @abstractmethod
    async def write_file(self, rel_path: str, content: str) -> None: ...

    @abstractmethod
    async def read_file(self, rel_path: str) -> str: ...
