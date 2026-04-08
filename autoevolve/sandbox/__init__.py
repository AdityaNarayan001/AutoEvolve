"""Sandboxed execution for agent tools.

Two implementations:
- DockerSandbox: each task gets a dedicated container with only its workspace
  bind-mounted. Network is off by default.
- LocalSandbox: macOS / Linux fallback using sandbox-exec or a confined
  subprocess. Workspace is the only writable path.

The orchestrator picks one via Settings.sandbox_mode = "auto" | "docker" | "local".
Both expose the same async interface so tools can stay agnostic."""

from .base import Sandbox, ExecResult
from .factory import make_sandbox

__all__ = ["Sandbox", "ExecResult", "make_sandbox"]
