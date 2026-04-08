from __future__ import annotations

import os
import shutil
from pathlib import Path

from ..config import SETTINGS
from .base import Sandbox


ORBSTACK_SOCKETS = [
    Path.home() / ".orbstack" / "run" / "docker.sock",
    Path("/var/run/docker.sock"),  # OrbStack also symlinks here on macOS
]


def _orbstack_available() -> bool:
    return any(p.exists() for p in ORBSTACK_SOCKETS) and shutil.which("docker") is not None


def _use_orbstack() -> None:
    """Point the docker CLI at OrbStack's socket so it doesn't talk to Docker Desktop."""
    for p in ORBSTACK_SOCKETS:
        if p.exists():
            os.environ["DOCKER_HOST"] = f"unix://{p}"
            return


def make_sandbox(workspace: Path) -> Sandbox:
    mode = SETTINGS.sandbox_mode
    if mode == "local":
        from .local import LocalSandbox

        return LocalSandbox(workspace)
    if mode == "orbstack":
        from .docker import DockerSandbox

        _use_orbstack()
        return DockerSandbox(workspace)
    if mode == "docker":
        from .docker import DockerSandbox

        return DockerSandbox(workspace)
    # auto: prefer OrbStack if it's running, then any docker, then local
    if _orbstack_available():
        from .docker import DockerSandbox

        _use_orbstack()
        return DockerSandbox(workspace)
    if shutil.which("docker"):
        from .docker import DockerSandbox

        return DockerSandbox(workspace)
    from .local import LocalSandbox

    return LocalSandbox(workspace)
