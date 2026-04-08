from __future__ import annotations

import shutil
from pathlib import Path

from ..config import SETTINGS
from .base import Sandbox


def make_sandbox(workspace: Path) -> Sandbox:
    mode = SETTINGS.sandbox_mode
    if mode == "local":
        from .local import LocalSandbox

        return LocalSandbox(workspace)
    if mode == "docker":
        from .docker import DockerSandbox

        return DockerSandbox(workspace)
    # auto
    if shutil.which("docker"):
        from .docker import DockerSandbox

        return DockerSandbox(workspace)
    from .local import LocalSandbox

    return LocalSandbox(workspace)
