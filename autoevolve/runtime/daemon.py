"""Background daemon: double-fork the orchestrator so it survives terminal close."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

from ..config import SETTINGS
from ..core.orchestrator import Orchestrator
from ..core.state import TaskState
from ..llm import get_backend


def _pidfile(task_id: str) -> Path:
    return SETTINGS.runs_dir / task_id / "orchestrator.pid"


def is_running(task_id: str) -> bool:
    pf = _pidfile(task_id)
    if not pf.exists():
        return False
    try:
        pid = int(pf.read_text().strip())
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def stop(task_id: str, force: bool = False) -> bool:
    pf = _pidfile(task_id)
    if not pf.exists():
        return False
    try:
        pid = int(pf.read_text().strip())
        os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)
        return True
    except Exception:
        return False


def spawn(task_id: str) -> int:
    """Double-fork; returns the daemon pid in the parent."""
    r, w = os.pipe()
    pid = os.fork()
    if pid > 0:
        os.close(w)
        os.waitpid(pid, 0)
        data = os.read(r, 64).decode().strip() or "0"
        os.close(r)
        return int(data)

    # First child
    os.close(r)
    os.setsid()
    pid2 = os.fork()
    if pid2 > 0:
        os.write(w, str(pid2).encode())
        os.close(w)
        os._exit(0)

    # Second child = the daemon
    os.close(w)
    log_path = SETTINGS.runs_dir / task_id / "log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(0)

    _pidfile(task_id).write_text(str(os.getpid()))
    try:
        asyncio.run(_run_daemon(task_id))
    finally:
        try:
            _pidfile(task_id).unlink()
        except Exception:
            pass
    os._exit(0)


async def _run_daemon(task_id: str) -> None:
    state = TaskState.load(task_id)
    backend = get_backend(state.task.backend or None)

    # Lazy import to avoid pulling telegram deps when not configured
    human_ask = None
    bridge = None
    if SETTINGS.telegram_token and SETTINGS.telegram_chat_id:
        try:
            from ..bridges.telegram import TelegramBridge

            bridge = TelegramBridge(state)
            await bridge.start()
            human_ask = bridge.ask_human
        except Exception as e:
            print(f"telegram bridge unavailable: {e}", flush=True)

    if human_ask is None:
        async def human_ask(q: str, urgency: str = "normal") -> str:  # noqa
            print(f"[human_ask:{urgency}] {q}", flush=True)
            return "ok"

    orch = Orchestrator(state, backend, human_ask=human_ask)

    if bridge is not None:
        bridge.attach_orchestrator(orch)

    await orch.run()

    if bridge is not None:
        await bridge.notify(f"Run {task_id} finished: {state.status}")
        await bridge.stop()
