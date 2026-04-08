"""`autoevolve watch` — a live TUI tailing all active runs.

Read-only: it never holds a handle into the orchestrator process. It just tails
events.jsonl files, so the UI can crash freely without affecting any run."""

from __future__ import annotations

import json
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table

from ..config import SETTINGS
from ..core.state import list_runs, TaskState
from ..runtime.daemon import is_running


def _runs_table() -> Table:
    t = Table(title="AutoEvolve runs", expand=True)
    t.add_column("task_id")
    t.add_column("status")
    t.add_column("iter", justify="right")
    t.add_column("score", justify="right")
    t.add_column("tokens", justify="right")
    t.add_column("daemon")
    for rid in list_runs():
        try:
            s = TaskState.load(rid)
        except Exception:
            continue
        last = s.iterations[-1] if s.iterations else None
        t.add_row(
            rid,
            s.status,
            str(len(s.iterations)),
            f"{last.judge_score:.0f}" if last else "-",
            f"{s.tokens_in}/{s.tokens_out}",
            "yes" if is_running(rid) else "no",
        )
    return t


def _tail_events(task_id: str | None, lines: int = 20) -> str:
    if not task_id:
        return ""
    p = SETTINGS.runs_dir / task_id / "events.jsonl"
    if not p.exists():
        return "(no events)"
    return "\n".join(p.read_text().splitlines()[-lines:])


def watch(focus: str | None = None) -> None:
    console = Console()
    layout = Layout()
    layout.split_column(
        Layout(name="runs", size=12),
        Layout(name="events"),
    )
    with Live(layout, console=console, refresh_per_second=2, screen=True):
        while True:
            target = focus or (list_runs() or [None])[-1]
            layout["runs"].update(Panel(_runs_table(), title="runs"))
            layout["events"].update(
                Panel(_tail_events(target, 30), title=f"events: {target}")
            )
            time.sleep(0.5)
