"""Typer CLI entry point.

Default behavior of `autoevolve run "<task>"` is to fork into a background
daemon and return immediately, since Telegram + `autoevolve watch` are the
expected interaction surfaces afterwards."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer
from rich import print as rprint

from .config import SETTINGS
from .core.state import DoneConfig, Task, TaskState, list_runs
from .runtime import daemon

app = typer.Typer(add_completion=False, help="AutoEvolve recursive build system")


@app.command()
def run(
    requirements: str = typer.Argument(..., help="Task description"),
    backend: str = typer.Option("", help="claude_cli | litellm_http"),
    domain: list[str] = typer.Option([], help="Domain hints (repeatable)"),
    max_iters: int = typer.Option(0, help="0 = unlimited"),
    judge: bool = typer.Option(True, help="Use LLM judge"),
    tests: str = typer.Option("", help="Optional test command for done check"),
    human: bool = typer.Option(False, help="Require human approval"),
    foreground: bool = typer.Option(False, help="Run in this terminal instead of detaching"),
):
    """Launch a new task. Detaches into a daemon by default."""
    done = DoneConfig(
        llm_judge=judge,
        executable_tests=bool(tests),
        test_command=tests,
        human_approval=human,
    )
    task = Task.new(
        requirements=requirements,
        domain_hints=list(domain),
        done=done,
        max_iters=max_iters,
        backend=backend,
    )
    state = TaskState(task=task)
    state.save()
    rprint(f"[green]created[/green] task_id={task.id}")

    if foreground:
        from .llm import get_backend
        from .core.orchestrator import Orchestrator

        async def _human(q: str, urgency: str = "normal") -> str:
            return input(f"[ask:{urgency}] {q}\n> ")

        orch = Orchestrator(state, get_backend(backend or None), human_ask=_human)
        asyncio.run(orch.run())
        rprint(f"[cyan]final status:[/cyan] {state.status}")
    else:
        pid = daemon.spawn(task.id)
        rprint(f"[green]daemon pid:[/green] {pid}")
        rprint(f"[dim]watch:[/dim] autoevolve watch")
        rprint(f"[dim]stop:[/dim]  autoevolve stop {task.id}")


@app.command("runs")
def runs_cmd():
    for rid in list_runs():
        try:
            s = TaskState.load(rid)
            running = "[running]" if daemon.is_running(rid) else ""
            rprint(f"{rid:40s} {s.status:10s} iters={len(s.iterations):3d} {running}")
        except Exception as e:
            rprint(f"{rid}  (error: {e})")


@app.command()
def status(task_id: str):
    s = TaskState.load(task_id)
    rprint(json.dumps({
        "id": s.task.id,
        "status": s.status,
        "iterations": len(s.iterations),
        "tokens_in": s.tokens_in,
        "tokens_out": s.tokens_out,
        "running": daemon.is_running(task_id),
    }, indent=2))


@app.command()
def stop(task_id: str, force: bool = typer.Option(False, "--force", "-f")):
    ok = daemon.stop(task_id, force=force)
    rprint("stopped" if ok else "no daemon found")


@app.command()
def watch(focus: str = typer.Argument("", help="task_id to focus, or empty for latest")):
    from .ui.dashboard import watch as _watch

    _watch(focus or None)


@app.command()
def branch(task_id: str, iteration: int):
    s = TaskState.load(task_id)
    new = s.branch_from(iteration)
    rprint(f"[green]branched[/green] -> {new.task.id}")


@app.command()
def web(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8765),
):
    """Launch the web UI (config + task input + live event stream)."""
    from .web.server import serve

    rprint(f"[green]AutoEvolve web UI:[/green] http://{host}:{port}")
    serve(host=host, port=port)


@app.command("ping-backend")
def ping_backend(backend: str = "litellm_http"):
    from .llm import get_backend
    from .llm.base import Message

    b = get_backend(backend)

    async def _go():
        r = await b.complete(system="", messages=[Message("user", "say pong")])
        print(r.text)

    asyncio.run(_go())


if __name__ == "__main__":
    app()
