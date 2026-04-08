"""The Watcher: a read-only subagent that answers user questions about a
running task in real time.

When the user sends a free-text message on Telegram (or asks "what is
happening?"), the bridge calls `answer_question(state, backend, question)`.
The Watcher snapshots the current task state — requirements, master plan,
forged roles, recent iterations, last 30 events, workspace listing — and asks
the LLM to answer in plain language. It never modifies anything."""

from __future__ import annotations

from ..llm.base import Backend, Message
from .state import TaskState


WATCHER_SYSTEM = """You are the Watcher of an autonomous build system.

A user is interacting with the system over Telegram and wants real answers
about what is happening, why decisions were made, what's left, what went
wrong, or what an artifact contains.

You are read-only. You will be given a snapshot of the current task state and
the user's question. Answer concisely, factually, and in plain language. If
the answer is not in the context, say so honestly — do not invent.

Keep responses under 1500 characters unless the user explicitly asks for
detail. Use short bullets for lists. Refer to roles, iterations, and events
by name when useful."""


async def answer_question(
    state: TaskState, backend: Backend, question: str
) -> str:
    s = state

    # Recent iterations
    iter_block = "\n".join(
        f"#{it.n} score={it.judge_score:.0f} passed={it.judge_passed}: {it.summary[:240]}"
        for it in s.iterations[-6:]
    ) or "(none yet)"

    # Recent events
    events_log = s.dir / "events.jsonl"
    events_tail = ""
    if events_log.exists():
        try:
            events_tail = "\n".join(events_log.read_text().splitlines()[-30:])
        except Exception:
            events_tail = "(unreadable)"
    events_tail = events_tail or "(none)"

    # Workspace files
    files: list[str] = []
    try:
        files = [p.name for p in s.workspace.iterdir()]
    except Exception:
        pass

    roles = ", ".join(r.get("title", "?") for r in s.role_specs) or "(none yet)"
    plan_summary = (s.master_plan or {}).get("summary", "(no plan yet)")
    plan_phases = (s.master_plan or {}).get("phases", []) or []
    phases_str = "\n".join(
        f"  - {p.get('name','?')}: {p.get('goal','')}" for p in plan_phases
    ) or "  (none)"

    context = (
        f"TASK ID: {s.task.id}\n"
        f"REQUIREMENTS: {s.task.requirements}\n"
        f"STATUS: {s.status}\n"
        f"BACKEND: {s.task.backend or '(default)'}\n"
        f"TOKENS: {s.tokens_in} in / {s.tokens_out} out\n"
        f"PLAN SUMMARY: {plan_summary}\n"
        f"PLAN PHASES:\n{phases_str}\n"
        f"FORGED ROLES SO FAR: {roles}\n"
        f"WORKSPACE FILES: {', '.join(files) or '(none)'}\n\n"
        f"RECENT ITERATIONS (newest last):\n{iter_block}\n\n"
        f"RECENT EVENTS (newest last):\n{events_tail}\n\n"
        f"USER QUESTION: {question}\n\n"
        "Answer the user."
    )

    resp = await backend.complete(
        system=WATCHER_SYSTEM, messages=[Message("user", context)]
    )
    return (resp.text or "(no response)").strip()
