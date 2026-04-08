"""The main control loop.

For each iteration:
  1. Manager agent (forged on first iter, reused after) produces a plan:
     a JSON list of subtasks, each with a worker role to spawn.
  2. For each subtask, the Role Forge generates a fresh expert role, an Agent
     runs in the same shared workspace sandbox, and produces an artifact summary.
  3. The Judge evaluates the iteration against DoneConfig.
  4. If passed, we stop. Otherwise, the manager sees the judge's critique and
     replans.

Stopping conditions: judge pass, max_iters reached (if >0), human stop signal,
unrecoverable error. There is no token/cost cap.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from ..llm.base import Backend, Message
from ..sandbox import make_sandbox, Sandbox
from ..tools import build_default_registry
from .agent import Agent
from .judge import Judge
from .role_forge import RoleForge, RoleSpec, _extract_json
from .state import Iteration, TaskState


MANAGER_SYSTEM = """You are the Manager of a recursive build system.
Given the task, prior history, and the latest critique (if any), produce the
next iteration plan as STRICT JSON:
{
  "thought": string,
  "subtasks": [
    {"id": string, "description": string, "domain": string}
  ],
  "expected_artifacts": [string, ...]
}
Be specific. Subtasks should be small enough that one expert can finish in one
turn. No commentary outside the JSON."""


HumanFn = Callable[[str, str], Awaitable[str]]


@dataclass
class StopFlag:
    stop: bool = False
    pause: bool = False
    nudge: str = ""  # injected user guidance


class Orchestrator:
    def __init__(
        self,
        state: TaskState,
        backend: Backend,
        human_ask: HumanFn | None = None,
        on_event: Callable[[str, dict], None] | None = None,
    ):
        self.state = state
        self.backend = backend
        self.human_ask = human_ask
        self.on_event = on_event or (lambda k, p: None)
        self.flag = StopFlag()
        self.sandbox: Sandbox = make_sandbox(state.workspace)
        self.tools = build_default_registry(human_ask_fn=human_ask)
        self.forge = RoleForge(backend)
        self.judge = Judge(backend, self.sandbox, human_ask)

    def _emit(self, kind: str, **payload) -> None:
        self.state.append_event(kind, **payload)
        self.on_event(kind, payload)

    async def run(self) -> None:
        self.state.status = "running"
        self.state.started_at = self.state.started_at or time.time()
        self.state.save()
        self._emit("run_start", task_id=self.state.task.id)

        try:
            await self.sandbox.start()
            await self._loop()
        except Exception as e:
            self.state.status = "error"
            self.state.last_error = f"{type(e).__name__}: {e}"
            self._emit("error", message=self.state.last_error)
        finally:
            try:
                await self.sandbox.stop()
            except Exception:
                pass
            self.state.finished_at = time.time()
            self.state.save()
            self._emit("run_end", status=self.state.status)

    async def _loop(self) -> None:
        critique = ""
        manager_role = await self._forge_manager()
        self.state.role_specs.append(manager_role.to_dict())

        while True:
            if self.flag.stop:
                self.state.status = "stopped"
                return
            while self.flag.pause:
                await asyncio.sleep(1)
            if self.state.task.max_iters and len(self.state.iterations) >= self.state.task.max_iters:
                self.state.status = "done"
                self._emit("max_iters_reached")
                return

            n = len(self.state.iterations) + 1
            self._emit("iteration_start", n=n)

            plan = await self._plan(critique)
            self._emit("plan", n=n, plan=plan)

            summaries: list[str] = []
            for st in plan.get("subtasks", []) or []:
                if self.flag.stop:
                    break
                role = await self.forge.forge(
                    self.state.task.requirements,
                    st.get("description", ""),
                    self.state.task.domain_hints + [st.get("domain", "")],
                )
                self.state.role_specs.append(role.to_dict())
                self._emit("role_forged", title=role.title)

                agent = Agent(role, self.backend, self.tools, self.sandbox)
                prompt = (
                    f"TASK: {self.state.task.requirements}\n"
                    f"SUBTASK: {st.get('description', '')}\n"
                    f"Use tools to make real progress. End with <final>summary of what you did</final>."
                )
                ar = await agent.run(prompt)
                self.state.tokens_in += ar.tokens_in
                self.state.tokens_out += ar.tokens_out
                summaries.append(f"[{role.title}] {ar.final_text}")
                self._emit(
                    "subtask_done",
                    role=role.title,
                    tool_calls=ar.tool_calls,
                    tokens_in=ar.tokens_in,
                    tokens_out=ar.tokens_out,
                )

            iteration_summary = "\n\n".join(summaries) or "(no work performed)"
            verdict = await self.judge.evaluate(
                self.state.task.requirements,
                iteration_summary,
                self.state.task.done,
            )

            it = Iteration(
                n=n,
                role="manager",
                summary=iteration_summary,
                judge_score=verdict.score,
                judge_passed=verdict.passed,
            )
            self.state.iterations.append(it)
            self.state.checkpoint(label=f"iter {n}")
            self.state.save()
            self._emit(
                "iteration_end",
                n=n,
                score=verdict.score,
                passed=verdict.passed,
                rationale=verdict.rationale,
            )

            if verdict.passed:
                self.state.status = "done"
                return

            critique = verdict.rationale
            if self.flag.nudge:
                critique += f"\nUSER NUDGE: {self.flag.nudge}"
                self.flag.nudge = ""

    async def _forge_manager(self) -> RoleSpec:
        return RoleSpec(
            title="Manager",
            expertise="planning, decomposition, coordination",
            system_prompt=MANAGER_SYSTEM,
            success_heuristics=["produce small, executable subtasks"],
            allowed_tools=[],
        )

    async def _plan(self, critique: str) -> dict:
        history = "\n".join(
            f"#{it.n} score={it.judge_score:.0f} passed={it.judge_passed}: {it.summary[:300]}"
            for it in self.state.iterations[-5:]
        )
        user = (
            f"TASK: {self.state.task.requirements}\n"
            f"DOMAIN HINTS: {', '.join(self.state.task.domain_hints) or 'none'}\n"
            f"HISTORY:\n{history or '(none)'}\n"
            f"LAST CRITIQUE:\n{critique or '(none)'}\n\n"
            "Produce the next iteration plan as JSON."
        )
        resp = await self.backend.complete(
            system=MANAGER_SYSTEM, messages=[Message("user", user)]
        )
        return _extract_json(resp.text) or {"subtasks": []}
