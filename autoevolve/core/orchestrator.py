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
import traceback
from dataclasses import dataclass
from typing import Awaitable, Callable

from ..llm.base import Backend, Message
from ..sandbox import make_sandbox, Sandbox
from ..tools import build_default_registry
from .agent import Agent
from .architect import build_architect_role, make_architect_tools
from .judge import Judge
from .role_forge import RoleForge, RoleSpec, _extract_json
from .state import Iteration, TaskState


PLANNER_SYSTEM = """You are a Strategic Planner. Given a high-level task, produce
a complete master plan BEFORE any execution begins. Think hard about phases,
risks, success criteria, and the right experts to involve.

Output STRICT JSON:
{
  "summary": string,                      // 1-2 sentence restatement of the goal
  "phases": [
    {
      "name": string,                     // e.g. "Modeling", "Simulation", "Tuning"
      "goal": string,
      "expected_roles": [string, ...],    // which kinds of experts will be needed
      "deliverables": [string, ...]
    }
  ],
  "risks": [string, ...],
  "success_criteria": [string, ...],
  "estimated_iterations": number
}
Be domain-aware: if the task involves electrical/mechanical/control systems,
include those experts in expected_roles. No commentary outside the JSON."""


MANAGER_SYSTEM = """You are the Manager of a recursive build system. A master
plan already exists. Your job is to translate the next slice of that plan into
concrete subtasks for one iteration.

Output STRICT JSON:
{
  "thought": string,
  "phase": string,                          // which master-plan phase you're in
  "subtasks": [
    {"id": string, "description": string, "domain": string}
  ],
  "expected_artifacts": [string, ...]
}
Subtasks must be small enough for one expert to finish in one turn. Reference
the master plan's phases. No commentary outside the JSON."""


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
            tb = traceback.format_exc()
            self.state.status = "error"
            self.state.last_error = f"{type(e).__name__}: {e}\n{tb}"
            self._emit("error", message=f"{type(e).__name__}: {e}", traceback=tb)
            print(tb, flush=True)
        finally:
            try:
                await self.sandbox.stop()
            except Exception:
                pass
            self.state.finished_at = time.time()
            self.state.save()
            self._emit("run_end", status=self.state.status)

    async def _loop(self) -> None:
        # Phase 1 — Plan mode: produce a master plan once before any execution.
        if not self.state.master_plan:
            self._emit("planning_start")
            self.state.master_plan = await self._make_master_plan()
            self.state.save()
            self._emit("master_plan", plan=self.state.master_plan)

        # Phase 2 — Architect mode: god of the run.
        # Build the Architect's tool registry: regular tools + the architect-only
        # tools (spawn_subagent, request_judgment, record_iteration). Same
        # workspace sandbox is shared with every spawned subagent.
        arch_registry = build_default_registry(human_ask_fn=self.human_ask)
        for t in make_architect_tools(
            forge=self.forge,
            backend=self.backend,
            base_registry=self.tools,
            sandbox=self.sandbox,
            state=self.state,
            judge=self.judge,
            should_continue=self._should_continue,
        ):
            arch_registry.register(t)
        self.architect_tools = arch_registry

        architect_role = build_architect_role(self.state.master_plan)
        if not any(r.get("title") == "Architect" for r in self.state.role_specs):
            self.state.role_specs.append(architect_role.to_dict())

        self._emit("architect_start")
        architect = Agent(
            role=architect_role,
            backend=self.backend,
            tools=arch_registry,
            sandbox=self.sandbox,
            max_steps=500,  # generous; the architect decides when to stop
            should_continue=self._should_continue,
            get_nudge=self._consume_nudge,
        )

        user_prompt = (
            f"USER TASK:\n{self.state.task.requirements}\n\n"
            f"DOMAIN HINTS: {', '.join(self.state.task.domain_hints) or 'none'}\n\n"
            "Deliver the perfect result. Spawn whatever experts you need, "
            "iterate as many times as needed, and stop only when the Judge has "
            "passed and you are confident the artifacts are excellent."
        )
        result = await architect.run(user_prompt)
        self.state.tokens_in += result.tokens_in
        self.state.tokens_out += result.tokens_out
        self._emit(
            "architect_end",
            tool_calls=result.tool_calls,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
        )

        # If the Architect emitted <final> without ever calling the Judge, run
        # one final judgment so the user gets a real verdict.
        recent_judged = any(
            it.judge_passed for it in self.state.iterations[-3:]
        )
        if not recent_judged:
            verdict = await self.judge.evaluate(
                self.state.task.requirements,
                result.final_text or "(architect emitted no summary)",
                self.state.task.done,
            )
            self._emit(
                "iteration_end",
                n=len(self.state.iterations) + 1,
                score=verdict.score,
                passed=verdict.passed,
                rationale=verdict.rationale,
            )
            self.state.iterations.append(
                Iteration(
                    n=len(self.state.iterations) + 1,
                    role="architect",
                    summary=result.final_text or "",
                    judge_score=verdict.score,
                    judge_passed=verdict.passed,
                )
            )
            self.state.save()

        self.state.status = "stopped" if self.flag.stop else "done"

    def _should_continue(self) -> bool:
        if self.flag.stop:
            return False
        # Honor pause: block here so the agent waits between steps
        while self.flag.pause and not self.flag.stop:
            import time as _t
            _t.sleep(0.5)
        return not self.flag.stop

    def _consume_nudge(self) -> str:
        n = self.flag.nudge
        self.flag.nudge = ""
        return n

    async def _make_master_plan(self) -> dict:
        user = (
            f"TASK: {self.state.task.requirements}\n"
            f"DOMAIN HINTS: {', '.join(self.state.task.domain_hints) or 'none'}\n\n"
            "Produce the master plan as JSON."
        )
        resp = await self.backend.complete(
            system=PLANNER_SYSTEM, messages=[Message("user", user)]
        )
        self.state.tokens_in += resp.input_tokens
        self.state.tokens_out += resp.output_tokens
        return _extract_json(resp.text) or {
            "summary": self.state.task.requirements,
            "phases": [],
            "risks": [],
            "success_criteria": [],
            "estimated_iterations": 0,
        }

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
        mp = json.dumps(self.state.master_plan)[:2000] if self.state.master_plan else "(none)"
        user = (
            f"TASK: {self.state.task.requirements}\n"
            f"DOMAIN HINTS: {', '.join(self.state.task.domain_hints) or 'none'}\n"
            f"MASTER PLAN:\n{mp}\n"
            f"HISTORY:\n{history or '(none)'}\n"
            f"LAST CRITIQUE:\n{critique or '(none)'}\n\n"
            "Produce the next iteration plan as JSON."
        )
        self._emit("planning")
        resp = await self.backend.complete(
            system=MANAGER_SYSTEM, messages=[Message("user", user)]
        )
        self.state.tokens_in += resp.input_tokens
        self.state.tokens_out += resp.output_tokens
        return _extract_json(resp.text) or {"subtasks": []}
