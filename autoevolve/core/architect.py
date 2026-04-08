"""The Architect: an autonomous agent that owns the build.

The Architect receives the master plan and decides everything: how to break the
work down, which experts to spawn, in what order, when to test, when to call
the Judge, and when the result is excellent enough to stop. It is itself an
Agent — but its tool registry is extended with `spawn_subagent`,
`request_judgment`, and `record_iteration` so it can drive the recursion.

The orchestrator becomes thin: build the Architect, run it, persist final
state. No fixed iteration loop, no fixed plan-then-execute split — the
Architect makes those calls itself."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..llm.base import ToolSpec
from ..sandbox import Sandbox
from ..tools import Tool, ToolRegistry
from .agent import Agent
from .role_forge import RoleForge, RoleSpec
from .state import Iteration, TaskState

if TYPE_CHECKING:
    from ..llm.base import Backend
    from .judge import Judge


ARCHITECT_SYSTEM = """You are the Architect of an autonomous build system.

You have already received a master plan. Your job is to deliver the *perfect*
final result for the user's task. You decide everything:

  - How to decompose the work
  - Which expert subagents to spawn (mechanical engineer, control systems
    engineer, simulation worker, code reviewer, whatever fits)
  - In what order, in how many passes
  - When to test, when to refine, when to call the Judge
  - When the result is excellent enough to stop

You have these tools beyond the usual shell/python/fs ones:

  - `spawn_subagent(role_description, task)` — forge a brand-new domain expert
    on the fly and run them to completion on a focused subtask. Returns their
    final output. Use this aggressively: a fresh expert is cheap, and quality
    comes from the right mind on the right slice of work.
  - `request_judgment(summary)` — ask the strict Judge to score current
    progress against the requirements. Use this whenever you think you might
    be close, or before claiming a milestone.
  - `record_iteration(summary, score, passed)` — log a meaningful unit of work
    so the user can see progress and so the run can be resumed.

You can also call `shell`, `python_exec`, and `fs_*` directly for quick checks.

Quality bar: the goal is the perfect output, not minimum steps. Loop as many
times as needed. Iterate until you and the Judge agree the result is excellent.

When (and only when) the Judge has passed AND you are confident the artifacts
fully satisfy the user, emit `<final>brief summary of what was delivered</final>`
to terminate."""


def make_architect_tools(
    forge: RoleForge,
    backend: "Backend",
    base_registry: ToolRegistry,
    sandbox: Sandbox,
    state: TaskState,
    judge: "Judge",
    should_continue,
) -> list[Tool]:
    """Build the Architect-only tools as closures over the live orchestration
    context. They share the same workspace sandbox so spawned subagents see the
    Architect's files and vice versa."""

    async def spawn(_sb: Sandbox, args: dict) -> str:
        role_desc = args.get("role_description", "").strip()
        sub_task = args.get("task", "").strip()
        if not role_desc or not sub_task:
            return "ERROR: spawn_subagent needs role_description and task"

        spec = await forge.forge(
            state.task.requirements,
            f"{role_desc}: {sub_task}",
            state.task.domain_hints,
        )
        state.role_specs.append(spec.to_dict())
        state.append_event("role_forged", title=spec.title)

        subagent = Agent(
            role=spec,
            backend=backend,
            tools=base_registry,
            sandbox=sandbox,
            max_steps=40,
            should_continue=should_continue,
        )
        ar = await subagent.run(sub_task)
        state.tokens_in += ar.tokens_in
        state.tokens_out += ar.tokens_out
        state.append_event(
            "subtask_done",
            role=spec.title,
            tool_calls=ar.tool_calls,
            tokens_in=ar.tokens_in,
            tokens_out=ar.tokens_out,
        )
        state.save()
        return f"[subagent: {spec.title}]\n{ar.final_text}"

    async def request_judgment(_sb: Sandbox, args: dict) -> str:
        summary = args.get("summary", "").strip()
        if not summary:
            return "ERROR: request_judgment needs a summary"
        verdict = await judge.evaluate(
            state.task.requirements, summary, state.task.done
        )
        state.append_event(
            "judgment",
            score=verdict.score,
            passed=verdict.passed,
            rationale=verdict.rationale,
        )
        state.save()
        return (
            f"score={verdict.score:.0f} passed={verdict.passed}\n"
            f"rationale: {verdict.rationale}"
        )

    async def record_iteration(_sb: Sandbox, args: dict) -> str:
        summary = args.get("summary", "")
        score = float(args.get("score", 0))
        passed = bool(args.get("passed", False))
        n = len(state.iterations) + 1
        it = Iteration(
            n=n,
            role="architect",
            summary=summary,
            judge_score=score,
            judge_passed=passed,
        )
        state.iterations.append(it)
        state.checkpoint(label=f"iter {n}")
        state.save()
        state.append_event(
            "iteration_end",
            n=n,
            score=score,
            passed=passed,
            rationale="architect-marked",
        )
        return f"recorded iteration {n}"

    return [
        Tool(
            spec=ToolSpec(
                name="spawn_subagent",
                description=(
                    "Forge a fresh expert role on the fly and run a subagent on "
                    "a focused subtask. Returns the subagent's final output. "
                    "Use this whenever a piece of work would benefit from a "
                    "different specialist mindset."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "role_description": {
                            "type": "string",
                            "description": "What kind of expert (e.g. 'mechanical engineer specializing in suspension dynamics')",
                        },
                        "task": {
                            "type": "string",
                            "description": "The exact task for that subagent",
                        },
                    },
                    "required": ["role_description", "task"],
                },
            ),
            fn=spawn,
        ),
        Tool(
            spec=ToolSpec(
                name="request_judgment",
                description="Ask the strict Judge to score current progress against the requirements.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Summary of what has been done so far",
                        }
                    },
                    "required": ["summary"],
                },
            ),
            fn=request_judgment,
        ),
        Tool(
            spec=ToolSpec(
                name="record_iteration",
                description="Mark a meaningful unit of work as complete in the task log so the user sees progress.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "score": {"type": "number"},
                        "passed": {"type": "boolean"},
                    },
                    "required": ["summary"],
                },
            ),
            fn=record_iteration,
        ),
    ]


def build_architect_role(master_plan: dict) -> RoleSpec:
    import json

    plan_str = json.dumps(master_plan, indent=2)[:3500] if master_plan else "(no plan)"
    return RoleSpec(
        title="Architect",
        expertise="autonomous orchestration of expert subagents",
        system_prompt=ARCHITECT_SYSTEM + "\n\nMASTER PLAN:\n" + plan_str,
        success_heuristics=[
            "spawn the right expert for each slice",
            "iterate until the Judge passes",
            "prioritize quality over speed",
        ],
        allowed_tools=[],  # all tools allowed
    )
