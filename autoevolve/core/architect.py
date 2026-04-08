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


ARCHITECT_SYSTEM = """You are the Architect — the god of this autonomous build
system. You have absolute authority and absolute responsibility for the final
result.

ABSOLUTE RULES (the system enforces these — you cannot bypass them):

  1. You MUST do real work using tools. Talking about a plan is not work.
     Writing files, running code, spawning subagents IS work.
  2. You MUST spawn at least one expert subagent via `spawn_subagent` for any
     task that has domain content. The user expects expert quality.
  3. You MUST call `request_judgment` and receive `passed=true` (score >= 85)
     before you are allowed to finish.
  4. You MUST verify real artifacts exist in the workspace (files, code,
     plots, reports — not just words) before finishing.
  5. If you emit `<final>` without satisfying rules 1-4, the system will
     reject it and force you to keep working. Don't waste turns trying.

YOUR TOOLS:

  Architect-only:
    - `spawn_subagent(role_description, task)` — forge a fresh domain expert
      on the fly and run them to completion. Returns their final output. Use
      this aggressively. Quality comes from the right mind on the right slice.
    - `request_judgment(summary)` — strict Judge scores progress vs.
      requirements. Required before finishing.
    - `record_iteration(summary, score, passed)` — log a meaningful unit so
      the user sees progress.

  General work tools:
    - `shell(cmd)` — run any command in the sandboxed workspace
    - `python_exec(code)` — run Python (numpy/scipy/matplotlib) for math/plots
    - `fs_write(path, content)` / `fs_read(path)` / `fs_list()` — workspace files
    - `human_ask(question)` — ask the user via Telegram if you're stuck

WORKFLOW:

  1. Read the master plan. Identify the first phase.
  2. For each phase, spawn the right expert subagent(s) to actually build
     the artifacts. They will write files into the shared workspace.
  3. Verify their output with `fs_list` / `shell`. If insufficient, spawn
     more subagents or refine.
  4. When you believe a phase is done, call `request_judgment(summary)`.
  5. If judge says not passed, read the rationale and keep working — spawn
     more experts, fix gaps, refine.
  6. Only after `request_judgment` returns `passed=true`, emit
     `<final>brief summary of artifacts delivered</final>`.

Quality bar: the goal is the *perfect* output, not the minimum steps. Loop as
many times as needed. Token cost is not a concern — correctness and
completeness are."""


def make_architect_validator(state: TaskState, arch_state: dict):
    """Build a validator the Agent calls each time the model emits <final>.
    Returns None to accept, or a string explaining the rejection."""

    def validate(result) -> str | None:
        # Rule 1: real work must have happened
        if result.tool_calls < 1:
            return (
                "You called zero tools. You must actually build something using "
                "spawn_subagent / shell / python_exec / fs_write before finishing."
            )
        # Rule 2: at least one expert must have been spawned
        if arch_state.get("subagents_spawned", 0) < 1:
            return (
                "You did not spawn any expert subagents. Use spawn_subagent to "
                "delegate domain work to the right specialist before finishing."
            )
        # Rule 3: judge must have passed at least once
        if not arch_state.get("last_judgment_passed", False):
            score = arch_state.get("last_judgment_score", 0)
            return (
                f"The Judge has not passed yet (last score={score:.0f}, need >=85). "
                "Call request_judgment(summary) and keep iterating until it passes."
            )
        # Rule 4: workspace must contain real artifacts
        try:
            files = [p for p in state.workspace.iterdir() if not p.name.startswith(".")]
        except Exception:
            files = []
        if not files:
            return (
                "The workspace is empty. No files have been created. Spawn a "
                "subagent to actually write the artifacts before finishing."
            )
        return None  # accept

    return validate


def make_architect_tools(
    forge: RoleForge,
    backend: "Backend",
    base_registry: ToolRegistry,
    sandbox: Sandbox,
    state: TaskState,
    judge: "Judge",
    should_continue,
    arch_state: dict | None = None,
) -> list[Tool]:
    """Build the Architect-only tools as closures over the live orchestration
    context. They share the same workspace sandbox so spawned subagents see the
    Architect's files and vice versa.

    `arch_state` is a mutable dict the Architect's validator inspects to decide
    whether the run is allowed to finish (real work happened, judge passed)."""
    if arch_state is None:
        arch_state = {}
    arch_state.setdefault("subagents_spawned", 0)
    arch_state.setdefault("last_judgment_passed", False)
    arch_state.setdefault("last_judgment_score", 0.0)

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
        arch_state["subagents_spawned"] = arch_state.get("subagents_spawned", 0) + 1
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
        arch_state["last_judgment_passed"] = verdict.passed
        arch_state["last_judgment_score"] = verdict.score
        state.append_event(
            "judgment",
            score=verdict.score,
            passed=verdict.passed,
            rationale=verdict.rationale,
        )
        state.save()
        return (
            f"score={verdict.score:.0f} passed={verdict.passed}\n"
            f"rationale: {verdict.rationale}\n"
            + ("You may now finish with <final>...</final>." if verdict.passed
               else "Not ready to finish — keep working until score >= 85.")
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
