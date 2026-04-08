"""Judge: decide whether the current iteration satisfies the task.

Combines (AND) any enabled DoneConfig criteria:
- llm_judge: an LLM scores artifacts vs. requirements (0-100, threshold 85)
- executable_tests: run a test command in the sandbox; pass = exit 0
- human_approval: pass through to a human (Telegram or local prompt)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Awaitable, Callable

from ..llm.base import Backend, Message
from ..sandbox import Sandbox
from .role_forge import _extract_json
from .state import DoneConfig


@dataclass
class Verdict:
    passed: bool
    score: float
    rationale: str


JUDGE_SYSTEM = """You are a strict acceptance Judge. Score (0-100) how well the
current artifacts and summary satisfy the requirements. Output STRICT JSON:
{"score": number, "passed": boolean, "rationale": string}
Pass only if score >= 85 AND no critical gaps."""


HumanFn = Callable[[str, str], Awaitable[str]]


class Judge:
    def __init__(
        self,
        backend: Backend,
        sandbox: Sandbox,
        human_ask: HumanFn | None = None,
    ):
        self.backend = backend
        self.sandbox = sandbox
        self.human_ask = human_ask

    async def evaluate(
        self,
        requirements: str,
        iteration_summary: str,
        done: DoneConfig,
    ) -> Verdict:
        results: list[Verdict] = []

        if done.llm_judge:
            results.append(await self._llm_judge(requirements, iteration_summary))

        if done.executable_tests and done.test_command:
            results.append(await self._tests(done.test_command))

        if done.human_approval and self.human_ask:
            ans = await self.human_ask(
                f"Approve current iteration? (yes/no)\n\n{iteration_summary[:1500]}",
                "high",
            )
            ok = ans.strip().lower().startswith("y")
            results.append(Verdict(ok, 100.0 if ok else 0.0, f"human: {ans[:200]}"))

        if not results:
            return Verdict(False, 0.0, "no done criteria enabled")

        passed = all(v.passed for v in results)
        score = sum(v.score for v in results) / len(results)
        rationale = " | ".join(v.rationale for v in results)
        return Verdict(passed, score, rationale)

    async def _llm_judge(self, requirements: str, summary: str) -> Verdict:
        # Cheap static gate first: if any .py file in the workspace fails to
        # parse, the iteration is broken regardless of what the LLM thinks.
        # Same for syntactically-checkable JSON. This saves a judge call on
        # obvious failures and gives the manager a precise rationale.
        syntax_err = await self._syntax_gate()
        if syntax_err:
            return Verdict(False, 0.0, f"syntax: {syntax_err[:400]}")

        # Build a richer workspace view: a tree (depth 3) + a small head of
        # each tracked source file so the judge can actually evaluate quality.
        tree = await self.sandbox.run(
            ["bash", "-lc", "find . -maxdepth 3 -not -path '*/.*' | head -120"]
        )
        excerpts = await self.sandbox.run(
            [
                "bash",
                "-lc",
                "for f in $(find . -maxdepth 3 -type f \\( -name '*.py' -o -name '*.md' "
                "-o -name '*.json' -o -name '*.txt' \\) -not -path '*/.*' | head -8); do "
                "echo '--- '\"$f\"' ---'; head -40 \"$f\"; done",
            ]
        )
        user = (
            f"REQUIREMENTS:\n{requirements}\n\n"
            f"ITERATION SUMMARY:\n{summary}\n\n"
            f"WORKSPACE TREE:\n{tree.stdout[:3000]}\n\n"
            f"FILE EXCERPTS:\n{excerpts.stdout[:6000]}"
        )
        resp = await self.backend.complete(
            system=JUDGE_SYSTEM, messages=[Message("user", user)]
        )
        data = _extract_json(resp.text) or {}
        return Verdict(
            passed=bool(data.get("passed", False)),
            score=float(data.get("score", 0)),
            rationale=str(data.get("rationale", resp.text[:200])),
        )

    async def _syntax_gate(self) -> str:
        """Compile every .py and parse every .json in the workspace. Returns an
        error string on the first failure, or '' if everything parses."""
        py = await self.sandbox.run(
            [
                "bash",
                "-lc",
                "set -e; for f in $(find . -name '*.py' -not -path '*/.*'); do "
                "python -m py_compile \"$f\" || { echo \"PY_FAIL $f\"; exit 1; }; done",
            ]
        )
        if py.exit_code != 0:
            return (py.stderr or py.stdout)[:400]
        js = await self.sandbox.run(
            [
                "bash",
                "-lc",
                "for f in $(find . -name '*.json' -not -path '*/.*'); do "
                "python -c \"import json,sys;json.load(open('$f'))\" || "
                "{ echo \"JSON_FAIL $f\"; exit 1; }; done",
            ]
        )
        if js.exit_code != 0:
            return (js.stderr or js.stdout)[:400]
        return ""

    async def _tests(self, cmd: str) -> Verdict:
        r = await self.sandbox.run(["bash", "-lc", cmd])
        ok = r.exit_code == 0
        # Surface the failing tail so the manager can actually fix it,
        # instead of just seeing an exit code.
        tail = ""
        if not ok:
            combined = (r.stdout or "") + "\n" + (r.stderr or "")
            tail = " | " + combined.strip().splitlines()[-20:].__str__()[:600]
        return Verdict(
            passed=ok,
            score=100.0 if ok else 0.0,
            rationale=f"tests exit={r.exit_code}{tail}",
        )
