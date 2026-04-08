"""Persistent state for a single task run.

Layout on disk:
  runs/<task_id>/
    state.json         — full task state, iteration history, role specs
    events.jsonl       — append-only event log (one JSON per line)
    orchestrator.pid   — pid of the running daemon (if any)
    workspace/         — bind-mounted into the sandbox; agents work here
    checkpoints/<n>/   — snapshot of state.json + workspace at iteration n
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import SETTINGS


@dataclass
class DoneConfig:
    llm_judge: bool = True
    executable_tests: bool = False
    test_command: str = ""
    human_approval: bool = False


@dataclass
class Task:
    id: str
    requirements: str
    domain_hints: list[str] = field(default_factory=list)
    done: DoneConfig = field(default_factory=DoneConfig)
    max_iters: int = 0  # 0 == unlimited
    backend: str = ""

    @staticmethod
    def new(requirements: str, **kw: Any) -> "Task":
        tid = kw.pop("id", None) or f"t-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        return Task(id=tid, requirements=requirements, **kw)


@dataclass
class Iteration:
    n: int
    role: str
    summary: str
    tokens_in: int = 0
    tokens_out: int = 0
    judge_score: float = 0.0
    judge_passed: bool = False
    artifacts: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class TaskState:
    task: Task
    status: str = "pending"  # pending|running|paused|done|stopped|error
    started_at: float = 0.0
    finished_at: float = 0.0
    iterations: list[Iteration] = field(default_factory=list)
    role_specs: list[dict] = field(default_factory=list)
    master_plan: dict = field(default_factory=dict)  # produced once in plan mode
    tokens_in: int = 0
    tokens_out: int = 0
    last_error: str = ""

    @property
    def dir(self) -> Path:
        return SETTINGS.runs_dir / self.task.id

    @property
    def workspace(self) -> Path:
        return self.dir / "workspace"

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "state.json", "w") as f:
            json.dump(_to_jsonable(self), f, indent=2)

    @classmethod
    def load(cls, task_id: str) -> "TaskState":
        d = SETTINGS.runs_dir / task_id
        with open(d / "state.json") as f:
            data = json.load(f)
        return _from_jsonable(data)

    def append_event(self, kind: str, **payload: Any) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        ev = {"t": time.time(), "kind": kind, **payload}
        with open(self.dir / "events.jsonl", "a") as f:
            f.write(json.dumps(ev) + "\n")

    def checkpoint(self, label: str = "") -> Path:
        n = len(self.iterations)
        cp_dir = self.dir / "checkpoints" / f"{n:04d}"
        cp_dir.mkdir(parents=True, exist_ok=True)
        (cp_dir / "state.json").write_text(json.dumps(_to_jsonable(self), indent=2))
        # Snapshot workspace (best-effort, ignore symlink loops)
        ws_snap = cp_dir / "workspace"
        if not ws_snap.exists() and self.workspace.exists():
            try:
                shutil.copytree(self.workspace, ws_snap)
            except Exception:
                pass
        if label:
            (cp_dir / "label.txt").write_text(label)
        return cp_dir

    def branch_from(self, iteration_n: int, new_id: str | None = None) -> "TaskState":
        cp = self.dir / "checkpoints" / f"{iteration_n:04d}"
        if not cp.exists():
            raise FileNotFoundError(f"no checkpoint at iteration {iteration_n}")
        new_id = new_id or f"{self.task.id}-b{iteration_n}-{uuid.uuid4().hex[:4]}"
        new_dir = SETTINGS.runs_dir / new_id
        new_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cp / "state.json", new_dir / "state.json")
        if (cp / "workspace").exists():
            shutil.copytree(cp / "workspace", new_dir / "workspace")
        # Rewrite the task id inside the loaded state
        st = TaskState.load(new_id)
        st.task.id = new_id
        st.status = "pending"
        st.save()
        return st


def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _from_jsonable(data: dict) -> TaskState:
    t = data["task"]
    done = DoneConfig(**t.get("done", {}))
    task = Task(
        id=t["id"],
        requirements=t["requirements"],
        domain_hints=t.get("domain_hints", []),
        done=done,
        max_iters=t.get("max_iters", 0),
        backend=t.get("backend", ""),
    )
    iters = [Iteration(**it) for it in data.get("iterations", [])]
    st = TaskState(
        task=task,
        status=data.get("status", "pending"),
        started_at=data.get("started_at", 0.0),
        finished_at=data.get("finished_at", 0.0),
        iterations=iters,
        role_specs=data.get("role_specs", []),
        master_plan=data.get("master_plan", {}) or {},
        tokens_in=data.get("tokens_in", 0),
        tokens_out=data.get("tokens_out", 0),
        last_error=data.get("last_error", ""),
    )
    return st


def list_runs() -> list[str]:
    if not SETTINGS.runs_dir.exists():
        return []
    return sorted(
        p.name for p in SETTINGS.runs_dir.iterdir()
        if p.is_dir() and (p / "state.json").exists()
    )
