"""Import + dataclass smoke tests. Real LLM calls are skipped here so the
suite runs offline; ping the live backend with `autoevolve ping-backend`."""

import asyncio
from pathlib import Path

from autoevolve.core.state import DoneConfig, Task, TaskState
from autoevolve.core.role_forge import _extract_json
from autoevolve.sandbox.local import LocalSandbox
from autoevolve.tools import build_default_registry


def test_state_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("autoevolve.config.SETTINGS.runs_dir", tmp_path)
    t = Task.new("build x", done=DoneConfig(llm_judge=True))
    s = TaskState(task=t)
    s.save()
    s2 = TaskState.load(t.id)
    assert s2.task.requirements == "build x"
    assert s2.task.done.llm_judge is True


def test_extract_json_tolerant():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('noise {"a": 2} more') == {"a": 2}
    assert _extract_json("not json") is None


def test_local_sandbox_run(tmp_path):
    sb = LocalSandbox(tmp_path / "ws")

    async def go():
        await sb.start()
        await sb.write_file("hello.txt", "hi")
        r = await sb.run(["bash", "-lc", "cat hello.txt"])
        assert r.exit_code == 0
        assert "hi" in r.stdout

    asyncio.run(go())


def test_default_registry_has_tools():
    reg = build_default_registry()
    names = {s.name for s in reg.specs()}
    assert {"shell", "fs_read", "fs_write", "python_exec"} <= names
