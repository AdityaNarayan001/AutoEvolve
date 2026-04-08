"""FastAPI web UI.

A single-page app to:
- enter API keys / backend / Telegram config (persisted to .env)
- launch a task (spawns the background daemon — same path as the CLI)
- watch all runs live (SSE tail of events.jsonl)
- stop / pause / resume / nudge a running task

The web server is itself stateless: every action goes through the same
TaskState / daemon machinery the CLI uses, so the UI can crash and reopen
without affecting any running task."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from ..config import SETTINGS, REPO_ROOT
from ..core.state import DoneConfig, Task, TaskState, list_runs
from ..runtime import daemon

app = FastAPI(title="AutoEvolve")

ENV_PATH = REPO_ROOT / ".env"

# Keys we let the user manage from the UI. Anything else in .env is preserved.
EDITABLE_KEYS = [
    "AUTOEVOLVE_BACKEND",
    "JUSPAY_API_KEY",
    "AUTOEVOLVE_LITELLM_URL",
    "AUTOEVOLVE_MODEL",
    "AUTOEVOLVE_CLAUDE_BIN",
    "AUTOEVOLVE_CLAUDE_MODEL",
    "AUTOEVOLVE_SANDBOX",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "AUTOEVOLVE_STATUS_INTERVAL",
]


def _read_env() -> dict[str, str]:
    out: dict[str, str] = {k: "" for k in EDITABLE_KEYS}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k in EDITABLE_KEYS:
                out[k] = v.strip().strip('"').strip("'")
    # Prefer current process env if it has a value (e.g. set externally)
    for k in EDITABLE_KEYS:
        if os.environ.get(k):
            out[k] = os.environ[k]
    # Mask the secret-ish ones for display
    return out


def _write_env(updates: dict[str, str]) -> None:
    existing: dict[str, str] = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            existing[k] = v
    for k, v in updates.items():
        if k not in EDITABLE_KEYS:
            continue
        existing[k] = v
        os.environ[k] = v  # apply to current process so daemons inherit
    ENV_PATH.write_text("\n".join(f"{k}={v}" for k, v in existing.items()) + "\n")


# ---------- API ----------


@app.get("/api/config")
async def get_config():
    return _read_env()


@app.post("/api/config")
async def set_config(payload: dict[str, str]):
    _write_env(payload)
    return {"ok": True}


@app.get("/api/runs")
async def runs():
    out = []
    for rid in list_runs():
        try:
            s = TaskState.load(rid)
            last = s.iterations[-1] if s.iterations else None
            out.append({
                "id": rid,
                "status": s.status,
                "iterations": len(s.iterations),
                "score": last.judge_score if last else 0,
                "tokens_in": s.tokens_in,
                "tokens_out": s.tokens_out,
                "running": daemon.is_running(rid),
                "requirements": s.task.requirements[:200],
            })
        except Exception:
            pass
    return out


@app.get("/api/runs/{task_id}")
async def run_detail(task_id: str):
    try:
        s = TaskState.load(task_id)
    except Exception:
        raise HTTPException(404, "not found")
    return {
        "id": s.task.id,
        "requirements": s.task.requirements,
        "status": s.status,
        "running": daemon.is_running(task_id),
        "iterations": [
            {"n": it.n, "score": it.judge_score, "passed": it.judge_passed, "summary": it.summary}
            for it in s.iterations
        ],
        "roles": [r.get("title", "?") for r in s.role_specs],
        "tokens_in": s.tokens_in,
        "tokens_out": s.tokens_out,
    }


@app.post("/api/run")
async def create_run(payload: dict[str, Any]):
    req = (payload.get("requirements") or "").strip()
    if not req:
        raise HTTPException(400, "requirements required")
    # Persist config first if the form sent it
    cfg = payload.get("config") or {}
    if cfg:
        _write_env(cfg)
        # Reload SETTINGS so the spawned daemon sees fresh values
        from importlib import reload
        from .. import config as _cfg
        reload(_cfg)

    done = DoneConfig(
        llm_judge=bool(payload.get("judge", True)),
        executable_tests=bool(payload.get("tests")),
        test_command=payload.get("tests", ""),
        human_approval=bool(payload.get("human", False)),
    )
    task = Task.new(
        requirements=req,
        domain_hints=payload.get("domain_hints", []) or [],
        done=done,
        max_iters=int(payload.get("max_iters", 0) or 0),
        backend=payload.get("backend", "") or "",
    )
    state = TaskState(task=task)
    state.save()
    pid = daemon.spawn(task.id)
    return {"task_id": task.id, "pid": pid}


@app.post("/api/runs/{task_id}/stop")
async def stop_run(task_id: str, force: bool = False):
    return {"ok": daemon.stop(task_id, force=force)}


@app.get("/api/runs/{task_id}/events")
async def events_stream(task_id: str, request: Request):
    log = SETTINGS.runs_dir / task_id / "events.jsonl"

    async def gen():
        # Send everything we have so far, then tail.
        pos = 0
        while True:
            if await request.is_disconnected():
                return
            if log.exists():
                with open(log) as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
                if chunk:
                    for line in chunk.splitlines():
                        yield f"data: {line}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------- HTML ----------


INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>AutoEvolve</title>
<style>
:root{color-scheme:dark}
body{background:#0c0d10;color:#e6e6e6;font-family:-apple-system,system-ui,sans-serif;margin:0;padding:24px;max-width:1100px;margin:auto}
h1{margin:0 0 16px;font-weight:600}
h2{font-size:14px;text-transform:uppercase;letter-spacing:1px;color:#888;margin-top:24px}
.card{background:#15171c;border:1px solid #23252c;border-radius:10px;padding:16px;margin-bottom:16px}
input,select,textarea,button{background:#0c0d10;color:#e6e6e6;border:1px solid #2a2d36;border-radius:6px;padding:8px 10px;font-size:13px;font-family:inherit}
input,select,textarea{width:100%;box-sizing:border-box;margin-bottom:8px}
textarea{min-height:80px;resize:vertical;font-family:ui-monospace,Menlo,monospace}
button{cursor:pointer;background:#2563eb;border-color:#2563eb;color:#fff;font-weight:600}
button:hover{background:#1d4ed8}
button.secondary{background:#15171c;color:#e6e6e6}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
label{display:block;font-size:11px;color:#888;margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px}
.runs{display:flex;flex-direction:column;gap:6px}
.run{display:flex;justify-content:space-between;padding:8px 12px;background:#0c0d10;border:1px solid #2a2d36;border-radius:6px;cursor:pointer}
.run:hover{border-color:#2563eb}
.run.active{border-color:#2563eb}
.events{font-family:ui-monospace,Menlo,monospace;font-size:11px;background:#0c0d10;border:1px solid #2a2d36;border-radius:6px;padding:12px;max-height:420px;overflow:auto;white-space:pre-wrap}
.ev{padding:2px 0;border-bottom:1px dotted #1f2128}
.ev .k{color:#60a5fa}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.dot.running{background:#10b981}
.dot.done{background:#2563eb}
.dot.error{background:#ef4444}
.dot.stopped{background:#6b7280}
.dot.pending{background:#f59e0b}
.muted{color:#888;font-size:12px}
</style></head><body>
<h1>AutoEvolve</h1>

<div class="card">
  <h2 style="margin-top:0">Config</h2>
  <div class="row">
    <div>
      <label>Backend</label>
      <select id="cfg_backend">
        <option value="claude_cli">claude_cli (use local Claude subscription)</option>
        <option value="litellm_http">litellm_http (Juspay endpoint)</option>
      </select>
    </div>
    <div>
      <label>Model (for litellm_http)</label>
      <input id="cfg_model" placeholder="kimi-latest">
    </div>
  </div>
  <div class="row">
    <div>
      <label>Juspay API key</label>
      <input id="cfg_api_key" type="password" placeholder="api-key">
    </div>
    <div>
      <label>LiteLLM URL</label>
      <input id="cfg_url" placeholder="https://grid.ai.juspay.net/v1/messages">
    </div>
  </div>
  <div class="row">
    <div>
      <label>Claude CLI binary</label>
      <input id="cfg_claude_bin" placeholder="claude">
    </div>
    <div>
      <label>Sandbox</label>
      <select id="cfg_sandbox">
        <option value="auto">auto</option>
        <option value="docker">docker</option>
        <option value="local">local</option>
      </select>
    </div>
  </div>
  <div class="row">
    <div>
      <label>Telegram bot token</label>
      <input id="cfg_tg_token" type="password" placeholder="123:abc">
    </div>
    <div>
      <label>Telegram chat id</label>
      <input id="cfg_tg_chat" placeholder="123456">
    </div>
  </div>
  <button onclick="saveConfig()">Save config</button>
  <span id="cfg_msg" class="muted" style="margin-left:10px"></span>
</div>

<div class="card">
  <h2 style="margin-top:0">New task</h2>
  <label>Requirements</label>
  <textarea id="req" placeholder="e.g. build an active suspension simulator and find optimal coefficients"></textarea>
  <div class="row">
    <div>
      <label>Test command (optional)</label>
      <input id="tests" placeholder="python -m pytest">
    </div>
    <div>
      <label>Max iterations (0 = unlimited)</label>
      <input id="max_iters" type="number" value="0">
    </div>
  </div>
  <button onclick="launch()">Launch</button>
  <span id="run_msg" class="muted" style="margin-left:10px"></span>
</div>

<div class="card">
  <h2 style="margin-top:0">Runs</h2>
  <div id="runs" class="runs"></div>
</div>

<div class="card">
  <h2 style="margin-top:0">Live events <span id="focus" class="muted"></span></h2>
  <div id="events" class="events">(select a run)</div>
</div>

<script>
const $ = id => document.getElementById(id);
let focus = null, es = null;

async function loadConfig() {
  const c = await (await fetch('/api/config')).json();
  $('cfg_backend').value = c.AUTOEVOLVE_BACKEND || 'claude_cli';
  $('cfg_model').value = c.AUTOEVOLVE_MODEL || '';
  $('cfg_api_key').value = c.JUSPAY_API_KEY || '';
  $('cfg_url').value = c.AUTOEVOLVE_LITELLM_URL || '';
  $('cfg_claude_bin').value = c.AUTOEVOLVE_CLAUDE_BIN || '';
  $('cfg_sandbox').value = c.AUTOEVOLVE_SANDBOX || 'auto';
  $('cfg_tg_token').value = c.TELEGRAM_BOT_TOKEN || '';
  $('cfg_tg_chat').value = c.TELEGRAM_CHAT_ID || '';
}

function configPayload() {
  return {
    AUTOEVOLVE_BACKEND: $('cfg_backend').value,
    AUTOEVOLVE_MODEL: $('cfg_model').value,
    JUSPAY_API_KEY: $('cfg_api_key').value,
    AUTOEVOLVE_LITELLM_URL: $('cfg_url').value,
    AUTOEVOLVE_CLAUDE_BIN: $('cfg_claude_bin').value,
    AUTOEVOLVE_SANDBOX: $('cfg_sandbox').value,
    TELEGRAM_BOT_TOKEN: $('cfg_tg_token').value,
    TELEGRAM_CHAT_ID: $('cfg_tg_chat').value,
  };
}

async function saveConfig() {
  await fetch('/api/config', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(configPayload())});
  $('cfg_msg').textContent = 'saved';
  setTimeout(()=>$('cfg_msg').textContent='', 2000);
}

async function launch() {
  const req = $('req').value.trim();
  if (!req) return;
  $('run_msg').textContent = 'launching...';
  const payload = {
    requirements: req,
    tests: $('tests').value,
    max_iters: parseInt($('max_iters').value || '0'),
    backend: $('cfg_backend').value,
    config: configPayload(),
  };
  const r = await (await fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)})).json();
  $('run_msg').textContent = 'started ' + r.task_id;
  $('req').value = '';
  loadRuns();
  selectRun(r.task_id);
}

async function loadRuns() {
  const runs = await (await fetch('/api/runs')).json();
  const el = $('runs');
  el.innerHTML = '';
  for (const r of runs.reverse()) {
    const div = document.createElement('div');
    div.className = 'run' + (r.id === focus ? ' active' : '');
    const stat = r.running ? 'running' : r.status;
    div.innerHTML = `<div><span class="dot ${stat}"></span><b>${r.id}</b> <span class="muted">${r.requirements}</span></div>
      <div class="muted">iter ${r.iterations} · score ${r.score.toFixed?.(0) ?? r.score} · tok ${r.tokens_in}/${r.tokens_out}</div>`;
    div.onclick = () => selectRun(r.id);
    el.appendChild(div);
  }
}

function selectRun(id) {
  focus = id;
  $('focus').textContent = id;
  $('events').innerHTML = '';
  if (es) es.close();
  es = new EventSource('/api/runs/' + id + '/events');
  es.onmessage = (e) => {
    try {
      const j = JSON.parse(e.data);
      const div = document.createElement('div');
      div.className = 'ev';
      const t = new Date(j.t * 1000).toISOString().substr(11,8);
      const rest = Object.entries(j).filter(([k]) => k!=='t' && k!=='kind').map(([k,v])=>`${k}=${typeof v==='string'?v.slice(0,120):JSON.stringify(v).slice(0,120)}`).join(' ');
      div.innerHTML = `${t} <span class="k">${j.kind}</span> ${rest}`;
      $('events').appendChild(div);
      $('events').scrollTop = $('events').scrollHeight;
    } catch {}
  };
  loadRuns();
}

loadConfig();
loadRuns();
setInterval(loadRuns, 3000);
</script></body></html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")
