# AutoEvolve

Recursive multi-agent build system. Give it a task — software, mechanical, electrical, mixed — and it spawns dynamically-generated expert roles (manager, workers, domain specialists) that build, simulate, test, and self-evaluate in a loop until the work meets the requirements.

Example: *"build an active suspension simulator and find optimal coefficients"* → manager plans subtasks, the Role Forge spawns a Mechanical Engineer + Control Systems Engineer + Simulation Worker + Verifier, they write `sim.py`, run it, plot results, tune coefficients, and the Judge keeps iterating until acceptance criteria pass.

## Features

- **Two LLM backends**: local `claude` CLI (uses your Claude Code subscription) or any Anthropic-compatible HTTP endpoint (default: Juspay LiteLLM with `kimi-latest`).
- **Dynamic roles**: every subtask gets a freshly forged expert system prompt — no fixed pipeline.
- **Sandboxed execution**: every task runs in its own Docker container (`--network=none`, workspace-only mount, dropped caps). Falls back to a workspace-confined local sandbox if Docker isn't available.
- **Configurable done criteria**: LLM judge, executable tests, human approval — combine any of them.
- **No token / cost cap** by default. Indefinite retry on rate limits.
- **Fire-and-forget background daemons**: `autoevolve run` detaches; the terminal can close.
- **Web UI** with config form, task input, runs list, and live event stream (SSE).
- **Telegram bridge**: `/status /usage /perf /logs /pause /resume /stop /kill /ask`, free-text nudges, and `human_ask` round-trips so agents can ask you questions while you're away.
- **Local TUI** (`autoevolve watch`) tailing all runs read-only.
- **Per-iteration checkpoints + branching** so you can fork a task from any past iteration to explore alternatives.
- **Isolated per-task storage** under `runs/<task_id>/` — nothing leaks between tasks.

## Install

Requires Python 3.10+. Docker is optional but recommended.

```bash
git clone <this-repo> AutoEvolve
cd AutoEvolve
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configure

You can configure either through the web UI (recommended) or via a `.env` file at the repo root:

```dotenv
# Backend: claude_cli  or  litellm_http
AUTOEVOLVE_BACKEND=litellm_http

# LiteLLM / Juspay (Anthropic-compatible)
JUSPAY_API_KEY=your-api-key
AUTOEVOLVE_LITELLM_URL=https://grid.ai.juspay.net/v1/messages
AUTOEVOLVE_MODEL=kimi-latest

# Or use the local Claude CLI subscription instead
AUTOEVOLVE_CLAUDE_BIN=claude

# Sandbox: auto | docker | local
AUTOEVOLVE_SANDBOX=auto

# Optional Telegram bridge
TELEGRAM_BOT_TOKEN=123:abc
TELEGRAM_CHAT_ID=123456
```

Smoke-test the backend:

```bash
autoevolve ping-backend litellm_http
```

## Usage

### Web UI (easiest)

```bash
autoevolve web
# → http://127.0.0.1:8765
```

A single page lets you fill in API keys / backend / Telegram, type your task, hit Launch, and watch the live event stream. Closing the browser does not stop the run.

### CLI

```bash
# Fire-and-forget; the orchestrator detaches into a background daemon.
autoevolve run "build an active suspension simulator and find optimal coefficients"

# With executable acceptance tests:
autoevolve run "write a Python fibonacci module with tests" --tests "python -m pytest"

# Stay attached in the current terminal instead of detaching:
autoevolve run "..." --foreground

# List all runs and their status
autoevolve runs

# Inspect one
autoevolve status <task_id>

# Stop / kill
autoevolve stop <task_id>
autoevolve stop <task_id> --force

# Branch from a past iteration to explore an alternative direction
autoevolve branch <task_id> <iteration_n>

# Live local TUI for all runs
autoevolve watch
```

### Telegram (optional)

Once `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set, every background daemon attaches a bot. Commands: `/status /runs /usage /perf /logs /tree /artifacts /pause /resume /stop /kill /ask <text> /reply <id> <text>`. Free text becomes a nudge to the manager. When an agent calls `human_ask`, the question is forwarded and the agent waits for your reply.

## How tasks are stored

Each task lives in its own directory:

```
runs/
  t-<unix>-<rand>/
    state.json          # requirements, status, iterations, role specs, tokens
    events.jsonl        # append-only event log
    log                 # daemon stdout/stderr
    orchestrator.pid    # while running
    workspace/          # files the agents created
    checkpoints/0001/   # per-iteration snapshot (state + workspace copy)
              /0002/
              ...
```

Task IDs are timestamped + random, so submissions never collide. Old runs stay around until you delete the directory.

## Project layout

```
autoevolve/
  cli.py              # Typer entrypoint
  config.py           # env-driven Settings
  llm/                # Backend abstraction + claude_cli + litellm_http
  sandbox/            # Docker + local fallback
  tools/              # shell, fs, python_exec, human_ask
  core/
    orchestrator.py   # main loop
    role_forge.py     # dynamic role generation
    agent.py          # tool-use loop
    judge.py          # done-criteria evaluator
    state.py          # persistent task state + checkpoints + branching
  runtime/daemon.py   # double-fork background daemon
  bridges/telegram.py # Telegram bridge
  ui/dashboard.py     # `autoevolve watch` TUI
  web/server.py       # FastAPI web UI
tests/
```

## Tests

```bash
pytest -q
```

Offline smoke tests cover state round-trip, JSON extraction, the local sandbox, and the tool registry. For real end-to-end runs, use `autoevolve run` against a configured backend.

## Status

Early. The core loop, sandboxing, persistence, web UI, Telegram bridge, and CLI all work. Concurrency between independent subtasks, voice-note transcription, multi-task scheduler, and the dedicated ODE simulator harness are deferred.
