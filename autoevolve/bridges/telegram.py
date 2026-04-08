"""Telegram bridge: the only live interaction surface for background runs.

Inbound commands are handled here; outbound notifications and human_ask
forwarding flow back to the user. The bridge holds a reference to the running
orchestrator so commands can pause/resume/stop/inject nudges.

Notification model:
- A rich welcome message goes out on `start()` so the user immediately knows
  what was launched and which commands are useful.
- The orchestrator's event stream is funneled through `_on_event`, which posts
  short one-liners for the events that matter (iteration_end, judge pass,
  errors, artifacts, run_end). Verbosity is configurable.
- A periodic snapshot still fires every N minutes as a fallback.
- User-defined triggers (`/trigger on iter=10 send status`) fire on matching
  events.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import SETTINGS
from ..core.state import TaskState, list_runs

if TYPE_CHECKING:
    from ..core.orchestrator import Orchestrator


try:
    from telegram import Update
    from telegram.constants import ParseMode
    from telegram.ext import (
        Application,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )
    HAS_TELEGRAM = True
except Exception:  # python-telegram-bot not installed
    HAS_TELEGRAM = False


HELP_TEXT = """*AutoEvolve commands*

*Observe*
`/status` — current iteration, score, tokens
`/runs` — all known runs
`/usage` — token totals
`/perf` — runtime, iters/min
`/tree` — forged roles so far
`/logs [n]` — last n events (default 20)
`/artifacts` — files in workspace

*Steer*
`/ask <text>` — inject guidance (nudge) the Architect picks up between steps
plain text → routed to the *Watcher* subagent which reads the live state and
answers your question (e.g. "what is happening?", "why did you spawn that role?")
reply to an `[ask:qN]` question → answers the waiting agent

*Lifecycle*
`/pause` `/resume`
`/stop` — clean shutdown
`/kill` — force kill
`/reply <id> <text>` — answer a pending agent question

*Triggers*
`/trigger on <cond> send <text|status|logs>` — fire a message when a condition holds
`/trigger list` — show active triggers
`/trigger clear` — remove all
Examples:
  `/trigger on iter=10 send status`
  `/trigger on score>=80 send "almost there"`
  `/trigger on error send logs`
  `/trigger on artifact send status`

*Verbosity*
`/verbose quiet|normal|loud`

`/help` — this message"""


@dataclass
class Trigger:
    cond_kind: str   # "iter" | "score" | "error" | "artifact" | "kind"
    op: str          # "==" | ">=" | "<=" | ">" | "<" | "any"
    value: str       # numeric or kind name
    action: str = "text"  # "status" | "logs" | "text"
    text: str = ""


@dataclass
class _State:
    verbosity: str = "normal"  # quiet | normal | loud
    triggers: list[Trigger] = field(default_factory=list)


class TelegramBridge:
    def __init__(self, state: TaskState):
        if not HAS_TELEGRAM:
            raise RuntimeError("python-telegram-bot is not installed")
        self.state = state
        self.chat_id = SETTINGS.telegram_chat_id
        self.app: Application = (
            Application.builder().token(SETTINGS.telegram_token).build()
        )
        self.orch: "Orchestrator | None" = None
        self._pending: dict[str, asyncio.Future] = {}
        self._counter = 0
        self._status_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._cfg = _State()
        self._last_score: float = 0.0
        self._register_handlers()

    def attach_orchestrator(self, orch: "Orchestrator") -> None:
        self.orch = orch
        # Hook the event stream so we can post smart notifications and fire triggers.
        orch.on_event = self._on_event_sync

    # ---------- lifecycle ----------

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
        self._status_task = asyncio.create_task(self._status_loop())
        await self._send_welcome()

    async def stop(self) -> None:
        if self._status_task:
            self._status_task.cancel()
        try:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
        except Exception:
            pass

    # ---------- outbound ----------

    async def notify(self, text: str, markdown: bool = False) -> None:
        try:
            await self.app.bot.send_message(
                chat_id=self.chat_id,
                text=text[:4000],
                parse_mode=ParseMode.MARKDOWN if markdown else None,
                disable_web_page_preview=True,
            )
        except Exception:
            # Retry without markdown in case of escaping issues
            try:
                await self.app.bot.send_message(chat_id=self.chat_id, text=text[:4000])
            except Exception:
                pass

    async def _send_welcome(self) -> None:
        s = self.state
        done_bits = []
        if s.task.done.llm_judge:
            done_bits.append("llm_judge")
        if s.task.done.executable_tests:
            done_bits.append(f"tests(`{s.task.done.test_command}`)")
        if s.task.done.human_approval:
            done_bits.append("human_approval")
        done_str = ", ".join(done_bits) or "none"

        msg = (
            f"*AutoEvolve task started*\n"
            f"`{s.task.id}`\n\n"
            f"*Requirements:* {s.task.requirements[:500]}\n"
            f"*Backend:* `{s.task.backend or SETTINGS.backend}`\n"
            f"*Sandbox:* `{SETTINGS.sandbox_mode}`\n"
            f"*Done criteria:* {done_str}\n"
            f"*Max iters:* {s.task.max_iters or 'unlimited'}\n\n"
            f"*Hot commands:*\n"
            f"`/status`  `/stop`  `/ask <text>`  `/logs`\n\n"
            f"Send `/help` for the full reference.\n"
            f"Plain text → answered live by the *Watcher* subagent (ask anything).\n"
            f"`/ask <text>` → nudge the *Architect* between steps."
        )
        await self.notify(msg, markdown=True)

    async def ask_human(self, question: str, urgency: str = "normal") -> str:
        self._counter += 1
        key = f"q{self._counter}"
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[key] = fut
        await self.notify(
            f"[ask:{key}:{urgency}] {question}\n\nReply with `/reply {key} <answer>` or just send a plain message."
        )
        try:
            return await fut
        finally:
            self._pending.pop(key, None)

    async def _status_loop(self) -> None:
        while True:
            await asyncio.sleep(SETTINGS.telegram_status_interval_s)
            if self._cfg.verbosity != "quiet":
                await self.notify(self._status_text())

    def _status_text(self) -> str:
        s = self.state
        n = len(s.iterations)
        last = s.iterations[-1] if s.iterations else None
        return (
            f"[{s.task.id}] {s.status} iter={n} "
            f"tokens={s.tokens_in}/{s.tokens_out} "
            f"last_score={last.judge_score if last else 0:.0f}"
        )

    # ---------- event hook (called from orchestrator's loop) ----------

    def _on_event_sync(self, kind: str, payload: dict) -> None:
        if not self._loop:
            return
        # Orchestrator runs in the same asyncio loop as the bridge, so the
        # cheapest correct path is `loop.create_task`. We fall back to the
        # thread-safe variant if (somehow) we're called from another thread.
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        try:
            if running is self._loop:
                self._loop.create_task(self._on_event(kind, payload))
            else:
                asyncio.run_coroutine_threadsafe(
                    self._on_event(kind, payload), self._loop
                )
        except Exception:
            pass

    async def _on_event(self, kind: str, payload: dict) -> None:
        v = self._cfg.verbosity

        # 0) plan-mode + lifecycle events (real-time, normal verbosity)
        if kind == "planning_start" and v != "quiet":
            await self.notify("🧭 entering plan mode…")
        elif kind == "master_plan":
            plan = payload.get("plan", {}) or {}
            phases = plan.get("phases", []) or []
            phases_str = "\n".join(
                f"  {i+1}. *{p.get('name','?')}* — {p.get('goal','')}"
                for i, p in enumerate(phases)
            )
            risks = plan.get("risks", []) or []
            risks_str = "\n".join(f"  ⚠ {r}" for r in risks[:5])
            await self.notify(
                f"📋 *Master plan*\n_{plan.get('summary','')}_\n\n"
                f"*Phases:*\n{phases_str or '  (none)'}\n\n"
                f"*Risks:*\n{risks_str or '  (none)'}\n\n"
                f"*Estimated iterations:* {plan.get('estimated_iterations','?')}",
                markdown=True,
            )
        elif kind == "iteration_start" and v != "quiet":
            await self.notify(f"🔄 iter {payload.get('n')} starting")
        elif kind == "planning" and v == "loud":
            await self.notify("  · manager planning…")
        elif kind == "plan" and v != "quiet":
            plan = payload.get("plan", {}) or {}
            subs = plan.get("subtasks", []) or []
            phase = plan.get("phase", "")
            head = f"📐 plan ({phase})" if phase else "📐 plan"
            body = "\n".join(f"  · {s.get('description','')[:120]}" for s in subs[:6])
            await self.notify(f"{head}\n{body}")
        elif kind == "role_forged" and v != "quiet":
            await self.notify(f"🧠 forged: {payload.get('title','?')}")
        elif kind == "subtask_done" and v != "quiet":
            await self.notify(
                f"  ✓ {payload.get('role','?')} done "
                f"(tools={payload.get('tool_calls',0)}, tok={payload.get('tokens_in',0)}/{payload.get('tokens_out',0)})"
            )

        # 1) standard notifications
        if kind == "iteration_end" and v != "quiet":
            score = float(payload.get("score", 0))
            arrow = ""
            if self._last_score:
                if score > self._last_score:
                    arrow = f" ↑{score - self._last_score:.0f}"
                elif score < self._last_score:
                    arrow = f" ↓{self._last_score - score:.0f}"
            self._last_score = score
            tag = "✅" if payload.get("passed") else "•"
            await self.notify(
                f"{tag} iter {payload.get('n')} score={score:.0f}{arrow}\n"
                f"{str(payload.get('rationale',''))[:300]}"
            )
        elif kind == "error":
            await self.notify(f"❌ error: {payload.get('message','')}")
        elif kind == "run_end":
            s = self.state
            runtime = max(1.0, (s.finished_at or time.time()) - (s.started_at or time.time()))
            ws_files = []
            try:
                ws_files = [p.name for p in s.workspace.iterdir()]
            except Exception:
                pass
            files_line = ", ".join(ws_files[:10]) or "(none)"
            await self.notify(
                f"*AutoEvolve done* `{s.task.id}`\n"
                f"status: `{payload.get('status','?')}`\n"
                f"runtime: {runtime:.0f}s · iters: {len(s.iterations)} · tokens: {s.tokens_in}/{s.tokens_out}\n"
                f"artifacts: {files_line}",
                markdown=True,
            )
        # 2) user-defined triggers
        for tr in list(self._cfg.triggers):
            if self._trigger_matches(tr, kind, payload):
                await self._fire_trigger(tr)

    def _trigger_matches(self, tr: Trigger, kind: str, payload: dict) -> bool:
        try:
            if tr.cond_kind == "iter" and kind == "iteration_end":
                return _cmp(int(payload.get("n", 0)), tr.op, float(tr.value))
            if tr.cond_kind == "score" and kind == "iteration_end":
                return _cmp(float(payload.get("score", 0)), tr.op, float(tr.value))
            if tr.cond_kind == "error" and kind == "error":
                return True
            if tr.cond_kind == "artifact" and kind == "subtask_done":
                return True
            if tr.cond_kind == "kind" and kind == tr.value:
                return True
        except Exception:
            return False
        return False

    async def _fire_trigger(self, tr: Trigger) -> None:
        if tr.action == "status":
            await self.notify(f"[trigger] {self._status_text()}")
        elif tr.action == "logs":
            log = self.state.dir / "events.jsonl"
            tail = "\n".join(log.read_text().splitlines()[-15:]) if log.exists() else "(no log)"
            await self.notify(f"[trigger]\n{tail[:3500]}")
        else:
            await self.notify(f"[trigger] {tr.text}")

    # ---------- inbound ----------

    def _register_handlers(self) -> None:
        h = self.app.add_handler
        h(CommandHandler("help", self._cmd_help))
        h(CommandHandler("start", self._cmd_help))
        h(CommandHandler("status", self._cmd_status))
        h(CommandHandler("runs", self._cmd_runs))
        h(CommandHandler("usage", self._cmd_usage))
        h(CommandHandler("perf", self._cmd_perf))
        h(CommandHandler("logs", self._cmd_logs))
        h(CommandHandler("tree", self._cmd_tree))
        h(CommandHandler("artifacts", self._cmd_artifacts))
        h(CommandHandler("pause", self._cmd_pause))
        h(CommandHandler("resume", self._cmd_resume))
        h(CommandHandler("stop", self._cmd_stop))
        h(CommandHandler("kill", self._cmd_kill))
        h(CommandHandler("ask", self._cmd_ask))
        h(CommandHandler("reply", self._cmd_reply))
        h(CommandHandler("trigger", self._cmd_trigger))
        h(CommandHandler("verbose", self._cmd_verbose))
        h(MessageHandler(filters.TEXT & ~filters.COMMAND, self._cmd_freeform))

    async def _reply_md(self, u: Update, text: str) -> None:
        try:
            await u.message.reply_text(text[:4000], parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await u.message.reply_text(text[:4000])

    async def _cmd_help(self, u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
        await self._reply_md(u, HELP_TEXT)

    async def _cmd_status(self, u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
        await u.message.reply_text(self._status_text())

    async def _cmd_runs(self, u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
        await u.message.reply_text("\n".join(list_runs()) or "(none)")

    async def _cmd_usage(self, u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
        s = self.state
        await u.message.reply_text(f"tokens in={s.tokens_in} out={s.tokens_out}")

    async def _cmd_perf(self, u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
        s = self.state
        runtime = max(1.0, time.time() - (s.started_at or time.time()))
        ipm = 60.0 * len(s.iterations) / runtime
        await u.message.reply_text(
            f"runtime={runtime:.0f}s iters={len(s.iterations)} ipm={ipm:.2f}"
        )

    async def _cmd_logs(self, u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
        n = int(c.args[0]) if c.args else 20
        log = self.state.dir / "events.jsonl"
        if not log.exists():
            await u.message.reply_text("(no events)")
            return
        lines = log.read_text().splitlines()[-n:]
        await u.message.reply_text("\n".join(lines)[:4000] or "(empty)")

    async def _cmd_tree(self, u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
        roles = "\n".join(f"- {r.get('title','?')}" for r in self.state.role_specs)
        await u.message.reply_text(roles or "(no roles yet)")

    async def _cmd_artifacts(self, u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
        ws = self.state.workspace
        if not ws.exists():
            await u.message.reply_text("(no workspace)")
            return
        files = "\n".join(p.name for p in ws.iterdir())
        await u.message.reply_text(files or "(empty)")

    async def _cmd_pause(self, u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
        if self.orch:
            self.orch.flag.pause = True
            await u.message.reply_text("paused")

    async def _cmd_resume(self, u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
        if self.orch:
            self.orch.flag.pause = False
            await u.message.reply_text("resumed")

    async def _cmd_stop(self, u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
        if self.orch:
            self.orch.flag.stop = True
            await u.message.reply_text("stopping cleanly")

    async def _cmd_kill(self, u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
        await u.message.reply_text("killing")
        os.kill(os.getpid(), signal.SIGKILL)

    async def _cmd_ask(self, u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
        nudge = " ".join(c.args)
        if self.orch:
            self.orch.flag.nudge = nudge
            await u.message.reply_text(f"nudge queued: {nudge}")

    async def _cmd_reply(self, u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
        if not c.args:
            return
        key, *rest = c.args
        ans = " ".join(rest)
        fut = self._pending.get(key)
        if fut and not fut.done():
            fut.set_result(ans)
            await u.message.reply_text("ok")
        else:
            await u.message.reply_text("no pending question with that id")

    async def _cmd_verbose(self, u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
        if not c.args or c.args[0] not in ("quiet", "normal", "loud"):
            await u.message.reply_text(f"current: {self._cfg.verbosity}\nuse: /verbose quiet|normal|loud")
            return
        self._cfg.verbosity = c.args[0]
        await u.message.reply_text(f"verbosity = {self._cfg.verbosity}")

    async def _cmd_trigger(self, u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
        args = c.args
        if not args or args[0] == "list":
            if not self._cfg.triggers:
                await u.message.reply_text("(no triggers)")
                return
            lines = [
                f"{i}. on {t.cond_kind}{t.op if t.op!='any' else ''}{t.value} send {t.action} {t.text}"
                for i, t in enumerate(self._cfg.triggers)
            ]
            await u.message.reply_text("\n".join(lines))
            return
        if args[0] == "clear":
            self._cfg.triggers.clear()
            await u.message.reply_text("cleared")
            return
        # /trigger on <cond> send <text|status|logs ...>
        try:
            assert args[0] == "on" and "send" in args
            send_idx = args.index("send")
            cond = " ".join(args[1:send_idx])
            tail = args[send_idx + 1:]
            tr = _parse_cond(cond)
            if tail and tail[0] in ("status", "logs"):
                tr.action = tail[0]
                tr.text = " ".join(tail[1:])
            else:
                tr.action = "text"
                tr.text = " ".join(tail).strip('"')
            self._cfg.triggers.append(tr)
            await u.message.reply_text(f"added: on {cond} send {tr.action} {tr.text}".strip())
        except Exception as e:
            await u.message.reply_text(f"parse error: {e}\nsee /help")

    async def _cmd_freeform(self, u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
        # Free text routing:
        #  1) If there's exactly one pending ask_human question, answer it.
        #  2) Otherwise, hand the message to the Watcher subagent which reads
        #     the live state and responds with a real answer.
        text = (u.message.text or "").strip()
        if not text:
            return
        if len(self._pending) == 1:
            key = next(iter(self._pending))
            self._pending[key].set_result(text)
            await u.message.reply_text(f"answered {key}")
            return
        await self._answer_with_watcher(u, text)

    async def _answer_with_watcher(self, u: Update, question: str) -> None:
        try:
            await u.message.reply_text("🔍 thinking…")
        except Exception:
            pass
        try:
            from ..core.watcher import answer_question

            backend = self.orch.backend if self.orch else None
            if backend is None:
                from ..llm import get_backend

                backend = get_backend(self.state.task.backend or None)
            ans = await answer_question(self.state, backend, question)
            await u.message.reply_text(ans[:4000])
        except Exception as e:
            await u.message.reply_text(f"watcher error: {e}")


# ---------- helpers ----------


def _cmp(a: float, op: str, b: float) -> bool:
    return {
        "==": a == b, "=": a == b,
        ">=": a >= b, "<=": a <= b,
        ">": a > b, "<": a < b,
    }.get(op, False)


def _parse_cond(cond: str) -> Trigger:
    """Parse 'iter=10', 'score>=80', 'error', 'artifact', 'kind=plan'."""
    cond = cond.strip()
    if cond in ("error", "artifact"):
        return Trigger(cond_kind=cond, op="any", value="")
    for op in (">=", "<=", "==", "=", ">", "<"):
        if op in cond:
            k, v = cond.split(op, 1)
            return Trigger(cond_kind=k.strip(), op=op, value=v.strip())
    raise ValueError(f"cannot parse condition: {cond}")
