"""Telegram bridge: the only live interaction surface for background runs.

Inbound commands are handled here; outbound notifications and human_ask
forwarding flow back to the user. The bridge holds a reference to the running
orchestrator so commands can pause/resume/stop/inject nudges."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import SETTINGS
from ..core.state import TaskState, list_runs

if TYPE_CHECKING:
    from ..core.orchestrator import Orchestrator


try:
    from telegram import Update
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
        self._register_handlers()

    def attach_orchestrator(self, orch: "Orchestrator") -> None:
        self.orch = orch

    # ---------- lifecycle ----------

    async def start(self) -> None:
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
        self._status_task = asyncio.create_task(self._status_loop())
        await self.notify(f"AutoEvolve started: {self.state.task.id}")

    async def stop(self) -> None:
        if self._status_task:
            self._status_task.cancel()
        await self.app.updater.stop()
        await self.app.stop()
        await self.app.shutdown()

    # ---------- outbound ----------

    async def notify(self, text: str) -> None:
        try:
            await self.app.bot.send_message(chat_id=self.chat_id, text=text[:4000])
        except Exception:
            pass

    async def ask_human(self, question: str, urgency: str = "normal") -> str:
        self._counter += 1
        key = f"q{self._counter}"
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[key] = fut
        await self.notify(f"[ask:{key}:{urgency}] {question}\nReply with: /reply {key} <answer>")
        try:
            return await fut
        finally:
            self._pending.pop(key, None)

    async def _status_loop(self) -> None:
        while True:
            await asyncio.sleep(SETTINGS.telegram_status_interval_s)
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

    # ---------- inbound ----------

    def _register_handlers(self) -> None:
        h = self.app.add_handler
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
        h(MessageHandler(filters.TEXT & ~filters.COMMAND, self._cmd_freeform))

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
        import os, signal
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

    async def _cmd_freeform(self, u: Update, c: ContextTypes.DEFAULT_TYPE) -> None:
        # Free text → if there is exactly one pending ask, answer it.
        # Otherwise treat as a nudge.
        text = u.message.text or ""
        if len(self._pending) == 1:
            key = next(iter(self._pending))
            self._pending[key].set_result(text)
            await u.message.reply_text(f"answered {key}")
            return
        if self.orch:
            self.orch.flag.nudge = text
            await u.message.reply_text("nudge queued")
