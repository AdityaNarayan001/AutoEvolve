"""Claude CLI backend — shells out to the local `claude` binary so the user's
existing subscription is reused. We use `claude -p <prompt> --output-format json`
which returns a single JSON document with the assistant text."""

from __future__ import annotations

import asyncio
import json
import shlex
import shutil

from ..config import SETTINGS
from .base import Backend, Message, Response, ToolSpec


class ClaudeCLIBackend(Backend):
    name = "claude_cli"

    def __init__(self, binary: str | None = None, model: str | None = None):
        self.binary = (binary or SETTINGS.claude_cli_bin or "claude").strip()
        self.model = model or SETTINGS.claude_cli_model
        if not self.binary:
            raise RuntimeError("claude_cli backend: binary path is empty")
        resolved = shutil.which(self.binary)
        if not resolved:
            raise RuntimeError(
                f"claude_cli backend: '{self.binary}' not found on PATH. "
                "Install Claude Code (`npm i -g @anthropic-ai/claude-code` or similar) "
                "or set AUTOEVOLVE_CLAUDE_BIN to the full path, "
                "or switch backend to 'litellm_http'."
            )
        self.binary = resolved

    async def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int | None = None,
    ) -> Response:
        # Flatten the conversation into a single prompt; the CLI is stateless per call.
        parts: list[str] = []
        if system:
            parts.append(f"[SYSTEM]\n{system}\n")
        for m in messages:
            tag = m.role.upper()
            parts.append(f"[{tag}]\n{m.content}\n")
        prompt = "\n".join(parts)

        argv = [self.binary, "-p", prompt, "--output-format", "json"]
        if self.model:
            argv += ["--model", self.model]

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"claude CLI failed ({proc.returncode}): {stderr.decode(errors='ignore')}\n"
                f"cmd: {shlex.join(argv[:2])} ..."
            )

        text = stdout.decode(errors="ignore").strip()
        try:
            data = json.loads(text)
            result = (
                data.get("result")
                or data.get("text")
                or data.get("content")
                or text
            )
            usage = data.get("usage", {}) or {}
            return Response(
                text=result if isinstance(result, str) else json.dumps(result),
                raw=data,
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                stop_reason=data.get("stop_reason", ""),
            )
        except json.JSONDecodeError:
            return Response(text=text)
