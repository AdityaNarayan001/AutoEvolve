"""Local Ollama backend. Talks to the Ollama HTTP server (default
http://localhost:11434) using its native /api/chat endpoint."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from ..config import SETTINGS
from .base import Backend, Message, Response, ToolSpec


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, url: str | None = None, model: str | None = None):
        self.url = (url or SETTINGS.ollama_url or "http://localhost:11434").rstrip("/")
        self.model = (model or SETTINGS.ollama_model or "").strip()
        if not self.model:
            raise RuntimeError(
                "ollama backend: AUTOEVOLVE_OLLAMA_MODEL is empty. "
                "Pick a model in the web Config card (or `ollama list`)."
            )
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))

    async def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int | None = None,
    ) -> Response:
        msgs: list[dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        for m in messages:
            if m.role in ("user", "assistant", "system"):
                msgs.append({"role": m.role, "content": m.content})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "stream": False,
        }
        if max_tokens:
            payload["options"] = {"num_predict": max_tokens}

        backoff = 2.0
        for _ in range(50):
            try:
                r = await self._client.post(f"{self.url}/api/chat", json=payload)
            except (httpx.TimeoutException, httpx.NetworkError):
                await asyncio.sleep(min(backoff, 30))
                backoff *= 1.5
                continue
            if r.status_code >= 500:
                await asyncio.sleep(min(backoff, 30))
                backoff *= 1.5
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"ollama {r.status_code}: {r.text[:500]}")
            data = r.json()
            msg = data.get("message") or {}
            return Response(
                text=msg.get("content", ""),
                raw=data,
                input_tokens=int(data.get("prompt_eval_count", 0) or 0),
                output_tokens=int(data.get("eval_count", 0) or 0),
                stop_reason=data.get("done_reason", ""),
            )
        raise RuntimeError("ollama: exhausted retries")


def list_local_models(url: str | None = None) -> list[str]:
    """Synchronously fetch installed models from a running Ollama daemon.
    Returns [] if the daemon isn't reachable — caller decides how to surface that."""
    base = (url or SETTINGS.ollama_url or "http://localhost:11434").rstrip("/")
    try:
        r = httpx.get(f"{base}/api/tags", timeout=2.0)
        if r.status_code != 200:
            return []
        return [m.get("name", "") for m in (r.json().get("models") or []) if m.get("name")]
    except Exception:
        return []
