"""HTTP backend for the Anthropic-compatible Juspay/LiteLLM endpoint."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from ..config import SETTINGS
from .base import Backend, Message, Response, ToolSpec


class LiteLLMHTTPBackend(Backend):
    name = "litellm_http"

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.url = url or SETTINGS.litellm_url
        self.api_key = api_key or SETTINGS.litellm_api_key
        self.model = model or SETTINGS.litellm_model
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))

    async def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int | None = None,
    ) -> Response:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or SETTINGS.litellm_max_tokens,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in messages
                if m.role in ("user", "assistant")
            ],
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        # Indefinite retry on transient errors (no token cap, no abort).
        backoff = 2.0
        for attempt in range(100):
            try:
                r = await self._client.post(self.url, json=payload, headers=headers)
                if r.status_code == 429 or r.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"{r.status_code}", request=r.request, response=r
                    )
                r.raise_for_status()
                data = r.json()
                return self._parse(data)
            except (httpx.HTTPError, httpx.TimeoutException):
                await asyncio.sleep(min(backoff, 60))
                backoff *= 1.5
        raise RuntimeError("LiteLLM HTTP backend exhausted retries")

    def _parse(self, data: dict[str, Any]) -> Response:
        text_parts: list[str] = []
        for block in data.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        usage = data.get("usage", {}) or {}
        return Response(
            text="".join(text_parts),
            raw=data,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            stop_reason=data.get("stop_reason", ""),
        )


async def _ping() -> None:
    b = LiteLLMHTTPBackend()
    r = await b.complete(system="", messages=[Message("user", "say pong")])
    print(r.text)


if __name__ == "__main__":
    import sys

    if "--ping" in sys.argv:
        asyncio.run(_ping())
