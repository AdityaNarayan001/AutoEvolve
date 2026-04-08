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
        self.url = (url or SETTINGS.litellm_url or "").strip()
        self.api_key = (api_key or SETTINGS.litellm_api_key or "").strip()
        self.model = (model or SETTINGS.litellm_model or "").strip()
        if not self.url:
            raise RuntimeError("litellm_http backend: AUTOEVOLVE_LITELLM_URL is empty")
        if not self.api_key:
            raise RuntimeError(
                "litellm_http backend: JUSPAY_API_KEY is empty. "
                "Set it in the web Config card or .env."
            )
        if not self.model:
            raise RuntimeError("litellm_http backend: AUTOEVOLVE_MODEL is empty")
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

        # Retry only on transient errors (network / timeout / 429 / 5xx).
        # 4xx auth/validation errors are permanent and raise immediately.
        backoff = 2.0
        for attempt in range(200):
            try:
                r = await self._client.post(self.url, json=payload, headers=headers)
            except (httpx.TimeoutException, httpx.NetworkError):
                await asyncio.sleep(min(backoff, 60))
                backoff *= 1.5
                continue

            if r.status_code == 429 or r.status_code >= 500:
                await asyncio.sleep(min(backoff, 60))
                backoff *= 1.5
                continue

            if r.status_code >= 400:
                # Permanent: bad key, bad payload, model not found, etc.
                raise RuntimeError(
                    f"litellm_http {r.status_code}: {r.text[:500]}"
                )

            return self._parse(r.json())

        raise RuntimeError("litellm_http: exhausted retries on transient errors")

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
