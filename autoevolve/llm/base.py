"""Backend abstraction. Both Claude CLI and the LiteLLM/Juspay HTTP endpoint
implement the same minimal interface, so the orchestrator stays backend-agnostic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class Response:
    text: str
    raw: dict[str, Any] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""


class Backend(ABC):
    """Backend ABC. Implementations must be safe to call from asyncio."""

    name: str = "abstract"

    @abstractmethod
    async def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int | None = None,
    ) -> Response: ...


def get_backend(name: str | None = None) -> Backend:
    """Factory. Imports lazily so missing optional deps don't break the package."""
    from ..config import SETTINGS

    n = (name or SETTINGS.backend).lower()
    if n in ("claude_cli", "claude-cli", "claude"):
        from .claude_cli import ClaudeCLIBackend

        return ClaudeCLIBackend()
    if n in ("litellm_http", "litellm", "juspay", "http"):
        from .litellm_http import LiteLLMHTTPBackend

        return LiteLLMHTTPBackend()
    if n in ("ollama", "local"):
        from .ollama import OllamaBackend

        return OllamaBackend()
    raise ValueError(f"unknown backend: {name!r}")
