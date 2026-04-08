"""Global configuration loaded from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(os.environ.get("AUTOEVOLVE_RUNS_DIR", REPO_ROOT / "runs"))
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _env(key: str, default: str = "") -> str:
    """Read an env var, treating empty strings as 'unset' so they fall back to default."""
    v = os.environ.get(key, "")
    return v if v else default


@dataclass
class Settings:
    # Backend selection: "claude_cli" or "litellm_http"
    backend: str = _env("AUTOEVOLVE_BACKEND", "claude_cli")

    # LiteLLM / Juspay endpoint
    litellm_url: str = _env(
        "AUTOEVOLVE_LITELLM_URL", "https://grid.ai.juspay.net/v1/messages"
    )
    litellm_api_key: str = _env("JUSPAY_API_KEY", "")
    litellm_model: str = _env("AUTOEVOLVE_MODEL", "kimi-latest")
    litellm_max_tokens: int = int(_env("AUTOEVOLVE_MAX_TOKENS", "4096"))

    # Claude CLI binary
    claude_cli_bin: str = _env("AUTOEVOLVE_CLAUDE_BIN", "claude")
    claude_cli_model: str = _env("AUTOEVOLVE_CLAUDE_MODEL", "")

    # Sandbox
    sandbox_mode: str = _env("AUTOEVOLVE_SANDBOX", "auto")  # auto|docker|local
    sandbox_image: str = _env("AUTOEVOLVE_SANDBOX_IMAGE", "python:3.11-slim")
    sandbox_cpu: str = _env("AUTOEVOLVE_SANDBOX_CPU", "2")
    sandbox_mem: str = _env("AUTOEVOLVE_SANDBOX_MEM", "2g")

    # Telegram
    telegram_token: str = _env("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = _env("TELEGRAM_CHAT_ID", "")
    telegram_status_interval_s: int = int(_env("AUTOEVOLVE_STATUS_INTERVAL", "300"))

    # Runtime
    runs_dir: Path = field(default_factory=lambda: RUNS_DIR)
    max_iters: int = int(_env("AUTOEVOLVE_MAX_ITERS", "0"))  # 0 == unlimited


SETTINGS = Settings()
