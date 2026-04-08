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


@dataclass
class Settings:
    # Backend selection: "claude_cli" or "litellm_http"
    backend: str = os.environ.get("AUTOEVOLVE_BACKEND", "claude_cli")

    # LiteLLM / Juspay endpoint
    litellm_url: str = os.environ.get(
        "AUTOEVOLVE_LITELLM_URL", "https://grid.ai.juspay.net/v1/messages"
    )
    litellm_api_key: str = os.environ.get("JUSPAY_API_KEY", "")
    litellm_model: str = os.environ.get("AUTOEVOLVE_MODEL", "kimi-latest")
    litellm_max_tokens: int = int(os.environ.get("AUTOEVOLVE_MAX_TOKENS", "4096"))

    # Claude CLI binary
    claude_cli_bin: str = os.environ.get("AUTOEVOLVE_CLAUDE_BIN", "claude")
    claude_cli_model: str = os.environ.get("AUTOEVOLVE_CLAUDE_MODEL", "")

    # Sandbox
    sandbox_mode: str = os.environ.get("AUTOEVOLVE_SANDBOX", "auto")  # auto|docker|local
    sandbox_image: str = os.environ.get(
        "AUTOEVOLVE_SANDBOX_IMAGE", "python:3.11-slim"
    )
    sandbox_cpu: str = os.environ.get("AUTOEVOLVE_SANDBOX_CPU", "2")
    sandbox_mem: str = os.environ.get("AUTOEVOLVE_SANDBOX_MEM", "2g")

    # Telegram
    telegram_token: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.environ.get("TELEGRAM_CHAT_ID", "")
    telegram_status_interval_s: int = int(
        os.environ.get("AUTOEVOLVE_STATUS_INTERVAL", "300")
    )

    # Runtime
    runs_dir: Path = field(default_factory=lambda: RUNS_DIR)
    max_iters: int = int(os.environ.get("AUTOEVOLVE_MAX_ITERS", "0"))  # 0 == unlimited


SETTINGS = Settings()
