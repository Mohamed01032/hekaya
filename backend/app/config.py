"""Runtime settings, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_env_file(path: Path) -> None:
    """Read KEY=value lines from backend/.env into the environment (no extra dependency).
    Real environment variables win over the file."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_env_file(ROOT / "backend" / ".env")


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("HEKAYA_MODEL", "claude-sonnet-5-5"))
    fast_model: str = field(default_factory=lambda: os.getenv("HEKAYA_FAST_MODEL", "claude-haiku-5-5"))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-flash-latest"))
    gemini_fast_model: str = field(default_factory=lambda: os.getenv("GEMINI_FAST_MODEL", "gemini-flash-latest"))
    provider: str = field(default_factory=lambda: os.getenv("HEKAYA_PROVIDER", "auto").lower())
    data_path: Path = field(
        default_factory=lambda: Path(os.getenv("HEKAYA_DATA", ROOT / "data" / "artifacts.json"))
    )
    db_path: Path = field(
        default_factory=lambda: Path(os.getenv("HEKAYA_DB", ROOT / "backend" / "hekaya.sqlite3"))
    )
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            o.strip() for o in os.getenv("HEKAYA_CORS", "*").split(",") if o.strip()
        )
    )
    # Requests per minute allowed per client IP for the AI endpoints.
    rate_limit_per_min: int = field(default_factory=lambda: int(os.getenv("HEKAYA_RATE_LIMIT", "30")))
    max_image_bytes: int = 8 * 1024 * 1024

    @property
    def ai_enabled(self) -> bool:
        return bool(self.anthropic_api_key or self.gemini_api_key)


def get_settings() -> Settings:
    return Settings()
