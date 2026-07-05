"""Configuration read from the environment (and an optional ``.env`` file).

The notebooks call :func:`load_config` at the top. If ``EM_API_KEY`` is set,
you get a live-API config; if it isn't, ``api_key`` is ``None`` and the
notebooks fall back to the bundled sample data so they still run offline.

Environment variables:
  EM_API_KEY        your submission's API key; sent as the ``X-API-Key`` header
  EM_API_BASE_URL   API base URL (default: beta stage)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_API_BASE_URL = "https://api-beta.explainingmarkets.ai/v1"


@dataclass(frozen=True)
class Config:
    """Resolved configuration for talking to the competition API.

    ``api_key`` is ``None`` when ``EM_API_KEY`` is unset — callers use
    :meth:`is_live` to decide between live calls and sample-data fallback.
    """

    api_key: str | None
    api_base_url: str

    @property
    def is_live(self) -> bool:
        """True when an API key is present, so live calls can be made."""
        return bool(self.api_key)

    def require_api_key(self) -> str:
        """Return the API key or raise a clear error if it is missing."""
        if not self.api_key:
            raise RuntimeError(
                "EM_API_KEY is not set. Copy .env.example to .env and paste your "
                "submission's API key (portal → Credentials), then restart the kernel. "
                "See the README."
            )
        return self.api_key


def load_config() -> Config:
    """Load configuration, reading a local ``.env`` file if one is present."""
    load_dotenv()
    return Config(
        api_key=os.environ.get("EM_API_KEY") or None,
        api_base_url=os.environ.get("EM_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/"),
    )
