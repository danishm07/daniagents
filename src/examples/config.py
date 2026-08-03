"""Configuration read from the environment (and an optional ``.env`` file).

The notebooks call :func:`load_config` at the top. If ``EM_API_KEY`` is set,
you get a live-API config; if it isn't, ``api_key`` is ``None`` and the
notebooks fall back to the bundled sample data so they still run offline.

Environment variables:
  EM_API_KEY        your submission's API key; sent as the ``X-API-Key`` header
  EM_API_BASE_URL   API base URL (default: beta stage)
  ANTHROPIC_API_KEY optional; only :mod:`examples.summary`'s live path uses it
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

DEFAULT_API_BASE_URL = "https://api-beta.explainingmarkets.ai/v1"


@dataclass(frozen=True)
class Config:
    """Resolved configuration for talking to the competition API.

    ``api_key`` is ``None`` when ``EM_API_KEY`` is unset — callers use
    :meth:`is_live` to decide between live calls and sample-data fallback.
    ``anthropic_api_key`` is separate and entirely optional: nothing in the
    competition workflow needs it, and it is used only to *demonstrate* the
    summarization call in :mod:`examples.summary`.
    """

    api_key: str | None
    api_base_url: str
    anthropic_api_key: str | None = None

    @property
    def is_live(self) -> bool:
        """True when an API key is present, so live calls can be made."""
        return bool(self.api_key)

    @property
    def has_anthropic(self) -> bool:
        """True when an Anthropic key is present, so the summarizer demo can run."""
        return bool(self.anthropic_api_key)

    def require_api_key(self) -> str:
        """Return the API key or raise a clear error if it is missing."""
        if not self.api_key:
            raise RuntimeError(
                "EM_API_KEY is not set. Copy .env.example to .env and paste your "
                "submission's API key (portal → Credentials), then restart the kernel. "
                "See the README."
            )
        return self.api_key

    def require_anthropic_key(self) -> str:
        """Return the Anthropic API key or raise a clear error if it is missing."""
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. This is only needed to run the "
                "summarization demo in notebooks/02_earnings_call_facts.ipynb against "
                "the real API — it is NOT a competition credential and nothing else in "
                "this repo requires it. Get one from console.anthropic.com, add it to "
                ".env (see .env.example), then restart the kernel."
            )
        return self.anthropic_api_key

    def anthropic_client(self) -> Any:
        """Build an ``anthropic.Anthropic`` client from :attr:`anthropic_api_key`.

        Mirrors :meth:`examples.client.Client.from_env`. Raises if the key is
        missing, so notebooks should gate on :attr:`has_anthropic` first.
        """
        import anthropic

        return anthropic.Anthropic(api_key=self.require_anthropic_key())


def load_config() -> Config:
    """Load configuration, reading a local ``.env`` file if one is present."""
    load_dotenv()
    return Config(
        api_key=os.environ.get("EM_API_KEY") or None,
        api_base_url=os.environ.get("EM_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
    )
