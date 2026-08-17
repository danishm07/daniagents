"""Configuration read from the environment.

In deployment these come from your local ``.env`` file, which Modal loads at
deploy time (see ``.env.example`` and ``modal_app.py``).

Required:
  EM_API_KEY         your submission's API key; sent as the X-API-Key header
  EM_WEBHOOK_SECRET  your signing secret (whsec_...); verifies incoming webhooks

Optional:
  EM_API_BASE_URL    API base URL (default: production)
  OPENAI_API_KEY     when set, predict.py makes real LLM calls; otherwise it
                     falls back to a 0.5 baseline so the round-trip still works
  OPENAI_MODEL       model name for predict.py (default: gpt-5.4-nano)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_API_BASE_URL = "https://api.explainingmarkets.ai/v1"
#: The deployed read. Measured 2026-08-16 over 2,093 archive events: 4.66% of
#: obtainable against gpt-5.4-nano's 3.52%, +0.0134 vs champion on 3/3 quarters
#: with a CI excluding zero. Routed through OpenRouter, which is where the
#: comparison was run and where strict structured outputs were verified.
DEFAULT_OPENAI_MODEL = "google/gemini-2.5-flash"

#: Fallback for a direct OpenAI key, kept so the starter still works without
#: OpenRouter configured.
FALLBACK_OPENAI_MODEL = "gpt-5.4-nano"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Config:
    api_key: str
    webhook_secret: str
    api_base_url: str

    @classmethod
    def from_env(cls) -> "Config":
        """Load and validate required config. Raises if a required var is missing."""
        return cls(
            api_key=_require("EM_API_KEY"),
            webhook_secret=_require("EM_WEBHOOK_SECRET"),
            api_base_url=os.environ.get("EM_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/"),
        )


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to your .env file "
            f"(copy .env.example to .env), then re-deploy. See the README."
        )
    return value


def openai_model() -> str:
    """The model id to call.

    ``OPENAI_MODEL`` overrides. Otherwise the OpenRouter default when an
    OpenRouter key is present, and the direct-OpenAI model when it is not — so a
    missing key degrades to a working configuration rather than a 404 on a
    vendor-prefixed id.
    """
    override = os.environ.get("OPENAI_MODEL")
    if override:
        return override
    return DEFAULT_OPENAI_MODEL if os.environ.get("OPENROUTER_API_KEY") else FALLBACK_OPENAI_MODEL


def openai_client_kwargs() -> dict:
    """Base URL and key for the LLM client.

    Prefers OpenRouter because that is where the model sweep ran: six models
    across four vendors, and the deployed choice was measured there. Falls back
    to the direct OpenAI account when no OpenRouter key is configured.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return {"base_url": OPENROUTER_BASE_URL, "api_key": key}
    return {}
