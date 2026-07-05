"""Helper package for the Explaining Markets notebook examples.

Reusable API/plotting/archive logic lives here so the notebooks stay short and
narrative. The notebooks import from this package rather than copy-pasting
boilerplate.

Quick tour:
  config    — load EM_API_KEY / base URL from the environment or a .env file
  client    — a small typed client for the read/trigger API endpoints
  schemas   — lenient pydantic models for the API responses
  frames    — turn responses into tidy, display-friendly pandas objects
  plotting  — a colorblind-safe event-calendar chart
  archive   — download, cache, and load the historical event archive
"""

from __future__ import annotations

from examples.client import ApiError, Client
from examples.config import Config, load_config

__all__ = ["ApiError", "Client", "Config", "load_config"]
