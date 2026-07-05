"""Config loading and the live / sample-mode switch."""

from __future__ import annotations

import pytest

from examples.config import DEFAULT_API_BASE_URL, Config, load_config


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate from any real .env on the developer's machine, so these tests
    exercise the environment we set — not whatever key happens to be configured."""
    monkeypatch.setattr("examples.config.load_dotenv", lambda *a, **k: False)


def test_load_config_without_key_is_sample_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EM_API_KEY", raising=False)
    monkeypatch.delenv("EM_API_BASE_URL", raising=False)
    cfg = load_config()
    assert cfg.api_key is None
    assert cfg.is_live is False
    assert cfg.api_base_url == DEFAULT_API_BASE_URL


def test_load_config_with_key_is_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EM_API_KEY", "abc123")
    monkeypatch.setenv("EM_API_BASE_URL", "https://api.example/v1/")
    cfg = load_config()
    assert cfg.api_key == "abc123"
    assert cfg.is_live is True
    # Trailing slash is stripped.
    assert cfg.api_base_url == "https://api.example/v1"


def test_require_api_key_raises_when_missing() -> None:
    cfg = Config(api_key=None, api_base_url=DEFAULT_API_BASE_URL)
    with pytest.raises(RuntimeError, match="EM_API_KEY"):
        cfg.require_api_key()
