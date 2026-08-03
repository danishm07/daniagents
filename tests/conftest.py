"""Shared test fixtures. All tests here are offline (mocked HTTP, sample data)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.config import Config

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = REPO_ROOT / "data" / "sample"
SAMPLE_EVENTS = SAMPLE_DIR / "events_sample.json"
SAMPLE_ARCHIVE = SAMPLE_DIR / "archive_EARNINGS_RELEASE_2025Q3.jsonl.gz"
SAMPLE_TRANSCRIPT = SAMPLE_DIR / "transcript_sample.json"
SAMPLE_SUMMARY = SAMPLE_DIR / "summary_sample.json"

TEST_BASE_URL = "https://api.test/v1"


@pytest.fixture
def config() -> Config:
    """A live-looking config pointed at a fake base URL for respx to intercept."""
    return Config(api_key="test-key", api_base_url=TEST_BASE_URL)


@pytest.fixture
def sample_events() -> list[dict]:
    return json.loads(SAMPLE_EVENTS.read_text())
