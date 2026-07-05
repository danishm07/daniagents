"""Client request shaping and error handling, with HTTP fully mocked (respx)."""

from __future__ import annotations

import httpx
import pytest
import respx

from examples.client import ApiError, Client
from examples.config import Config
from tests.conftest import TEST_BASE_URL


@respx.mock
def test_events_sends_key_and_date_params(config: Config) -> None:
    route = respx.get(f"{TEST_BASE_URL}/events").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "event_id": "e1",
                    "event_type": "EARNINGS_RELEASE",
                    "timing_category": "SCHEDULED",
                    "event_datetime": "2026-01-27T21:00:00Z",
                    "focal_assets": [{"identifier_type": "TICKER", "identifier_value": "MSFT"}],
                }
            ],
        )
    )
    with Client(config) as client:
        events = client.events(start_date="2026-01-01T00:00:00Z", end_date="2026-02-01T00:00:00Z")

    assert [e.event_type for e in events] == ["EARNINGS_RELEASE"]
    assert events[0].focal_assets[0].identifier_value == "MSFT"
    request = route.calls.last.request
    assert request.headers["x-api-key"] == "test-key"
    assert request.url.params["start_date"] == "2026-01-01T00:00:00Z"
    assert request.url.params["end_date"] == "2026-02-01T00:00:00Z"


@respx.mock
def test_events_omits_absent_date_params(config: Config) -> None:
    route = respx.get(f"{TEST_BASE_URL}/events").mock(return_value=httpx.Response(200, json=[]))
    with Client(config) as client:
        client.events()
    assert "start_date" not in route.calls.last.request.url.params
    assert "end_date" not in route.calls.last.request.url.params


@respx.mock
def test_archive_manifest_parses_files(config: Config) -> None:
    respx.get(f"{TEST_BASE_URL}/archive").mock(
        return_value=httpx.Response(
            200,
            json={
                "schema_version": "1.0",
                "files": [
                    {
                        "event_type": "EARNINGS_RELEASE",
                        "quarter": "2025Q3",
                        "key": "archive/v1/EARNINGS_RELEASE/2025Q3.jsonl.gz",
                        "url": "https://cdn.test/signed",
                        "events": 5,
                        "bytes": 863,
                    }
                ],
            },
        )
    )
    with Client(config) as client:
        manifest = client.archive_manifest()
    assert len(manifest.files) == 1
    assert manifest.files[0].filename == "EARNINGS_RELEASE_2025Q3.jsonl.gz"


@respx.mock
def test_send_test_event_returns_status(config: Config) -> None:
    respx.post(f"{TEST_BASE_URL}/webhook/test").mock(return_value=httpx.Response(202))
    with Client(config) as client:
        assert client.send_test_event() == 202


@respx.mock
def test_non_2xx_raises_api_error(config: Config) -> None:
    respx.get(f"{TEST_BASE_URL}/health").mock(return_value=httpx.Response(401, text="unauthorized"))
    with Client(config) as client, pytest.raises(ApiError) as excinfo:
        client.health()
    assert excinfo.value.status_code == 401


def test_client_requires_api_key() -> None:
    cfg = Config(api_key=None, api_base_url=TEST_BASE_URL)
    with pytest.raises(RuntimeError, match="EM_API_KEY"):
        Client(cfg)
