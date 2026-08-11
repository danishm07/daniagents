"""The ASGI app guards the webhook route and ACKs before predicting.

Modal wraps `web` and `predict_and_submit` in `modal.Function` objects;
`get_raw_f()` unwraps them so the handler logic can be exercised offline,
without a Modal connection. `.spawn()` is patched out — what matters here is
that the route ACKs and hands the work off, not that Modal dispatches it.
"""

from __future__ import annotations

import base64
import hmac
import json
import time
from hashlib import sha256

import pytest
from fastapi.testclient import TestClient

import modal_app

SECRET = "whsec_" + base64.urlsafe_b64encode(b"k" * 32).decode().rstrip("=")


def _signed(payload: dict) -> tuple[bytes, dict]:
    raw_body = json.dumps(payload).encode()
    ts = int(time.time())
    key = base64.urlsafe_b64decode(SECRET[len("whsec_"):] + "==")
    signed = f"{payload['id']}.{ts}.".encode() + raw_body
    signature = "v1," + base64.b64encode(hmac.new(key, signed, sha256).digest()).decode()
    return raw_body, {
        "webhook-id": payload["id"],
        "webhook-timestamp": str(ts),
        "webhook-signature": signature,
    }


def _test_event(webhook_id: str = "evt_test_abc") -> dict:
    return {
        "id": webhook_id,
        "event_id": "test_abc",
        "event_type": "TEST",
        "focal_assets": [{"identifier_type": "TICKER", "identifier_value": "TEST"}],
        "prediction_deadline": "2099-01-01T00:00:00+00:00",
    }


class _Put:
    """Callable with an `.aio` attribute, mirroring modal.Dict.put."""

    def __init__(self, store):
        self.store = store

    def __call__(self, key, value, *, skip_if_exists=False):
        if skip_if_exists and key in self.store:
            return False
        self.store[key] = value
        return True

    async def aio(self, key, value, *, skip_if_exists=False):
        return self(key, value, skip_if_exists=skip_if_exists)


class FakeDict:
    """Stand-in for modal.Dict with the same atomic-claim semantics."""

    def __init__(self):
        self.data: dict[str, str] = {}
        self.put = _Put(self.data)

    def __setitem__(self, key, value):
        self.data[key] = value

    def pop(self, key, default=None):
        return self.data.pop(key, default)


class _Spawn:
    """Callable with an `.aio` attribute, mirroring modal.Function.spawn."""

    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(args)

    async def aio(self, *args, **kwargs):
        return self(*args, **kwargs)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("EM_API_KEY", "test-key")
    monkeypatch.setenv("EM_WEBHOOK_SECRET", SECRET)
    monkeypatch.setattr(modal_app, "seen_webhooks", FakeDict())
    return TestClient(modal_app.web.get_raw_f()())


@pytest.fixture
def spawned(monkeypatch):
    """Record spawn() calls instead of dispatching them to Modal."""
    spawn = _Spawn()
    monkeypatch.setattr(modal_app.predict_and_submit, "spawn", spawn)
    return spawn.calls


def test_health_ok(client) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_unsigned_post_is_rejected(client, spawned) -> None:
    for path in ("/", "/competition/webhook"):
        assert client.post(path, content=b"{}").status_code == 401
    assert spawned == []


def test_signed_event_acks_and_spawns_the_work(client, spawned) -> None:
    raw_body, headers = _signed(_test_event())
    resp = client.post("/", content=raw_body, headers=headers)
    assert resp.status_code == 200
    # The route did not predict inline — it handed off exactly one job.
    assert len(spawned) == 1
    event, webhook_id = spawned[0]
    assert event["event_id"] == "test_abc"
    assert webhook_id == "evt_test_abc"
    assert modal_app.seen_webhooks.data["evt_test_abc"] == "in_flight"


def test_duplicate_while_in_flight_is_skipped(client, spawned) -> None:
    raw_body, headers = _signed(_test_event())
    assert client.post("/", content=raw_body, headers=headers).status_code == 200
    assert client.post("/", content=raw_body, headers=headers).status_code == 200
    assert len(spawned) == 1  # the redelivery spawned nothing


def test_route_uses_the_async_modal_interfaces() -> None:
    """Regression guard: blocking Modal calls inside the `async def` route stall
    the event loop -- the exact failure ACKing first is meant to remove. Modal
    only warns about this at runtime, so pin it here."""
    import inspect

    src = inspect.getsource(modal_app.web.get_raw_f())
    assert "await _claim_aio(" in src
    assert "await predict_and_submit.spawn.aio(" in src
    assert "seen_webhooks.put(" not in src


def test_claim_release_state_machine(monkeypatch) -> None:
    """in_flight -> done on success; in_flight -> absent on failure."""
    fake = FakeDict()
    monkeypatch.setattr(modal_app, "seen_webhooks", fake)

    assert modal_app._claim("evt_1") is True
    assert modal_app._claim("evt_1") is False  # already in flight
    modal_app._release("evt_1", True)
    assert fake.data["evt_1"] == "done"
    assert modal_app._claim("evt_1") is False  # done is permanent

    assert modal_app._claim("evt_2") is True
    modal_app._release("evt_2", False)
    assert "evt_2" not in fake.data  # released, so a redelivery can retry
    assert modal_app._claim("evt_2") is True


def test_worker_submits_and_marks_done(monkeypatch) -> None:
    """The spawned function submits a neutral prediction for a TEST event and
    marks the claim done; a raising submit releases it instead."""
    monkeypatch.setenv("EM_API_KEY", "test-key")
    monkeypatch.setenv("EM_WEBHOOK_SECRET", SECRET)
    fake = FakeDict()
    monkeypatch.setattr(modal_app, "seen_webhooks", fake)

    submitted: list[dict] = []
    import explaining_markets.client as client_mod

    monkeypatch.setattr(
        client_mod,
        "submit_predictions",
        lambda *, event_id, predictions, config: submitted.append(
            {"event_id": event_id, "predictions": predictions}
        ),
    )

    worker = modal_app.predict_and_submit.get_raw_f()
    fake.put("evt_test_abc", "in_flight")
    worker(_test_event(), "evt_test_abc")

    assert submitted == [
        {
            "event_id": "test_abc",
            "predictions": [{"identifier_value": "TEST", "predicted_percentile": 0.5}],
        }
    ]
    assert fake.data["evt_test_abc"] == "done"

    def broken(**kwargs):
        raise RuntimeError("api down")

    monkeypatch.setattr(client_mod, "submit_predictions", broken)
    fake.put("evt_test_def", "in_flight")
    worker(_test_event("evt_test_def"), "evt_test_def")
    assert "evt_test_def" not in fake.data
