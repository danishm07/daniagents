"""The degradation ladder: every failure still submits something.

Before this ladder existed, any exception in the worker submitted *nothing* —
and because the delivery is already ACKed by then, nothing upstream retries it,
so the event was lost permanently. The contest metric mean-fills events we miss,
so a neutral 0.5 scores zero but a miss scores worse.

The case that matters most is the last one: the ``information_url`` fetch is a
network call with a 15s timeout, so it is the most likely thing to fail, and the
neutral rung has to be reachable *without* it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modal_app


@pytest.fixture
def event():
    return {
        "event_id": "ea_TEST_Q1_2026",
        "event_type": "EARNINGS_RELEASE",
        "focal_assets": [{"identifier_type": "TICKER", "identifier_value": "AAA"}],
        "information_url": "https://example.invalid/bundle.json",
    }


@pytest.fixture
def patched(monkeypatch):
    """Swap the two boundaries — the model call and the submit POST."""
    calls = {"predict": 0, "submit": 0, "payloads": []}

    def set_predict(fn):
        def wrapped(event):
            calls["predict"] += 1
            return fn(event)

        monkeypatch.setattr("predict.predict", wrapped)

    def set_submit(fn):
        def wrapped(*, event_id, predictions, config):
            calls["submit"] += 1
            calls["payloads"].append(predictions)
            return fn(event_id, predictions)

        monkeypatch.setattr("explaining_markets.client.submit_predictions", wrapped)

    monkeypatch.setattr(modal_app, "SUBMIT_BACKOFF_SECONDS", 0.0)
    return calls, set_predict, set_submit


def good(_event):
    return [{"identifier_value": "AAA", "predicted_percentile": 0.62}]


def test_happy_path_submits_the_model_output(event, patched):
    calls, set_predict, set_submit = patched
    set_predict(good)
    set_submit(lambda event_id, predictions: None)

    predictions, rung, _ = modal_app._predict_with_ladder(event)
    assert rung == "full"
    assert predictions == [{"identifier_value": "AAA", "predicted_percentile": 0.62}]
    assert calls["predict"] == 1


def test_bundle_fetch_failure_still_submits_neutral(event, patched):
    """THE case: the information_url fetch is what fails.

    ``predict()`` raises before any facts exist, so the neutral rung must be
    built from the webhook body alone.
    """
    calls, set_predict, set_submit = patched

    def fetch_died(_event):
        raise ConnectionError("information_url unreachable")

    set_predict(fetch_died)
    set_submit(lambda event_id, predictions: None)

    predictions, rung, detail = modal_app._predict_with_ladder(event)
    assert rung == "neutral"
    assert predictions == [{"identifier_value": "AAA", "predicted_percentile": 0.5}]
    assert calls["predict"] == 2, "should try twice before giving up"
    assert detail


def test_transient_failure_recovers_on_the_retry_rung(event, patched):
    calls, set_predict, set_submit = patched

    def flaky(evt):
        if calls["predict"] == 1:
            raise TimeoutError("model call timed out")
        return good(evt)

    set_predict(flaky)
    set_submit(lambda event_id, predictions: None)

    predictions, rung, _ = modal_app._predict_with_ladder(event)
    assert rung == "retry"
    assert predictions[0]["predicted_percentile"] == 0.62


def test_out_of_range_output_is_replaced_not_submitted(event, patched):
    """An out-of-range percentile is scored as a miss, so it must not go out."""
    calls, set_predict, set_submit = patched
    set_predict(lambda _e: [{"identifier_value": "AAA", "predicted_percentile": 42.0}])
    set_submit(lambda event_id, predictions: None)

    predictions, rung, _ = modal_app._predict_with_ladder(event)
    assert rung == "neutral"
    assert predictions[0]["predicted_percentile"] == 0.5


def test_submit_retries_until_it_sticks(event, patched):
    """A prediction we already paid for must not be lost to one failed POST."""
    calls, set_predict, set_submit = patched
    set_predict(good)

    def flaky_submit(event_id, predictions):
        if calls["submit"] < 3:
            raise ConnectionError("competition API 503")

    set_submit(flaky_submit)

    attempt = modal_app._submit_with_retries("ea_X", good(event), config=None)
    assert attempt == 3
    assert calls["submit"] == 3


def test_submit_raises_only_after_exhausting_attempts(event, patched):
    calls, set_predict, set_submit = patched
    set_submit(lambda event_id, predictions: (_ for _ in ()).throw(ConnectionError("down")))

    with pytest.raises(ConnectionError):
        modal_app._submit_with_retries("ea_X", good(event), config=None)
    assert calls["submit"] == modal_app.SUBMIT_ATTEMPTS


def test_test_events_take_the_neutral_path_without_calling_the_model(patched):
    calls, set_predict, set_submit = patched
    set_predict(good)
    test_event = {
        "event_id": "ea_TEST",
        "event_type": "TEST",
        "focal_assets": [{"identifier_type": "TICKER", "identifier_value": "ZZZ"}],
    }

    predictions, rung, _ = modal_app._predict_with_ladder(test_event)
    assert rung == "test"
    assert predictions == [{"identifier_value": "ZZZ", "predicted_percentile": 0.5}]
    assert calls["predict"] == 0, "a TEST event must never spend a model call"


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        "not a list",
        [{"identifier_value": "AAA", "predicted_percentile": None}],
        [{"identifier_value": "AAA", "predicted_percentile": 1.5}],
        [{"identifier_value": "AAA", "predicted_percentile": -0.1}],
        [{"identifier_value": "AAA", "predicted_percentile": True}],
        [{"identifier_value": 123, "predicted_percentile": 0.5}],
        [{"predicted_percentile": 0.5}],
    ],
)
def test_usable_rejects_anything_the_scorer_would_call_a_miss(bad):
    assert not modal_app._usable(bad)


@pytest.mark.parametrize(
    "ok",
    [
        [{"identifier_value": "AAA", "predicted_percentile": 0.0}],
        [{"identifier_value": "AAA", "predicted_percentile": 1.0}],
        [{"identifier_value": "AAA", "predicted_percentile": 0.5}],
        [
            {"identifier_value": "AAA", "predicted_percentile": 0.2},
            {"identifier_value": "BBB", "predicted_percentile": 0.8},
        ],
    ],
)
def test_usable_accepts_valid_shapes(ok):
    assert modal_app._usable(ok)
