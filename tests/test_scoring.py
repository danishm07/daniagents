"""The scoring transform — percentile ranks and the analysis-frame assembly."""

from __future__ import annotations

import math

import pandas as pd

from examples.scoring import (
    add_percentiles,
    event_asset_rows,
    outcomes_frame,
    percentile_ranks,
)


def test_percentile_ranks_basic() -> None:
    assert percentile_ranks([]) == []
    assert percentile_ranks([42.0]) == [0.5]
    # min -> 0, max -> 1, evenly spaced in between.
    assert percentile_ranks([30.0, 10.0, 20.0]) == [1.0, 0.0, 0.5]


def test_percentile_ranks_ties_share_average() -> None:
    # Two tied at the bottom (positions 0,1 -> avg 0.5 -> 0.5/2 = 0.25), one top.
    assert percentile_ranks([1.0, 1.0, 2.0]) == [0.25, 0.25, 1.0]


def _record(event_id: str, ticker: str, car1, surprise, *, surprise_ok=True, preds=None) -> dict:
    return {
        "event_id": event_id,
        "focal_assets": [{"identifier_type": "TICKER", "identifier_value": ticker}],
        "event_returns": {ticker: {"car1": car1, "return_status": "ok"}},
        "metrics": {
            "earnings_surprise": {
                "surprise": surprise,
                "surprise_status": "ok" if surprise_ok else "unavailable",
            }
        },
        "baseline_predictions": preds or {},
    }


def test_event_asset_rows_maps_fields() -> None:
    rec = _record(
        "e1",
        "AAPL",
        car1=0.05,
        surprise=0.01,
        preds={
            "gemini/ea-explain-contemp-summary": {"AAPL": 0.7},
            "openai/ea-explain-contemp-summary": {"AAPL": 0.6},
        },
    )
    (row,) = list(event_asset_rows(rec))
    assert row["car1"] == 0.05
    assert row["surprise"] == 0.01
    assert row["Gemini 2.5 Flash-Lite"] == 0.7
    assert row["GPT-5 nano"] == 0.6


def test_surprise_dropped_when_status_not_ok() -> None:
    rec = _record("e1", "AAPL", car1=0.05, surprise=0.01, surprise_ok=False)
    (row,) = list(event_asset_rows(rec))
    assert row["surprise"] is None


def test_asset_without_car1_is_skipped() -> None:
    rec = _record("e1", "AAPL", car1=None, surprise=0.01)
    assert list(event_asset_rows(rec)) == []


def test_add_percentiles_ranks_car1_and_surprise() -> None:
    records = [
        _record("e1", "A", car1=0.01, surprise=0.05),
        _record("e2", "B", car1=0.03, surprise=None, surprise_ok=False),
        _record("e3", "C", car1=0.02, surprise=0.01),
    ]
    df = add_percentiles(outcomes_frame(records))
    # y ranks car1 (0.01,0.03,0.02) -> (0.0, 1.0, 0.5)
    assert df.set_index("event_id")["y"].to_dict() == {"e1": 0.0, "e3": 0.5, "e2": 1.0}
    # surprise_pct ranks only the two rows carrying a surprise (e3=0.01 < e1=0.05
    # -> 0.0, 1.0); the surprise-less row is NaN.
    sp = df.set_index("event_id")["surprise_pct"].to_dict()
    assert sp["e3"] == 0.0 and sp["e1"] == 1.0
    assert math.isnan(sp["e2"])


def test_outcomes_frame_empty() -> None:
    assert outcomes_frame([]).empty


def test_percentiles_align_with_pandas_index() -> None:
    # A non-trivial index must not scramble the positional rank assignment.
    df = pd.DataFrame(
        {
            "event_id": ["a", "b"],
            "identifier_value": ["X", "Y"],
            "car1": [9.0, 1.0],
            "surprise": [1.0, 2.0],
        }
    )
    out = add_percentiles(df)
    assert out.loc[out["car1"] == 1.0, "y"].iloc[0] == 0.0
    assert out.loc[out["car1"] == 9.0, "y"].iloc[0] == 1.0
