"""The scoring transform — percentile ranks and the analysis-frame assembly."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from examples.scoring import (
    add_percentiles,
    event_asset_rows,
    impute_predictions,
    ols_fit,
    ols_fit2,
    outcomes_frame,
    percentile_ranks,
    rank_submissions,
    score_submission,
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


# ----- OLS ports --------------------------------------------------------------


def test_ols_fit_known_line() -> None:
    fit = ols_fit([(1.0, 3.0), (2.0, 5.0), (3.0, 7.0)])  # y = 2x + 1 exactly
    assert fit is not None
    assert fit.beta == pytest.approx(2.0)
    assert fit.alpha == pytest.approx(1.0)
    assert fit.r_squared == pytest.approx(1.0)


def test_ols_fit_degenerate_is_none() -> None:
    assert ols_fit([(0.1, 0.2)]) is None
    assert ols_fit([(0.3, 0.1), (0.3, 0.9)]) is None  # zero variance in x


def test_ols_fit2_recovers_known_plane() -> None:
    def y(x1: float, x2: float) -> float:
        return 0.2 + 0.5 * x1 + 0.3 * x2

    pts = [(x1, x2, y(x1, x2)) for x1, x2 in [(0, 0), (1, 0.5), (0.5, 1), (0.2, 0.9)]]
    fit = ols_fit2(pts)
    assert fit is not None
    assert fit.beta == pytest.approx(0.5)
    assert fit.beta_surprise == pytest.approx(0.3)
    assert fit.alpha == pytest.approx(0.2)
    assert fit.r_squared == pytest.approx(1.0)


def test_ols_fit2_singular_is_none() -> None:
    # x2 perfectly collinear with x1 -> det == 0 -> unranked, no fallback.
    pts = [(x, 2 * x, 0.5 * x) for x in (0.1, 0.4, 0.8)]
    assert ols_fit2(pts) is None


# ----- Contest imputation + ranking (Official Rules §5) -------------------------


def _contest_frame() -> pd.DataFrame:
    """5 common-sample events; sub_a predicts 3, sub_b predicts all 5,
    sub_c predicts none."""
    records = [
        _record("e1", "A", car1=-0.10, surprise=0.05),
        _record("e2", "B", car1=0.00, surprise=0.01),
        _record("e3", "C", car1=0.10, surprise=0.09),
        _record("e4", "D", car1=0.20, surprise=0.02),
        _record("e5", "E", car1=0.05, surprise=0.07),
    ]
    df = add_percentiles(outcomes_frame(records))
    df["sub_a"] = [0.2, float("nan"), 0.8, float("nan"), 0.5]
    df["sub_b"] = [0.1, 0.9, 0.4, 0.6, 0.3]
    df["sub_c"] = [float("nan")] * 5
    return df


def test_impute_predictions_mean_before_fill() -> None:
    df = _contest_frame()
    imp = impute_predictions(df, "sub_a")
    assert imp is not None
    assert imp.imputed_mean == pytest.approx(0.5)  # (0.2 + 0.8 + 0.5) / 3
    assert imp.imputed_event_count == 2
    assert imp.values == pytest.approx([0.2, 0.5, 0.8, 0.5, 0.5])


def test_impute_predictions_no_coverage_ineligible() -> None:
    assert impute_predictions(_contest_frame(), "sub_c") is None


def test_score_submission_imputed_matches_hand_built_fill() -> None:
    df = _contest_frame()
    scores = score_submission(df, "sub_a")
    assert scores["n_obs"] == 3
    assert scores["imputed_mean"] == pytest.approx(0.5)
    assert scores["imputed_event_count"] == 2

    filled = [0.2, 0.5, 0.8, 0.5, 0.5]
    pts = list(zip(filled, df["surprise_pct"], df["y"], strict=True))
    want = ols_fit2(pts)
    want_surprise = ols_fit(list(zip(df["surprise_pct"], df["y"], strict=True)))
    assert want is not None and want_surprise is not None
    assert scores["r_squared_imputed"] == pytest.approx(want.r_squared)
    assert scores["r_squared_surprise_imputed"] == pytest.approx(want_surprise.r_squared)
    assert scores["delta_r_squared_imputed"] == pytest.approx(
        want.r_squared - want_surprise.r_squared
    )


def test_score_submission_full_coverage_families_identical() -> None:
    scores = score_submission(_contest_frame(), "sub_b")
    assert scores["imputed_event_count"] == 0
    for name in ("r_squared", "r_squared_surprise", "delta_r_squared", "beta", "mse"):
        assert scores[f"{name}_imputed"] == pytest.approx(scores[name])


def test_score_submission_surprise_benchmark_submission_invariant() -> None:
    df = _contest_frame()
    a = score_submission(df, "sub_a")
    b = score_submission(df, "sub_b")
    # The imputed benchmark is fit on the full common sample -> identical.
    assert a["r_squared_surprise_imputed"] == pytest.approx(b["r_squared_surprise_imputed"])


def test_score_submission_no_coverage_all_none() -> None:
    scores = score_submission(_contest_frame(), "sub_c")
    assert scores["n_obs"] == 0
    assert scores["imputed_mean"] is None
    assert scores["r_squared"] is None
    assert scores["delta_r_squared_imputed"] is None


def test_rank_submissions_order_gate_and_tiebreak() -> None:
    scores = pd.DataFrame(
        [
            # Exact metric tie: s_zzz wins on higher pre-imputation coverage.
            {"submission": "s_zzz", "n_obs": 40, "delta_r_squared_imputed": 0.25},
            {"submission": "s_aaa", "n_obs": 10, "delta_r_squared_imputed": 0.25},
            {"submission": "s_top", "n_obs": 5, "delta_r_squared_imputed": 0.90},
            # Undefined metric -> unranked, dropped.
            {"submission": "s_none", "n_obs": 50, "delta_r_squared_imputed": None},
        ]
    )
    ranked = rank_submissions(scores)
    assert ranked["submission"].tolist() == ["s_top", "s_zzz", "s_aaa"]
    assert ranked["rank"].tolist() == [1, 2, 3]
