"""Exact port of the Explaining Markets competition scoring transform.

Scoring happens within a **period** (a calendar quarter, or the official
contest Scoring Period). For each realized ``(event, focal asset)`` outcome the
competition converts the one-day cumulative abnormal return (``car1``) into a
percentile rank ``y`` in ``[0, 1]`` across the period's cross-section, and
likewise percentile-ranks the earnings surprise. A submission's period fit is
the two-regressor OLS

    y = alpha + beta1 * predicted_percentile + beta2 * surprise_percentile

so its R^2 measures how much of the realized abnormal-return *ranking* the
submission's predicted percentiles explain, over and above the naive
earnings-surprise benchmark (the same sample refit on the surprise alone;
``delta_r_squared`` is the difference between the two R^2 values).

**Contest scoring — the imputed family (Official Rules §5).** For the contest,
every submission is evaluated on the same common set of valid scored events
(the "Scored Events"). First the arithmetic mean of the submission's timely,
valid predictions on that common set is computed — *before* any filling. Every
Scored Event the submission did not (validly) predict is then imputed with
that submission-specific mean, and the identical two-regressor OLS is fit over
the *full* common set. These metrics carry an ``_imputed`` suffix
(``r_squared_imputed``, ``r_squared_surprise_imputed``,
``delta_r_squared_imputed`` — the contest ranking metric). A submission with
no timely, valid prediction on any Scored Event has an undefined mean and is
not eligible (no imputed metrics). An exact tie on the ranking metric resolves
first to the higher timely-prediction coverage before imputation (``n_obs``);
prizes for any remaining exact tie are combined and split equally — no random
tie-break.

:func:`percentile_ranks`, :func:`ols_fit`, and :func:`ols_fit2` are verbatim
ports of the production scorer — tie handling, degenerate-fit guards and all —
so offline numbers match what the leaderboard computes float-for-float. The
rest assembles the per-``(event, asset)`` analysis frame from archive records
(as produced by :func:`examples.archive.load_archive` /
:func:`examples.archive.read_jsonl_gz`) and applies the imputation/ranking
rules above.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import pandas as pd

# The reference LLM baselines woven into each archive line, mapped to display
# names. Keys match the archive's ``baseline_predictions`` blocks.
BASELINE_LABELS: dict[str, str] = {
    "gemini/ea-explain-contemp-summary": "Gemini 2.5 Flash-Lite",
    "openai/ea-explain-contemp-summary": "GPT-5 nano",
}


def percentile_ranks(values: Sequence[float]) -> list[float]:
    """Map each value to its percentile rank in ``[0, 1]`` (min -> 0, max -> 1).

    Ties share the average rank. With a single value the rank is ``0.5`` (no
    spread to rank against); an empty input returns ``[]``. This is a verbatim
    port of the competition scorer, so ranks match the leaderboard exactly.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    denom = n - 1
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        # positions i..j (0-indexed) tie; average position is (i + j) / 2.
        rank = ((i + j) / 2.0) / denom
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def event_asset_rows(record: dict) -> Iterator[dict]:
    """Yield one analysis row per focal asset of an archive event that has a ``car1``.

    Mirrors the production scorer's outcome construction: the event's earnings
    surprise is attached to each focal asset when its ``surprise_status == "ok"``
    (else ``None``), and each reference baseline's predicted percentile is pulled
    in under its display label. Assets whose return leg carries no ``car1`` are
    skipped — they cannot be scored.
    """
    es = (record.get("metrics") or {}).get("earnings_surprise") or {}
    surprise = es.get("surprise") if es.get("surprise_status") == "ok" else None
    returns = record.get("event_returns") or {}
    baselines = record.get("baseline_predictions") or {}
    for asset in record.get("focal_assets") or []:
        ticker = asset.get("identifier_value")
        leg = returns.get(ticker) or {}
        car1 = leg.get("car1")
        if car1 is None:
            continue
        row: dict = {
            "event_id": record.get("event_id"),
            "identifier_value": ticker,
            "car1": float(car1),
            "surprise": float(surprise) if surprise is not None else None,
        }
        for key, label in BASELINE_LABELS.items():
            preds = baselines.get(key) or {}
            if ticker in preds:
                row[label] = float(preds[ticker])
        yield row


def outcomes_frame(records: Iterable[dict]) -> pd.DataFrame:
    """Assemble the per-``(event, asset)`` analysis frame from archive records.

    Pass the records for **one period** (quarter): percentiles are later ranked
    within whatever you pass, matching the competition's per-period scoring. Each
    row carries ``car1``, ``surprise`` (may be ``NaN``), and one column per
    reference baseline (its predicted percentile).
    """
    rows = [row for record in records for row in event_asset_rows(record)]
    return pd.DataFrame(rows).reset_index(drop=True)


def add_percentiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the competition's scoring targets to an :func:`outcomes_frame`.

    Adds two columns:

    - ``y`` — percentile rank of ``car1`` across the whole frame. This is the
      dependent variable: the realized abnormal-return percentile (``r_abn``).
    - ``surprise_pct`` — percentile rank of ``surprise``, ranked over only the
      rows that carry a numeric surprise (others left ``NaN``), exactly as the
      scorer treats the surprise regressor.
    """
    out = frame.copy()
    out["y"] = percentile_ranks(out["car1"].tolist())

    surprise = out["surprise"].tolist()
    have = [i for i, v in enumerate(surprise) if pd.notna(v)]
    ranked = percentile_ranks([surprise[i] for i in have])
    col = [float("nan")] * len(out)
    for i, r in zip(have, ranked, strict=True):
        col[i] = r
    out["surprise_pct"] = col
    return out


# ----- OLS (verbatim ports of the production scorer) ------------------------


@dataclass(frozen=True)
class OLSFit:
    """Result of regressing ``y`` (percentile-ranked CAR) on the predictors.

    ``beta`` is the coefficient on the prediction (or, for the surprise-only
    benchmark fit, on the single regressor). ``beta_surprise`` is the
    coefficient on the percentile-ranked earnings surprise in the two-regressor
    model — ``None`` for a univariate fit.
    """

    n: int
    alpha: float
    beta: float
    r_squared: float
    mse: float
    beta_surprise: float | None = None


def ols_fit(points: list[tuple[float, float]]) -> OLSFit | None:
    """Fit ``y = alpha + beta*x`` over ``(x, y)`` points.

    Returns ``None`` when the regression is undefined — fewer than 2 points, or
    zero variance in ``x`` (beta/R^2 indeterminate). R^2 is the squared Pearson
    correlation, identically ``1 - SS_res/SS_tot``. Verbatim port of the
    production scorer.
    """
    n = len(points)
    if n < 2:
        return None
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    s_xx = sum((p[0] - mean_x) ** 2 for p in points)
    s_yy = sum((p[1] - mean_y) ** 2 for p in points)
    s_xy = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points)
    if s_xx == 0.0:
        # No spread in predictions — slope and explained variance undefined.
        return None
    beta = s_xy / s_xx
    alpha = mean_y - beta * mean_x
    ss_res = sum((p[1] - (alpha + beta * p[0])) ** 2 for p in points)
    mse = ss_res / n
    # SS_tot == 0 (all outcomes identical) leaves R^2 undefined; report 0.0 — a
    # constant target carries no variance for predictions to explain.
    r_squared = 0.0 if s_yy == 0.0 else 1.0 - ss_res / s_yy
    return OLSFit(n=n, alpha=alpha, beta=beta, r_squared=r_squared, mse=mse)


def ols_fit2(points: list[tuple[float, float, float]]) -> OLSFit | None:
    """Fit ``y = alpha + beta1*x1 + beta2*x2`` over ``(x1, x2, y)`` points
    (x1 = the submission's prediction, x2 = percentile-ranked surprise).

    Solves the centered 2x2 normal equations in closed form (Cramer's rule).
    ``beta`` carries beta1 (the prediction coefficient), ``beta_surprise``
    carries beta2, ``r_squared`` is the multiple R^2. Returns ``None`` with
    fewer than 3 points, or when the 2x2 system is singular — a regressor with
    no spread, or perfect collinearity — in which case the submission is left
    unranked (there is deliberately NO univariate fallback). Verbatim port of
    the production scorer.
    """
    n = len(points)
    if n < 3:
        return None
    mean_x1 = sum(p[0] for p in points) / n
    mean_x2 = sum(p[1] for p in points) / n
    mean_y = sum(p[2] for p in points) / n
    s11 = sum((p[0] - mean_x1) ** 2 for p in points)
    s22 = sum((p[1] - mean_x2) ** 2 for p in points)
    s12 = sum((p[0] - mean_x1) * (p[1] - mean_x2) for p in points)
    s1y = sum((p[0] - mean_x1) * (p[2] - mean_y) for p in points)
    s2y = sum((p[1] - mean_x2) * (p[2] - mean_y) for p in points)
    s_yy = sum((p[2] - mean_y) ** 2 for p in points)
    det = s11 * s22 - s12 * s12
    if det == 0.0:
        return None
    beta1 = (s22 * s1y - s12 * s2y) / det
    beta2 = (s11 * s2y - s12 * s1y) / det
    alpha = mean_y - beta1 * mean_x1 - beta2 * mean_x2
    ss_res = sum(
        (p[2] - (alpha + beta1 * p[0] + beta2 * p[1])) ** 2 for p in points
    )
    mse = ss_res / n
    r_squared = 0.0 if s_yy == 0.0 else 1.0 - ss_res / s_yy
    return OLSFit(
        n=n, alpha=alpha, beta=beta1, r_squared=r_squared, mse=mse, beta_surprise=beta2
    )


# ----- Contest imputation + scoring (Official Rules §5) ----------------------


@dataclass(frozen=True)
class ImputedPredictions:
    """A submission's mean-filled prediction vector over the common sample.

    ``values`` has one entry per common-sample row (in row order): the
    submission's own prediction where it made one, else ``imputed_mean`` — the
    arithmetic mean of its timely, valid predictions computed BEFORE filling.
    ``imputed_event_count`` is how many rows were filled.
    """

    values: list[float]
    imputed_mean: float
    imputed_event_count: int


def impute_predictions(
    common: pd.DataFrame, prediction_col: str
) -> ImputedPredictions | None:
    """Mean-impute a submission's missing predictions over the common sample.

    ``common`` is the common-sample frame (every Scored Event row — see
    :func:`score_submission`); ``prediction_col`` holds the submission's
    predicted percentiles with ``NaN`` where it made no timely, valid
    prediction. Per Rules §5 the fill value is the mean of the *present*
    predictions, computed before filling. Returns ``None`` when the submission
    predicted nothing on the common sample — the mean is undefined and the
    submission is not eligible for imputed metrics (or prizes).
    """
    raw = common[prediction_col].tolist()
    present = [float(v) for v in raw if pd.notna(v)]
    if not present:
        return None
    mean = sum(present) / len(present)
    values = [float(v) if pd.notna(v) else mean for v in raw]
    return ImputedPredictions(
        values=values,
        imputed_mean=mean,
        imputed_event_count=len(raw) - len(present),
    )


def score_submission(frame: pd.DataFrame, prediction_col: str) -> dict:
    """Both metric families for one submission over one period's frame.

    ``frame`` is an :func:`add_percentiles` output; ``prediction_col`` holds the
    submission's predicted percentiles (``NaN`` = no timely, valid prediction).

    The **common sample** (the Rules' "Scored Events") is every row carrying
    both ``y`` and ``surprise_pct``. The no-fill family fits only the rows the
    submission actually predicted; the ``_imputed`` family mean-fills the rest
    (:func:`impute_predictions`) and fits the full common sample. The
    surprise-only benchmark for the imputed family is fit on the full common
    sample, so it is identical for every submission.

    Returns a flat dict: ``n_obs`` (timely predictions on the common sample —
    the pre-imputation coverage and the first tie-break), the no-fill
    ``r_squared_surprise / r_squared / delta_r_squared / beta / beta_surprise /
    alpha / mse``, their ``_imputed`` twins, and ``imputed_mean`` /
    ``imputed_event_count``. Undefined metrics are ``None``.
    """
    common = frame[frame["surprise_pct"].notna() & frame["y"].notna()]
    xs = common[prediction_col].tolist()
    ss = common["surprise_pct"].tolist()
    ys = common["y"].tolist()

    out: dict = {
        "n_obs": 0,
        "imputed_mean": None,
        "imputed_event_count": None,
    }
    names = (
        "r_squared_surprise",
        "r_squared",
        "delta_r_squared",
        "beta",
        "beta_surprise",
        "alpha",
        "mse",
    )
    for name in names:
        out[name] = None
        out[f"{name}_imputed"] = None

    def emit(fit: OLSFit | None, surprise_fit: OLSFit | None, suffix: str) -> None:
        if fit is None:
            return
        out[f"r_squared{suffix}"] = fit.r_squared
        out[f"beta{suffix}"] = fit.beta
        out[f"beta_surprise{suffix}"] = fit.beta_surprise
        out[f"alpha{suffix}"] = fit.alpha
        out[f"mse{suffix}"] = fit.mse
        if surprise_fit is not None:
            out[f"r_squared_surprise{suffix}"] = surprise_fit.r_squared
            out[f"delta_r_squared{suffix}"] = fit.r_squared - surprise_fit.r_squared

    # No-fill family: only the rows the submission actually predicted.
    points = [
        (float(x), s, y) for x, s, y in zip(xs, ss, ys, strict=True) if pd.notna(x)
    ]
    out["n_obs"] = len(points)
    fit = ols_fit2(points)
    emit(fit, ols_fit([(s, y) for _x, s, y in points]) if fit else None, "")

    # Imputed family (Rules §5): mean-fill, refit over the FULL common sample.
    imputed = impute_predictions(common, prediction_col)
    if imputed is not None:
        out["imputed_mean"] = imputed.imputed_mean
        out["imputed_event_count"] = imputed.imputed_event_count
        ipoints = list(zip(imputed.values, ss, ys, strict=True))
        ifit = ols_fit2(ipoints)
        emit(ifit, ols_fit(list(zip(ss, ys, strict=True))) if ifit else None, "_imputed")

    return out


def rank_submissions(
    scores: pd.DataFrame, metric: str = "delta_r_squared_imputed"
) -> pd.DataFrame:
    """Order a per-submission scores frame by the contest ranking rules.

    ``scores`` needs one row per submission with columns ``submission``,
    ``n_obs``, and ``metric`` (e.g. the frame of :func:`score_submission`
    outputs). Rows where ``metric`` is undefined (``NaN``/``None``) are dropped
    — an undefined fit is unranked. Order: the metric descending (ascending
    for the ``mse`` family), then higher ``n_obs`` (timely coverage before
    imputation), then ``submission`` ascending. A ``rank`` column numbers the
    result 1..n.

    Note per Rules §5 an exact residual tie after the ``n_obs`` tie-break does
    NOT reorder further by any random device — tied positions share the
    combined prize equally. The deterministic ``submission`` sort here is
    display order only.
    """
    ranked = scores[scores[metric].notna()].copy()
    ascending = metric in ("mse", "mse_imputed")
    ranked = ranked.sort_values(
        by=[metric, "n_obs", "submission"],
        ascending=[ascending, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked
