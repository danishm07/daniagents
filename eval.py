"""The single entry point every experiment goes through.

``harness.py`` answers "what does the competition scorer say about this column?".
This module answers the question one level up: "should I believe it, and what did
it cost?" — and it writes the answer down whether or not the run succeeded.

Four things it adds over calling :func:`harness.backtest` directly.

1. **2026Q3 is held out.** :data:`DEV_QUARTERS` excludes it and :func:`run`
   refuses to touch it unless you pass ``final=True``. Every such touch is
   logged, so "we only evaluated on the holdout once" is a checkable claim
   rather than a memory.

2. **Partial correlation alongside ΔR², everywhere.** ΔR² is a squared
   quantity, so it throws away the sign and compresses differences near zero.
   The partial correlation of the prediction with ``y`` after projecting out
   ``surprise_pct`` is the same information, signed and on a readable scale —
   and it is what the ``ρ/√ρ_b`` combination arithmetic below is stated in.
   Identity worth knowing: ``ΔR² = pc² · (1 − R²_surprise)``, asserted in
   ``__main__``.

3. **Inter-view correlation is a first-class output, not an afterthought.**
   Combining ``k`` views whose individual correlation with the target is ``ρ``
   and whose mutual correlation is ``ρ_b`` asymptotes to ``ρ/√ρ_b`` — no matter
   how many you add. With a single LLM read at ρ ≈ 0.269, mutual correlation
   above ~0.17 puts leader-level performance out of reach at any ``k``. So a
   view's *decorrelation* from what we already have is as decision-relevant as
   its standalone score, and both get reported together.

4. **Every run is appended to a JSONL log — including failures.** Config hash,
   feature set, hyperparameters, all metrics, runtime, cost, git commit,
   traceback. The reason is honesty about multiple comparisons: a headline
   number means nothing without the count of configurations tried to reach it,
   and that count is only trustworthy if the failures are in the log too.

Typical use::

    import eval

    def tfidf(events, quarter):
        train = eval.training_data(quarter)      # prior quarters only
        ...
        return preds

    eval.run(tfidf, "tfidf residual", config={"alpha": 1.0, "w": 0.25})

    # several views at once: each is scored, and the correlation matrix
    # between them is part of the output
    eval.run({"llm": llm_view, "tfidf": tfidf, "embed": embed_view},
             "three views", combine=eval.zblend)
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import subprocess
import time
import traceback
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import harness
from examples.scoring import no_spread
from harness import GEMINI, GPT, QUARTERS, load, training_data  # noqa: F401  (re-exported)

#: Views the caller derived from other views. Excluded from the combination
#: arithmetic — a blend is correlated with its own inputs by construction, and
#: counting it as an independent view inflates every ceiling in the table.
DERIVED_VIEWS = {"combined"}

#: The champion joins every correlation matrix under this key. It is a
#: reference point, not a candidate: it gets no ceiling row of its own and is
#: kept out of peer ρ_b, but ρ_b *against it* is the number that says whether a
#: candidate is a new channel or a paraphrase of what we already run.
CHAMPION_KEY = "champion"

#: Never evaluated until the final model is chosen. One shot, at the end.
HOLDOUT = "2026Q3"

#: What every experiment runs on.
DEV_QUARTERS = [q for q in QUARTERS if q != HOLDOUT]

RUN_LOG = Path(__file__).parent / "runs" / "eval_runs.jsonl"

#: Fallback champion: the archive's GPT baseline. A *proxy* — different model
#: (``gpt-5-nano-2025-08-07`` vs the deployed ``gpt-5.4-nano``), different
#: prompt. Used only until ``champion.py`` has replayed the real thing.
CHAMPION_PROXY = GPT


def default_champion(quiet: bool = False) -> str:
    """The real champion column if it has been generated, else the proxy.

    Resolved per call rather than at import, so a run started after
    ``champion.py`` finishes picks it up without anyone remembering to.
    """
    if harness.CHAMPION_FILE.exists():
        return harness.CHAMPION_COLUMN
    if not quiet:
        print(
            "!! no champion column — falling back to the GPT baseline proxy. "
            "rho_b and every floor will be measured against a different model "
            "running a different prompt. Run `uv run python champion.py`."
        )
    return CHAMPION_PROXY

#: Single-test one-sided 95% bar. The best-of-K term below overtakes it around
#: K≈13; until then this is what keeps the floor off the ground.
Z_SINGLE_TEST = 1.645

#: Configurations tried before this log existed. The 2026-08-11 signal study
#: swept seven candidate signals plus blend weights, rank transforms, isotonic
#: variants, a dispersion battery and five surprise transforms — order 40 looks
#: at the same data. Starting K at zero would pretend those looks were free and
#: set the promotion floor too low for every candidate that follows them.
K_PRIOR = 40

#: A view function takes the leakage-safe event list plus the quarter it is
#: predicting, and returns one float per event, in order. Single-argument
#: functions are accepted too — see :func:`_call_view`.
ViewFn = Callable[..., Sequence[float]]


# --------------------------------------------------------------------------
# The two statistics the scorer's headline number hides
# --------------------------------------------------------------------------


def _residualize(values: np.ndarray, surprise: np.ndarray) -> np.ndarray:
    """``values`` with ``surprise`` projected out by OLS.

    Everything the contest pays for lives in this residual space: the benchmark
    already owns whatever the surprise explains.
    """
    slope, intercept = np.polyfit(surprise, values, 1)
    return values - (slope * surprise + intercept)


def partial_corr(pred: np.ndarray, y: np.ndarray, surprise: np.ndarray) -> float:
    """Correlation of ``pred`` with ``y``, controlling for ``surprise``.

    Signed, unlike ΔR². A view can be genuinely informative and *inverted*; the
    scorer is indifferent (affine invariance) but you are not, because a sign
    flip between quarters means something different from a consistent edge.
    """
    ok = np.isfinite(pred) & np.isfinite(y) & np.isfinite(surprise)
    if ok.sum() < 3:
        return float("nan")
    if _degenerate(pred[ok]) or _degenerate(y[ok]):
        return 0.0
    pr = _residualize(pred[ok], surprise[ok])
    yr = _residualize(y[ok], surprise[ok])
    if _degenerate(pr) or _degenerate(yr):
        return 0.0
    return float(np.corrcoef(pr, yr)[0, 1])


def _degenerate(values: np.ndarray) -> bool:
    """Zero spread, by the scorer's own definition.

    Reusing :func:`examples.scoring.no_spread` rather than testing ``std == 0``
    matters: residualizing a constant vector leaves floating-point dust with a
    nonzero standard deviation, and correlating that dust against ``y`` produced
    a spurious ±0.011 partial correlation for a constant 0.5 — a prediction the
    scorer correctly calls exactly zero.
    """
    centred = values - values.mean()
    return no_spread(float(centred @ centred), len(values))


def obtainable(quarter: str) -> float:
    """How much ΔR² is on the table at all: ``1 − R²_surprise`` for the quarter.

    Feed ``y`` back in as the prediction and the scorer returns exactly this
    (asserted in ``__main__``), so it is the ceiling in the literal sense —
    perfect foresight scores it and nothing scores more. Dev-quarter values are
    0.9506 / 0.9422 / 0.9311, mean **0.9413**.

    Reporting a result as a fraction of it is what makes a number legible:
    ΔR² 0.06 is 6% of what is obtainable, the LLM-read family's ceiling of 0.077
    is 8%, and the leaders' 0.378 is 40%. "8% of what is available" settles
    whether a family is worth pursuing in a way that "0.077" does not.

    The fraction is also not merely rhetorical — it is an identity. Since
    ``ΔR² = pc² · (1 − R²_surprise)``, the fraction of obtainable **is the
    squared partial correlation**. The two headline numbers are one number.
    """
    frame = load(quarter)
    return 1.0 - harness.evaluate(frame, GEMINI)["r_squared_surprise"]


def as_pct_obtainable(delta_r2: float, r2_surprise: float) -> float:
    """``delta_r2`` as a fraction of what perfect foresight would score."""
    room = 1.0 - r2_surprise
    return delta_r2 / room if room else float("nan")


def combination_ceiling(rho: float, rho_b: float) -> float:
    """Best correlation reachable by combining infinitely many views.

    ``rho`` is each view's correlation with the target, ``rho_b`` their mutual
    correlation. The limit of an equal-weight average is ``ρ/√ρ_b`` — adding
    views past that point buys nothing, which is why a decorrelated weak view
    beats another strong-but-redundant one.

    Returns ``inf`` when ``rho_b <= 0``: uncorrelated views have no ceiling in
    this arithmetic, and that is a modelling signal, not a number to trust.
    """
    if rho_b <= 0:
        return float("inf")
    return rho / math.sqrt(rho_b)


# --------------------------------------------------------------------------
# How big a gain has to be before it means anything
# --------------------------------------------------------------------------
#
# The old bar — "2 SE, where SE = 0.010" — was wrong twice over. 0.010 is the
# bootstrap sd of Gemini's ΔR² *level*, and promotion is a paired comparison:
# challenger and champion are scored on the same events, so quarter difficulty
# cancels out of the difference. Most of the 0.0114 spread in Gemini's levels is
# real difficulty drift (the benchmark's own R² climbs 0.049 → 0.078), not noise
# in the comparison. The sd of the *difference* is a different and much smaller
# quantity, it depends on how correlated the challenger is with the champion,
# and there is no reason to guess it when it can be measured per candidate.


def _delta_r2_fast(pred: np.ndarray, surprise: np.ndarray, y: np.ndarray) -> float:
    """ΔR² by closed-form OLS — the scorer's number, fast enough to bootstrap.

    Verified against :func:`harness.evaluate` to 1e-9 in ``__main__``. The
    scorer stays the source of truth for anything reported; this exists only so
    the inner loop of a few thousand resamples finishes in seconds.
    """
    ok = np.isfinite(pred) & np.isfinite(surprise) & np.isfinite(y)
    pred, surprise, y = pred[ok], surprise[ok], y[ok]
    if len(y) < 3:
        return float("nan")
    tss = float(((y - y.mean()) ** 2).sum())
    if tss == 0:
        return float("nan")

    def rss(*columns: np.ndarray) -> float:
        design = np.column_stack([*columns, np.ones(len(y))])
        residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        return float(residual @ residual)

    return (rss(surprise) - rss(surprise, pred)) / tss


def bootstrap_diff(
    challenger: Mapping[str, np.ndarray],
    champion: Mapping[str, np.ndarray],
    frames: Mapping[str, pd.DataFrame],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Resample events to get the sampling distribution of ``vs_champion``.

    Both models are rescored on the *same* resampled events every replicate —
    that pairing is the whole point, and it is what removes quarter difficulty
    from the comparison.

    Returns per-quarter sds plus ``se_mean``, the sd of the dev-quarter mean
    difference, which is the quantity the promotion floor is stated in.
    """
    rng = np.random.default_rng(seed)
    quarters = list(frames)
    draws = np.empty((n_boot, len(quarters)))
    paired_n = {}

    for j, quarter in enumerate(quarters):
        frame = frames[quarter]
        surprise = frame["surprise_pct"].to_numpy(dtype=float)
        y = frame["y"].to_numpy(dtype=float)
        a, b = challenger[quarter], champion[quarter]
        # Same rows for both, or the "difference" partly measures coverage
        # rather than quality. The official baselines are NaN on ~5% of events
        # in two quarters, so this is not hypothetical.
        keep = np.isfinite(a) & np.isfinite(b) & np.isfinite(surprise) & np.isfinite(y)
        a, b, surprise, y = a[keep], b[keep], surprise[keep], y[keep]
        paired_n[quarter] = int(keep.sum())
        n = len(y)
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            draws[i, j] = _delta_r2_fast(a[idx], surprise[idx], y[idx]) - _delta_r2_fast(
                b[idx], surprise[idx], y[idx]
            )

    return {
        "per_quarter_sd": {q: float(draws[:, j].std(ddof=1)) for j, q in enumerate(quarters)},
        "se_mean": float(draws.mean(axis=1).std(ddof=1)),
        "paired_n": paired_n,
        "n_boot": n_boot,
    }


def expected_best_of_k(k: int) -> float:
    """E[max] of ``k`` iid standard normals, by quadrature.

    This is the multiple-comparisons correction in its most literal form: run
    ``k`` worthless configurations and the best of them still looks this good.
    Sign consistency does not survive it — at k=40 the chance that *something*
    goes 3/3 by luck is 99.5%, so 3/3 is a gate, not evidence.
    """
    if k <= 1:
        return 0.0
    grid = np.linspace(-6.0, 8.0, 40_001)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(grid / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * grid**2) / math.sqrt(2.0 * math.pi)
    return float(np.trapezoid(grid * k * cdf ** (k - 1) * pdf, grid))


def config_count(log: pd.DataFrame | None = None) -> int:
    """Distinct configurations tried so far — the ``K`` the floor is scaled by.

    Counts distinct ``config_hash`` over every logged *run*, failures included —
    a configuration that crashed still consumed a look at the data if it had
    been scored, and pretending otherwise is how K gets quietly understated.
    Promotion decisions are excluded: they re-state a config that was already
    counted when it ran.
    """
    frame = runs() if log is None else log
    if frame.empty or "config_hash" not in frame:
        return K_PRIOR
    if "kind" in frame:
        frame = frame[frame.kind != "promotion_decision"]
    return K_PRIOR + int(frame.config_hash.nunique())


def promotion_floor(se_mean: float, k: int) -> float:
    """The gain a challenger must clear: the best draw ``K`` null configs would produce.

    Treats the ``K`` configurations as independent, which they are not — most
    are variants of each other, and correlated draws have a lower expected
    maximum. The floor is therefore conservative. That is the right direction to
    err, but it means a rejected candidate is "not proven", not "disproven".
    """
    return se_mean * max(expected_best_of_k(k), Z_SINGLE_TEST)


# --------------------------------------------------------------------------
# Running an experiment
# --------------------------------------------------------------------------


def _call_view(fn: ViewFn, events: list[dict], quarter: str) -> list[float]:
    """Call a view with ``(events, quarter)``, or ``(events)`` if that is all it takes.

    The two-argument form is the one to write: a model that fits on prior
    quarters needs to know which quarter it is predicting so it can ask
    :func:`harness.training_data` for the right slice.
    """
    try:
        n_params = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        n_params = 2
    return list(fn(events, quarter) if n_params >= 2 else fn(events))


def zblend(weights: Mapping[str, float] | None = None) -> Callable[[dict], np.ndarray]:
    """Additive blend of z-scored views — the combination that has actually worked.

    Z-scoring first is what makes the weights comparable; the scorer's affine
    invariance means the z-scoring itself is free.
    """

    def combine(views: dict[str, np.ndarray]) -> np.ndarray:
        out = np.zeros(len(next(iter(views.values()))))
        for name, values in views.items():
            w = 1.0 if weights is None else weights.get(name, 0.0)
            sd = np.nanstd(values)
            out = out + w * ((values - np.nanmean(values)) / (sd if sd else 1.0))
        return out

    return combine


@dataclass
class RunResult:
    """Everything one experiment produced, plus the record that was logged."""

    name: str
    run_id: str
    config_hash: str
    per_quarter: pd.DataFrame
    summary: pd.DataFrame
    correlations: dict[str, pd.DataFrame] = field(default_factory=dict)
    ceilings: pd.DataFrame | None = None
    runtime_s: float = 0.0
    cost_usd: float | None = None
    status: str = "ok"
    champion: str = ""
    #: ``{quarter: {view: predictions}}``, kept so a promotion decision can
    #: bootstrap without re-running the model. For an LLM view that is the
    #: difference between a free decision and paying for the run twice.
    predictions: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)

    def report(self) -> None:
        print(f"\n{self.name}   [{self.run_id[:8]}  cfg {self.config_hash[:8]}]")
        print("-" * 78)
        print(self.per_quarter.to_string(index=False, float_format=_fmt))
        print()
        print(self.summary.to_string(index=False, float_format=_fmt))
        for label, matrix in self.correlations.items():
            print(f"\ninter-view correlation ({label}, surprise projected out)")
            print(matrix.to_string(float_format=_fmt))
        if self.ceilings is not None and len(self.ceilings):
            print("\ncombination ceiling  rho / sqrt(rho_b)")
            print(self.ceilings.to_string(index=False, float_format=_fmt))
        print(f"\nruntime {self.runtime_s:.1f}s"
              + (f"   cost ${self.cost_usd:.2f}" if self.cost_usd is not None else ""))


def run(
    views: ViewFn | Mapping[str, ViewFn],
    name: str,
    *,
    combine: Callable[[dict[str, np.ndarray]], np.ndarray] | None = None,
    config: Mapping | None = None,
    quarters: Sequence[str] | None = None,
    champion: str | None = None,
    cost_usd: float | None = None,
    notes: str = "",
    final: bool = False,
    log: bool = True,
) -> RunResult:
    """Score one or more views per quarter, log the run, return the result.

    ``views`` is a single function or a name → function mapping. Each is scored
    on its own; ``combine`` additionally scores a combination of them. Nothing
    is combined implicitly — an unweighted average of views you have not looked
    at is a guess, not a result.

    ``config`` is free-form and is hashed into ``config_hash``: put every
    hyperparameter and feature-set choice in it, because that hash is what makes
    the run log a record of *how many* configurations were tried.

    ``final=True`` is the only way to touch :data:`HOLDOUT`, and it is recorded.
    """
    champion = default_champion() if champion is None else champion
    if quarters is None:
        quarters = QUARTERS if final else DEV_QUARTERS
    quarters = list(quarters)

    touching_holdout = HOLDOUT in quarters
    if touching_holdout and not final:
        raise ValueError(
            f"{HOLDOUT} is the holdout. Pass final=True to spend it — one evaluation, "
            f"after the model is chosen. Default dev quarters are {DEV_QUARTERS}."
        )
    if touching_holdout:
        print(f"!! SPENDING THE HOLDOUT: {HOLDOUT} is being evaluated. This is logged.")

    view_map: dict[str, ViewFn] = (
        dict(views) if isinstance(views, Mapping) else {"model": views}
    )
    if CHAMPION_KEY in view_map:
        raise ValueError(f"{CHAMPION_KEY!r} is reserved — the champion joins every run under it")
    config = dict(config or {})
    # ``name`` is part of the hash. Without it, two different candidate
    # functions run with the same empty config collapse to one hash, and K —
    # the whole point of the log — silently undercounts.
    config_hash = hashlib.sha256(
        json.dumps(
            {"name": name, "views": sorted(view_map), "config": config},
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    run_id = uuid.uuid4().hex
    started = time.time()

    record = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "run",
        "name": name,
        "status": "running",
        "config_hash": config_hash,
        "config": config,
        "views": sorted(view_map),
        "quarters": quarters,
        "spent_holdout": touching_holdout,
        "champion": champion,
        "cost_usd": cost_usd,
        "notes": notes,
        "git": _git_commit(),
    }

    try:
        rows, matrices, kept_preds, kept_frames = [], {}, {}, {}
        for quarter in quarters:
            frame = load(quarter).copy()
            events = harness.events_for(quarter)
            y = frame["y"].to_numpy(dtype=float)
            surprise = frame["surprise_pct"].to_numpy(dtype=float)

            preds: dict[str, np.ndarray] = {}
            for view_name, fn in view_map.items():
                values = _call_view(fn, events, quarter)
                if len(values) != len(frame):
                    raise ValueError(
                        f"view {view_name!r} returned {len(values)} predictions for "
                        f"{quarter}, expected {len(frame)}"
                    )
                preds[view_name] = np.asarray(values, dtype=float)

            if combine is not None:
                preds["combined"] = np.asarray(combine(dict(preds)), dtype=float)

            champion_values = frame[champion].to_numpy(dtype=float)
            for view_name, values in preds.items():
                frame["_pred"] = values
                scored = harness.evaluate(frame, "_pred")
                # vs_champion is scored on the rows *both* models predicted.
                # Comparing each on its own rows would fold coverage into the
                # difference, and the archive's baselines are NaN on ~5% of
                # events in 2026Q1 and 2026Q2.
                paired = frame[np.isfinite(values) & np.isfinite(champion_values)]
                vs_champion = (
                    harness.evaluate(paired, "_pred")["delta_r_squared"]
                    - harness.evaluate(paired, champion)["delta_r_squared"]
                )
                rows.append(
                    {
                        "quarter": quarter,
                        "view": view_name,
                        "n": scored["n_obs"],
                        "n_paired": len(paired),
                        "r2_surprise": scored["r_squared_surprise"],
                        "delta_r2": scored["delta_r_squared"],
                        "pct_obtainable": as_pct_obtainable(
                            scored["delta_r_squared"], scored["r_squared_surprise"]
                        ),
                        "delta_r2_imp": scored["delta_r_squared_imputed"],
                        "partial_corr": partial_corr(values, y, surprise),
                        "vs_champion": vs_champion,
                    }
                )

            # The champion joins the correlation matrix as a view: ρ_b against
            # what we actually run is the number that decides whether a
            # candidate is a new channel or a paraphrase of the current one.
            preds[CHAMPION_KEY] = frame[champion].to_numpy(dtype=float)
            kept_preds[quarter] = preds
            kept_frames[quarter] = frame
            if len(preds) > 1:
                matrices[quarter] = _correlation_matrix(preds, surprise)

        per_quarter = pd.DataFrame(rows)
        summary = _summarize(per_quarter)
        pooled = _pool(matrices)
        result = RunResult(
            name=name,
            run_id=run_id,
            config_hash=config_hash,
            per_quarter=per_quarter,
            summary=summary,
            correlations=({"pooled": pooled} if pooled is not None else {}),
            ceilings=_ceilings(summary, pooled, float(per_quarter.r2_surprise.mean())),
            runtime_s=time.time() - started,
            cost_usd=cost_usd,
            champion=champion,
            predictions=kept_preds,
            frames=kept_frames,
        )
        record.update(
            status="ok",
            runtime_s=result.runtime_s,
            per_quarter=per_quarter.to_dict("records"),
            summary=summary.to_dict("records"),
            correlations={q: m.to_dict() for q, m in matrices.items()},
            ceilings=None if result.ceilings is None else result.ceilings.to_dict("records"),
        )
        return result

    except Exception as exc:  # logged, then re-raised — a failed run still counts
        record.update(
            status="error",
            runtime_s=time.time() - started,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        if log:
            _append(record)


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------


def _correlation_matrix(preds: Mapping[str, np.ndarray], surprise: np.ndarray) -> pd.DataFrame:
    """Pairwise correlation between views, in residual space.

    Residualized against the surprise first, because two views can look
    correlated purely by both tracking the benchmark — which says nothing about
    whether they add to *each other* on the part that is actually scored.
    """
    residuals = {}
    for view_name, values in preds.items():
        ok = np.isfinite(values) & np.isfinite(surprise)
        column = np.full(len(values), np.nan)
        if ok.sum() >= 3:
            column[ok] = _residualize(values[ok], surprise[ok])
        residuals[view_name] = column
    return pd.DataFrame(residuals).corr()


def _pool(matrices: Mapping[str, pd.DataFrame]) -> pd.DataFrame | None:
    """Mean correlation matrix across quarters, averaged in Fisher-z space.

    Correlations do not average linearly — the arithmetic mean of r is biased
    toward zero, and at r ≈ 0.7 the bias is large enough to matter for a ρ_b
    that decides whether a whole family is worth pursuing.
    """
    if not matrices:
        return None
    stack = np.stack([m.to_numpy(dtype=float) for m in matrices.values()])
    z = np.arctanh(np.clip(stack, -0.999999, 0.999999))
    pooled = np.tanh(z.mean(axis=0))
    np.fill_diagonal(pooled, 1.0)
    first = next(iter(matrices.values()))
    return pd.DataFrame(pooled, index=first.index, columns=first.columns)


def _summarize(per_quarter: pd.DataFrame) -> pd.DataFrame:
    """Per-view means. Deliberately does **not** render a verdict.

    Whether a gain is real depends on the measured sd of the paired difference
    and on how many configurations have been tried — neither of which is
    visible from one run's means. :func:`decide` is the only thing that rules.
    """
    rows = []
    for view_name, group in per_quarter.groupby("view", sort=False):
        wins = int((group.vs_champion > 0).sum())
        rows.append(
            {
                "view": view_name,
                "mean_delta_r2": group.delta_r2.mean(),
                "pct_obtainable": group.pct_obtainable.mean(),
                "mean_partial_corr": group.partial_corr.mean(),
                "mean_vs_champion": group.vs_champion.mean(),
                "signs": f"{wins}/{len(group)}",
            }
        )
    return pd.DataFrame(rows)


def _ceilings(
    summary: pd.DataFrame, pooled: pd.DataFrame | None, r2_surprise: float
) -> pd.DataFrame | None:
    """Per-view: how far combining more views *like this one* could ever get.

    ``rho_b`` is the view's mean correlation with the other **independent**
    views; anything in :data:`DERIVED_VIEWS` is dropped first, since a blend
    correlates with its own inputs by construction and would inflate every
    ceiling in the table.

    ``implied_delta_r2`` converts the ceiling correlation back to the scorer's
    scale via ``ΔR² = pc² · (1 − R²_surprise)``, so it is directly comparable to
    a leaderboard number rather than being an unscaled squared correlation.
    """
    if pooled is None:
        return None
    candidates = [c for c in pooled.columns if c not in DERIVED_VIEWS and c != CHAMPION_KEY]
    if not candidates:
        return None
    has_champion = CHAMPION_KEY in pooled.columns
    rows = []
    for view_name in candidates:
        peers = [c for c in candidates if c != view_name]
        rho_b_champion = (
            float(pooled.loc[view_name, CHAMPION_KEY]) if has_champion else float("nan")
        )
        rho_b_peers = float(pooled.loc[view_name, peers].mean()) if peers else float("nan")
        # Against the champion when we have it: the live question is whether
        # adding this to what we already run buys anything, not whether a
        # hypothetical fleet of copies of it would.
        rho_b = rho_b_champion if has_champion else rho_b_peers
        match = summary.loc[summary.view == view_name, "mean_partial_corr"]
        rho = float(match.iloc[0]) if len(match) else float("nan")
        ceiling = combination_ceiling(abs(rho), rho_b)
        rows.append(
            {
                "view": view_name,
                "rho": rho,
                "rho_b_champion": rho_b_champion,
                "rho_b_peers": rho_b_peers,
                "corr_ceiling": ceiling,
                "implied_delta_r2": (
                    ceiling**2 * (1 - r2_surprise) if math.isfinite(ceiling) else float("inf")
                ),
                # ceiling**2 IS the fraction of obtainable — see obtainable()
                "implied_pct": ceiling**2 if math.isfinite(ceiling) else float("inf"),
            }
        )
    return pd.DataFrame(rows)


def _short(label: str) -> str:
    return label.split()[0].lower()


def _fmt(v: float) -> str:
    return f"{v:+.4f}" if isinstance(v, (int, float)) and math.isfinite(v) else str(v)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip() or None
    except Exception:
        return None


def _append(record: Mapping) -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def decide(
    result: RunResult,
    view: str = "model",
    *,
    n_boot: int = 2000,
    k: int | None = None,
    seed: int = 0,
    log: bool = True,
) -> dict:
    """Should this view replace the champion? Measured, logged, and K-aware.

    Three numbers decide it, and all three are recorded:

    ``se_mean``   sd of the mean paired difference, bootstrapped over events.
                  Paired, so quarter difficulty cancels — this is much smaller
                  than the spread in either model's ΔR² *levels*, and it is
                  specific to this candidate's correlation with the champion.
    ``k``         distinct configurations tried, read off the run log.
    ``floor``     ``se_mean × E[max of k null draws]`` — what the luckiest of
                  ``k`` worthless candidates would score.

    Sign consistency is reported but carries almost no weight: at k=40 the
    probability that *something* goes 3/3 on three dev quarters is 99.5%.
    """
    if not result.predictions:
        raise ValueError("run() kept no predictions — cannot decide without them")
    quarters = list(result.frames)
    challenger = {q: result.predictions[q][view] for q in quarters}
    incumbent = {q: result.predictions[q][CHAMPION_KEY] for q in quarters}

    boot = bootstrap_diff(challenger, incumbent, result.frames, n_boot=n_boot, seed=seed)
    k = config_count() if k is None else k
    floor = promotion_floor(boot["se_mean"], k)

    rows = result.per_quarter[result.per_quarter.view == view]
    if rows.empty:
        raise ValueError(f"no scored rows for view {view!r} — views are {sorted(result.summary.view)}")
    mean_gain = float(rows.vs_champion.mean())
    wins = int((rows.vs_champion > 0).sum())
    ship = mean_gain > floor

    decision = {
        "run_id": result.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "promotion_decision",
        "name": result.name,
        "view": view,
        "champion": result.champion,
        "config_hash": result.config_hash,
        "quarters": quarters,
        "mean_vs_champion": mean_gain,
        "per_quarter_vs_champion": dict(zip(rows.quarter, rows.vs_champion, strict=True)),
        "mean_delta_r2": float(rows.delta_r2.mean()),
        "mean_pct_obtainable": float(rows.pct_obtainable.mean()),
        "mean_delta_r2_imputed": float(rows.delta_r2_imp.mean()),
        "signs": f"{wins}/{len(rows)}",
        "sign_gate_passed": wins == len(rows),
        "se_mean": boot["se_mean"],
        "per_quarter_sd": boot["per_quarter_sd"],
        "paired_n": boot["paired_n"],
        "n_boot": boot["n_boot"],
        "k_configs": k,
        "best_of_k_multiplier": max(expected_best_of_k(k), Z_SINGLE_TEST),
        "floor": floor,
        "ship": ship,
    }
    if log:
        _append(decision)

    print(
        f"\npromotion: {result.name} [{view}] vs {result.champion}\n"
        f"  mean vs champion  {mean_gain:+.4f}   signs {wins}/{len(rows)}\n"
        f"  bootstrapped se   {boot['se_mean']:.4f}  ({n_boot} resamples, paired)\n"
        f"  configs tried K   {k}  ->  best-of-K multiplier "
        f"{decision['best_of_k_multiplier']:.2f}\n"
        f"  floor             {floor:+.4f}\n"
        f"  own score         {rows.delta_r2.mean():+.4f} = "
        f"{rows.pct_obtainable.mean():.1%} of obtainable (0.9413)\n"
        f"  => {'SHIP' if ship else 'DO NOT SHIP'}"
    )
    return decision


def runs(status: str | None = None) -> pd.DataFrame:
    """The run log as a frame — the audit trail for "how many things did we try?"."""
    if not RUN_LOG.exists():
        return pd.DataFrame()
    records = [json.loads(line) for line in RUN_LOG.open() if line.strip()]
    frame = pd.DataFrame(records)
    return frame if status is None else frame[frame.status == status]


# --------------------------------------------------------------------------
# Self-test — the module's own correctness gate
# --------------------------------------------------------------------------

if __name__ == "__main__":
    harness.compare()

    null = run(lambda events: [0.5] * len(events), "constant 0.5 (must be exactly 0)", log=False)
    null.report()
    assert abs(null.per_quarter.delta_r2).max() < 1e-12, "constant prediction must score 0"

    # ΔR² = pc² · (1 − R²_surprise). If this identity fails, one of the two
    # numbers this module reports is not what it claims to be.
    for quarter in DEV_QUARTERS:
        frame = load(quarter)
        pred = frame[GEMINI].to_numpy(dtype=float)
        y = frame["y"].to_numpy(dtype=float)
        surprise = frame["surprise_pct"].to_numpy(dtype=float)
        scored = harness.evaluate(frame, GEMINI)
        implied = partial_corr(pred, y, surprise) ** 2 * (1 - scored["r_squared_surprise"])
        assert abs(implied - scored["delta_r_squared"]) < 1e-9, (
            f"{quarter}: partial corr and delta_r2 disagree "
            f"({implied:.6f} vs {scored['delta_r_squared']:.6f})"
        )
    print(f"\nidentity check passed on {DEV_QUARTERS}: delta_r2 == pc^2 * (1 - r2_surprise)")

    # The bootstrap's fast ΔR² must agree with the scorer, or every promotion
    # decision is made against a number the competition does not compute.
    for quarter in DEV_QUARTERS:
        frame = load(quarter)
        fast = _delta_r2_fast(
            frame[GEMINI].to_numpy(dtype=float),
            frame["surprise_pct"].to_numpy(dtype=float),
            frame["y"].to_numpy(dtype=float),
        )
        exact = harness.evaluate(frame, GEMINI)["delta_r_squared"]
        assert abs(fast - exact) < 1e-9, f"{quarter}: fast {fast} vs scorer {exact}"
    print("fast delta_r2 agrees with the scorer to 1e-9 — safe to bootstrap with")

    # Perfect foresight: feed y back in as the prediction. R^2 must be exactly 1
    # and delta_r2 exactly 1 - r2_surprise. This is the cleanest end-to-end check
    # available — it exercises ranking, alignment, the benchmark fit and the
    # subtraction in one line, and if it drifts, every number this module has
    # ever produced is suspect.
    for quarter in DEV_QUARTERS:
        frame = load(quarter).copy()
        frame["_perfect"] = frame["y"]
        scored = harness.evaluate(frame, "_perfect")
        assert abs(scored["r_squared"] - 1.0) < 1e-9, f"{quarter}: perfect R2 {scored['r_squared']}"
        assert (
            abs(scored["delta_r_squared"] - (1 - scored["r_squared_surprise"])) < 1e-9
        ), quarter
    print("perfect foresight scores exactly 1 - r2_surprise (mean ceiling ~0.941)")

    for k in (1, 3, 10, 40, 100):
        print(f"  K={k:<4} best-of-K multiplier {max(expected_best_of_k(k), Z_SINGLE_TEST):.3f}")

    # Two views, one of them deliberately redundant, to exercise the matrix and
    # the ceiling arithmetic.
    def gemini_view(events, quarter):
        return load(quarter)[GEMINI].tolist()

    def gpt_view(events, quarter):
        return load(quarter)[GPT].tolist()

    # log=False throughout the self-test: the run log is the audit trail for how
    # many configurations were tried to reach a headline number, and a module
    # smoke test is not one of them.
    both = run(
        {"gemini": gemini_view, "gpt": gpt_view},
        "official baselines as two views",
        combine=zblend(),
        config={"note": "known to correlate ~0.8 — the matrix should show it"},
        log=False,
    )
    both.report()

    # A real candidate: a different vendor's read of the same ten facts. The sd
    # of the paired difference is measured here, not assumed — and it is
    # specific to this pair, because it depends on how correlated the two are.
    decide(both, "gemini", n_boot=2000, log=False)

    # The null that needs no invention: replaying the champion's own column.
    # Gain is exactly zero, so any positive floor must reject it.
    def replay_champion(events, quarter):
        return load(quarter)[default_champion(quiet=True)].tolist()

    replayed = run(replay_champion, "replay of the champion — gain must be exactly zero", log=False)
    assert abs(replayed.per_quarter.vs_champion).max() < 1e-12, "replay must tie with itself"
    null_decision = decide(replayed, "model", n_boot=500, log=False)
    assert not null_decision["ship"], "a zero-gain candidate must not ship"

    try:
        run(lambda events: [0.5] * len(events), "holdout guard", quarters=[HOLDOUT], log=False)
    except ValueError as exc:
        print(f"\nholdout guard works: {exc}")
